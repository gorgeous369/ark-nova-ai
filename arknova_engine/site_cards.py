"""Fan Hub card dataset fetcher/parser.

This module scrapes public Ark Nova card pages from:
- https://ark-nova.ender-wiggin.com/cards
- https://ark-nova.ender-wiggin.com/card/<id>

The output is a normalized JSON-friendly structure used by local tooling.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import re
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "https://ark-nova.ender-wiggin.com"
USER_AGENT = "ark-nova-ai/0.1 (local data fetcher)"

CARD_TAG_RE = re.compile(r'<div id="card-([^"]+)" data-id="([^"]+)" class="([^"]+)"[^>]*>')
DIV_TOKEN_RE = re.compile(r"<div\b|</div>")
EFFECT_RE = re.compile(r'<li class="effect-([a-zA-Z0-9_-]+)">(.*?)</li>', flags=re.S)
ENCLOSURE_ICON_RE = re.compile(
    r'<div class="arknova-icon ([^"]*icon-enclosure-[^"]*)"[^>]*>(.*?)</div>',
    flags=re.S,
)
BADGE_ICON_RE = re.compile(r'badge-icon" data-type="([^"]+)"')
ICON_DIV_RE = re.compile(r'<div class="([^"]*arknova-icon[^"]*)"[^>]*>(.*?)</div>', flags=re.S)
EFFECT_FALLBACK_RE = re.compile(
    r'<(?:li|div|span)[^>]*class="[^"]*effect-([a-zA-Z0-9_-]+)[^"]*"[^>]*>(.*?)</(?:li|div|span)>',
    flags=re.S,
)
NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', flags=re.S)
CHUNK_SRC_RE = re.compile(r'<script[^>]+src="([^"]*_next/static/chunks/[^"]+\.js)"', flags=re.S)
SPONSOR_ENTRY_RE = re.compile(
    r'\{id:"(?P<id>2\d{2})".*?name:"(?P<name>[^"]+)".*?strength:(?P<strength>-?\d+).*?'
    r'requirements:\[(?P<requirements>.*?)\],\s*tags:\[(?P<tags>.*?)\],\s*effects:\[(?P<effects>.*?)\],\s*'
    r'reputation:(?P<reputation>-?\d+),\s*appeal:(?P<appeal>-?\d+),\s*conservationPoint:(?P<conservation>-?\d+),\s*'
    r'source:[^}]+\}',
    flags=re.S,
)
QQ_TOKEN_RE = re.compile(r'Qq\.([A-Za-z0-9_]+)')
SPONSOR_EFFECT_ENTRY_RE = re.compile(
    r'\{[^{}]*effectType:o\.A\.([A-Z]+)[^{}]*effectDesc:"([^"]+)"[^{}]*\}',
    flags=re.S,
)


def _strip_markup(fragment: str) -> str:
    text = re.sub(r"<!--.*?-->", " ", fragment, flags=re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _extract_first_text(body: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, body, flags=re.S)
    if not match:
        return None
    text = _strip_markup(match.group(1))
    return text or None


def _extract_first_int(body: str, pattern: str) -> Optional[int]:
    text = _extract_first_text(body, pattern)
    if text is None:
        return None
    digits = re.search(r"-?\d+", text)
    if not digits:
        return None
    return int(digits.group(0))


def _card_type(card_class: str) -> str:
    if "animal-card" in card_class:
        return "animal"
    if "sponsor-card" in card_class:
        return "sponsor"
    if "project-card" in card_class:
        return "conservation_project"
    if "scoring-card" in card_class:
        return "final_scoring"
    return "unknown"


def _extract_enclosure_icons(block_html: str) -> List[tuple[str, Optional[int]]]:
    icons: List[tuple[str, Optional[int]]] = []
    for classes_raw, body_raw in ENCLOSURE_ICON_RE.findall(block_html):
        icon_class = None
        for token in classes_raw.split():
            if token.startswith("icon-enclosure-"):
                icon_class = token
                break
        if not icon_class:
            continue
        size_text = _strip_markup(body_raw)
        size_match = re.search(r"-?\d+", size_text)
        enclosure_size = int(size_match.group(0)) if size_match else None
        icons.append((icon_class, enclosure_size))
    return icons


def _extract_primary_enclosure_icon(block_html: str) -> tuple[Optional[str], Optional[int]]:
    for icon_class, enclosure_size in _extract_enclosure_icons(block_html):
        if icon_class.startswith("icon-enclosure-regular") or icon_class.startswith("icon-enclosure-forbidden"):
            return icon_class, enclosure_size
    return None, None


def _extract_special_enclosure_sizes(block_html: str) -> Dict[str, int]:
    special_sizes: Dict[str, int] = {}
    for icon_class, enclosure_size in _extract_enclosure_icons(block_html):
        if icon_class == "icon-enclosure-special-reptile-house":
            special_sizes["reptile_house_size"] = enclosure_size if enclosure_size is not None else 0
        elif icon_class == "icon-enclosure-special-large-bird-aviary":
            special_sizes["large_bird_aviary_size"] = enclosure_size if enclosure_size is not None else 0
    return special_sizes


def _enclosure_adjacency_requirements(icon_class: Optional[str]) -> tuple[int, int]:
    if not icon_class or not icon_class.startswith("icon-enclosure-"):
        return 0, 0

    parts = icon_class[len("icon-enclosure-") :].split("-")
    if not parts or parts[0] not in {"regular", "forbidden"}:
        return 0, 0

    rock = sum(1 for token in parts[1:] if token == "rock")
    water = sum(1 for token in parts[1:] if token == "water")
    return rock, water


def _fallback_card_number(data_id: str) -> Optional[int]:
    match = re.search(r"[A-Z](\d+)_", data_id)
    if not match:
        return None
    return int(match.group(1))


def _extract_div_block(page_html: str, start_index: int) -> Optional[str]:
    depth = 0
    saw_start = False
    for token in DIV_TOKEN_RE.finditer(page_html, start_index):
        if token.group(0) == "<div":
            depth += 1
            saw_start = True
        else:
            depth -= 1
            if saw_start and depth == 0:
                return page_html[start_index : token.end()]
    return None


def _extract_first_div_block_by_class(body: str, class_name: str) -> Optional[str]:
    escaped = re.escape(class_name)
    match = re.search(rf'<div class="[^"]*\b{escaped}\b[^"]*"[^>]*>', body)
    if not match:
        return None
    return _extract_div_block(body, match.start())


def _normalize_icon_name(icon_name: str) -> str:
    key = icon_name.strip().lower()
    aliases = {
        "americas": "america",
        "oceania": "australia",
    }
    return aliases.get(key, key)


def _extract_required_icons_from_prerequisites(block_html: str) -> Dict[str, int]:
    prerequisites_block = _extract_first_div_block_by_class(block_html, "zoo-card-prerequisites")
    if not prerequisites_block:
        return {}
    counter: Counter[str] = Counter()
    for icon_name in BADGE_ICON_RE.findall(prerequisites_block):
        normalized = _normalize_icon_name(icon_name)
        if normalized:
            counter[normalized] += 1
    # Some sponsor cards encode prerequisites with arknova-icon classes instead of badge-icon data-type.
    for classes_raw, body_raw in ICON_DIV_RE.findall(prerequisites_block):
        tokens = classes_raw.split()
        icon_token = next((token for token in tokens if token.startswith("icon-")), "")
        if not icon_token:
            continue
        icon_name = icon_token[len("icon-") :].strip().lower().replace("-", "_")
        if not icon_name:
            continue
        icon_name = _normalize_icon_name(icon_name)
        amount_match = re.search(r"-?\d+", _strip_markup(body_raw))
        amount = int(amount_match.group(0)) if amount_match else 1
        if amount <= 0:
            continue
        counter[icon_name] += amount
    return dict(sorted(counter.items()))


def _extract_icon_value_from_block(body: str, icon_prefix: str) -> Optional[int]:
    for classes_raw, value_raw in ICON_DIV_RE.findall(body):
        tokens = classes_raw.split()
        if not any(token == icon_prefix or token.startswith(icon_prefix + "-") for token in tokens):
            continue
        value_text = _strip_markup(value_raw)
        match = re.search(r"-?\d+", value_text)
        if match:
            return int(match.group(0))
    return None


def _extract_effects(block_html: str) -> List[Dict[str, str]]:
    effects: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for matcher in (EFFECT_RE, EFFECT_FALLBACK_RE):
        for kind_raw, text_raw in matcher.findall(block_html):
            kind = str(kind_raw).strip().lower()
            text = _strip_markup(text_raw)
            if not kind or not text:
                continue
            key = (kind, text)
            if key in seen:
                continue
            seen.add(key)
            effects.append({"kind": kind, "text": text})
    return effects


def _canonical_requirement_token(raw_token: str) -> str:
    token = raw_token.strip()
    if not token:
        return ""
    token_lower = token.lower()
    explicit = {
        "science": "science",
        "appeal": "appeal",
        "conservationpoint": "conservation",
        "conservation": "conservation",
        "money": "money",
        "x": "x",
        "xtoken": "x_token",
        "sponsorsii": "sponsorsii",
        "associationii": "associationii",
        "buildii": "buildii",
        "cardsii": "cardsii",
        "animalsii": "animalsii",
        "africa": "africa",
        "europe": "europe",
        "asia": "asia",
        "america": "america",
        "americas": "america",
        "australia": "australia",
        "oceania": "australia",
        "bird": "bird",
        "reptile": "reptile",
        "primate": "primate",
        "predator": "predator",
        "herbivore": "herbivore",
    }
    if token_lower in explicit:
        return explicit[token_lower]
    return _normalize_icon_name(token_lower)


def _extract_sponsor_texts_from_next_data(page_html: str) -> Dict[str, str]:
    match = NEXT_DATA_RE.search(page_html)
    if not match:
        return {}
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return {}
    try:
        i18n_store = payload["props"]["pageProps"]["_nextI18Next"]["initialI18nStore"]
    except (TypeError, KeyError):
        return {}
    locale_payload: Dict[str, Any]
    if isinstance(i18n_store, dict) and "en" in i18n_store and isinstance(i18n_store["en"], dict):
        locale_payload = i18n_store["en"]
    elif isinstance(i18n_store, dict):
        first = next((value for value in i18n_store.values() if isinstance(value, dict)), {})
        locale_payload = first if isinstance(first, dict) else {}
    else:
        return {}
    common = locale_payload.get("common")
    if not isinstance(common, dict):
        return {}
    sponsors = common.get("sponsors")
    if not isinstance(sponsors, dict):
        return {}
    return {str(key): str(value) for key, value in sponsors.items() if isinstance(value, str)}


def _extract_chunk_urls(page_html: str) -> List[str]:
    urls: List[str] = []
    seen: set[str] = set()
    for raw_url in CHUNK_SRC_RE.findall(page_html):
        if raw_url.startswith("http://") or raw_url.startswith("https://"):
            url = raw_url
        elif raw_url.startswith("/"):
            url = f"{BASE_URL}{raw_url}"
        else:
            url = f"{BASE_URL}/{raw_url}"
        if url in seen:
            continue
        seen.add(url)
        urls.append(url)
    return urls


def _parse_sponsor_chunk_records(chunk_text: str, sponsor_texts: Dict[str, str]) -> Dict[int, Dict[str, object]]:
    parsed: Dict[int, Dict[str, object]] = {}
    for match in SPONSOR_ENTRY_RE.finditer(chunk_text):
        number = int(match.group("id"))
        title = str(match.group("name")).strip()
        level = int(match.group("strength"))

        requirements_counter: Counter[str] = Counter()
        for token in QQ_TOKEN_RE.findall(match.group("requirements")):
            normalized = _canonical_requirement_token(token)
            if normalized:
                requirements_counter[normalized] += 1

        tags: List[str] = []
        for token in QQ_TOKEN_RE.findall(match.group("tags")):
            normalized = _canonical_requirement_token(token)
            if not normalized:
                continue
            tag = normalized[0].upper() + normalized[1:]
            if tag not in tags:
                tags.append(tag)

        effects: List[Dict[str, str]] = []
        for kind_raw, desc_key_raw in SPONSOR_EFFECT_ENTRY_RE.findall(match.group("effects")):
            kind = str(kind_raw).strip().lower()
            desc_key = str(desc_key_raw).strip()
            if desc_key.startswith("sponsors."):
                lookup_key = desc_key.split(".", 1)[1]
            else:
                lookup_key = desc_key
            text = sponsor_texts.get(lookup_key) or sponsor_texts.get(desc_key) or desc_key
            text = _strip_markup(text)
            if not kind or not text:
                continue
            effects.append({"kind": kind, "text": text})

        record: Dict[str, object] = {
            "number": number,
            "title": title,
            "level": level,
        }
        if requirements_counter:
            record["required_icons"] = dict(sorted(requirements_counter.items()))
        if tags:
            record["badges"] = tags
        if effects:
            record["effects"] = effects
        reputation = int(match.group("reputation"))
        appeal = int(match.group("appeal"))
        conservation = int(match.group("conservation"))
        if reputation != 0:
            record["reputation"] = reputation
        if appeal != 0:
            record["appeal"] = appeal
        if conservation != 0:
            record["conservation"] = conservation
        parsed[number] = record
    return parsed


def fetch_sponsor_fallback_records() -> Dict[int, Dict[str, object]]:
    """Fetch structured sponsor metadata from frontend chunk payloads as fallback."""

    page_html = fetch_html(f"{BASE_URL}/card/206")
    if not page_html:
        return {}

    sponsor_texts = _extract_sponsor_texts_from_next_data(page_html)
    chunk_urls = _extract_chunk_urls(page_html)
    for chunk_url in chunk_urls:
        chunk_text = fetch_html(chunk_url)
        if not chunk_text:
            continue
        if "S201_ScienceLab" not in chunk_text and "S206_MedicalBreakthrough" not in chunk_text:
            continue
        parsed = _parse_sponsor_chunk_records(chunk_text, sponsor_texts)
        if parsed:
            return parsed
    return {}


def _card_detail_score(card: Dict[str, object]) -> int:
    score = 0
    for key in (
        "title",
        "subtitle",
        "cost",
        "level",
        "appeal",
        "reputation",
        "conservation",
        "required_icons",
        "ability_title",
        "ability_text",
        "effects",
        "description",
    ):
        value = card.get(key)
        if value in (None, "", [], {}):
            continue
        score += 1
    return score


def _merge_card_records(existing: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    # Keep the richer record as base and fill missing fields from the other source.
    existing_score = _card_detail_score(existing)
    incoming_score = _card_detail_score(incoming)
    if incoming_score > existing_score:
        primary, secondary = incoming, existing
    else:
        primary, secondary = existing, incoming

    merged: Dict[str, object] = dict(primary)
    for key, value in secondary.items():
        if key not in merged or merged[key] in (None, "", [], {}):
            merged[key] = value
    return merged


def _extract_animal_badges(block_html: str) -> List[str]:
    # Card icon badges are rendered in the top-right area; prerequisites are top-left.
    top_right_block = _extract_first_div_block_by_class(block_html, "ark-card-top-right")
    if top_right_block:
        return [name for name in BADGE_ICON_RE.findall(top_right_block) if name.strip()]
    # Fallback for simplified HTML fixtures that omit top-right wrappers.
    return [name for name in BADGE_ICON_RE.findall(block_html) if name.strip()]


def parse_first_card_from_page(page_html: str, source_url: str) -> Optional[Dict[str, object]]:
    """Parse the first rendered card from a card detail page HTML."""

    match = CARD_TAG_RE.search(page_html)
    if not match:
        return None

    block = _extract_div_block(page_html, match.start())
    if block is None:
        return None
    return parse_card_block(match.group(2), match.group(3), block, source_url)


def parse_best_card_from_page(
    page_html: str,
    source_url: str,
    expected_card_id: Optional[int] = None,
) -> Optional[Dict[str, object]]:
    """Parse and select the best matching card block from a detail page."""

    candidates: List[Dict[str, object]] = []
    for match in CARD_TAG_RE.finditer(page_html):
        block = _extract_div_block(page_html, match.start())
        if block is None:
            continue
        card = parse_card_block(match.group(2), match.group(3), block, source_url)
        if not card:
            continue
        candidates.append(card)

    if not candidates:
        return None

    def rank(item: Dict[str, object]) -> tuple[int, int]:
        number = item.get("number")
        id_match = 1 if expected_card_id is not None and isinstance(number, int) and number == expected_card_id else 0
        return (id_match, _card_detail_score(item))

    return max(candidates, key=rank)


def parse_all_cards_from_cards_page(page_html: str, source_url: str) -> List[Dict[str, object]]:
    """Parse all rendered cards from the /cards page HTML."""

    cards: List[Dict[str, object]] = []
    for match in CARD_TAG_RE.finditer(page_html):
        block = _extract_div_block(page_html, match.start())
        if block is None:
            continue
        cards.append(parse_card_block(match.group(2), match.group(3), block, source_url))
    return cards


def parse_card_block(
    data_id: str,
    card_class: str,
    block_html: str,
    source_url: str,
) -> Dict[str, object]:
    """Normalize one card HTML block into a structured dict."""

    card: Dict[str, object] = {
        "data_id": data_id,
        "type": _card_type(card_class),
        "class_name": card_class,
        "source_url": source_url,
    }

    number = _extract_first_int(block_html, r'<div class="ark-card-number[^"]*"[^>]*>(.*?)</div>')
    if number is None:
        number = _fallback_card_number(data_id)
    card["number"] = number

    title = _extract_first_text(block_html, r'<div class="ark-card-title[^"]*"[^>]*>(.*?)</div>')
    if title:
        card["title"] = title

    subtitle = _extract_first_text(block_html, r'<div class="ark-card-subtitle[^"]*"[^>]*>(.*?)</div>')
    if subtitle:
        card["subtitle"] = subtitle

    if card["type"] == "animal":
        top_left = _extract_first_div_block_by_class(block_html, "ark-card-top-left") or block_html
        cost = _extract_first_int(block_html, r'<div class="arknova-icon icon-money original-cost">(.*?)</div>')
        if cost is None:
            cost = _extract_icon_value_from_block(top_left, "icon-money")
        if cost is not None:
            card["cost"] = cost

        enclosure_icon, enclosure_size = _extract_primary_enclosure_icon(block_html)
        if enclosure_size is not None:
            card["enclosure_size"] = enclosure_size
        required_rock_adjacency, required_water_adjacency = _enclosure_adjacency_requirements(
            enclosure_icon
        )
        if required_rock_adjacency > 0:
            card["required_rock_adjacency"] = required_rock_adjacency
        if required_water_adjacency > 0:
            card["required_water_adjacency"] = required_water_adjacency
        special_enclosure_sizes = _extract_special_enclosure_sizes(block_html)
        if "reptile_house_size" in special_enclosure_sizes:
            card["reptile_house_size"] = special_enclosure_sizes["reptile_house_size"]
        if "large_bird_aviary_size" in special_enclosure_sizes:
            card["large_bird_aviary_size"] = special_enclosure_sizes["large_bird_aviary_size"]

        appeal = _extract_first_int(block_html, r'<div class="zoo-card-bonus appeal">(.*?)</div>')
        if appeal is not None:
            card["appeal"] = appeal
        reputation = _extract_first_int(block_html, r'<div class="zoo-card-bonus reputation">(.*?)</div>')
        if reputation is not None:
            card["reputation"] = reputation
        conservation = _extract_first_int(block_html, r'<div class="zoo-card-bonus conservation">(.*?)</div>')
        if conservation is not None:
            card["conservation"] = conservation

        required_icons = _extract_required_icons_from_prerequisites(block_html)
        if required_icons:
            card["required_icons"] = required_icons

        badges = _extract_animal_badges(block_html)
        if badges:
            card["badges"] = badges

        ability_title = _extract_first_text(block_html, r'<h6 class="animal-ability-title">(.*?)</h6>')
        if ability_title:
            card["ability_title"] = ability_title

        ability_text = _extract_first_text(block_html, r'<div class="animal-ability-desc">(.*?)</div>')
        if ability_text:
            card["ability_text"] = ability_text

    elif card["type"] == "sponsor":
        top_left = _extract_first_div_block_by_class(block_html, "ark-card-top-left") or block_html
        level = _extract_first_int(block_html, r'<div class="arknova-icon icon-level[^"]*"[^>]*>(.*?)</div>')
        if level is None:
            level = _extract_icon_value_from_block(top_left, "icon-level")
        if level is not None:
            card["level"] = level

        cost = _extract_first_int(block_html, r'<div class="arknova-icon icon-money original-cost">(.*?)</div>')
        if cost is None:
            cost = _extract_icon_value_from_block(top_left, "icon-money")
        if cost is not None:
            card["cost"] = cost

        appeal = _extract_first_int(block_html, r'<div class="zoo-card-bonus appeal">(.*?)</div>')
        if appeal is not None:
            card["appeal"] = appeal
        reputation = _extract_first_int(block_html, r'<div class="zoo-card-bonus reputation">(.*?)</div>')
        if reputation is not None:
            card["reputation"] = reputation
        conservation = _extract_first_int(block_html, r'<div class="zoo-card-bonus conservation">(.*?)</div>')
        if conservation is not None:
            card["conservation"] = conservation

        required_icons = _extract_required_icons_from_prerequisites(block_html)
        if required_icons:
            card["required_icons"] = required_icons

        ability_title = _extract_first_text(block_html, r'<h6 class="sponsor-ability-title">(.*?)</h6>')
        if ability_title is None:
            ability_title = _extract_first_text(block_html, r'<h6 class="animal-ability-title">(.*?)</h6>')
        if ability_title:
            card["ability_title"] = ability_title

        ability_text = _extract_first_text(block_html, r'<div class="sponsor-ability-desc">(.*?)</div>')
        if ability_text is None:
            ability_text = _extract_first_text(block_html, r'<div class="animal-ability-desc">(.*?)</div>')
        if ability_text:
            card["ability_text"] = ability_text

        effects = _extract_effects(block_html)
        if effects:
            card["effects"] = effects

    elif card["type"] in {"conservation_project", "final_scoring"}:
        description = _extract_first_text(block_html, r'<div class="project-card-description[^"]*"[^>]*>(.*?)</div>')
        if description:
            card["description"] = description

    return card


def fetch_html(url: str, timeout: float = 15.0) -> Optional[str]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.read().decode("utf-8", "ignore")
    except HTTPError as exc:
        if exc.code == 404:
            return None
        raise
    except URLError:
        return None


def fetch_zoo_cards(max_card_id: int = 1300, workers: int = 32) -> List[Dict[str, object]]:
    """Fetch Sponsor/Animal cards from /card/<id> pages."""

    def worker(card_id: int) -> Optional[Dict[str, object]]:
        source_url = f"{BASE_URL}/card/{card_id}"
        page_html = fetch_html(source_url)
        if not page_html:
            return None

        card = parse_best_card_from_page(page_html, source_url=source_url, expected_card_id=card_id)
        if not card:
            return None
        card_type = card.get("type")
        if card_type not in {"animal", "sponsor"}:
            return None

        card["site_card_id"] = card_id
        return card

    cards: List[Dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(worker, card_id) for card_id in range(1, max_card_id + 1)]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                cards.append(result)
    return cards


def build_cards_dataset(max_card_id: int = 1300, workers: int = 32) -> Dict[str, object]:
    """Build a merged dataset from the public card pages."""

    cards_page_url = f"{BASE_URL}/cards"
    cards_page_html = fetch_html(cards_page_url)
    if cards_page_html is None:
        raise RuntimeError(f"Unable to fetch {cards_page_url}")

    project_and_final_cards = parse_all_cards_from_cards_page(cards_page_html, source_url=cards_page_url)
    zoo_cards = fetch_zoo_cards(max_card_id=max_card_id, workers=workers)
    sponsor_fallback_by_number = fetch_sponsor_fallback_records()

    merged_by_data_id: Dict[str, Dict[str, object]] = {}
    for card in project_and_final_cards + zoo_cards:
        data_id = str(card["data_id"])
        current = merged_by_data_id.get(data_id)
        if current is None:
            merged_by_data_id[data_id] = card
            continue
        merged_by_data_id[data_id] = _merge_card_records(current, card)

    if sponsor_fallback_by_number:
        for data_id, card in list(merged_by_data_id.items()):
            if str(card.get("type")) != "sponsor":
                continue
            number = card.get("number")
            if not isinstance(number, int):
                continue
            fallback = sponsor_fallback_by_number.get(number)
            if not fallback:
                continue
            merged_by_data_id[data_id] = _merge_card_records(card, fallback)

    cards = list(merged_by_data_id.values())
    cards.sort(key=lambda item: (str(item.get("type", "")), int(item.get("number") or 0), str(item["data_id"])))

    type_counter = Counter(str(item.get("type", "unknown")) for item in cards)
    dataset = {
        "source": BASE_URL,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "max_card_id_scanned": max_card_id,
        "cards": cards,
        "stats": {
            "total": len(cards),
            "by_type": dict(sorted(type_counter.items())),
        },
    }
    return dataset


def write_dataset(path: Path, dataset: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=True, indent=2, sort_keys=False) + "\n", encoding="utf-8")

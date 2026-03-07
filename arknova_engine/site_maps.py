"""Public map dataset fetcher/parser.

This module scrapes map metadata from:
- https://ark-nova.ender-wiggin.com/maps
- maps page JS bundle referenced by that page
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Dict, List, Optional
from urllib.parse import urljoin
from urllib.request import Request, urlopen

from arknova_engine.site_cards import BASE_URL, fetch_html


MAPS_PAGE_URL = f"{BASE_URL}/maps"
MAP_IMAGE_BASE_URL = f"{BASE_URL}/img/maps"

MAPS_CHUNK_RE = re.compile(r'src="(/_next/static/chunks/pages/maps-[^"]+\.js)"')
MAP_ENTRY_RE = re.compile(
    r'\{id:"([^"]+)",name:"([^"]+)",image:"([^"]+)",cardSource:r\.t\.([A-Z_]+),description:\[([^\]]*)\]\}'
)
NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>', flags=re.S)


CARD_SOURCE_LABELS = {
    "BASE": "Base",
    "MARINE_WORLD": "Marine World",
    "PROMO": "Promo",
    "FAN_MADE": "Fan Made",
    "ALTERNATIVE": "Alternative",
    "BEGINNER": "Beginner",
}


def resolve_maps_translation(translations: Dict[str, str], key: str, max_hops: int = 3) -> str:
    """Resolve translation key/value for maps payload.

    The source uses keys like "maps.name1", while translation dictionary keys are "name1".
    Some translation values can also point to another maps key (e.g. "maps.desc1-2").
    """

    current = key
    for _ in range(max_hops):
        normalized = current[5:] if current.startswith("maps.") else current
        value = translations.get(normalized)
        if value is None:
            return current
        if value.startswith("maps."):
            current = value
            continue
        return value
    return current


def _extract_map_chunk_url(maps_page_html: str) -> str:
    match = MAPS_CHUNK_RE.search(maps_page_html)
    if not match:
        raise RuntimeError("Cannot find maps page JS chunk URL.")
    return urljoin(BASE_URL, match.group(1))


def parse_map_entries_from_chunk(chunk_js: str) -> List[Dict[str, object]]:
    """Parse map entries from the maps page JS bundle."""

    entries: List[Dict[str, object]] = []
    for map_id, name_key, image_name, card_source_key, desc_blob in MAP_ENTRY_RE.findall(chunk_js):
        description_keys = re.findall(r'"([^"]+)"', desc_blob)
        entry = {
            "id": map_id,
            "name_key": name_key,
            "image_name": image_name,
            "card_source_key": card_source_key,
            "card_source": CARD_SOURCE_LABELS.get(card_source_key, card_source_key),
            "description_keys": description_keys,
            "image_url": f"{MAP_IMAGE_BASE_URL}/{image_name}.jpg",
        }
        entries.append(entry)

    # Keep deterministic order matching bundle order and avoid accidental duplicates.
    unique_by_id: Dict[str, Dict[str, object]] = {}
    for entry in entries:
        unique_by_id[str(entry["id"])] = entry
    return list(unique_by_id.values())


def parse_maps_translations_from_page(maps_page_html: str) -> Dict[str, str]:
    """Extract maps translations from __NEXT_DATA__ on the maps page."""

    match = NEXT_DATA_RE.search(maps_page_html)
    if not match:
        raise RuntimeError("Cannot find __NEXT_DATA__ payload on maps page.")

    payload = json.loads(match.group(1))
    common = payload["props"]["pageProps"]["_nextI18Next"]["initialI18nStore"]["en"]["common"]
    maps_translations = common.get("maps")
    if not isinstance(maps_translations, dict):
        raise RuntimeError("Cannot find maps translations in __NEXT_DATA__ payload.")

    out: Dict[str, str] = {}
    for key, value in maps_translations.items():
        if isinstance(key, str) and isinstance(value, str):
            out[key] = value
    return out


def build_maps_dataset() -> Dict[str, object]:
    """Build merged maps dataset."""

    maps_page_html = fetch_html(MAPS_PAGE_URL)
    if maps_page_html is None:
        raise RuntimeError(f"Unable to fetch {MAPS_PAGE_URL}")

    maps_chunk_url = _extract_map_chunk_url(maps_page_html)
    chunk_js = fetch_html(maps_chunk_url)
    if chunk_js is None:
        raise RuntimeError(f"Unable to fetch maps chunk: {maps_chunk_url}")

    translations = parse_maps_translations_from_page(maps_page_html)
    entries = parse_map_entries_from_chunk(chunk_js)

    maps: List[Dict[str, object]] = []
    for entry in entries:
        name_key = str(entry["name_key"])
        description_keys = list(entry["description_keys"])
        descriptions = [resolve_maps_translation(translations, str(key)) for key in description_keys]

        item = dict(entry)
        item["name"] = resolve_maps_translation(translations, name_key)
        item["descriptions"] = descriptions
        maps.append(item)

    source_counter = Counter(str(item["card_source"]) for item in maps)
    dataset = {
        "source": BASE_URL,
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "maps_page_url": MAPS_PAGE_URL,
        "maps_chunk_url": maps_chunk_url,
        "maps": maps,
        "stats": {
            "total": len(maps),
            "by_card_source": dict(sorted(source_counter.items())),
        },
    }
    return dataset


def write_dataset(path: Path, dataset: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dataset, ensure_ascii=True, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def download_map_images(dataset: Dict[str, object], images_dir: Path) -> Dict[str, object]:
    """Download map images referenced by the dataset into a local directory."""

    images_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0

    for item in dataset.get("maps", []):
        image_name = str(item.get("image_name", ""))
        image_url = str(item.get("image_url", ""))
        if not image_name or not image_url:
            failed += 1
            continue

        target_path = images_dir / f"{image_name}.jpg"
        item["image_path"] = str(target_path.as_posix())

        if target_path.exists() and target_path.stat().st_size > 0:
            skipped += 1
            continue

        request = Request(image_url, headers={"User-Agent": "ark-nova-ai/0.1 (map image downloader)"})
        try:
            with urlopen(request, timeout=20) as response:
                content = response.read()
            if not content:
                failed += 1
                continue
            target_path.write_bytes(content)
            downloaded += 1
        except Exception:
            failed += 1

    return {
        "downloaded": downloaded,
        "skipped_existing": skipped,
        "failed": failed,
        "images_dir": str(images_dir.as_posix()),
    }

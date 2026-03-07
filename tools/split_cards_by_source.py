#!/usr/bin/env python3
"""Split cards.json into base / marine_world / promo groups.

Source labels are read from Ark Nova Fan Hub frontend chunks:
- animals: chunk id 176
- sponsors: chunk id 307
- conservation+endgame: chunk id 148
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Dict, List, Optional, Tuple
from urllib.request import Request, urlopen


SITE_BASE_URL = "https://ark-nova.ender-wiggin.com"
USER_AGENT = "ark-nova-ai/0.1 (source splitter)"

CHUNK_SRC_RE = re.compile(r'src="(/_next/static/chunks/[^"]+\.js)"')
ID_SOURCE_RE = re.compile(
    r'id:"(?P<id>\d+)".*?source:[A-Za-z_$][A-Za-z0-9_$]*\.t\.(?P<source>BASE|MARINE_WORLD|PROMO)',
    flags=re.S,
)

SOURCE_NORMALIZE = {
    "BASE": "base",
    "MARINE_WORLD": "marine_world",
    "PROMO": "promo",
}
OUTPUT_ORDER = ("base", "marine_world", "promo")


def fetch_text(url: str, timeout: float) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", "ignore")


def _pick_chunk(paths: List[str], chunk_prefix: str) -> str:
    for path in paths:
        name = Path(path).name
        if name.startswith(f"{chunk_prefix}-"):
            return path
    raise RuntimeError(f"Chunk {chunk_prefix}-*.js not found in card page script list.")


def discover_chunk_urls(base_url: str, timeout: float) -> Dict[str, str]:
    html = fetch_text(f"{base_url}/card/401", timeout=timeout)
    paths = sorted(set(CHUNK_SRC_RE.findall(html)))
    return {
        "animals": f"{base_url}{_pick_chunk(paths, '176')}",
        "sponsors": f"{base_url}{_pick_chunk(paths, '307')}",
        "projects_endgame": f"{base_url}{_pick_chunk(paths, '148')}",
    }


def parse_id_source_map(chunk_text: str) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for match in ID_SOURCE_RE.finditer(chunk_text):
        card_id = int(match.group("id"))
        source = SOURCE_NORMALIZE[match.group("source")]
        mapping[card_id] = source
    return mapping


def load_source_maps(base_url: str, timeout: float) -> Dict[str, Dict[int, str]]:
    chunk_urls = discover_chunk_urls(base_url=base_url, timeout=timeout)
    return {
        key: parse_id_source_map(fetch_text(url, timeout=timeout))
        for key, url in chunk_urls.items()
    }


def infer_card_number(card: Dict[str, object]) -> Optional[int]:
    number = card.get("number")
    if isinstance(number, int):
        return number
    if isinstance(number, str) and number.isdigit():
        return int(number)

    data_id = str(card.get("data_id", ""))
    match = re.search(r"[A-Z](\d+)", data_id)
    if match:
        return int(match.group(1))
    return None


def assign_source_group(
    card: Dict[str, object],
    source_maps: Dict[str, Dict[int, str]],
) -> Optional[str]:
    data_id = str(card.get("data_id", ""))
    prefix = data_id[:1]
    number = infer_card_number(card)
    if number is None:
        return None

    if prefix == "A":
        return source_maps["animals"].get(number)
    if prefix == "S":
        return source_maps["sponsors"].get(number)
    if prefix in {"P", "F"}:
        return source_maps["projects_endgame"].get(number)
    return None


def build_output_payload(
    base_payload: Dict[str, object],
    cards: List[Dict[str, object]],
    source_group: str,
) -> Dict[str, object]:
    by_type = Counter(str(card.get("type", "unknown")) for card in cards)
    payload: Dict[str, object] = {
        "source": base_payload.get("source"),
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "split_source_group": source_group,
        "cards": cards,
        "stats": {
            "total": len(cards),
            "by_type": dict(sorted(by_type.items())),
        },
    }
    for keep_key in ("max_card_id_scanned",):
        if keep_key in base_payload:
            payload[keep_key] = base_payload[keep_key]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split cards.json by source group.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/cards/cards.json"),
        help="Input dataset (default: data/cards/cards.json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/cards"),
        help="Output directory (default: data/cards)",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=SITE_BASE_URL,
        help="Fan Hub base URL (default: https://ark-nova.ender-wiggin.com)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="HTTP timeout in seconds (default: 20)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cards = payload.get("cards")
    if not isinstance(cards, list):
        raise RuntimeError("Input JSON does not contain a valid 'cards' list.")

    source_maps = load_source_maps(base_url=args.base_url, timeout=args.timeout)
    groups: Dict[str, List[Dict[str, object]]] = {name: [] for name in OUTPUT_ORDER}
    unknown_ids: List[str] = []

    for raw_card in cards:
        if not isinstance(raw_card, dict):
            continue
        source_group = assign_source_group(raw_card, source_maps=source_maps)
        if source_group not in groups:
            unknown_ids.append(str(raw_card.get("data_id", "<missing-data-id>")))
            continue
        groups[source_group].append(raw_card)

    if unknown_ids:
        raise RuntimeError(
            "Could not map source group for {} cards: {}".format(
                len(unknown_ids),
                ", ".join(sorted(unknown_ids)[:10]),
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for source_group in OUTPUT_ORDER:
        output_path = args.output_dir / f"cards.{source_group}.json"
        output_payload = build_output_payload(
            base_payload=payload,
            cards=groups[source_group],
            source_group=source_group,
        )
        output_path.write_text(
            json.dumps(output_payload, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {len(groups[source_group])} cards -> {output_path}")


if __name__ == "__main__":
    main()

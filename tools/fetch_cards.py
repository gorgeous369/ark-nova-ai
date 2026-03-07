#!/usr/bin/env python3
"""Fetch and write Ark Nova card dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arknova_engine.site_cards import build_cards_dataset, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Ark Nova cards from ark-nova.ender-wiggin.com")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cards/cards.json"),
        help="Output JSON path (default: data/cards/cards.json)",
    )
    parser.add_argument(
        "--max-card-id",
        type=int,
        default=1300,
        help="Highest /card/<id> to scan (default: 1300)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=32,
        help="Parallel fetch workers for /card/<id> scanning (default: 32)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_cards_dataset(max_card_id=args.max_card_id, workers=args.workers)
    write_dataset(args.output, dataset)
    print(f"Wrote {dataset['stats']['total']} cards to {args.output}")
    print(f"Type stats: {dataset['stats']['by_type']}")


if __name__ == "__main__":
    main()

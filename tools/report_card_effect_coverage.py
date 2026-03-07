#!/usr/bin/env python3
"""Report card-effect coverage for the lightweight main runner."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arknova_engine.card_effects import resolve_card_effect
from main import build_deck


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report supported/unsupported zoo card effects.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/cards/card_effect_coverage.tsv"),
        help="Output TSV path (default: data/cards/card_effect_coverage.tsv)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cards = build_deck()
    rows = ["number\tinstance_id\ttype\tname\teffect_code\tsupported\tability_title\tability_text"]
    for card in cards:
        resolved = resolve_card_effect(card)
        rows.append(
            "\t".join(
                [
                    str(card.number),
                    str(card.instance_id),
                    str(card.card_type),
                    str(card.name).replace("\t", " "),
                    str(resolved.code).replace("\t", " "),
                    "1" if resolved.supported else "0",
                    str(card.ability_title).replace("\t", " "),
                    str(card.ability_text).replace("\t", " "),
                ]
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(rows) + "\n", encoding="utf-8")
    supported = sum(1 for card in cards if resolve_card_effect(card).supported and resolve_card_effect(card).code != "none")
    unsupported = sum(1 for card in cards if not resolve_card_effect(card).supported)
    no_effect = sum(1 for card in cards if resolve_card_effect(card).code == "none")
    print(f"cards={len(cards)} supported={supported} unsupported={unsupported} no_effect={no_effect}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()

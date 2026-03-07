#!/usr/bin/env python3
"""Create a machine-readable tile template for one Ark Nova map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arknova_engine.map_tiles import create_map_tile_template


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a map tile template JSON.")
    parser.add_argument("--map-id", required=True, help="Map id, e.g. m1a")
    parser.add_argument("--map-name", required=True, help="Human-readable map name")
    parser.add_argument("--image-name", required=True, help="Image basename, e.g. plan1a")
    parser.add_argument(
        "--image-path",
        default="",
        help="Optional local image path (e.g. data/maps/images/plan1a.jpg)",
    )
    parser.add_argument(
        "--effect",
        action="append",
        default=[],
        help="Map-level effect text; repeat this flag for multiple lines.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSON path, e.g. data/maps/tiles/plan1a.tiles.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = create_map_tile_template(
        map_id=args.map_id,
        map_name=args.map_name,
        image_name=args.image_name,
        image_path=args.image_path,
        map_effects=list(args.effect),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote template with {len(payload['tiles'])} tiles to {args.output}")


if __name__ == "__main__":
    main()

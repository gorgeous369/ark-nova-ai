#!/usr/bin/env python3
"""Fetch and write Ark Nova maps dataset."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arknova_engine.site_maps import build_maps_dataset, download_map_images, write_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Ark Nova maps from ark-nova.ender-wiggin.com")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/maps/maps.json"),
        help="Output JSON path (default: data/maps/maps.json)",
    )
    parser.add_argument(
        "--download-images",
        action="store_true",
        help="Also download map images to --images-dir",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/maps/images"),
        help="Image output directory (default: data/maps/images)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_maps_dataset()
    if args.download_images:
        image_stats = download_map_images(dataset, images_dir=args.images_dir)
        print(f"Image download stats: {image_stats}")
    write_dataset(args.output, dataset)
    print(f"Wrote {dataset['stats']['total']} maps to {args.output}")
    print(f"Source stats: {dataset['stats']['by_card_source']}")


if __name__ == "__main__":
    main()

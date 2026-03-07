#!/usr/bin/env python3
"""Validate map tile JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arknova_engine.map_tiles import validate_map_tile_payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate map tile JSON.")
    parser.add_argument("path", type=Path, help="Path to map tile JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    issues = validate_map_tile_payload(payload)
    if issues:
        print(f"Validation FAILED for {args.path}:")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"Validation OK: {args.path}")


if __name__ == "__main__":
    main()

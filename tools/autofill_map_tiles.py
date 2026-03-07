#!/usr/bin/env python3
"""Autofill map tile JSON files from map images (first-pass heuristics).

This script is intentionally conservative:
- It preserves `plan1a.tiles.json` by default (already manually verified).
- It only fills fields that can be inferred with simple image heuristics.
- Output is meant for human review, not as final ground truth.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arknova_engine.map_tiles import ALLOWED_PLACEMENT_BONUS_KINDS


REFERENCE_WIDTH = 4028.0
REFERENCE_HEIGHT = 2126.0

# Calibrated from the verified plan1a overlay.
REFERENCE_X0 = 270.0
REFERENCE_Y0 = 320.0
REFERENCE_STEP_X = 430.0
REFERENCE_STEP_Y = 210.0
REFERENCE_HEX_RADIUS = 105.0


@dataclass(frozen=True)
class GridParams:
    x0: float
    y0: float
    step_x: float
    step_y: float
    hex_radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autofill terrain/build2/placement_bonus for map tile JSON files."
    )
    parser.add_argument(
        "--tiles-dir",
        type=Path,
        default=Path("data/maps/tiles"),
        help="Directory containing *.tiles.json files.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("data/maps/images"),
        help="Directory containing map images (jpg).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Optional image_name to include. Repeat for multiple maps.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=["plan1a"],
        help="Optional image_name to exclude. Repeat for multiple maps.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not write files; print summary only.",
    )
    return parser.parse_args()


def grid_params_for_image(width: int, height: int) -> GridParams:
    sx = width / REFERENCE_WIDTH
    sy = height / REFERENCE_HEIGHT
    return GridParams(
        x0=REFERENCE_X0 * sx,
        y0=REFERENCE_Y0 * sy,
        step_x=REFERENCE_STEP_X * sx,
        step_y=REFERENCE_STEP_Y * sy,
        hex_radius=REFERENCE_HEX_RADIUS * ((sx + sy) / 2.0),
    )


def tile_center(x: int, y: int, params: GridParams) -> Tuple[float, float]:
    px = params.x0 + (x + y) * params.step_x
    py = params.y0 + (x - y) * (params.step_y / 2.0)
    return px, py


def hex_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(img)
    points: List[Tuple[float, float]] = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        px = cx + radius * math.cos(angle)
        py = cy + radius * math.sin(angle)
        points.append((px, py))
    draw.polygon(points, fill=255)
    return np.array(img) > 0


def circle_mask(height: int, width: int, cx: float, cy: float, radius: float) -> np.ndarray:
    yy, xx = np.ogrid[:height, :width]
    return ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius


def classify_terrain(pixels: np.ndarray) -> str:
    r = pixels[:, 0]
    g = pixels[:, 1]
    b = pixels[:, 2]

    # Water-like pixels (blue/cyan dominance).
    water_1 = ((b > r + 16) & (g > r + 6) & (b > 95)).mean()
    water_2 = ((b > g + 10) & (b > r + 10) & (b > 80)).mean()
    water_score = max(float(water_1), float(water_2))

    # Rock-like pixels (darker neutral/violet textures).
    rock_1 = (
        (r < 190)
        & (g < 180)
        & (b < 180)
        & (np.abs(r - b) < 34)
        & (r > g - 12)
        & (r > 55)
        & (g > 40)
        & (b > 45)
    ).mean()
    dark_ratio = ((r < 120) & (g < 120) & (b < 120)).mean()
    rock_score = float(rock_1) + 0.35 * float(dark_ratio)

    # Keep thresholds conservative to reduce false positives.
    if water_score >= 0.58 and water_score >= rock_score + 0.10:
        return "water"
    if rock_score >= 0.34 and rock_score >= water_score + 0.05:
        return "rock"
    return "plain"


def detect_build2_required(center_pixels: np.ndarray) -> bool:
    r = center_pixels[:, 0]
    g = center_pixels[:, 1]
    b = center_pixels[:, 2]
    magenta_ratio = ((r > 140) & (b > 120) & (g < 130) & (r > g + 25) & (b > g + 15)).mean()
    return bool(magenta_ratio >= 0.18)


def detect_bonus_kind(center_pixels: np.ndarray, build2_required: bool) -> str | None:
    r = center_pixels[:, 0]
    g = center_pixels[:, 1]
    b = center_pixels[:, 2]

    yellow_ratio = ((r > 165) & (g > 135) & (b < 120)).mean()
    black_ratio = ((r < 70) & (g < 70) & (b < 70)).mean()
    blue_ratio = ((b > 110) & (b > g + 10) & (b > r + 20)).mean()
    white_ratio = ((r > 205) & (g > 205) & (b > 205)).mean()
    magenta_ratio = ((r > 140) & (b > 120) & (g < 130) & (r > g + 25) & (b > g + 15)).mean()

    # Yellow token families: x-token / 5 coins / reputation.
    if yellow_ratio >= 0.11:
        if blue_ratio >= 0.12:
            return "5coins"
        if black_ratio >= 0.16:
            # Reputation icon tends to have larger dark filled area.
            return "reputation" if black_ratio >= 0.26 else "x_token"

    # White card token family.
    if white_ratio >= 0.24 and blue_ratio >= 0.03:
        return "card_in_reputation_range"

    # Action-to-slot-1 token can appear magenta-like; ignore build2 spaces.
    if (not build2_required) and magenta_ratio >= 0.23:
        return "action_to_slot_1"

    return None


def autofill_payload(payload: Dict[str, object], image_array: np.ndarray) -> Dict[str, object]:
    tiles: List[Dict[str, object]] = payload["tiles"]  # type: ignore[assignment]
    height, width = image_array.shape[:2]
    params = grid_params_for_image(width=width, height=height)

    for tile in tiles:
        x = int(tile["x"])
        y = int(tile["y"])
        cx, cy = tile_center(x, y, params)

        terrain_mask = hex_mask(height, width, cx, cy, params.hex_radius * 0.98)
        center_mask = circle_mask(height, width, cx, cy, params.hex_radius * 0.40)

        terrain_pixels = image_array[terrain_mask]
        center_pixels = image_array[center_mask]
        if len(terrain_pixels) == 0 or len(center_pixels) == 0:
            continue

        terrain = classify_terrain(terrain_pixels)
        build2_required = detect_build2_required(center_pixels)
        bonus = detect_bonus_kind(center_pixels, build2_required=build2_required)

        if bonus not in ALLOWED_PLACEMENT_BONUS_KINDS:
            bonus = None

        tile["terrain"] = terrain
        tile["build2_required"] = build2_required and terrain == "plain"
        tile["placement_bonus"] = bonus
        tile["tags"] = []

    return payload


def should_process(image_name: str, include: Iterable[str], exclude: Iterable[str]) -> bool:
    include_set = {name.strip() for name in include if name.strip()}
    exclude_set = {name.strip() for name in exclude if name.strip()}
    if include_set and image_name not in include_set:
        return False
    return image_name not in exclude_set


def main() -> None:
    args = parse_args()
    tiles_files = sorted(args.tiles_dir.glob("*.tiles.json"))
    if not tiles_files:
        raise SystemExit(f"No tiles JSON files found in: {args.tiles_dir}")

    summaries: List[Tuple[str, int, int, int, int]] = []

    for path in tiles_files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        image_name = str(payload.get("image_name", "")).strip()
        if not image_name:
            continue
        if not should_process(image_name, include=args.include, exclude=args.exclude):
            continue

        image_path = args.images_dir / f"{image_name}.jpg"
        if not image_path.exists():
            print(f"SKIP {path.name}: missing image {image_path}")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        payload = autofill_payload(payload, image_array=image)
        tiles: List[Dict[str, object]] = payload["tiles"]  # type: ignore[assignment]
        terrain_rock = sum(1 for t in tiles if t.get("terrain") == "rock")
        terrain_water = sum(1 for t in tiles if t.get("terrain") == "water")
        build2_count = sum(1 for t in tiles if bool(t.get("build2_required")))
        bonus_count = sum(1 for t in tiles if t.get("placement_bonus") is not None)

        summaries.append((image_name, terrain_rock, terrain_water, build2_count, bonus_count))
        if not args.dry_run:
            path.write_text(
                json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=False) + "\n",
                encoding="utf-8",
            )

    for image_name, rock_count, water_count, build2_count, bonus_count in summaries:
        print(
            f"{image_name}: rock={rock_count}, water={water_count}, "
            f"build2={build2_count}, bonus={bonus_count}"
        )

    mode = "DRY-RUN" if args.dry_run else "UPDATED"
    print(f"{mode}: {len(summaries)} map file(s)")


if __name__ == "__main__":
    main()

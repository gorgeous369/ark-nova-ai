#!/usr/bin/env python3
"""Render an SVG overlay with (x,y) labels for map-tile annotation.

No external dependencies are required. This tool helps manual annotation by
drawing the Ark Nova hex grid over a map image.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List


TERRAIN_COLORS = {
    "plain": "#ffffff00",
    "rock": "#8b5e3c66",
    "water": "#1d9bf066",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render coordinate overlay SVG for map tile annotation.")
    parser.add_argument("tiles_json", type=Path, help="Input map tile JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Output SVG path.")
    parser.add_argument("--image", default="", help="Background image href (default: image_path from JSON).")
    parser.add_argument("--image-width", type=int, default=4028, help="Background image width in px.")
    parser.add_argument("--image-height", type=int, default=2126, help="Background image height in px.")
    parser.add_argument("--x0", type=float, default=270.0, help="Center X for tile (x=0,y=0).")
    parser.add_argument("--y0", type=float, default=320.0, help="Center Y for tile (x=0,y=0).")
    parser.add_argument("--step-x", type=float, default=430.0, help="Horizontal component for x/y axis steps.")
    parser.add_argument(
        "--step-y",
        type=float,
        default=210.0,
        help="Vertical component for x/y axis steps.",
    )
    parser.add_argument("--hex-radius", type=float, default=105.0, help="Hex corner radius in px.")
    return parser.parse_args()


def tile_center(x: int, y: int, x0: float, y0: float, step_x: float, step_y: float) -> tuple[float, float]:
    # x+ is right-down, y+ is right-up.
    px = x0 + (x + y) * step_x
    py = y0 + (x - y) * (step_y / 2.0)
    return px, py


def hex_points(cx: float, cy: float, radius: float) -> str:
    points: List[str] = []
    for i in range(6):
        angle = math.radians(30 + i * 60)
        px = cx + radius * math.cos(angle)
        py = cy + radius * math.sin(angle)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def main() -> None:
    args = parse_args()
    payload: Dict[str, object] = json.loads(args.tiles_json.read_text(encoding="utf-8"))
    tiles: List[Dict[str, object]] = payload["tiles"]  # type: ignore[assignment]
    image_href = args.image or str(payload.get("image_path", ""))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{args.image_width}" height="{args.image_height}" '
            f'viewBox="0 0 {args.image_width} {args.image_height}">\n'
        )
        if image_href:
            f.write(
                f'  <image href="{image_href}" x="0" y="0" width="{args.image_width}" height="{args.image_height}" />\n'
            )

        for tile in tiles:
            x = int(tile["x"])
            y = int(tile["y"])
            terrain = str(tile.get("terrain", "plain"))
            build2_required = bool(tile.get("build2_required", False))
            fill = TERRAIN_COLORS.get(terrain, "#ff000055")
            cx, cy = tile_center(x, y, args.x0, args.y0, args.step_x, args.step_y)
            points = hex_points(cx, cy, args.hex_radius)
            bonus = tile.get("placement_bonus")
            bonus_label = ""
            if isinstance(bonus, str) and bonus:
                bonus_label = bonus

            stroke = "#ffffff"
            stroke_width = 4
            stroke_opacity = "0.8"
            if build2_required:
                # Build-2 required spaces are highlighted with a magenta border/overlay.
                stroke = "#d63384"
                stroke_width = 8
                if terrain == "plain":
                    fill = "#d6338466"
            f.write(
                f'  <polygon points="{points}" fill="{fill}" stroke="{stroke}" '
                f'stroke-width="{stroke_width}" stroke-opacity="{stroke_opacity}"/>\n'
            )
            f.write(
                f'  <text x="{cx:.2f}" y="{cy:.2f}" text-anchor="middle" dominant-baseline="middle" '
                f'font-size="34" font-family="monospace" fill="#111">({x},{y})</text>\n'
            )
            if build2_required:
                f.write(
                    f'  <text x="{cx:.2f}" y="{cy - 34:.2f}" text-anchor="middle" dominant-baseline="middle" '
                    f'font-size="22" font-family="monospace" fill="#d63384">B2</text>\n'
                )
            if bonus_label:
                f.write(
                    f'  <text x="{cx:.2f}" y="{cy + 38:.2f}" text-anchor="middle" dominant-baseline="middle" '
                    f'font-size="24" font-family="monospace" fill="#111">{bonus_label}</text>\n'
                )

        f.write("</svg>\n")

    print(f"Wrote overlay SVG: {args.output}")


if __name__ == "__main__":
    main()

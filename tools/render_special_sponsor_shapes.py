#!/usr/bin/env python3
"""Render sponsor special-building shapes (#243-#257) for visual verification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFont


ShapeSpec = Dict[str, object]


SHAPES: List[ShapeSpec] = [
    {"number": 243, "name": "MEERKAT DEN", "cells": [(0, 0), (0, 1), (1, 1)], "req": "rock>=1"},
    {"number": 244, "name": "PENGUIN POOL", "cells": [(0, 0), (1, 0), (2, 0), (1, 1)], "req": "water>=1"},
    {"number": 245, "name": "AQUARIUM", "cells": [(0, 0), (0, 1), (1, 1), (2, 0)], "req": "water>=2"},
    {"number": 246, "name": "CABLE CAR", "cells": [(0, 0), (0, 1), (0, 2), (0, 3)], "req": "rock>=2"},
    {"number": 247, "name": "BABOON ROCK", "cells": [(0, 0), (1, 0), (2, 0), (2, -1)], "req": "rock>=1"},
    {"number": 248, "name": "RHESUS MONKEY PARK", "cells": [(0, 0), (0, 1), (1, 1), (2, 1)], "req": "-"},
    {"number": 249, "name": "BARRED OWL HUT", "cells": [(0, 0), (1, 0), (2, 0)], "req": "-"},
    {"number": 250, "name": "SEA TURTLE TANK", "cells": [(0, 0), (1, 0), (1, 1), (2, -1)], "req": "water>=1"},
    {"number": 251, "name": "POLAR BEAR EXHIBIT", "cells": [(0, 0), (0, 1), (1, 1), (1, 2)], "req": "water>=1"},
    {"number": 252, "name": "SPOTTED HYENA COMPOUND", "cells": [(0, 0), (1, 0), (1, 1), (1, 2)], "req": "rock>=1"},
    {"number": 253, "name": "OKAPI STABLE", "cells": [(0, 0), (1, 0), (1, 1), (2, 1)], "req": "-"},
    {"number": 254, "name": "ZOO SCHOOL", "cells": [(0, 0), (1, 0), (1, 1)], "req": "border>=2"},
    {"number": 255, "name": "ADVENTURE PLAYGROUND", "cells": [(0, 0), (1, 0)], "req": "rock>=1"},
    {"number": 256, "name": "WATER PLAYGROUND", "cells": [(0, 0), (1, 0)], "req": "water>=1"},
    {"number": 257, "name": "SIDE ENTRANCE", "cells": [(0, 0), (1, 0)], "req": "border>=2, no-adj-required"},
]


def center_xy(x: int, y: int, size: float) -> Tuple[float, float]:
    # x-axis: right-down, y-axis: right-up
    # Board-aligned projection (flat-top look in rendered preview).
    cx = (x + y) * 1.5 * size
    cy = (x - y) * (math.sqrt(3) / 2.0) * size
    return cx, cy


def hex_points(cx: float, cy: float, radius: float) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for deg in (0, 60, 120, 180, 240, 300):
        rad = math.radians(deg)
        pts.append((cx + radius * math.cos(rad), cy + radius * math.sin(rad)))
    return pts


def draw_shape_panel(
    draw: ImageDraw.ImageDraw,
    panel_xy: Tuple[int, int],
    panel_wh: Tuple[int, int],
    shape: ShapeSpec,
    font: ImageFont.ImageFont,
    font_small: ImageFont.ImageFont,
) -> None:
    px, py = panel_xy
    pw, ph = panel_wh
    draw.rounded_rectangle([px, py, px + pw, py + ph], radius=14, outline=(90, 90, 90), width=2, fill=(248, 248, 248))

    number = int(shape["number"])  # type: ignore[index]
    name = str(shape["name"])  # type: ignore[index]
    req = str(shape["req"])  # type: ignore[index]
    cells = list(shape["cells"])  # type: ignore[index]

    draw.text((px + 12, py + 8), f"#{number} {name}", fill=(30, 30, 30), font=font)
    draw.text((px + 12, py + 34), f"req: {req}", fill=(55, 55, 55), font=font_small)

    size = 22.0
    centers = [center_xy(int(x), int(y), size) for x, y in cells]
    min_cx = min(c[0] for c in centers)
    max_cx = max(c[0] for c in centers)
    min_cy = min(c[1] for c in centers)
    max_cy = max(c[1] for c in centers)
    shape_w = max_cx - min_cx + size * 2.4
    shape_h = max_cy - min_cy + size * 2.4

    anchor_x = px + (pw - shape_w) / 2.0 - min_cx + size * 1.2
    anchor_y = py + 72 + (ph - 96 - shape_h) / 2.0 - min_cy + size * 1.2

    # faint local grid
    xs = [int(x) for x, _ in cells]
    ys = [int(y) for _, y in cells]
    for gx in range(min(xs) - 1, max(xs) + 2):
        for gy in range(min(ys) - 1, max(ys) + 2):
            cx_rel, cy_rel = center_xy(gx, gy, size)
            cx = anchor_x + cx_rel
            cy = anchor_y + cy_rel
            draw.polygon(hex_points(cx, cy, size * 0.92), outline=(220, 220, 220), fill=None)

    # actual cells
    for x, y in cells:
        cx_rel, cy_rel = center_xy(int(x), int(y), size)
        cx = anchor_x + cx_rel
        cy = anchor_y + cy_rel
        is_origin = (int(x), int(y)) == (0, 0)
        fill = (255, 215, 170) if not is_origin else (255, 170, 170)
        outline = (130, 95, 55) if not is_origin else (180, 60, 60)
        draw.polygon(hex_points(cx, cy, size * 0.9), outline=outline, fill=fill)
        text = f"({x},{y})"
        tw, th = draw.textbbox((0, 0), text, font=font_small)[2:]
        draw.text((cx - tw / 2.0, cy - th / 2.0), text, fill=(15, 15, 15), font=font_small)


def main() -> None:
    cols = 3
    rows = 5
    panel_w, panel_h = 520, 220
    margin = 20
    gap_x, gap_y = 16, 16
    canvas_w = margin * 2 + cols * panel_w + (cols - 1) * gap_x
    canvas_h = margin * 2 + rows * panel_h + (rows - 1) * gap_y

    image = Image.new("RGB", (canvas_w, canvas_h), (238, 242, 246))
    draw = ImageDraw.Draw(image)

    try:
        font = ImageFont.truetype("Arial Unicode.ttf", 18)
        font_small = ImageFont.truetype("Arial Unicode.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
        font_small = ImageFont.load_default()

    for idx, shape in enumerate(SHAPES):
        r = idx // cols
        c = idx % cols
        x = margin + c * (panel_w + gap_x)
        y = margin + r * (panel_h + gap_y)
        draw_shape_panel(draw, (x, y), (panel_w, panel_h), shape, font, font_small)

    out_dir = Path("data/cards")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sponsor_special_buildings_243_257.png"
    image.save(out_path)
    print(out_path)


if __name__ == "__main__":
    main()

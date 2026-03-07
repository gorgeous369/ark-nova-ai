"""Map tile template and validation helpers.

This module defines a machine-readable map-tile schema based on the existing
ArkNovaMap axial (x, y) grid.
"""

from __future__ import annotations

from typing import Dict, List, Set, Tuple

from arknova_engine.map_model import ArkNovaMap, HexTile


ALLOWED_TERRAINS: Set[str] = {
    "plain",
    "rock",
    "water",
}

ALLOWED_PLACEMENT_BONUS_KINDS: Set[str] = {
    "x_token",
    "5coins",
    "card_in_reputation_range",
    "reputation",
    "action_to_slot_1",
}


def sorted_grid_tiles() -> List[HexTile]:
    grid = ArkNovaMap().grid
    return sorted(grid, key=lambda t: (t.x, t.y))


def create_map_tile_template(
    map_id: str,
    map_name: str,
    image_name: str,
    image_path: str,
    map_effects: List[str],
) -> Dict[str, object]:
    tiles = [
        {
            "x": tile.x,
            "y": tile.y,
            "terrain": "plain",
            "build2_required": False,
            "placement_bonus": None,
            "tags": [],
        }
        for tile in sorted_grid_tiles()
    ]
    return {
        "map_id": map_id,
        "map_name": map_name,
        "image_name": image_name,
        "image_path": image_path,
        "coordinate_system": "axial_xy",
        "axis_definition": {
            "x_positive_direction": "right_down",
            "y_positive_direction": "right_up",
            "origin": "top_left",
        },
        "grid_version": "arknova_map_v1",
        "map_effects": map_effects,
        "tiles": tiles,
    }


def _expected_coord_set() -> Set[Tuple[int, int]]:
    return {(tile.x, tile.y) for tile in ArkNovaMap().grid}


def validate_map_tile_payload(payload: Dict[str, object]) -> List[str]:
    issues: List[str] = []
    tiles = payload.get("tiles")
    if not isinstance(tiles, list):
        return ["'tiles' must be a list."]

    seen: Set[Tuple[int, int]] = set()
    expected = _expected_coord_set()
    for index, tile in enumerate(tiles):
        if not isinstance(tile, dict):
            issues.append(f"tile[{index}] must be an object.")
            continue

        x = tile.get("x")
        y = tile.get("y")
        terrain = tile.get("terrain")
        build2_required = tile.get("build2_required")
        if not isinstance(x, int) or not isinstance(y, int):
            issues.append(f"tile[{index}] must contain integer x/y.")
            continue

        coord = (x, y)
        if coord in seen:
            issues.append(f"duplicate tile coordinate: {coord}.")
        seen.add(coord)

        if coord not in expected:
            issues.append(f"tile coordinate {coord} is outside ArkNovaMap grid.")

        if not isinstance(terrain, str) or terrain not in ALLOWED_TERRAINS:
            issues.append(
                f"tile[{index}] terrain '{terrain}' is invalid; allowed={sorted(ALLOWED_TERRAINS)}."
            )
        if not isinstance(build2_required, bool):
            issues.append(f"tile[{index}] build2_required must be boolean.")
        elif build2_required and terrain in {"rock", "water"}:
            issues.append(f"tile[{index}] build2_required cannot be true on terrain '{terrain}'.")

        placement_bonus = tile.get("placement_bonus")
        if placement_bonus is not None:
            if not isinstance(placement_bonus, str):
                issues.append(f"tile[{index}] placement_bonus must be null or bonus-kind string.")
            elif placement_bonus not in ALLOWED_PLACEMENT_BONUS_KINDS:
                issues.append(
                    f"tile[{index}] placement_bonus '{placement_bonus}' is invalid; "
                    f"allowed={sorted(ALLOWED_PLACEMENT_BONUS_KINDS)}."
                )

        tags = tile.get("tags")
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            issues.append(f"tile[{index}] tags must be a list of strings.")

    missing = expected - seen
    extra = seen - expected
    if missing:
        issues.append(f"missing coordinates: {sorted(missing)}")
    if extra:
        issues.append(f"extra coordinates: {sorted(extra)}")
    if len(tiles) != len(expected):
        issues.append(f"tile count is {len(tiles)}, expected {len(expected)}.")

    return issues

"""Map and building placement model for Ark Nova.

This module ports the map/building placement core from the MIT-licensed
`jbargu/TabletopGames` implementation under `games/arknova`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, List, Optional, Set, Tuple


MINIMUM_KIOSK_DISTANCE = 3


@dataclass(frozen=True)
class HexTile:
    x: int
    y: int

    @staticmethod
    def from_legacy_qr(q: int, r: int) -> "HexTile":
        """Convert old (q,r) coordinates to new (x,y).

        New axes:
        - x positive: right-down
        - y positive: right-up
        """
        return HexTile(q + r, -r)

    def to_legacy_qr(self) -> Tuple[int, int]:
        q = self.x + self.y
        r = -self.y
        return q, r

    def add(self, other: "HexTile") -> "HexTile":
        return HexTile(self.x + other.x, self.y + other.y)

    def subtract(self, other: "HexTile") -> "HexTile":
        return HexTile(self.x - other.x, self.y - other.y)

    def distance(self, other: "HexTile") -> int:
        dx = self.x - other.x
        dy = self.y - other.y
        dz = -(dx + dy)
        return (abs(dx) + abs(dy) + abs(dz)) // 2

    def rotate_left(self) -> "HexTile":
        q, r = self.to_legacy_qr()
        s = -q - r
        return HexTile.from_legacy_qr(-s, -q)

    def rotate_right(self) -> "HexTile":
        q, r = self.to_legacy_qr()
        s = -q - r
        return HexTile.from_legacy_qr(-r, -s)

    def doubled_coordinates(self) -> Tuple[int, int]:
        q, r = self.to_legacy_qr()
        return q, 2 * r + q

    def neighbors(self) -> List["HexTile"]:
        return [self.add(delta) for delta in HEX_NEIGHBORS]


HEX_NEIGHBORS: Tuple[HexTile, ...] = (
    HexTile(1, 0),  # right-down
    HexTile(0, 1),  # right-up
    HexTile(-1, 1),  # up
    HexTile(-1, 0),  # left-up
    HexTile(0, -1),  # left-down
    HexTile(1, -1),  # down
)


def legacy_hex(q: int, r: int) -> HexTile:
    return HexTile.from_legacy_qr(q, r)


class Terrain(str, Enum):
    ROCK = "rock"
    WATER = "water"
    BUILD_2_REQUIRED = "build_2_required"


class BonusResource(str, Enum):
    REPUTATION = "reputation"


@dataclass(frozen=True)
class GainBonus:
    resource: BonusResource
    automatic: bool
    amount: int


class BuildingSubType(str, Enum):
    ENCLOSURE_BASIC = "enclosure_basic"
    ENCLOSURE_SPECIAL = "enclosure_special"
    UNIQUE = "unique"
    KIOSK = "kiosk"
    PAVILION = "pavilion"
    SPONSOR_BUILDING = "sponsor_building"


class Rotation(Enum):
    ROT_0 = 0
    ROT_60 = 60
    ROT_120 = 120
    ROT_180 = 180
    ROT_240 = 240
    ROT_300 = 300


class BuildingType(Enum):
    SIZE_1 = (
        BuildingSubType.ENCLOSURE_BASIC,
        ((0, 0),),
        1,
        False,
    )
    SIZE_2 = (
        BuildingSubType.ENCLOSURE_BASIC,
        ((0, 0), (0, -1)),
        1,
        False,
    )
    SIZE_3 = (
        BuildingSubType.ENCLOSURE_BASIC,
        ((0, 0), (0, -1), (1, -1)),
        1,
        False,
    )
    SIZE_4 = (
        BuildingSubType.ENCLOSURE_BASIC,
        ((0, 0), (0, -1), (1, -2), (1, -1)),
        1,
        False,
    )
    SIZE_5 = (
        BuildingSubType.ENCLOSURE_BASIC,
        ((0, 0), (0, -1), (1, -2), (1, -1), (0, -2)),
        1,
        False,
    )
    PETTING_ZOO = (
        BuildingSubType.ENCLOSURE_SPECIAL,
        ((0, 0), (0, -1), (1, -2)),
        3,
        True,
    )
    REPTILE_HOUSE = (
        BuildingSubType.ENCLOSURE_SPECIAL,
        ((0, 0), (0, -1), (1, -1), (2, -2), (2, -1)),
        5,
        True,
    )
    LARGE_BIRD_AVIARY = (
        BuildingSubType.ENCLOSURE_SPECIAL,
        ((0, 0), (0, -1), (1, -2), (1, -1), (2, -1)),
        5,
        True,
    )
    PAVILION = (
        BuildingSubType.PAVILION,
        ((0, 0),),
        -1,
        False,
    )
    KIOSK = (
        BuildingSubType.KIOSK,
        ((0, 0),),
        -1,
        False,
    )

    @property
    def subtype(self) -> BuildingSubType:
        return self.value[0]

    @property
    def layout(self) -> Tuple[HexTile, ...]:
        return tuple(legacy_hex(q, r) for q, r in self.value[1])

    @property
    def max_capacity(self) -> int:
        return self.value[2]

    @property
    def unique_per_zoo(self) -> bool:
        return self.value[3]


@lru_cache(maxsize=None)
def _rotated_building_layout(building_type: BuildingType, rotation: Rotation) -> Tuple[HexTile, ...]:
    layout = building_type.layout
    steps_right = (int(rotation.value) // 60) % 6
    rotated = layout
    for _ in range(steps_right):
        rotated = tuple(tile.rotate_right() for tile in rotated)
    return rotated


@dataclass
class Building:
    type: BuildingType
    origin_hex: HexTile
    rotation: Rotation = Rotation.ROT_0
    layout: List[HexTile] = field(default_factory=list)
    empty_spaces: int = 0

    def __post_init__(self) -> None:
        self.empty_spaces = self.type.max_capacity
        self.layout = [
            self.origin_hex.add(tile) for tile in _rotated_building_layout(self.type, self.rotation)
        ]

    def apply_rotation(self, new_rotation: Rotation) -> None:
        if new_rotation == self.rotation:
            return
        self.layout = [
            self.origin_hex.add(tile) for tile in _rotated_building_layout(self.type, new_rotation)
        ]
        self.rotation = new_rotation

    def rotate_left(self) -> None:
        rotations = [
            Rotation.ROT_0,
            Rotation.ROT_60,
            Rotation.ROT_120,
            Rotation.ROT_180,
            Rotation.ROT_240,
            Rotation.ROT_300,
        ]
        current_index = rotations.index(self.rotation)
        self.apply_rotation(rotations[(current_index - 1) % len(rotations)])

    def rotate_right(self) -> None:
        rotations = [
            Rotation.ROT_0,
            Rotation.ROT_60,
            Rotation.ROT_120,
            Rotation.ROT_180,
            Rotation.ROT_240,
            Rotation.ROT_300,
        ]
        current_index = rotations.index(self.rotation)
        self.apply_rotation(rotations[(current_index + 1) % len(rotations)])


def _building_footprint_key(building: Building) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((tile.x, tile.y) for tile in building.layout))


@lru_cache(maxsize=None)
def _unique_rotations_for_building_type(building_type: BuildingType) -> Tuple[Rotation, ...]:
    origin = HexTile(0, 0)
    unique_rotations: List[Rotation] = []
    seen_layouts: Set[Tuple[Tuple[int, int], ...]] = set()
    for rotation in Rotation:
        building = Building(building_type, origin, rotation)
        footprint = _building_footprint_key(building)
        if footprint in seen_layouts:
            continue
        seen_layouts.add(footprint)
        unique_rotations.append(rotation)
        if len(building.layout) == 1:
            break
    return tuple(unique_rotations)


@dataclass(frozen=True)
class MapData:
    name: str
    terrain: Dict[HexTile, Terrain]
    placement_bonuses: Dict[HexTile, GainBonus]


MAP_7 = MapData(
    name="Ice Cream parlors",
    terrain={
        legacy_hex(0, 0): Terrain.ROCK,
        legacy_hex(0, 1): Terrain.ROCK,
        legacy_hex(1, 3): Terrain.ROCK,
        legacy_hex(2, 3): Terrain.ROCK,
        legacy_hex(3, 0): Terrain.ROCK,
        legacy_hex(4, -1): Terrain.ROCK,
        legacy_hex(4, -2): Terrain.ROCK,
        legacy_hex(4, 2): Terrain.ROCK,
        legacy_hex(5, 1): Terrain.ROCK,
        legacy_hex(7, -4): Terrain.WATER,
        legacy_hex(8, -4): Terrain.WATER,
        legacy_hex(8, -3): Terrain.WATER,
        legacy_hex(8, -2): Terrain.WATER,
        legacy_hex(7, -2): Terrain.WATER,
        legacy_hex(7, 1): Terrain.WATER,
        legacy_hex(8, 1): Terrain.WATER,
        legacy_hex(3, 2): Terrain.BUILD_2_REQUIRED,
        legacy_hex(3, 3): Terrain.BUILD_2_REQUIRED,
        legacy_hex(3, 4): Terrain.BUILD_2_REQUIRED,
    },
    placement_bonuses={
        legacy_hex(0, 2): GainBonus(BonusResource.REPUTATION, True, 1),
    },
)


def map_data_from_tiles_payload(payload: Dict[str, object]) -> MapData:
    """Build MapData from a `data/maps/tiles/*.tiles.json` payload.

    Mapping rules:
    - `terrain=rock|water` -> ROCK|WATER
    - `build2_required=true` on plain -> BUILD_2_REQUIRED
    - `placement_bonus=reputation` -> +1 reputation gain bonus
    """

    map_name = str(payload.get("map_name", "Custom Map"))
    raw_tiles = payload.get("tiles", [])
    if not isinstance(raw_tiles, list):
        raise ValueError("map tile payload must contain a 'tiles' list.")

    terrain: Dict[HexTile, Terrain] = {}
    placement_bonuses: Dict[HexTile, GainBonus] = {}

    for raw_tile in raw_tiles:
        if not isinstance(raw_tile, dict):
            continue
        x = raw_tile.get("x")
        y = raw_tile.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        tile = HexTile(x, y)

        terrain_name = raw_tile.get("terrain")
        build2_required = bool(raw_tile.get("build2_required", False))
        placement_bonus_name = raw_tile.get("placement_bonus")

        if terrain_name == "rock":
            terrain[tile] = Terrain.ROCK
        elif terrain_name == "water":
            terrain[tile] = Terrain.WATER
        elif build2_required:
            terrain[tile] = Terrain.BUILD_2_REQUIRED

        if placement_bonus_name == "reputation":
            placement_bonuses[tile] = GainBonus(BonusResource.REPUTATION, True, 1)

    return MapData(
        name=map_name,
        terrain=terrain,
        placement_bonuses=placement_bonuses,
    )


@lru_cache(maxsize=64)
def load_map_data_from_tiles_file(path: str) -> MapData:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return map_data_from_tiles_payload(payload)


@lru_cache(maxsize=64)
def load_map_data_by_image_name(image_name: str) -> MapData:
    root = Path(__file__).resolve().parents[1]
    tiles_path = root / "data" / "maps" / "tiles" / f"{image_name}.tiles.json"
    if not tiles_path.exists():
        raise FileNotFoundError(f"Map tiles file not found: {tiles_path}")
    return load_map_data_from_tiles_file(str(tiles_path))


class ArkNovaMap:
    WIDTH = 8
    HEIGHT = 5

    def __init__(self, map_data: MapData = MAP_7):
        self.map_data = map_data
        self.grid: Set[HexTile] = set()
        self.buildings: Dict[HexTile, Building] = {}
        self._legal_building_placements_cache: Dict[
            Tuple[
                Tuple[Tuple[str, int, int, str], ...],
                bool,
                bool,
                int,
                Tuple[str, ...],
            ],
            Tuple[Tuple[str, int, int, str], ...],
        ] = {}

        for q in range(self.WIDTH + 1):
            q_offset = (q + 1) // 2
            for r in range(-q_offset, self.HEIGHT - q_offset + 1):
                self.grid.add(legacy_hex(q, r))
            if q % 2 == 1:
                self.grid.add(legacy_hex(q, self.HEIGHT - q_offset + 1))

    def add_building(self, building: Building) -> None:
        self.buildings[building.origin_hex] = building
        self._legal_building_placements_cache.clear()

    def _building_state_signature(self) -> Tuple[Tuple[str, int, int, str], ...]:
        return tuple(
            sorted(
                (
                    building.type.name,
                    building.origin_hex.x,
                    building.origin_hex.y,
                    building.rotation.name,
                )
                for building in self.buildings.values()
            )
        )

    def _legal_placement_cache_key(
        self,
        *,
        is_build_upgraded: bool,
        has_diversity_researcher: bool,
        max_building_size: int,
        already_built_buildings: Set[BuildingType],
    ) -> Tuple[
        Tuple[Tuple[str, int, int, str], ...],
        bool,
        bool,
        int,
        Tuple[str, ...],
    ]:
        return (
            self._building_state_signature(),
            bool(is_build_upgraded),
            bool(has_diversity_researcher),
            int(max_building_size),
            tuple(sorted(building_type.name for building_type in already_built_buildings)),
        )

    def _placement_specs_to_buildings(
        self,
        specs: Tuple[Tuple[str, int, int, str], ...],
    ) -> List[Building]:
        return [
            Building(
                BuildingType[type_name],
                HexTile(origin_x, origin_y),
                Rotation[rotation_name],
            )
            for type_name, origin_x, origin_y, rotation_name in specs
        ]

    def covered_hexes(self) -> Set[HexTile]:
        covered: Set[HexTile] = set()
        for building in self.buildings.values():
            covered.update(building.layout)
        return covered

    def border_tiles(self, ignore_terrain: bool) -> Set[HexTile]:
        tiles: Set[HexTile] = set()
        for tile in self.grid:
            terrain = self.map_data.terrain.get(tile)
            if not ignore_terrain and terrain in {Terrain.WATER, Terrain.ROCK}:
                continue
            x, y = tile.doubled_coordinates()
            if x in {0, self.WIDTH} or y in {-1, 0, 2 * self.HEIGHT, 2 * self.HEIGHT + 1}:
                tiles.add(tile)
        return tiles

    def can_build_on_hex(
        self,
        tile: HexTile,
        covered_hexes: Set[HexTile],
        build_upgraded: bool,
        has_diversity_researcher: bool,
    ) -> bool:
        if tile not in self.grid or tile in covered_hexes:
            return False

        terrain = self.map_data.terrain.get(tile)
        if terrain == Terrain.BUILD_2_REQUIRED and not build_upgraded:
            return False

        return has_diversity_researcher or terrain not in {Terrain.WATER, Terrain.ROCK}

    def possible_starting_building_hexes(
        self,
        is_build_upgraded: bool,
        has_diversity_researcher: bool,
        covered_hexes: Set[HexTile],
    ) -> Set[HexTile]:
        if not covered_hexes:
            return self.border_tiles(has_diversity_researcher)

        possible: Set[HexTile] = set()
        for covered_hex in covered_hexes:
            for neighbor in covered_hex.neighbors():
                if self.can_build_on_hex(
                    neighbor,
                    covered_hexes,
                    is_build_upgraded,
                    has_diversity_researcher,
                ):
                    possible.add(neighbor)
        return possible

    def legal_building_placements(
        self,
        is_build_upgraded: bool,
        has_diversity_researcher: bool,
        max_building_size: int,
        already_built_buildings: Optional[Set[BuildingType]] = None,
    ) -> List[Building]:
        if already_built_buildings is None:
            already_built_buildings = set()

        cache_key = self._legal_placement_cache_key(
            is_build_upgraded=is_build_upgraded,
            has_diversity_researcher=has_diversity_researcher,
            max_building_size=max_building_size,
            already_built_buildings=already_built_buildings,
        )
        cached = self._legal_building_placements_cache.get(cache_key)
        if cached is not None:
            return self._placement_specs_to_buildings(cached)

        covered_hexes = self.covered_hexes()
        border_tiles = self.border_tiles(has_diversity_researcher) if not covered_hexes else set()
        starts = self.possible_starting_building_hexes(
            is_build_upgraded,
            has_diversity_researcher,
            covered_hexes,
        )
        existing_special = {
            b.type for b in self.buildings.values() if b.type.subtype == BuildingSubType.ENCLOSURE_SPECIAL
        }
        existing_kiosk_origins = {
            b.origin_hex for b in self.buildings.values() if b.type == BuildingType.KIOSK
        }

        candidate_types = [
            b_type
            for b_type in BuildingType
            if len(b_type.layout) <= max_building_size
            and b_type not in already_built_buildings
            and b_type.subtype != BuildingSubType.SPONSOR_BUILDING
            and not (b_type.subtype == BuildingSubType.ENCLOSURE_SPECIAL and b_type in existing_special)
        ]

        if not is_build_upgraded and max_building_size >= len(BuildingType.LARGE_BIRD_AVIARY.layout):
            candidate_types = [
                b for b in candidate_types if b not in {BuildingType.LARGE_BIRD_AVIARY, BuildingType.REPTILE_HOUSE}
            ]

        placements: List[Building] = []
        seen_placements: Set[Tuple[BuildingType, Tuple[Tuple[int, int], ...]]] = set()
        placement_specs: List[Tuple[str, int, int, str]] = []
        candidate_rotation_layouts = {
            b_type: tuple(
                (rotation, _rotated_building_layout(b_type, rotation))
                for rotation in _unique_rotations_for_building_type(b_type)
            )
            for b_type in candidate_types
        }

        for starting_hex in starts:
            for b_type in candidate_types:
                rotation_layouts = candidate_rotation_layouts[b_type]
                if b_type == BuildingType.KIOSK:
                    if any(
                        kiosk_origin.distance(starting_hex) < MINIMUM_KIOSK_DISTANCE
                        for kiosk_origin in existing_kiosk_origins
                    ):
                        continue

                for rotation, offset_layout in rotation_layouts:
                    layout = [starting_hex.add(tile) for tile in offset_layout]
                    footprint_key = tuple(sorted((tile.x, tile.y) for tile in layout))
                    placement_key = (b_type, footprint_key)
                    if placement_key in seen_placements:
                        continue
                    if all(
                        self.can_build_on_hex(
                            tile,
                            covered_hexes,
                            is_build_upgraded,
                            has_diversity_researcher,
                        )
                        for tile in layout
                    ):
                        if not covered_hexes:
                            if not any(tile in border_tiles for tile in layout):
                                continue
                        else:
                            touches_existing = any(
                                any(neighbor in covered_hexes for neighbor in tile.neighbors())
                                for tile in layout
                            )
                            if not touches_existing:
                                continue
                        building = Building(b_type, starting_hex, rotation)
                        seen_placements.add(placement_key)
                        placements.append(building)
                        placement_specs.append(
                            (
                                b_type.name,
                                starting_hex.x,
                                starting_hex.y,
                                rotation.name,
                            )
                        )

        self._legal_building_placements_cache[cache_key] = tuple(placement_specs)
        return placements


def building_origins_of_type(buildings: Iterable[Building], building_type: BuildingType) -> List[HexTile]:
    return [building.origin_hex for building in buildings if building.type == building_type]

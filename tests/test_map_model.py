from arknova_engine.map_model import (
    ArkNovaMap,
    Building,
    BuildingType,
    HexTile,
    Rotation,
    building_origins_of_type,
)


def test_hex_distance():
    assert HexTile(6, -1).distance(HexTile(6, -1)) == 0
    assert HexTile(0, 1).distance(HexTile(7, 1)) == 7
    assert HexTile(9, -2).distance(HexTile(7, -3)) == 3


def test_get_border_tiles_without_ignoring_terrain():
    game_map = ArkNovaMap()
    border = game_map.border_tiles(ignore_terrain=False)

    expected = {
        HexTile(0, 1),
        HexTile(1, 1),
        HexTile(1, 2),
        HexTile(2, 3),
        HexTile(3, 3),
        HexTile(2, -2),
        HexTile(3, -3),
        HexTile(4, -4),
        HexTile(5, -5),
        HexTile(7, 1),
        HexTile(8, 0),
        HexTile(6, -5),
        HexTile(6, -4),
        HexTile(7, -4),
        HexTile(7, -3),
        HexTile(8, -3),
        HexTile(8, -2),
        HexTile(9, -2),
    }

    assert border == expected
    assert HexTile(0, 0) not in border


def test_get_border_tiles_ignoring_terrain():
    game_map = ArkNovaMap()
    border = game_map.border_tiles(ignore_terrain=True)

    expected = {
        HexTile(0, 0),
        HexTile(1, -1),
        HexTile(9, -1),
        HexTile(6, 2),
        HexTile(5, 3),
        HexTile(4, 4),
        HexTile(3, 4),
        HexTile(2, 2),
        HexTile(0, 1),
        HexTile(1, 1),
        HexTile(1, 2),
        HexTile(2, 3),
        HexTile(3, 3),
        HexTile(2, -2),
        HexTile(3, -3),
        HexTile(4, -4),
        HexTile(5, -5),
        HexTile(7, 1),
        HexTile(8, 0),
        HexTile(6, -5),
        HexTile(6, -4),
        HexTile(7, -4),
        HexTile(7, -3),
        HexTile(8, -3),
        HexTile(8, -2),
        HexTile(9, -2),
    }

    assert border == expected


def test_can_build_on_hex():
    game_map = ArkNovaMap()
    rock_tile = HexTile(0, 0)
    assert not game_map.can_build_on_hex(rock_tile, set(), False, False)
    assert game_map.can_build_on_hex(rock_tile, set(), False, True)

    game_map.add_building(Building(BuildingType.SIZE_1, rock_tile, Rotation.ROT_0))
    assert not game_map.can_build_on_hex(rock_tile, game_map.covered_hexes(), False, True)

    build2_tile = HexTile(7, -4)
    assert not game_map.can_build_on_hex(build2_tile, game_map.covered_hexes(), False, True)
    assert game_map.can_build_on_hex(build2_tile, game_map.covered_hexes(), True, True)


def test_kiosk_distance_rule():
    game_map = ArkNovaMap()
    game_map.add_building(Building(BuildingType.KIOSK, HexTile(9, -2), Rotation.ROT_0))

    kiosks = building_origins_of_type(
        game_map.legal_building_placements(
            is_build_upgraded=True,
            has_diversity_researcher=False,
            max_building_size=100,
            already_built_buildings=set(),
        ),
        BuildingType.KIOSK,
    )
    assert not kiosks

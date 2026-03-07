import copy

import pytest

from arknova_engine.base_game import BuildSelection, MainAction, create_base_game
from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation


def _selection_from_building(building: Building) -> BuildSelection:
    return BuildSelection(building.type, building.origin_hex, building.rotation)


def test_build_side_i_single_build_and_action_rotation():
    state = create_base_game(num_players=2, seed=1)
    player = state.current()
    player.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    player.action_upgraded[MainAction.BUILD] = False

    legal = player.zoo_map.legal_building_placements(
        is_build_upgraded=False,
        has_diversity_researcher=False,
        max_building_size=2,
        already_built_buildings=set(),
    )
    picked = legal[0]
    cost = len(picked.layout) * 2
    start_money = player.money

    state.perform_build_action([_selection_from_building(picked)])

    assert player.money == start_money - cost
    assert player.action_order[0] == MainAction.BUILD
    assert len(player.zoo_map.buildings) == 1
    assert state.current_player == 1


def test_build_side_i_rejects_multiple_buildings():
    state = create_base_game(num_players=2, seed=2)
    player = state.current()
    player.action_order = [
        MainAction.CARDS,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
        MainAction.BUILD,
    ]
    player.action_upgraded[MainAction.BUILD] = False

    legal = player.zoo_map.legal_building_placements(
        is_build_upgraded=False,
        has_diversity_researcher=False,
        max_building_size=5,
        already_built_buildings=set(),
    )
    with pytest.raises(ValueError, match="side I"):
        state.perform_build_action(
            [
                _selection_from_building(legal[0]),
                _selection_from_building(legal[1]),
            ]
        )


def test_build_side_ii_allows_multiple_different_buildings():
    state = create_base_game(num_players=2, seed=3)
    player = state.current()
    player.action_order = [
        MainAction.CARDS,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
        MainAction.BUILD,
    ]
    player.action_upgraded[MainAction.BUILD] = True

    legal_first = player.zoo_map.legal_building_placements(
        is_build_upgraded=True,
        has_diversity_researcher=False,
        max_building_size=5,
        already_built_buildings=set(),
    )
    first = next(
        b
        for b in legal_first
        if b.type == BuildingType.KIOSK and b.origin_hex != HexTile(2, -2)
    )

    scratch = copy.deepcopy(player.zoo_map)
    scratch.add_building(copy.deepcopy(first))
    legal_second = scratch.legal_building_placements(
        is_build_upgraded=True,
        has_diversity_researcher=False,
        max_building_size=4,
        already_built_buildings={first.type},
    )
    second = next(
        b
        for b in legal_second
        if b.type == BuildingType.PAVILION and b.origin_hex != HexTile(2, -2)
    )

    start_money = player.money
    state.perform_build_action([_selection_from_building(first), _selection_from_building(second)])

    assert len(player.zoo_map.buildings) == 2
    assert player.appeal == 1  # pavilion immediate bonus
    assert player.money == start_money - 4


def test_build_placement_bonus_grants_reputation():
    state = create_base_game(num_players=2, seed=4)
    player = state.current()
    player.action_order = [
        MainAction.BUILD,
        MainAction.CARDS,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    player.action_upgraded[MainAction.BUILD] = False

    legal = player.zoo_map.legal_building_placements(
        is_build_upgraded=False,
        has_diversity_researcher=False,
        max_building_size=1,
        already_built_buildings=set(),
    )
    reputation_hexes = set(player.zoo_map.map_data.placement_bonuses)
    picked = next(
        building
        for building in legal
        if building.type == BuildingType.SIZE_1 and building.origin_hex in reputation_hexes
    )
    selection = BuildSelection(
        building_type=picked.type,
        origin_hex=picked.origin_hex,
        rotation=picked.rotation,
    )
    state.perform_build_action([selection])

    assert player.reputation == 1

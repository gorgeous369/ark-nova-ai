import pytest

from arknova_engine.base_game import (
    AnimalCard,
    AnimalPlaySelection,
    CardSource,
    MainAction,
    create_base_game,
)
from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation, Terrain


def _set_animals_strength_slot(state, slot_index: int) -> None:
    order = [MainAction.CARDS, MainAction.BUILD, MainAction.ASSOCIATION, MainAction.SPONSORS]
    order.insert(slot_index - 1, MainAction.ANIMALS)
    state.current().action_order = order


def _animal(
    card_id: str,
    cost: int = 8,
    min_size: int = 1,
    appeal: int = 1,
    **kwargs,
) -> AnimalCard:
    return AnimalCard(
        card_id=card_id,
        cost=cost,
        min_enclosure_size=min_size,
        appeal_gain=appeal,
        **kwargs,
    )


def test_animals_side_i_strength_1_cannot_play_any_animal():
    state = create_base_game(num_players=2, seed=21)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(state, 1)
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, HexTile(0, 1), Rotation.ROT_0))
    player.hand = [_animal("a1")]

    with pytest.raises(ValueError, match="at most 0"):
        state.perform_animals_action(
            [
                AnimalPlaySelection(
                    source=CardSource.HAND,
                    source_index=0,
                    enclosure_origin=HexTile(0, 1),
                )
            ]
        )


def test_animals_side_i_strength_5_can_play_two_from_hand():
    state = create_base_game(num_players=2, seed=22)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(state, 5)

    first_enclosure = HexTile(0, 1)
    second_enclosure = HexTile(1, 1)
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, first_enclosure, Rotation.ROT_0))
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, second_enclosure, Rotation.ROT_0))
    player.hand = [
        _animal(
            "a1",
            cost=8,
            appeal=2,
            continent_icons={"europe": 1},
            category_icons={"predator": 1},
            reputation_gain=1,
        ),
        _animal(
            "a2",
            cost=5,
            appeal=3,
            continent_icons={"asia": 1},
            category_icons={"bird": 1},
            conservation_gain=1,
        ),
    ]

    state.perform_animals_action(
        [
            AnimalPlaySelection(CardSource.HAND, 0, first_enclosure),
            AnimalPlaySelection(CardSource.HAND, 0, second_enclosure),
        ]
    )

    assert player.money == 12
    assert player.appeal == 5
    assert player.reputation == 1
    assert player.conservation == 1
    assert player.hand == []
    assert len(player.played_animals) == 2
    assert player.zoo_icons["europe"] == 1
    assert player.zoo_icons["asia"] == 1
    assert player.zoo_icons["predator"] == 1
    assert player.zoo_icons["bird"] == 1
    assert player.zoo_map.buildings[first_enclosure].empty_spaces == 0
    assert player.zoo_map.buildings[second_enclosure].empty_spaces == 0
    assert state.current_player == 1
    assert player.action_order[0] == MainAction.ANIMALS


def test_animals_upgraded_display_play_cost_and_strength5_rep_bonus():
    state = create_base_game(num_players=2, seed=23)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = True
    _set_animals_strength_slot(state, 5)
    player.reputation = 1
    player.partner_zoos.add("europe")
    player.hand = []
    enclosure_origin = HexTile(1, 1)
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, enclosure_origin, Rotation.ROT_0))

    from_display = _animal("disp", cost=10, appeal=2, continent_icons={"europe": 1})
    state.display = ["x1", "x2", from_display, "x4", "x5", "x6"]
    state.deck = ["new_top"]

    with pytest.raises(ValueError, match="outside reputation range"):
        state.perform_animals_action(
            [AnimalPlaySelection(CardSource.DISPLAY, 2, enclosure_origin)],
            take_strength5_reputation_bonus=False,
        )

    state.perform_animals_action(
        [AnimalPlaySelection(CardSource.DISPLAY, 2, enclosure_origin)],
        take_strength5_reputation_bonus=True,
    )

    # base 10 - partner discount 3 + folder cost 3 = 10
    assert player.money == 15
    assert player.appeal == 2
    assert player.reputation == 2
    assert player.hand == []
    assert state.display == ["x1", "x2", "x4", "x5", "x6", "new_top"]


def test_animals_condition_requires_upgraded_action():
    card = _animal("gated", requires_upgraded_animals_action=True)

    state = create_base_game(num_players=2, seed=24)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(state, 2)
    origin = HexTile(0, 1)
    player.zoo_map.add_building(Building(BuildingType.SIZE_1, origin, Rotation.ROT_0))
    player.hand = [card]

    with pytest.raises(ValueError, match="requires upgraded Animals action"):
        state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, origin)])

    upgraded_state = create_base_game(num_players=2, seed=25)
    upgraded_player = upgraded_state.current()
    upgraded_player.action_upgraded[MainAction.ANIMALS] = True
    _set_animals_strength_slot(upgraded_state, 2)
    upgraded_player.zoo_map.add_building(Building(BuildingType.SIZE_1, origin, Rotation.ROT_0))
    upgraded_player.hand = [card]

    upgraded_state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, origin)])
    assert len(upgraded_player.played_animals) == 1


def test_animals_petting_zoo_and_special_enclosure_capacity():
    state = create_base_game(num_players=2, seed=26)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(state, 2)

    petting_origin = HexTile(2, 0)
    standard_origin = HexTile(0, 1)
    player.zoo_map.add_building(Building(BuildingType.PETTING_ZOO, petting_origin, Rotation.ROT_0))
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, standard_origin, Rotation.ROT_0))

    petting_card = _animal("pet", is_petting_zoo_animal=True, min_size=1)
    player.hand = [petting_card]

    with pytest.raises(ValueError, match="Petting Zoo"):
        state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, standard_origin)])

    state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, petting_origin)])
    assert player.zoo_map.buildings[petting_origin].empty_spaces == 2

    special_state = create_base_game(num_players=2, seed=27)
    special_player = special_state.current()
    special_player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(special_state, 2)
    reptile_origin = HexTile(2, 1)
    special_player.zoo_map.add_building(
        Building(BuildingType.REPTILE_HOUSE, reptile_origin, Rotation.ROT_0)
    )
    reptile_card = _animal(
        "rep",
        special_enclosure_spaces={BuildingType.REPTILE_HOUSE: 2},
        min_size=4,
    )
    special_player.hand = [reptile_card]
    special_state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, reptile_origin)])
    assert special_player.zoo_map.buildings[reptile_origin].empty_spaces == 3


def test_animals_terrain_adjacency_requirement():
    state = create_base_game(num_players=2, seed=28)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(state, 2)

    plain_tiles = sorted(
        (
            tile
            for tile in player.zoo_map.grid
            if player.zoo_map.map_data.terrain.get(tile) not in {Terrain.ROCK, Terrain.WATER}
        ),
        key=lambda t: (t.x, t.y),
    )

    def water_adjacency(tile: HexTile) -> int:
        return sum(
            1
            for neighbor in tile.neighbors()
            if player.zoo_map.map_data.terrain.get(neighbor) == Terrain.WATER
        )

    dry_origin = next(tile for tile in plain_tiles if water_adjacency(tile) == 0)
    wet_origin = next(tile for tile in plain_tiles if water_adjacency(tile) >= 1)
    player.zoo_map.add_building(Building(BuildingType.SIZE_1, dry_origin, Rotation.ROT_0))
    player.zoo_map.add_building(Building(BuildingType.SIZE_1, wet_origin, Rotation.ROT_0))
    player.hand = [_animal("w", required_water_adjacency=1)]

    with pytest.raises(ValueError, match="water adjacency"):
        state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, dry_origin)])

    state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, wet_origin)])
    assert len(player.played_animals) == 1


def test_animals_add_water_and_rock_icons_and_track_placement():
    state = create_base_game(num_players=2, seed=29)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    _set_animals_strength_slot(state, 2)

    plain_tiles = sorted(
        (
            tile
            for tile in player.zoo_map.grid
            if player.zoo_map.map_data.terrain.get(tile) not in {Terrain.ROCK, Terrain.WATER}
        ),
        key=lambda t: (t.x, t.y),
    )

    def adjacency(tile: HexTile, terrain: Terrain) -> int:
        return sum(
            1
            for neighbor in tile.neighbors()
            if player.zoo_map.map_data.terrain.get(neighbor) == terrain
        )

    origin = next(
        tile
        for tile in plain_tiles
        if adjacency(tile, Terrain.WATER) >= 1 and adjacency(tile, Terrain.ROCK) >= 1
    )
    player.zoo_map.add_building(Building(BuildingType.SIZE_3, origin, Rotation.ROT_0))
    player.hand = [
        _animal(
            "adjacent_icons",
            min_size=3,
            required_water_adjacency=1,
            required_rock_adjacency=1,
        )
    ]

    state.perform_animals_action([AnimalPlaySelection(CardSource.HAND, 0, origin)])

    assert player.zoo_icons["water"] == 1
    assert player.zoo_icons["rock"] == 1
    placement = player.animal_placements["adjacent_icons"]
    assert placement.enclosure_origin == origin
    assert placement.enclosure_type == BuildingType.SIZE_3
    assert placement.spaces_used == 1

from arknova_engine.actions import (
    run_animals_action,
    run_association_action,
    run_build_action,
    run_cards_action,
    run_sponsors_action,
    run_x_token_action,
)
from arknova_engine.base_game import (
    AnimalCard,
    AnimalPlaySelection,
    AssociationTask,
    AssociationTaskSelection,
    BuildSelection,
    CardSource,
    MainAction,
    SponsorCard,
    SponsorPlaySelection,
    create_base_game,
)
from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation


def test_run_build_action_module_entrypoint():
    state = create_base_game(num_players=2, seed=71)
    player = state.current()
    player.action_upgraded[MainAction.BUILD] = False
    player.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    legal = player.zoo_map.legal_building_placements(
        is_build_upgraded=False,
        has_diversity_researcher=False,
        max_building_size=2,
        already_built_buildings=set(),
    )
    selected = legal[0]
    run_build_action(
        state,
        selections=[BuildSelection(selected.type, selected.origin_hex, selected.rotation)],
    )
    assert len(player.zoo_map.buildings) == 1


def test_run_cards_action_module_entrypoint():
    state = create_base_game(num_players=2, seed=72)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = False
    player.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    player.hand = ["keep", "drop"]
    state.deck = ["d1", "d2"]
    run_cards_action(state, from_deck_count=1, discard_hand_indices=[1])
    assert player.hand == ["keep", "d1"]


def test_run_animals_action_module_entrypoint():
    state = create_base_game(num_players=2, seed=73)
    player = state.current()
    player.action_upgraded[MainAction.ANIMALS] = False
    player.action_order = [
        MainAction.CARDS,
        MainAction.ANIMALS,
        MainAction.BUILD,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    enclosure_origin = HexTile(0, 1)
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, enclosure_origin, Rotation.ROT_0))
    player.hand = [AnimalCard(card_id="a1", cost=7, min_enclosure_size=2, appeal_gain=2)]
    run_animals_action(
        state,
        selections=[AnimalPlaySelection(CardSource.HAND, 0, enclosure_origin)],
    )
    assert len(player.played_animals) == 1
    assert player.appeal == 2


def test_run_association_action_module_entrypoint():
    state = create_base_game(num_players=2, seed=74)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.action_order = [
        MainAction.CARDS,
        MainAction.ASSOCIATION,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.SPONSORS,
    ]
    run_association_action(
        state,
        tasks=[AssociationTaskSelection(task=AssociationTask.REPUTATION)],
    )
    assert player.reputation == 2


def test_run_sponsors_action_module_entrypoint():
    state = create_base_game(num_players=2, seed=75)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = False
    player.reputation = 1
    player.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.SPONSORS,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
    ]
    player.hand = [SponsorCard(card_id="s1", level=3, min_reputation=1, appeal_gain=1)]
    run_sponsors_action(
        state,
        selections=[SponsorPlaySelection(CardSource.HAND, 0)],
    )
    assert len(player.played_sponsors) == 1
    assert player.appeal == 1


def test_run_x_token_action_module_entrypoint():
    state = create_base_game(num_players=2, seed=76)
    player = state.current()
    player.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.SPONSORS,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
    ]
    player.x_tokens = 1
    run_x_token_action(state, chosen_action=MainAction.SPONSORS)
    assert player.x_tokens == 2
    assert player.action_order[0] == MainAction.SPONSORS

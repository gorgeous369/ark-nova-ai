from arknova_engine.base_game import MainAction, create_base_game
from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation


def test_break_resolution_sequence():
    state = create_base_game(num_players=2, seed=5)
    p0 = state.player(0)
    p1 = state.player(1)
    state.current_player = 0
    p0.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    p1.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]

    # Controlled break setup
    state.break_progress = 8
    state.display = ["d1", "d2", "d3", "d4", "d5", "d6"]
    state.deck = ["n1", "n2", "n3"]
    state.discard = []
    state.appeal_income_table = [3, 4, 5, 6, 7, 8, 9, 11] + [11] * 200

    p0.hand = ["a", "b", "c", "d"]
    p1.hand = ["e", "f", "g", "h"]
    p0.hand_limit = 3
    p1.hand_limit = 3

    p0.multiplier_tokens_on_actions[MainAction.CARDS] = 1
    p1.venom_tokens_on_actions[MainAction.BUILD] = 2
    p1.constriction_tokens_on_actions[MainAction.SPONSORS] = 1

    p0.active_workers = 1
    p1.active_workers = 1
    p0.workers_on_association_board = 2
    p1.workers_on_association_board = 1

    p0.appeal = 7
    p1.appeal = 0
    p0.recurring_break_income = 2
    p1.recurring_break_income = 0

    p0.zoo_map.add_building(Building(BuildingType.KIOSK, HexTile(0, 1), Rotation.ROT_0))
    p0.zoo_map.add_building(Building(BuildingType.PAVILION, HexTile(1, 1), Rotation.ROT_0))

    money0 = p0.money
    money1 = p1.money

    # Cards action advances break by 2. From 8, this triggers break at 9.
    state.perform_cards_action_stub(break_steps=2)

    assert state.break_progress == 0
    assert state.break_pending is False
    assert state.break_trigger_player is None
    assert state.current_player == 1

    assert p0.x_tokens == 1  # trigger reward
    assert len(p0.hand) == 3
    assert len(p1.hand) == 3
    assert p0.workers_on_association_board == 0
    assert p1.workers_on_association_board == 0
    assert p0.active_workers == 3
    assert p1.active_workers == 2

    assert state.display == ["d3", "d4", "d5", "d6", "n1", "n2"]
    assert {"d1", "d2", "d", "h"}.issubset(set(state.discard))

    # p0 income = appeal(11) + kiosk(1 from adjacent pavilion) + recurring(2)
    assert p0.money == money0 + 14
    # p1 income = appeal(3)
    assert p1.money == money1 + 3

    assert p0.multiplier_tokens_on_actions[MainAction.CARDS] == 0
    assert p1.venom_tokens_on_actions[MainAction.BUILD] == 0
    assert p1.constriction_tokens_on_actions[MainAction.SPONSORS] == 0


def test_break_trigger_x_token_cap():
    state = create_base_game(num_players=2, seed=6)
    p0 = state.player(0)
    p0.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    p0.x_tokens = 5
    state.break_progress = 8
    state.perform_cards_action_stub(break_steps=2)
    assert p0.x_tokens == 5

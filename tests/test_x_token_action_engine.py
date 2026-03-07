import pytest

from arknova_engine.base_game import MainAction, create_base_game


def test_x_token_action_gains_one_and_rotates_chosen_card():
    state = create_base_game(num_players=2, seed=61)
    player = state.current()
    player.action_order = [
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    player.x_tokens = 2

    state.perform_x_token_action(MainAction.ASSOCIATION)

    assert player.x_tokens == 3
    assert player.action_order == [
        MainAction.ASSOCIATION,
        MainAction.CARDS,
        MainAction.BUILD,
        MainAction.ANIMALS,
        MainAction.SPONSORS,
    ]
    assert state.current_player == 1


def test_x_token_action_respects_maximum_cap():
    state = create_base_game(num_players=2, seed=62)
    player = state.current()
    player.x_tokens = 5

    with pytest.raises(ValueError, match="maximum X-token limit"):
        state.perform_x_token_action(MainAction.CARDS)


def test_x_token_action_resolves_pending_break_if_trigger_player_acts():
    state = create_base_game(num_players=2, seed=63)
    player = state.current()
    player.action_order = [
        MainAction.BUILD,
        MainAction.CARDS,
        MainAction.ANIMALS,
        MainAction.ASSOCIATION,
        MainAction.SPONSORS,
    ]
    state.break_pending = True
    state.break_trigger_player = 0
    state.break_progress = state.break_max
    state.display = ["d1", "d2", "d3", "d4", "d5", "d6"]
    state.deck = ["n1", "n2"]
    state.appeal_income_table = [3] + [3] * 200

    state.perform_x_token_action(MainAction.BUILD)

    assert state.break_pending is False
    assert state.break_trigger_player is None
    assert state.break_progress == 0
    assert state.display == ["d3", "d4", "d5", "d6", "n1", "n2"]

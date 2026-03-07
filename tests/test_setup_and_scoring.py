import pytest

from arknova_engine.base_game import MainAction, create_base_game


def test_create_base_game_uses_plan1a_two_player_setup():
    state = create_base_game(seed=101)

    assert len(state.players) == 2
    assert state.map_image_name == "plan1a"
    assert state.break_max == 9
    assert len(state.display) == 6
    assert len(state.final_scoring_deck) == 7

    # 212 zoo cards total: 16 drafted + 6 display + 190 remaining.
    assert len(state.deck) == 190
    # 8 drafted cards discarded during 8-choose-4 opening.
    assert len(state.discard) == 8

    p0 = state.player(0)
    p1 = state.player(1)

    assert p0.zoo_map.map_data.name == "Observation Tower"
    assert p1.zoo_map.map_data.name == "Observation Tower"
    assert p0.appeal == 0
    assert p1.appeal == 1
    assert p0.money == 25
    assert p1.money == 25
    assert p0.conservation == 0
    assert p1.conservation == 0
    assert p0.reputation == 0
    assert p1.reputation == 0
    assert len(p0.hand) == 4
    assert len(p1.hand) == 4
    assert len(p0.final_scoring_cards) == 2
    assert len(p1.final_scoring_cards) == 2
    assert p0.player_tokens_in_supply == 18
    assert p1.player_tokens_in_supply == 18

    for player in state.players:
        assert player.action_order[0].value == "animals"
        assert sorted(action.value for action in player.action_order) == sorted(
            ["animals", "cards", "build", "association", "sponsors"]
        )


def test_bga_scoring_formula_and_winner():
    state = create_base_game(seed=102)
    p0 = state.player(0)
    p1 = state.player(1)

    # BGA scoring requested:
    # - appeal: 1 ticket = 1 point
    # - first 10 conservation = 2 points each
    # - conservation above 10 = 3 points each
    p0.appeal = 36
    p0.conservation = 8
    p1.appeal = 30
    p1.conservation = 12

    assert state.bga_conservation_points(0) == 0
    assert state.bga_conservation_points(10) == 20
    assert state.bga_conservation_points(12) == 26

    s0 = state.bga_final_score(player_id=0)
    s1 = state.bga_final_score(player_id=1)
    assert s0.total_points == 52  # 36 + (8*2)
    assert s1.total_points == 56  # 30 + (10*2 + 2*3)

    with_bonus = state.bga_final_score(
        player_id=0,
        final_scoring_conservation_bonus=2,
        end_game_bonus_points=3,
    )
    assert with_bonus.total_points == 59  # 36 + (10*2) + 3

    assert state.bga_winner_player_ids() == [1]


def test_end_game_trigger_uses_bga_progress_threshold():
    state = create_base_game(seed=103)
    p0 = state.player(0)
    p0.appeal = 80
    p0.conservation = 10  # 80 + (10 * 2) = 100

    state.perform_x_token_action(chosen_action=MainAction.ANIMALS)

    assert state.end_game_triggered is True
    assert state.end_game_trigger_player == 0
    assert state.end_game_pending_players == {1}
    assert state.game_over is False


def test_end_game_not_triggered_below_threshold():
    state = create_base_game(seed=104)
    p0 = state.player(0)
    p0.appeal = 79
    p0.conservation = 10  # 79 + 20 = 99

    state.perform_x_token_action(chosen_action=MainAction.ANIMALS)

    assert state.end_game_triggered is False
    assert state.end_game_trigger_player is None
    assert state.end_game_pending_players == set()
    assert state.game_over is False


def test_end_game_after_one_extra_turn_for_other_player():
    state = create_base_game(seed=105)
    p0 = state.player(0)
    p0.appeal = 100

    # Triggering player finishes their current turn.
    state.perform_x_token_action(chosen_action=MainAction.ANIMALS)
    assert state.current_player == 1
    assert state.end_game_pending_players == {1}

    # Other player gets exactly one final turn.
    state.perform_x_token_action(chosen_action=MainAction.ANIMALS)
    assert state.game_over is True
    assert state.end_game_pending_players == set()

    with pytest.raises(ValueError, match="already over"):
        state.perform_x_token_action(chosen_action=MainAction.ANIMALS)

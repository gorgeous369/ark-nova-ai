import pytest

from arknova_engine.base_game import MainAction, create_base_game


def _set_cards_strength_slot(state, slot_index: int) -> None:
    player = state.current()
    order = [MainAction.ANIMALS, MainAction.BUILD, MainAction.ASSOCIATION, MainAction.SPONSORS]
    order.insert(slot_index - 1, MainAction.CARDS)
    player.action_order = order


def test_cards_side_i_strength_1_draw_1_discard_1():
    state = create_base_game(num_players=2, seed=11)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = False
    _set_cards_strength_slot(state, 1)
    player.hand = ["keep", "drop"]
    state.deck = ["d1", "d2", "d3"]
    state.display = ["s1", "s2", "s3", "s4", "s5", "s6"]

    state.perform_cards_action(from_deck_count=1, discard_hand_indices=[1])

    assert player.hand == ["keep", "d1"]
    assert "drop" in state.discard
    assert state.deck == ["d2", "d3"]
    # no display draw in non-upgraded mode, so display should remain unchanged
    assert state.display == ["s1", "s2", "s3", "s4", "s5", "s6"]


def test_cards_side_i_snap_requires_strength_5():
    state = create_base_game(num_players=2, seed=12)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = False
    _set_cards_strength_slot(state, 4)
    with pytest.raises(ValueError, match="Snap is not available"):
        state.perform_cards_action(snap_display_index=0)

    state2 = create_base_game(num_players=2, seed=13)
    player2 = state2.current()
    player2.action_upgraded[MainAction.CARDS] = False
    _set_cards_strength_slot(state2, 5)
    state2.display = ["a", "b", "c", "d", "e", "f"]
    state2.deck = ["n1", "n2"]
    state2.perform_cards_action(snap_display_index=4)
    assert "e" in player2.hand
    assert state2.display == ["a", "b", "c", "d", "f", "n1"]


def test_cards_side_ii_strength_3_snap_allowed():
    state = create_base_game(num_players=2, seed=14)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = True
    _set_cards_strength_slot(state, 3)
    player.hand = []
    state.display = ["x1", "x2", "x3", "x4", "x5", "x6"]
    state.deck = ["z1"]

    state.perform_cards_action(snap_display_index=5)
    assert player.hand == ["x6"]
    assert state.display == ["x1", "x2", "x3", "x4", "x5", "z1"]


def test_cards_side_ii_draw_with_reputation_range_only():
    state = create_base_game(num_players=2, seed=15)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = True
    player.reputation = 1  # folders 1-2 in range
    _set_cards_strength_slot(state, 4)  # upgraded draw target = 3, discard=1
    player.hand = ["h1", "h2"]
    state.display = ["s1", "s2", "s3", "s4", "s5", "s6"]
    state.deck = ["d1", "d2", "d3"]

    with pytest.raises(ValueError, match="outside reputation range"):
        state.perform_cards_action(
            from_display_indices=[2],
            from_deck_count=2,
            discard_hand_indices=[0],
        )

    state.perform_cards_action(
        from_display_indices=[0, 1],
        from_deck_count=1,
        discard_hand_indices=[0],
    )
    # drawn s1,s2,d1 then discarded one card
    assert len(player.hand) == 4
    assert state.display == ["s3", "s4", "s5", "s6", "d2", "d3"]


def test_cards_action_respects_max_draw_from_table():
    state = create_base_game(num_players=2, seed=16)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = False
    _set_cards_strength_slot(state, 2)  # side I draw target=1
    with pytest.raises(ValueError, match="at most 1"):
        state.perform_cards_action(from_deck_count=2)


def test_cards_action_x_token_increases_strength_table():
    state = create_base_game(num_players=2, seed=17)
    player = state.current()
    player.action_upgraded[MainAction.CARDS] = False
    player.x_tokens = 1
    _set_cards_strength_slot(state, 4)  # base draw target=2 discard=0, with x -> strength 5 draw=3 discard=1
    player.hand = ["h1", "h2"]
    state.deck = ["d1", "d2", "d3", "d4"]
    state.perform_cards_action(from_deck_count=3, discard_hand_indices=[0], x_spent=1)

    assert player.x_tokens == 0
    assert len(player.hand) == 4  # started 2 +3 -1

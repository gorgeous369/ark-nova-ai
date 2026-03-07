import pytest

from arknova_engine.base_game import (
    CardSource,
    MainAction,
    SponsorCard,
    SponsorPlaySelection,
    create_base_game,
)


def _set_sponsors_strength_slot(state, slot_index: int) -> None:
    order = [MainAction.ANIMALS, MainAction.CARDS, MainAction.BUILD, MainAction.ASSOCIATION]
    order.insert(slot_index - 1, MainAction.SPONSORS)
    state.current().action_order = order


def _sponsor(
    card_id: str,
    level: int,
    **kwargs,
) -> SponsorCard:
    return SponsorCard(card_id=card_id, level=level, **kwargs)


def test_sponsors_side_i_single_card_from_hand():
    state = create_base_game(num_players=2, seed=41)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = False
    _set_sponsors_strength_slot(state, 3)

    card = _sponsor(
        "s1",
        level=3,
        min_reputation=1,
        appeal_gain=2,
        reputation_gain=1,
        recurring_break_income_gain=1,
        granted_icons={"herbivore": 1},
    )
    player.reputation = 1
    player.hand = [card]

    state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])

    assert len(player.played_sponsors) == 1
    assert player.hand == []
    assert player.appeal == 2
    assert player.reputation == 2
    assert player.recurring_break_income == 1
    assert player.zoo_icons["herbivore"] == 1
    assert player.action_order[0] == MainAction.SPONSORS
    assert state.current_player == 1


def test_sponsors_side_i_rejects_multiple_or_display():
    state = create_base_game(num_players=2, seed=42)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = False
    _set_sponsors_strength_slot(state, 5)
    player.hand = [_sponsor("a", 1), _sponsor("b", 1)]

    with pytest.raises(ValueError, match="exactly one sponsor"):
        state.perform_sponsors_action(
            [
                SponsorPlaySelection(CardSource.HAND, 0),
                SponsorPlaySelection(CardSource.HAND, 1),
            ]
        )

    state2 = create_base_game(num_players=2, seed=43)
    player2 = state2.current()
    player2.action_upgraded[MainAction.SPONSORS] = False
    _set_sponsors_strength_slot(state2, 5)
    state2.display = [_sponsor("d1", 1), "x2", "x3", "x4", "x5", "x6"]
    with pytest.raises(ValueError, match="side I cannot play from display"):
        state2.perform_sponsors_action([SponsorPlaySelection(CardSource.DISPLAY, 0)])


def test_sponsors_side_ii_multiple_level_cap_and_display_cost():
    state = create_base_game(num_players=2, seed=44)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = True
    _set_sponsors_strength_slot(state, 4)
    player.reputation = 2
    player.money = 25
    player.hand = [_sponsor("h1", level=2, appeal_gain=1)]
    state.display = ["x1", "x2", _sponsor("d3", level=2, reputation_gain=1), "x4", "x5", "x6"]
    state.deck = ["new1"]

    state.perform_sponsors_action(
        [
            SponsorPlaySelection(CardSource.DISPLAY, 2),
            SponsorPlaySelection(CardSource.HAND, 0),
        ]
    )

    # level cap upgraded at strength 4 is 5
    assert len(player.played_sponsors) == 2
    assert player.money == 22  # folder 3 cost
    assert player.appeal == 1
    assert player.reputation == 3
    assert state.display == ["x1", "x2", "x4", "x5", "x6", "new1"]

    fail_state = create_base_game(num_players=2, seed=45)
    fail_player = fail_state.current()
    fail_player.action_upgraded[MainAction.SPONSORS] = True
    _set_sponsors_strength_slot(fail_state, 4)
    fail_player.hand = [_sponsor("a", 3), _sponsor("b", 3)]
    with pytest.raises(ValueError, match="exceeds allowed maximum"):
        fail_state.perform_sponsors_action(
            [
                SponsorPlaySelection(CardSource.HAND, 0),
                SponsorPlaySelection(CardSource.HAND, 0),
            ]
        )


def test_sponsors_display_requires_reputation_range():
    state = create_base_game(num_players=2, seed=46)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = True
    _set_sponsors_strength_slot(state, 4)
    player.reputation = 1
    state.display = ["x1", "x2", _sponsor("d3", 1), "x4", "x5", "x6"]

    with pytest.raises(ValueError, match="outside reputation range"):
        state.perform_sponsors_action([SponsorPlaySelection(CardSource.DISPLAY, 2)])


def test_sponsors_break_alternative_base_and_upgraded():
    base_state = create_base_game(num_players=2, seed=47)
    base_player = base_state.current()
    base_player.action_upgraded[MainAction.SPONSORS] = False
    _set_sponsors_strength_slot(base_state, 4)
    base_state.break_progress = 2
    money_before = base_player.money
    base_state.perform_sponsors_action(use_break_ability=True)
    assert base_player.money == money_before + 4
    assert base_state.break_progress == 6

    up_state = create_base_game(num_players=2, seed=48)
    up_player = up_state.current()
    up_player.action_upgraded[MainAction.SPONSORS] = True
    _set_sponsors_strength_slot(up_state, 4)
    up_state.break_progress = 2
    money_before_up = up_player.money
    up_state.perform_sponsors_action(use_break_ability=True)
    assert up_player.money == money_before_up + 8
    assert up_state.break_progress == 6


def test_sponsors_condition_checks():
    state = create_base_game(num_players=2, seed=49)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = False
    _set_sponsors_strength_slot(state, 5)
    player.hand = [
        _sponsor(
            "gated",
            level=3,
            min_reputation=2,
            required_partner_zoos={"africa"},
            required_icons={"science": 1},
        )
    ]

    with pytest.raises(ValueError, match="Reputation is too low"):
        state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])

    player.reputation = 2
    with pytest.raises(ValueError, match="Required partner zoo is missing"):
        state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])

    player.partner_zoos.add("africa")
    with pytest.raises(ValueError, match="Not enough icon 'science'"):
        state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])

    player.universities.add("university_1")
    state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])
    assert len(player.played_sponsors) == 1


def test_sponsors_upgraded_only_condition():
    state = create_base_game(num_players=2, seed=50)
    player = state.current()
    player.action_upgraded[MainAction.SPONSORS] = False
    _set_sponsors_strength_slot(state, 5)
    player.hand = [_sponsor("u", level=1, requires_upgraded_sponsors_action=True)]
    with pytest.raises(ValueError, match="requires upgraded Sponsors action"):
        state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])

    up_state = create_base_game(num_players=2, seed=51)
    up_player = up_state.current()
    up_player.action_upgraded[MainAction.SPONSORS] = True
    _set_sponsors_strength_slot(up_state, 1)
    up_player.hand = [_sponsor("u", level=1, requires_upgraded_sponsors_action=True)]
    up_state.perform_sponsors_action([SponsorPlaySelection(CardSource.HAND, 0)])
    assert len(up_player.played_sponsors) == 1

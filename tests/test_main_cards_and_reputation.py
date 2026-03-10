import pytest

from tests.helpers import make_state

from main import (
    Action,
    ActionType,
    _break_income_from_appeal,
    _prompt_cards_action_details_for_human,
    _increase_reputation,
    _reputation_display_limit,
    _validate_card_zones,
    apply_action,
    build_deck,
    card_zone_report,
    legal_actions,
)


def test_reputation_display_limit_mapping():
    assert _reputation_display_limit(0) == 0
    assert _reputation_display_limit(1) == 1
    assert _reputation_display_limit(2) == 2
    assert _reputation_display_limit(3) == 2
    assert _reputation_display_limit(4) == 3
    assert _reputation_display_limit(6) == 3
    assert _reputation_display_limit(7) == 4
    assert _reputation_display_limit(9) == 4
    assert _reputation_display_limit(10) == 5
    assert _reputation_display_limit(12) == 5
    assert _reputation_display_limit(13) == 6
    assert _reputation_display_limit(15) == 6


def test_break_income_from_appeal_piecewise_track_mapping():
    expected = {
        0: 5,
        4: 9,
        5: 10,
        6: 10,
        7: 11,
        16: 15,
        17: 16,
        19: 16,
        20: 17,
        31: 20,
        32: 21,
        35: 21,
        36: 22,
        55: 26,
        56: 27,
        60: 27,
        61: 28,
        95: 34,
        96: 35,
        101: 35,
        102: 36,
        113: 37,
    }
    for appeal, income in expected.items():
        assert _break_income_from_appeal(appeal) == income


def test_cards_action_non_upgraded_cannot_draw_from_display():
    state = make_state(501)
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "build", "animals", "association", "sponsors"]
    p0.action_upgraded["cards"] = False

    with pytest.raises(ValueError, match="Non-upgraded Cards cannot draw from display"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="cards",
                details={"from_display_indices": [0], "from_deck_count": 0},
            ),
        )


def test_cards_action_upgraded_respects_reputation_range_for_display_draw():
    state = make_state(502)
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["animals", "build", "cards", "association", "sponsors"]  # cards strength=3
    p0.action_upgraded["cards"] = True
    p0.reputation = 4  # display range = 3

    display_card_name = state.zoo_display[2].name
    hand_before = len(p0.hand)
    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="cards",
            details={"from_display_indices": [2], "from_deck_count": 0},
        ),
    )
    assert len(p0.hand) == hand_before + 1
    assert any(card.name == display_card_name for card in p0.hand)

    state_2 = make_state(503)
    p0_2 = state_2.players[0]
    state_2.current_player = 0
    p0_2.action_order = ["animals", "build", "cards", "association", "sponsors"]
    p0_2.action_upgraded["cards"] = True
    p0_2.reputation = 4  # range = 3; index 3 means folder 4, out of range
    with pytest.raises(ValueError, match="outside reputation range"):
        apply_action(
            state_2,
            Action(
                ActionType.MAIN_ACTION,
                card_name="cards",
                details={"from_display_indices": [3], "from_deck_count": 0},
            ),
        )


def test_cards_action_draw_and_discard_goes_to_global_discard():
    state = make_state(504)
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "build", "animals", "association", "sponsors"]
    p0.action_upgraded["cards"] = False
    state.break_progress = 0

    deck_before = len(state.zoo_deck)
    discard_before = len(state.zoo_discard)
    hand_before = len(p0.hand)
    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))
    pending_action = legal_actions(p0, state=state, player_id=state.current_player)[0]
    apply_action(state, pending_action)

    assert len(state.zoo_deck) == max(0, deck_before - 1)
    assert len(state.zoo_discard) >= discard_before + 1
    assert len(p0.hand) == hand_before


def test_cards_action_non_upgraded_requires_exact_deck_draw_count():
    state = make_state(506)
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["animals", "build", "cards", "association", "sponsors"]  # cards strength=3
    p0.action_upgraded["cards"] = False

    with pytest.raises(ValueError, match="must draw exactly 2 card"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="cards",
                details={"from_deck_count": 1, "discard_hand_indices": [0]},
            ),
        )


def test_cards_prompt_non_upgraded_does_not_ask_deck_count(monkeypatch):
    state = make_state(507)
    p0 = state.players[0]
    p0.action_upgraded["cards"] = False

    prompts = []

    def _fake_input(prompt: str) -> str:
        prompts.append(prompt)
        return "1"

    monkeypatch.setattr("builtins.input", _fake_input)
    details = _prompt_cards_action_details_for_human(state=state, player=p0, strength=3)

    assert details["from_deck_count"] == 2
    assert details["discard_hand_indices"] == [0]
    assert not any("Draw how many from deck?" in prompt for prompt in prompts)


def test_reputation_milestones_grant_expected_rewards(monkeypatch):
    state = make_state(505)
    p0 = state.players[0]
    p0.reputation = 4
    p0.workers = 1
    p0.conservation = 0
    p0.x_tokens = 0
    p0.action_upgraded = {action: False for action in p0.action_upgraded}
    p0.action_upgraded["association"] = True
    hand_before = len(p0.hand)

    monkeypatch.setattr("builtins.input", lambda _: "2")
    _increase_reputation(state, p0, 11, allow_interactive=True)  # 4 -> 15, crossing 5/8/10/11/12/13/14/15

    assert p0.reputation == 15
    assert p0.action_upgraded["cards"] is True  # from reputation 5 milestone
    assert p0.workers == 2  # from reputation 8 milestone
    assert p0.conservation == 2  # from 11 and 14 milestones
    assert p0.x_tokens == 2  # from 12 and 15 milestones
    assert len(p0.hand) >= hand_before + 2  # from 10 and 13 draw bonus


def test_reputation_caps_at_9_without_association_upgrade():
    state = make_state(509)
    p0 = state.players[0]
    p0.reputation = 8
    p0.action_upgraded["association"] = False

    _increase_reputation(state, p0, 3)

    assert p0.reputation == 9
    assert 10 not in p0.claimed_reputation_milestones


def test_reputation_overflow_above_15_converts_to_appeal_with_association_upgrade():
    state = make_state(510)
    p0 = state.players[0]
    p0.action_upgraded["association"] = True
    p0.reputation = 14
    p0.appeal = 3

    _increase_reputation(state, p0, 4)

    assert p0.reputation == 15
    assert p0.appeal == 6


def test_cards_use_number_and_instance_id_and_zone_integrity():
    deck = build_deck()
    assert deck
    assert all(card.number >= 0 for card in deck)
    assert all(card.instance_id for card in deck)
    assert len({card.instance_id for card in deck}) == len(deck)

    state = make_state(508)
    _validate_card_zones(state)
    report = card_zone_report(state)
    all_entries = [entry for zone in report.values() for entry in zone]
    zone_card_ids = [entry.split(":", 1)[0] for entry in all_entries]
    assert len(zone_card_ids) == len(set(zone_card_ids))

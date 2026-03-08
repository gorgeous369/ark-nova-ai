import pytest

import main

from main import (
    Action,
    ActionType,
    AnimalCard,
    Enclosure,
    HumanPlayer,
    MAIN_ACTION_CARDS,
    SetupCardRef,
    _rank_scores,
    _perform_animals_action_effect,
    _resolve_break,
    _prompt_sponsors_action_details_for_human,
    _resolve_manual_opening_drafts,
    apply_action,
    legal_actions,
    setup_game,
)


def test_setup_uses_8_choose_4_opening_and_break_9_in_two_player_game():
    state = setup_game(seed=41, player_names=["P1", "P2"])

    assert len(state.players) == 2
    assert state.break_max == 9
    assert state.break_progress == 0
    assert len(state.zoo_discard) == 8

    for player in state.players:
        assert len(player.opening_draft_drawn) == 8
        assert len(player.hand) == 4
        assert len(player.discard) == 0
        assert len(player.enclosures) == 0
        assert len(player.action_order) == 5
        assert player.action_order[0] == "animals"
        assert set(player.action_order) == set(MAIN_ACTION_CARDS)


def test_legal_actions_are_5_main_actions_plus_x_action():
    state = setup_game(seed=42, player_names=["P1", "P2"])
    player = state.players[state.current_player]
    actions = legal_actions(player)

    main_actions = [action for action in actions if action.type == ActionType.MAIN_ACTION]
    x_actions = [action for action in actions if action.type == ActionType.X_TOKEN]

    assert len(main_actions) == 5
    assert len(x_actions) == 1
    assert {action.card_name for action in main_actions} == set(player.action_order)


def test_legal_actions_expand_each_main_action_by_available_x_spend():
    state = setup_game(seed=142, player_names=["P1", "P2"])
    player = state.players[state.current_player]
    player.x_tokens = 2

    actions = legal_actions(player)
    main_actions = [action for action in actions if action.type == ActionType.MAIN_ACTION]
    x_actions = [action for action in actions if action.type == ActionType.X_TOKEN]

    assert len(main_actions) == 5 * 3
    assert len(x_actions) == 1
    for card_name in player.action_order:
        spends = sorted({int(action.value or 0) for action in main_actions if action.card_name == card_name})
        assert spends == [0, 1, 2]


def test_format_card_line_shows_sponsor_level_instead_of_zero_cost():
    sponsor = AnimalCard(
        "SPONSORSHIP: VULTURES",
        3,
        0,
        0,
        0,
        card_type="sponsor",
        required_icons=(("bird", 1),),
        number=233,
        instance_id="s-233",
    )

    rendered = main._format_card_line(sponsor)

    assert "level=3" in rendered
    assert "cost=" not in rendered
    assert "size=0" not in rendered


def test_state_legal_actions_expand_x_targets_when_state_is_provided():
    state = setup_game(seed=242, player_names=["P1", "P2"])
    player = state.players[state.current_player]

    actions = legal_actions(player, state=state, player_id=state.current_player)
    x_actions = [action for action in actions if action.type == ActionType.X_TOKEN]

    assert len(x_actions) == len(player.action_order)
    assert {action.card_name for action in x_actions} == set(player.action_order)


def test_cards_main_action_advances_break_by_2():
    state = setup_game(seed=43, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]

    hand_size_before = len(player.hand)
    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))
    pending_actions = legal_actions(player, state=state, player_id=state.current_player)

    assert state.break_progress == 2
    assert state.current_player == 0
    assert state.turn_index == 0
    assert state.pending_decision_kind == "cards_discard"
    assert player.action_order[0] == "cards"
    assert len(player.hand) == hand_size_before + 1
    assert pending_actions
    assert all(action.type == ActionType.PENDING_DECISION for action in pending_actions)


def test_main_action_spends_x_and_uses_higher_strength():
    state = setup_game(seed=143, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]  # sponsors base strength=5
    player.x_tokens = 2

    money_before = player.money
    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            value=2,
            details={"use_break_ability": True, "sponsor_selections": []},
        ),
    )

    assert player.money == money_before + 7
    assert player.x_tokens == 0
    assert state.break_progress == 7
    assert player.action_order[0] == "sponsors"


def test_build_action_requires_explicit_selection_when_multiple_placements_exist():
    state = setup_game(seed=144, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    order_before = list(player.action_order)

    with pytest.raises(
        ValueError,
        match="Build action requires explicit selections when multiple affordable legal placements exist.",
    ):
        apply_action(state, Action(ActionType.MAIN_ACTION, card_name="build"))

    assert player.action_order == order_before
    assert state.current_player == 0
    assert state.turn_index == 0


def test_x_action_moves_selected_action_to_slot_1_and_gains_x_token():
    state = setup_game(seed=44, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]

    apply_action(state, Action(ActionType.X_TOKEN, card_name="sponsors"))

    assert player.x_tokens == 1
    assert player.action_order[0] == "sponsors"
    assert state.current_player == 1
    assert state.turn_index == 1


def test_break_resolves_at_9_and_triggering_player_gets_1_x_token(monkeypatch):
    state = setup_game(seed=45, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    state.break_progress = 8
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.x_tokens = 0

    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))
    pending_action = legal_actions(player, state=state, player_id=state.current_player)[0]
    monkeypatch.setattr("builtins.input", lambda _: "1")
    apply_action(state, pending_action)

    assert state.break_progress == 0
    assert player.x_tokens == 1


def test_human_cards_choice_keeps_selected_x_spend(monkeypatch):
    state = setup_game(seed=145, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.x_tokens = 1

    actions = legal_actions(player)
    monkeypatch.setattr("main._prompt_cards_action_details_for_human", lambda **kwargs: {})
    monkeypatch.setattr("builtins.input", lambda _: "2")
    chosen = HumanPlayer().choose_action(state, actions)

    assert chosen.type == ActionType.MAIN_ACTION
    assert chosen.card_name == "cards"
    assert chosen.value == 1


def test_human_build_choice_keeps_selected_x_spend(monkeypatch):
    state = setup_game(seed=146, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.x_tokens = 1

    actions = legal_actions(player)
    monkeypatch.setattr("main._prompt_build_action_details_for_human", lambda **kwargs: {"selections": []})
    monkeypatch.setattr("builtins.input", lambda _: "4")
    chosen = HumanPlayer().choose_action(state, actions)

    assert chosen.type == ActionType.MAIN_ACTION
    assert chosen.card_name == "build"
    assert chosen.value == 1


def test_human_association_choice_keeps_selected_x_spend(monkeypatch):
    state = setup_game(seed=147, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "association", "animals", "sponsors"]
    player.x_tokens = 1

    actions = legal_actions(player)
    monkeypatch.setattr("main._prompt_association_action_details_for_human", lambda **kwargs: {})
    monkeypatch.setattr("builtins.input", lambda _: "6")
    chosen = HumanPlayer().choose_action(state, actions)

    assert chosen.type == ActionType.MAIN_ACTION
    assert chosen.card_name == "association"
    assert chosen.value == 1


def test_manual_opening_draft_allows_user_to_choose_kept_cards(monkeypatch):
    state = setup_game(
        seed=77,
        player_names=["P1", "P2"],
        manual_opening_draft_player_names={"P1"},
    )
    p1 = state.players[0]
    assert p1.opening_draft_kept_indices == []
    assert p1.hand == []
    assert p1.discard == []

    monkeypatch.setattr("builtins.input", lambda _: "1 2 3 4")
    _resolve_manual_opening_drafts(state, {"P1"})

    assert p1.opening_draft_kept_indices == [0, 1, 2, 3]
    assert [card.name for card in p1.hand] == [card.name for card in p1.opening_draft_drawn[:4]]


def test_sponsors_prompt_auto_falls_back_to_break_when_no_playable_card(monkeypatch):
    state = setup_game(seed=178, player_names=["P1", "P2"])
    player = state.players[0]
    sponsor_233 = next(card for card in state.zoo_deck if card.number == 233)
    state.zoo_deck.remove(sponsor_233)
    player.hand = [sponsor_233]
    player.zoo_cards = []
    player.action_upgraded["sponsors"] = False

    def _no_input(_: str) -> str:
        raise AssertionError("input() should not be called when only break option is legal.")

    monkeypatch.setattr("builtins.input", _no_input)
    details = _prompt_sponsors_action_details_for_human(
        state=state,
        player=player,
        strength=4,
    )

    assert details["use_break_ability"] is True
    assert details["sponsor_selections"] == []


def test_sponsors_prompt_lists_only_playable_candidates(monkeypatch):
    state = setup_game(seed=181, player_names=["P1", "P2"])
    player = state.players[0]
    sponsor_206 = next(card for card in state.zoo_deck if card.number == 206)
    sponsor_238 = next(card for card in state.zoo_deck if card.number == 238)
    state.zoo_deck.remove(sponsor_206)
    state.zoo_deck.remove(sponsor_238)
    player.hand = [sponsor_206, sponsor_238]
    player.zoo_cards = []
    player.action_upgraded["sponsors"] = False

    answers = iter(["p", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(answers))

    details = _prompt_sponsors_action_details_for_human(
        state=state,
        player=player,
        strength=5,
    )

    selection = details["sponsor_selections"][0]
    assert details["use_break_ability"] is False
    assert selection["card_instance_id"] == sponsor_238.instance_id
    assert selection["source_index"] == 1


def test_state_legal_actions_expand_sponsors_to_concrete_play_and_break():
    state = setup_game(seed=281, player_names=["P1", "P2"])
    player = state.players[0]
    sponsor_206 = next(card for card in state.zoo_deck if card.number == 206)
    sponsor_238 = next(card for card in state.zoo_deck if card.number == 238)
    state.zoo_deck.remove(sponsor_206)
    state.zoo_deck.remove(sponsor_238)
    player.hand = [sponsor_206, sponsor_238]
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]

    actions = legal_actions(player, state=state, player_id=0)
    sponsor_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "sponsors" and int(action.value or 0) == 0
    ]

    assert len(sponsor_actions) == 2
    assert any(action.details.get("use_break_ability") is True for action in sponsor_actions)
    concrete_plays = [
        action for action in sponsor_actions
        if action.details.get("use_break_ability") is False
    ]
    assert len(concrete_plays) == 1
    selection = concrete_plays[0].details["sponsor_selections"][0]
    assert selection["card_instance_id"] == sponsor_238.instance_id
    assert "ORNITHOLOGIST" in str(concrete_plays[0])
    assert "MEDICAL BREAKTHROUGH" not in " | ".join(str(action) for action in sponsor_actions)


def test_state_legal_actions_expand_cards_to_draw_actions_only():
    state = setup_game(seed=280, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["animals", "build", "cards", "association", "sponsors"]  # cards strength=3

    actions = legal_actions(player, state=state, player_id=0)
    cards_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards" and int(action.value or 0) == 0
    ]

    assert len(cards_actions) == 1
    assert cards_actions[0].details.get("from_deck_count") == 2
    assert "discard[" not in str(cards_actions[0])


def test_state_legal_actions_expand_project_support_with_map_unlock_draw_choices():
    state = setup_game(seed=286, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "sponsors", "association"]
    state.opening_setup.base_conservation_projects = [
        SetupCardRef(data_id="P101_SpeciesDiversity", title="SPECIES DIVERSITY")
    ]
    state.opening_setup.two_player_blocked_project_levels = []
    state.conservation_project_slots = {
        "P101_SpeciesDiversity": {
            "left_level": None,
            "middle_level": None,
            "right_level": None,
        }
    }
    player.zoo_cards = [
        AnimalCard("Bird", 0, 1, 0, 0, badges=("Bird",), instance_id="bird-1"),
        AnimalCard("Reptile", 0, 1, 0, 0, badges=("Reptile",), instance_id="reptile-1"),
        AnimalCard("Predator", 0, 1, 0, 0, badges=("Predator",), instance_id="pred-1"),
    ]

    actions = legal_actions(player, state=state, player_id=0)
    project_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "association"
        and "P101_SpeciesDiversity" in str(action)
    ]

    assert len(project_actions) == 1
    assert any("right_level: need>=3 -> +2 CP" in str(action) for action in project_actions)
    assert any("unlock: draw display[1]" in str(action) for action in project_actions)

    display_action = project_actions[0]
    expected_display_card = state.zoo_display[0]
    hand_before = len(player.hand)
    apply_action(state, display_action)

    assert player.map_left_track_unlocked_count == 1
    assert player.map_left_track_unlocked_effects == ["draw_1_card_deck_or_reputation_range"]
    assert len(player.hand) == hand_before + 1
    assert any(card.instance_id == expected_display_card.instance_id for card in player.hand)


def test_state_legal_actions_expand_build_to_concrete_selection():
    state = setup_game(seed=282, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]

    actions = legal_actions(player, state=state, player_id=0)
    build_action = next(
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "build" and int(action.value or 0) == 0
    )

    assert build_action.details.get("concrete") is True
    assert isinstance(build_action.details.get("selections"), list)
    assert len(build_action.details["selections"]) == 1


def test_state_legal_actions_keep_build_ii_parameterized():
    state = setup_game(seed=284, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_upgraded["build"] = True
    player.action_order = ["animals", "cards", "association", "sponsors", "build"]

    actions = legal_actions(player, state=state, player_id=0)
    build_action = next(
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "build" and int(action.value or 0) == 0
    )

    assert build_action.details.get("concrete") is not True
    assert "selections" not in (build_action.details or {})


def test_legal_actions_marks_sponsors_break_only_when_no_playable_sponsor():
    state = setup_game(seed=179, player_names=["P1", "P2"])
    player = state.players[0]
    sponsor_233 = next(card for card in state.zoo_deck if card.number == 233)
    state.zoo_deck.remove(sponsor_233)
    player.hand = [sponsor_233]
    player.zoo_cards = []
    player.action_order = ["association", "cards", "sponsors", "build", "animals"]  # sponsors strength=3

    actions = legal_actions(player, state=state, player_id=0)
    sponsors_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "sponsors" and int(action.value or 0) == 0
    ]
    assert len(sponsors_actions) == 1
    sponsors_action = sponsors_actions[0]
    assert sponsors_action.details.get("sponsors_break_only") is True
    assert str(sponsors_action) == "sponsors_break_only(strength=3)"


def test_legal_actions_hide_association_when_no_workers():
    state = setup_game(seed=180, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["cards", "association", "animals", "build", "sponsors"]  # association strength=2
    player.workers = 0
    player.workers_on_association_board = 1
    player.x_tokens = 1

    actions = legal_actions(player, state=state, player_id=0)
    association_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "association"
    ]
    assert association_actions == []


def test_break_resolves_in_rule_order_for_display_workers_and_temp_tokens(monkeypatch):
    state = setup_game(seed=181, player_names=["P1", "P2"])
    p0, p1 = state.players

    state.zoo_display = [
        AnimalCard("D1", 0, 0, 0, 0, number=9001, instance_id="d1"),
        AnimalCard("D2", 0, 0, 0, 0, number=9002, instance_id="d2"),
        AnimalCard("D3", 0, 0, 0, 0, number=9003, instance_id="d3"),
        AnimalCard("D4", 0, 0, 0, 0, number=9004, instance_id="d4"),
        AnimalCard("D5", 0, 0, 0, 0, number=9005, instance_id="d5"),
        AnimalCard("D6", 0, 0, 0, 0, number=9006, instance_id="d6"),
    ]
    state.zoo_deck = [
        AnimalCard("N1", 0, 0, 0, 0, number=9011, instance_id="n1"),
        AnimalCard("N2", 0, 0, 0, 0, number=9012, instance_id="n2"),
    ]
    state.zoo_discard = []

    p0.hand_limit = 3
    p0.hand = [
        AnimalCard("H1", 0, 0, 0, 0, number=9101, instance_id="h1"),
        AnimalCard("H2", 0, 0, 0, 0, number=9102, instance_id="h2"),
        AnimalCard("H3", 0, 0, 0, 0, number=9103, instance_id="h3"),
        AnimalCard("H4", 0, 0, 0, 0, number=9104, instance_id="h4"),
        AnimalCard("H5", 0, 0, 0, 0, number=9105, instance_id="h5"),
    ]
    p0.multiplier_tokens_on_actions["cards"] = 1
    p0.venom_tokens_on_actions["cards"] = 1
    p0.venom_tokens_on_actions["build"] = 1
    p0.constriction_tokens_on_actions["association"] = 1
    p0.constriction_tokens_on_actions["sponsors"] = 1
    p0.constriction_tokens_on_actions["animals"] = 1
    p0.workers = 0
    p0.workers_on_association_board = 2
    p0.association_workers_by_task["reputation"] = 2
    p0.partner_zoos = {"asia"}
    p1.partner_zoos = {"asia"}

    responses = iter(["4 5", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    _resolve_break(state)

    assert [card.instance_id for card in state.zoo_display] == ["d3", "d4", "d5", "d6", "n1", "n2"]
    assert "d1" in {card.instance_id for card in state.zoo_discard}
    assert "d2" in {card.instance_id for card in state.zoo_discard}
    assert "h4" in {card.instance_id for card in state.zoo_discard}
    assert "h5" in {card.instance_id for card in state.zoo_discard}
    assert len(p0.hand) == 3

    assert p0.multiplier_tokens_on_actions["cards"] == 0
    assert sum(p0.venom_tokens_on_actions.values()) == 0
    assert sum(p0.constriction_tokens_on_actions.values()) == 0

    assert p0.workers == 2
    assert p0.workers_on_association_board == 0
    assert p0.association_workers_by_task["reputation"] == 0
    assert "asia" not in state.available_partner_zoos


def test_break_income_order_starts_from_trigger_player_when_contested():
    state = setup_game(seed=182, player_names=["P1", "P2"])
    p0, p1 = state.players
    p0.hand = []
    p1.hand = []
    state.zoo_deck = []
    state.zoo_discard = []
    state.zoo_display = [
        AnimalCard("A", 0, 0, 0, 0, number=9201, instance_id="a"),
        AnimalCard("B", 0, 0, 0, 0, number=9202, instance_id="b"),
        AnimalCard("C", 0, 0, 0, 0, number=9203, instance_id="c"),
        AnimalCard("D", 0, 0, 0, 0, number=9204, instance_id="d"),
    ]
    p0.zoo_cards = [AnimalCard("S201-0", 0, 0, 0, 0, card_type="sponsor", number=201, instance_id="s201-0")]
    p1.zoo_cards = [AnimalCard("S201-1", 0, 0, 0, 0, card_type="sponsor", number=201, instance_id="s201-1")]

    state.break_trigger_player = 1
    _resolve_break(state)

    # After step 4, display becomes [C, D]. Trigger player (P2) takes C first, then P1 takes D.
    assert [card.instance_id for card in p1.hand] == ["c"]
    assert [card.instance_id for card in p0.hand] == ["d"]


def test_legal_actions_show_strength_reduced_by_constriction():
    state = setup_game(seed=183, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.constriction_tokens_on_actions["cards"] = 1
    player.x_tokens = 1

    actions = legal_actions(player, state=state, player_id=0)
    cards_zero = next(
        action for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards" and int(action.value or 0) == 0
    )
    cards_one = next(
        action for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards" and int(action.value or 0) == 1
    )

    assert cards_zero.details["effective_strength"] == 0
    assert cards_one.details["effective_strength"] == 0
    assert str(cards_zero) == "cards(strength=0)"


def test_human_choose_action_uses_concrete_state_actions_without_secondary_prompt(monkeypatch):
    state = setup_game(seed=283, player_names=["P1", "P2"])
    player = state.players[0]
    sponsor_206 = next(card for card in state.zoo_deck if card.number == 206)
    sponsor_238 = next(card for card in state.zoo_deck if card.number == 238)
    state.zoo_deck.remove(sponsor_206)
    state.zoo_deck.remove(sponsor_238)
    player.hand = [sponsor_206, sponsor_238]
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]

    actions = legal_actions(player, state=state, player_id=0)
    concrete_index = next(
        idx
        for idx, action in enumerate(actions, start=1)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "sponsors"
        and "ORNITHOLOGIST" in str(action)
    )

    monkeypatch.setattr(
        "main._prompt_sponsors_action_details_for_human",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("secondary sponsors prompt should not be used")),
    )
    monkeypatch.setattr("builtins.input", lambda _: str(concrete_index))

    chosen = HumanPlayer().choose_action(state, actions)

    assert chosen.type == ActionType.MAIN_ACTION
    assert chosen.card_name == "sponsors"
    assert chosen.details.get("concrete") is True
    assert chosen.details.get("_interactive") is True
    assert chosen.details["sponsor_selections"][0]["card_instance_id"] == sponsor_238.instance_id


def test_venom_penalty_applies_if_turn_ends_with_remaining_venom_tokens():
    state = setup_game(seed=184, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.money = 10
    player.venom_tokens_on_actions["build"] = 1

    apply_action(state, Action(ActionType.X_TOKEN, card_name="cards"))

    assert player.money == 8
    assert player.x_tokens == 1
    assert player.venom_tokens_on_actions["build"] == 1


def test_using_venom_marked_action_discards_token_and_avoids_penalty():
    state = setup_game(seed=185, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.money = 10
    player.venom_tokens_on_actions["cards"] = 1

    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))
    assert state.pending_decision_kind == "cards_discard"
    assert player.money == 10
    assert player.venom_tokens_on_actions["cards"] == 0

    pending_action = legal_actions(player, state=state, player_id=state.current_player)[0]
    apply_action(state, pending_action)

    assert player.money == 10
    assert player.venom_tokens_on_actions["cards"] == 0


def test_cards_pending_discard_actions_use_card_instance_ids_and_finalize_turn():
    state = setup_game(seed=285, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]

    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))

    assert state.pending_decision_kind == "cards_discard"
    pending_actions = legal_actions(player, state=state, player_id=state.current_player)
    assert pending_actions
    assert all(action.type == ActionType.PENDING_DECISION for action in pending_actions)
    assert all(action.details.get("pending_kind") == "cards_discard" for action in pending_actions)
    assert all(len(action.details.get("discard_card_instance_ids") or []) == 1 for action in pending_actions)
    assert all("discard #" in str(action) for action in pending_actions)

    chosen = pending_actions[0]
    discard_id = chosen.details["discard_card_instance_ids"][0]
    assert any(card.instance_id == discard_id for card in player.hand)

    apply_action(state, chosen)

    assert state.pending_decision_kind == ""
    assert state.current_player == 1
    assert state.turn_index == 1


def test_sell_hand_cards_effect_prompts_for_specific_cards(monkeypatch):
    state = setup_game(seed=291, player_names=["P1", "P2"])
    player = state.players[0]
    player.money = 5
    player.hand = [
        AnimalCard(
            "Seller",
            0,
            1,
            0,
            0,
            ability_title="Sun Bathing 2",
            ability_text="You may sell up to 2 card(s) from your hand for 4",
            instance_id="seller",
        ),
        AnimalCard("Keep A", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-a"),
        AnimalCard("Keep B", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-b"),
    ]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    monkeypatch.setattr("builtins.input", lambda _: "1 2")

    _perform_animals_action_effect(
        state=state,
        player=player,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0, "_interactive": True},
    )

    assert player.money == 13
    assert player.hand == []
    assert {card.instance_id for card in state.zoo_discard} >= {"keep-a", "keep-b"}


def test_sell_hand_cards_effect_expands_animals_legal_actions_into_card_subsets():
    state = setup_game(seed=292, player_names=["P1", "P2"])
    player = state.players[0]
    player.money = 5
    player.hand = [
        AnimalCard(
            "Seller",
            0,
            1,
            0,
            0,
            ability_title="Sun Bathing 2",
            ability_text="You may sell up to 2 card(s) from your hand for 4",
            instance_id="seller",
        ),
        AnimalCard("Keep A", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-a"),
        AnimalCard("Keep B", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-b"),
        AnimalCard("Keep C", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-c"),
    ]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "Seller" in str((action.details or {}).get("action_label") or "")
    ]

    assert len(animal_actions) == 7

    sell_choice_sets = {
        tuple(choice.get("card_instance_ids") or [])
        for action in animal_actions
        for choice in list((action.details or {}).get("sell_hand_card_choices") or [])
    }
    assert sell_choice_sets == {
        (),
        ("keep-a",),
        ("keep-b",),
        ("keep-c",),
        ("keep-a", "keep-b"),
        ("keep-a", "keep-c"),
        ("keep-b", "keep-c"),
    }

    chosen = next(
        action
        for action in animal_actions
        if [tuple(choice.get("card_instance_ids") or []) for choice in list((action.details or {}).get("sell_hand_card_choices") or [])]
        == [("keep-a", "keep-c")]
    )

    apply_action(state, chosen)

    assert player.money == 13
    assert [card.instance_id for card in player.hand] == ["keep-b"]
    assert {card.instance_id for card in state.zoo_discard} >= {"keep-a", "keep-c"}


def test_pouch_effect_expands_animals_legal_actions_and_moves_cards_under_host():
    state = setup_game(seed=293, player_names=["P1", "P2"])
    player = state.players[0]
    player.money = 5
    player.hand = [
        AnimalCard(
            "Poucher",
            0,
            1,
            0,
            0,
            ability_title="Pouch 1",
            ability_text="You may place 1 card(s) from your hand under this card to gain 2",
            instance_id="poucher",
        ),
        AnimalCard("Keep A", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-a"),
    ]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "Poucher" in str((action.details or {}).get("action_label") or "")
    ]

    assert len(animal_actions) == 2
    assert {
        tuple(choice.get("card_instance_ids") or [])
        for action in animal_actions
        for choice in list((action.details or {}).get("pouch_hand_card_choices") or [])
    } == {(), ("keep-a",)}

    chosen = next(
        action
        for action in animal_actions
        if [tuple(choice.get("card_instance_ids") or []) for choice in list((action.details or {}).get("pouch_hand_card_choices") or [])]
        == [("keep-a",)]
    )

    apply_action(state, chosen)

    assert player.appeal == 2
    assert player.hand == []
    assert [card.instance_id for card in player.pouched_cards] == ["keep-a"]
    assert [card.instance_id for card in player.pouched_cards_by_host["poucher"]] == ["keep-a"]
    assert all(card.instance_id != "keep-a" for card in state.zoo_discard)


def test_concrete_sell_choice_does_not_prompt_again_when_interactive(monkeypatch):
    state = setup_game(seed=294, player_names=["P1", "P2"])
    player = state.players[0]
    player.money = 5
    player.hand = [
        AnimalCard(
            "Seller",
            0,
            1,
            0,
            0,
            ability_title="Sun Bathing 1",
            ability_text="You may sell up to 1 card(s) from your hand for 4",
            instance_id="seller",
        ),
        AnimalCard("Keep A", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-a"),
    ]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    chosen = next(
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and [tuple(choice.get("card_instance_ids") or []) for choice in list((action.details or {}).get("sell_hand_card_choices") or [])]
        == [("keep-a",)]
    )

    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(AssertionError("unexpected prompt")))

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            value=chosen.value,
            card_name=chosen.card_name,
            details={**dict(chosen.details or {}), "_interactive": True},
        ),
    )

    assert player.money == 9
    assert player.hand == []
    assert {card.instance_id for card in state.zoo_discard} >= {"keep-a"}


def test_sponsors_concrete_actions_prune_higher_x_duplicates_with_same_payload():
    state = setup_game(seed=295, player_names=["P1", "P2"])
    player = state.players[0]
    player.x_tokens = 2
    player.action_order = ["cards", "build", "sponsors", "animals", "association"]
    player.hand = [
        AnimalCard("SPONSORSHIP: LIONS", 3, 0, 0, 0, card_type="sponsor", required_icons=(("predator", 1),), number=234, instance_id="s-234")
    ]
    player.zoo_cards = [
        AnimalCard("Predator Host", 0, 2, 0, 0, badges=("Predator",), card_type="animal", instance_id="predator-host")
    ]

    actions = legal_actions(player, state=state, player_id=0)
    sponsor_play_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "sponsors"
        and not bool((action.details or {}).get("use_break_ability"))
    ]

    assert len(sponsor_play_actions) == 1
    assert sponsor_play_actions[0].value in {None, 0}
    assert "SPONSORSHIP: LIONS" in str(sponsor_play_actions[0])


def test_association_concrete_actions_prune_higher_x_duplicates_with_same_payload():
    state = setup_game(seed=296, player_names=["P1", "P2"])
    player = state.players[0]
    player.x_tokens = 1
    player.money = 20
    player.workers = 1
    player.action_upgraded["association"] = True
    player.action_order = ["cards", "build", "animals", "sponsors", "association"]

    actions = legal_actions(player, state=state, player_id=0)
    association_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "association"
    ]

    assert association_actions
    assert all(int(action.value or 0) == 0 for action in association_actions)


def test_build_concrete_actions_prune_higher_x_duplicates_but_keep_new_size_options():
    state = setup_game(seed=297, player_names=["P1", "P2"])
    player = state.players[0]
    player.x_tokens = 1
    player.money = 50
    player.action_order = ["build", "animals", "cards", "sponsors", "association"]

    actions = legal_actions(player, state=state, player_id=0)
    build_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "build"
    ]

    assert any(int(action.value or 0) == 1 for action in build_actions)

    duplicate_small_builds = [
        action
        for action in build_actions
        if int(action.value or 0) == 1
        and (
            (action.details or {}).get("selections", [{}])[0].get("building_type")
            in {"SIZE_1", "KIOSK", "PAVILION"}
        )
    ]
    assert duplicate_small_builds == []

    size2_actions = [
        action
        for action in build_actions
        if (action.details or {}).get("selections", [{}])[0].get("building_type") == "SIZE_2"
    ]
    assert size2_actions
    assert any(int(action.value or 0) == 1 for action in size2_actions)


def test_playing_venom_animal_marks_leftmost_actions_of_higher_appeal_zoo():
    state = setup_game(seed=186, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Venom Beast",
            0,
            1,
            0,
            0,
            ability_title="Venom 2",
            number=9901,
            instance_id="venom-1",
        )
    ]
    actor.money = 10
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.action_order = ["cards", "build", "animals", "association", "sponsors"]

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert target.venom_tokens_on_actions["cards"] == 1
    assert target.venom_tokens_on_actions["build"] == 1
    assert sum(target.venom_tokens_on_actions.values()) == 2


def test_playing_constriction_animal_uses_only_appeal_and_conservation_tracks():
    state = setup_game(seed=187, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Constriction Beast",
            0,
            1,
            0,
            0,
            ability_title="Constriction",
            number=9902,
            instance_id="constrict-1",
        )
    ]
    actor.money = 10
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 0
    target.conservation = 0
    target.reputation = 7
    target.action_order = ["cards", "build", "animals", "association", "sponsors"]

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert sum(target.constriction_tokens_on_actions.values()) == 0


def test_playing_constriction_animal_marks_rightmost_action_when_target_is_ahead():
    state = setup_game(seed=287, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Constriction Beast",
            0,
            1,
            0,
            0,
            ability_title="Constriction",
            number=9905,
            instance_id="constrict-2",
        )
    ]
    actor.money = 10
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 4
    target.action_order = ["cards", "build", "animals", "association", "sponsors"]

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert target.constriction_tokens_on_actions["sponsors"] == 1
    assert sum(target.constriction_tokens_on_actions.values()) == 1


def test_hypnosis_uses_target_action_card_and_rotates_target_card_to_slot_1(monkeypatch):
    state = setup_game(seed=188, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Hypnosis Beast",
            0,
            1,
            0,
            0,
            ability_title="Hypnosis 3",
            number=9903,
            instance_id="hyp-1",
        )
    ]
    actor.money = 10
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.action_order = ["cards", "build", "animals", "association", "sponsors"]

    monkeypatch.setattr(main, "_prompt_build_action_details_for_human", lambda **kwargs: {"selections": []})
    monkeypatch.setattr("builtins.input", lambda _: "2")

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0, "_interactive": True},
    )

    assert target.action_order[0] == "build"


def test_pilfering_takes_five_money_from_highest_appeal_target():
    state = setup_game(seed=189, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=9904,
            instance_id="pilfer-1",
        )
    ]
    actor.money = 2
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.money = 11
    target.hand = []

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert actor.money == 7
    assert target.money == 6


def test_pilfering_noninteractive_requires_explicit_choice_when_both_losses_are_legal():
    state = setup_game(seed=190, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=9905,
            instance_id="pilfer-2",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    actor.money = 2
    target.appeal = 5
    target.money = 11
    target.hand = [
        AnimalCard("Expensive Animal", 12, 4, 3, 1, number=9910, instance_id="expensive"),
        AnimalCard("Cheap Animal", 1, 1, 0, 0, number=9911, instance_id="cheap"),
    ]

    with pytest.raises(ValueError, match="explicit pilfering_choices"):
        _perform_animals_action_effect(
            state=state,
            player=actor,
            player_id=0,
            strength=3,
            details={"animals_sequence_index": 0},
        )


def test_pilfering_noninteractive_uses_explicit_card_choice():
    state = setup_game(seed=190, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=9905,
            instance_id="pilfer-2",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    actor.money = 2
    target.appeal = 5
    target.money = 11
    target.hand = [
        AnimalCard("Expensive Animal", 12, 4, 3, 1, number=9910, instance_id="expensive"),
        AnimalCard("Cheap Animal", 1, 1, 0, 0, number=9911, instance_id="cheap"),
    ]

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={
            "animals_sequence_index": 0,
            "pilfering_choices": [{"choice": "card", "card_instance_id": "cheap"}],
        },
    )

    assert actor.money == 2
    assert target.money == 11
    assert [card.instance_id for card in actor.hand] == ["cheap"]
    assert [card.instance_id for card in target.hand] == ["expensive"]


def test_pilfering_interactive_target_can_choose_specific_card(monkeypatch):
    state = setup_game(seed=191, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=9906,
            instance_id="pilfer-3",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.money = 11
    target.hand = [
        AnimalCard("Card A", 2, 1, 0, 0, number=9912, instance_id="card-a"),
        AnimalCard("Card B", 8, 3, 0, 0, number=9913, instance_id="card-b"),
    ]

    responses = iter(["2", "2"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0, "_interactive": True},
    )

    assert actor.money == 25
    assert target.money == 11
    assert [card.instance_id for card in actor.hand] == ["card-b"]
    assert [card.instance_id for card in target.hand] == ["card-a"]


def test_pilfering_does_not_prompt_when_only_card_loss_is_available(monkeypatch):
    state = setup_game(seed=192, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=9907,
            instance_id="pilfer-4",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.money = 4
    target.hand = [
        AnimalCard("Only Card", 3, 2, 0, 0, number=9914, instance_id="only-card"),
    ]

    def _unexpected_input(_: str) -> str:
        raise AssertionError("input() should not be called when there is no pilfering choice.")

    monkeypatch.setattr("builtins.input", _unexpected_input)

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0, "_interactive": True},
    )

    assert actor.money == 25
    assert target.money == 4
    assert [card.instance_id for card in actor.hand] == ["only-card"]
    assert target.hand == []


def test_game_ends_after_rest_of_round_when_score_reaches_100():
    state = setup_game(seed=193, player_names=["P1", "P2"])
    state.players[0].appeal = 100
    state.current_player = 0

    apply_action(state, Action(ActionType.X_TOKEN, card_name="cards"))

    assert state.endgame_trigger_player == 0
    assert state.current_player == 1
    assert not state.game_over()

    apply_action(state, Action(ActionType.X_TOKEN, card_name="cards"))

    assert state.current_player == 0
    assert state.game_over()


def test_rank_scores_assigns_same_rank_to_ties():
    ranking = _rank_scores({"P1": 100, "P2": 95, "P3": 95, "P4": 80})

    assert ranking == [
        (1, "P1", 100),
        (2, "P2", 95),
        (2, "P3", 95),
        (4, "P4", 80),
    ]

import main

from main import (
    Action,
    ActionType,
    AnimalCard,
    Enclosure,
    HumanPlayer,
    MAIN_ACTION_CARDS,
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


def test_cards_main_action_advances_break_by_2():
    state = setup_game(seed=43, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]

    hand_size_before = len(player.hand)
    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))

    assert state.break_progress == 2
    assert state.current_player == 1
    assert state.turn_index == 1
    assert player.action_order[0] == "cards"
    assert len(player.hand) == hand_size_before


def test_main_action_spends_x_and_uses_higher_strength():
    state = setup_game(seed=143, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]  # sponsors base strength=5
    player.x_tokens = 2

    money_before = player.money
    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="sponsors", value=2))

    assert player.money == money_before + 7
    assert player.x_tokens == 0
    assert state.break_progress == 7
    assert player.action_order[0] == "sponsors"


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

    monkeypatch.setattr("builtins.input", lambda _: "1")
    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="cards"))

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

    assert player.money == 10
    assert player.venom_tokens_on_actions["cards"] == 0


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


def test_pilfering_noninteractive_target_can_choose_card_instead_of_money():
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
        details={"animals_sequence_index": 0},
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

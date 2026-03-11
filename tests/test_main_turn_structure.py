import copy

import pytest

import main

from tests.helpers import find_action, has_action, make_state, materialize_first_action

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
    _prompt_break_discard_indices,
    _prompt_opening_draft_indices,
    _resolve_break,
    _prompt_sponsors_action_details_for_human,
    _resolve_manual_opening_drafts,
    apply_action,
    legal_actions,
    setup_game,
)


def test_setup_uses_8_choose_4_opening_and_break_9_in_two_player_game():
    state = make_state(41)

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
    state = make_state(42)
    player = state.players[state.current_player]
    actions = legal_actions(player)

    main_actions = [action for action in actions if action.type == ActionType.MAIN_ACTION]
    x_actions = [action for action in actions if action.type == ActionType.X_TOKEN]

    assert len(main_actions) == 5
    assert len(x_actions) == 1
    assert {action.card_name for action in main_actions} == set(player.action_order)


def test_legal_actions_expand_each_main_action_by_available_x_spend():
    state = make_state(142)
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


def test_conservation_space_2_pending_offers_upgrade_and_activate_worker_choices():
    state = make_state(2041)
    p0 = state.players[0]
    money_before = p0.money
    p0.conservation = 2

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    assert state.pending_decision_kind == "conservation_reward"

    actions = legal_actions(p0, state=state, player_id=0)
    assert has_action(actions, reward="activate_association_worker")
    assert has_action(actions, reward="upgrade_action_card", upgraded_action="animals")

    worker_action = find_action(actions, reward="activate_association_worker")
    apply_action(state, worker_action)

    assert p0.workers == 2
    assert p0.money == money_before
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_5_pending_expands_choiceful_bonus_tiles():
    state = make_state(2042)
    p0 = state.players[0]
    p0.conservation = 5
    p0.claimed_conservation_reward_spaces.add(2)
    p0.hand.clear()
    p0.hand.append(
        AnimalCard(
            "ORNITHOLOGIST",
            4,
            0,
            0,
            0,
            card_type="sponsor",
            number=238,
            instance_id="s-238",
        )
    )
    state.shared_conservation_bonus_tiles[5] = [
        "size_3_enclosure",
        "partner_zoo",
        "university",
        "x2_multiplier",
        "sponsor_card",
        "10_money",
        "2_reputation",
        "3_x_tokens",
        "3_cards",
    ]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    assert has_action(actions, reward="5coins")
    assert has_action(actions, reward="size_3_enclosure")
    assert has_action(actions, reward="partner_zoo", partner_zoo="africa")
    assert has_action(actions, reward="university", university="reputation_1_hand_limit_5")
    assert has_action(
        actions,
        reward="x2_multiplier",
        predicate=lambda action: str((action.details or {}).get("multiplier_action") or "") in MAIN_ACTION_CARDS,
    )
    assert has_action(
        actions,
        reward="sponsor_card",
        predicate=lambda action: bool((action.details or {}).get("sponsor_details")),
    )
    assert has_action(actions, reward="10_money")
    assert has_action(actions, reward="2_reputation")
    assert has_action(actions, reward="3_x_tokens")
    assert has_action(actions, reward="3_cards")


def test_conservation_space_5_reputation_reward_expands_reputation_milestone_upgrade_choices():
    state = make_state(2044)
    p0 = state.players[0]
    p0.conservation = 5
    p0.claimed_conservation_reward_spaces.add(2)
    p0.reputation = 4
    state.shared_conservation_bonus_tiles[5] = ["2_reputation"]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    chosen = find_action(
        actions,
        reward="2_reputation",
        predicate=lambda action: any(
            item.get("upgraded_action") == "build"
            for item in list((action.details or {}).get("reputation_milestone_reward_choices") or [])
            if isinstance(item, dict)
        ),
    )
    apply_action(state, chosen)

    assert p0.reputation == 6
    assert p0.action_upgraded["build"] is True
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_5_university_reward_expands_university_reputation_milestone_choices():
    state = make_state(2047)
    p0 = state.players[0]
    p0.conservation = 5
    p0.claimed_conservation_reward_spaces.add(2)
    p0.reputation = 4
    state.shared_conservation_bonus_tiles[5] = ["university"]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    chosen = find_action(
        actions,
        reward="university",
        university="reputation_1_hand_limit_5",
        predicate=lambda action: any(
            item.get("upgraded_action") == "build"
            for item in list((action.details or {}).get("reputation_milestone_reward_choices") or [])
            if isinstance(item, dict)
        ),
    )
    apply_action(state, chosen)

    assert p0.reputation == 5
    assert "reputation_1_hand_limit_5" in p0.universities
    assert p0.action_upgraded["build"] is True
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_5_reputation_reward_rep8_worker_uses_post_gain_reputation_range():
    state = setup_game(seed=2048, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.conservation = 5
    p0.claimed_conservation_reward_spaces.add(2)
    p0.reputation = 6
    state.shared_conservation_bonus_tiles[5] = ["2_reputation"]
    state.map_rules["worker_gain_rewards"] = [{"effect": "draw_1_card_deck_or_reputation_range"}]
    hand_before = len(p0.hand)

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    labels = [str((action.details or {}).get("action_label") or "") for action in actions]
    assert any("rep8 worker(display[4]" in label for label in labels)

    chosen = next(
        action
        for action in actions
        if "rep8 worker(display[4]" in str((action.details or {}).get("action_label") or "")
    )
    apply_action(state, chosen)

    assert p0.reputation == 8
    assert p0.workers == 2
    assert len(p0.hand) == hand_before + 1
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_5_partner_threshold_gain_worker_expands_worker_reward_choices():
    state = setup_game(seed=2049, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.conservation = 5
    p0.claimed_conservation_reward_spaces.add(2)
    state.shared_conservation_bonus_tiles[5] = ["partner_zoo"]
    state.map_rules["partner_zoo_threshold_rewards"] = [{"count": 1, "effect": "gain_worker_1"}]
    state.map_rules["worker_gain_rewards"] = [{"effect": "upgrade_action_card"}]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    labels = [str((action.details or {}).get("action_label") or "") for action in actions]
    assert "cons[5] | tile(partner(Africa)) ; partner-threshold-1(+1 worker) ; worker-reward(build)" in labels

    chosen = next(
        action
        for action in actions
        if (action.details or {}).get("action_label")
        == "cons[5] | tile(partner(Africa)) ; partner-threshold-1(+1 worker) ; worker-reward(build)"
    )
    apply_action(state, chosen)

    assert p0.workers == 2
    assert p0.action_upgraded["build"] is True
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_5_choice_can_flatten_followup_space_8_reward_in_same_action():
    state = setup_game(seed=2050, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.conservation = 6
    p0.claimed_conservation_reward_spaces.add(2)
    state.shared_conservation_bonus_tiles[5] = ["partner_zoo"]
    state.shared_conservation_bonus_tiles[8] = ["10_money"]
    state.map_rules["partner_zoo_threshold_rewards"] = [{"count": 1, "effect": "gain_conservation_2"}]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    labels = [str((action.details or {}).get("action_label") or "") for action in actions]
    assert "cons[5] | tile(partner(Africa)) ; partner-threshold-1(+2 conservation) ; cons[8] | +10 money" in labels

    chosen = next(
        action
        for action in actions
        if (action.details or {}).get("action_label")
        == "cons[5] | tile(partner(Africa)) ; partner-threshold-1(+2 conservation) ; cons[8] | +10 money"
    )
    apply_action(state, chosen)

    assert p0.conservation == 8
    assert "africa" in {value.lower() for value in p0.partner_zoos}
    assert p0.money >= 10
    assert p0.claimed_conservation_reward_spaces.issuperset({2, 5, 8})
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_2_activate_worker_expands_map_worker_reward_choices():
    state = setup_game(seed=2045, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.conservation = 2
    state.map_rules["worker_gain_rewards"] = [{"effect": "upgrade_action_card"}]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    labels = [str((action.details or {}).get("action_label") or "") for action in actions]

    assert "cons[2] | activate worker ; worker-reward(build)" in labels

    chosen = next(
        action
        for action in actions
        if (action.details or {}).get("action_label") == "cons[2] | activate worker ; worker-reward(build)"
    )
    apply_action(state, chosen)

    assert p0.workers == 2
    assert p0.action_upgraded["build"] is True
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_conservation_space_5_partner_zoo_expands_map_threshold_reward_choices():
    state = setup_game(seed=2046, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.conservation = 5
    p0.claimed_conservation_reward_spaces.add(2)
    state.shared_conservation_bonus_tiles[5] = ["partner_zoo"]
    state.map_rules["partner_zoo_threshold_rewards"] = [{"count": 1, "effect": "upgrade_action_card"}]

    started = main._maybe_begin_conservation_reward_pending(
        state,
        player_id=0,
        resume_kind="turn_finalize",
    )

    assert started is True
    actions = legal_actions(p0, state=state, player_id=0)
    labels = [str((action.details or {}).get("action_label") or "") for action in actions]

    assert "cons[5] | tile(partner(Africa)) ; partner-threshold-1(build)" in labels

    chosen = next(
        action
        for action in actions
        if (action.details or {}).get("action_label") == "cons[5] | tile(partner(Africa)) ; partner-threshold-1(build)"
    )
    apply_action(state, chosen)

    assert "africa" in {value.lower() for value in p0.partner_zoos}
    assert p0.action_upgraded["build"] is True
    assert state.pending_decision_kind == ""
    assert state.current_player == 1


def test_break_conservation_rewards_pause_in_order_and_shared_tile_is_consumed():
    state = setup_game(seed=2043, player_names=["P1", "P2"])
    p0, p1 = state.players
    state.break_trigger_player = 0
    state.break_progress = state.break_max
    state.shared_conservation_bonus_tiles[5] = ["10_money"]

    for idx, player in enumerate((p0, p1), start=1):
        player.conservation = 4
        player.claimed_conservation_reward_spaces.add(2)
        player.appeal = 0
        player.hand.clear()
        player.zoo_cards.append(
            AnimalCard(
                f"Medical-{idx}",
                0,
                0,
                0,
                0,
                card_type="sponsor",
                number=206,
                instance_id=f"s-206-{idx}",
            )
        )

    _resolve_break(state, use_pending=True)

    assert state.pending_decision_kind == "conservation_reward"
    assert state.pending_decision_player_id == 0

    p0_actions = legal_actions(p0, state=state, player_id=0)
    tile_action = next(action for action in p0_actions if (action.details or {}).get("reward") == "10_money")
    apply_action(state, tile_action)

    assert state.pending_decision_kind == "conservation_reward"
    assert state.pending_decision_player_id == 1

    p1_actions = legal_actions(p1, state=state, player_id=1)
    p1_rewards = {str((action.details or {}).get("reward") or "") for action in p1_actions}

    assert "10_money" not in p1_rewards
    assert "5coins" in p1_rewards

    money_action = next(action for action in p1_actions if (action.details or {}).get("reward") == "5coins")
    apply_action(state, money_action)

    assert state.pending_decision_kind == ""
    assert state.break_progress == 0
    assert state.break_trigger_player is None


def test_player_snapshot_lists_only_in_zoo_animals_and_separates_pouched_cards(capsys):
    state = setup_game(seed=1001, player_names=["P1", "P2"])
    player = state.players[0]
    player.hand = []
    state.zoo_display = []
    wombat = AnimalCard(
        "COMMON WOMBAT",
        9,
        2,
        4,
        0,
        card_type="animal",
        number=450,
        badges=("Australia", "Herbivore"),
        instance_id="wombat-host",
    )
    caracal = AnimalCard(
        "CARACAL",
        9,
        2,
        4,
        0,
        card_type="animal",
        number=404,
        badges=("Africa", "Predator"),
        instance_id="caracal-zoo",
    )
    ornithologist = AnimalCard(
        "ORNITHOLOGIST",
        4,
        0,
        0,
        0,
        card_type="sponsor",
        number=238,
        instance_id="ornithologist-sponsor",
    )
    pouched = AnimalCard(
        "GOLDEN EAGLE",
        20,
        5,
        7,
        0,
        card_type="animal",
        number=509,
        badges=("Bird",),
        instance_id="golden-eagle-pouched",
    )
    player.zoo_cards = [wombat, ornithologist, caracal]
    player.pouched_cards = [pouched]
    player.pouched_cards_by_host = {"wombat-host": [pouched]}

    HumanPlayer()._print_player_snapshot(state, player)
    output = capsys.readouterr().out

    assert "Zoo animals:" in output
    assert "#450 COMMON WOMBAT size=2 release=2-" in output
    assert "#404 CARACAL size=2 release=2-" in output
    assert "ORNITHOLOGIST" not in output
    assert "Pouched cards (not in zoo):" in output
    assert "#509 GOLDEN EAGLE under[COMMON WOMBAT]" in output


def test_prompt_opening_draft_indices_uses_combination_choices(monkeypatch):
    drafted_cards = [
        AnimalCard(f"Card {idx}", idx, 1, 0, 0, number=400 + idx, instance_id=f"draft-{idx}")
        for idx in range(8)
    ]
    monkeypatch.setattr("builtins.input", lambda _: "1")

    kept = _prompt_opening_draft_indices("P1", drafted_cards)

    assert kept == [0, 1, 2, 3]


def test_prompt_break_discard_indices_uses_combination_choices(monkeypatch):
    state = setup_game(seed=1002, player_names=["P1", "P2"])
    player = state.players[0]
    player.hand = [
        AnimalCard(f"Card {idx}", idx, 1, 0, 0, number=500 + idx, instance_id=f"hand-{idx}")
        for idx in range(5)
    ]
    monkeypatch.setattr("builtins.input", lambda _: "1")

    picked = _prompt_break_discard_indices(player, 2)

    assert picked == [0, 1]


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
    apply_action(state, pending_action)

    while state.pending_decision_kind == "break_discard":
        pending_player = state.players[state.pending_decision_player_id]
        break_pending_actions = legal_actions(
            pending_player,
            state=state,
            player_id=state.pending_decision_player_id,
        )
        apply_action(state, break_pending_actions[0])

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


def test_cards_legal_actions_do_not_include_strength_above_five():
    state = setup_game(seed=1451, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["build", "cards", "animals", "association", "sponsors"]
    player.x_tokens = 5

    abstract_actions = legal_actions(player)
    abstract_card_actions = [
        action
        for action in abstract_actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards"
    ]

    assert abstract_card_actions
    assert {int(action.details.get("effective_strength", 0)) for action in abstract_card_actions} == {2, 3, 4, 5}
    assert all(int(action.details.get("effective_strength", 0)) <= 5 for action in abstract_card_actions)

    concrete_actions = legal_actions(player, state=state, player_id=0)
    concrete_card_actions = [
        action
        for action in concrete_actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards"
    ]
    assert concrete_card_actions
    assert all("strength=6" not in str(action) for action in concrete_card_actions)
    assert all(int(action.details.get("effective_strength", 0)) <= 5 for action in concrete_card_actions)


def test_animals_legal_actions_do_not_include_strength_above_five():
    state = setup_game(seed=1452, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["build", "cards", "animals", "association", "sponsors"]
    player.x_tokens = 5
    player.hand = [
        AnimalCard(
            "Simple Animal",
            1,
            1,
            0,
            0,
            card_type="animal",
            instance_id="simple-animal",
        )
    ]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    abstract_actions = legal_actions(player)
    abstract_animal_actions = [
        action
        for action in abstract_actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"
    ]

    assert abstract_animal_actions
    assert {int(action.details.get("effective_strength", 0)) for action in abstract_animal_actions} == {3, 4, 5}
    assert all(int(action.details.get("effective_strength", 0)) <= 5 for action in abstract_animal_actions)

    concrete_actions = legal_actions(player, state=state, player_id=0)
    concrete_animal_actions = [
        action
        for action in concrete_actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"
    ]
    assert concrete_animal_actions
    assert all("strength=6" not in str(action) for action in concrete_animal_actions)
    assert all(int(action.details.get("effective_strength", 0)) <= 5 for action in concrete_animal_actions)


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


def test_state_legal_actions_expand_upgraded_build_into_ordered_concrete_sequences(monkeypatch):
    state = setup_game(seed=1461, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.action_upgraded["build"] = True

    option_a = {
        "index": 1,
        "building_type": "SIZE_1",
        "building_label": "enclosure_1",
        "cells": [(0, 0)],
        "size": 1,
        "cost": 2,
        "placement_bonuses": [],
    }
    option_b = {
        "index": 2,
        "building_type": "PAVILION",
        "building_label": "pavilion",
        "cells": [(1, 0)],
        "size": 1,
        "cost": 2,
        "placement_bonuses": [],
    }

    def _fake_list_legal_build_options(*, already_built_types=None, strength, **kwargs):
        built = {item.name for item in set(already_built_types or set())}
        options = []
        if strength >= 1 and "SIZE_1" not in built:
            options.append(copy.deepcopy(option_a))
        if strength >= 1 and "PAVILION" not in built:
            options.append(copy.deepcopy(option_b))
        return options

    monkeypatch.setattr(main, "list_legal_build_options", _fake_list_legal_build_options)
    monkeypatch.setattr(main, "_perform_build_action_effect", lambda *args, **kwargs: None)

    actions = legal_actions(player, state=state, player_id=0)
    build_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "build"
        and int(action.value or 0) == 0
    ]

    assert build_actions
    assert all(bool((action.details or {}).get("concrete")) for action in build_actions)
    assert any(len((action.details or {}).get("selections") or []) >= 2 for action in build_actions)
    assert any(" ; then " in str(action) for action in build_actions if len((action.details or {}).get("selections") or []) >= 2)

    def _selection_key(action: Action) -> tuple:
        selections = list((action.details or {}).get("selections") or [])
        return tuple(
            (
                str(selection.get("building_type") or ""),
                tuple(tuple(cell) for cell in list(selection.get("cells") or [])),
            )
            for selection in selections
        )

    two_step_keys = [
        _selection_key(action)
        for action in build_actions
        if len((action.details or {}).get("selections") or []) == 2
    ]
    assert any(key[::-1] in two_step_keys and key[::-1] != key for key in two_step_keys)


def test_build_card_bonus_expands_to_reputation_range_display_and_deck(monkeypatch):
    state = setup_game(seed=1462, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.reputation = 4
    state.zoo_display = [
        AnimalCard(f"Display {idx}", 0, 1, 0, 0, number=700 + idx, instance_id=f"display-{idx}")
        for idx in range(1, 5)
    ]

    option = {
        "index": 1,
        "building_type": "SIZE_2",
        "building_label": "enclosure_2",
        "cells": [(4, 1), (5, 1)],
        "size": 2,
        "cost": 4,
        "placement_bonuses": ["card_in_reputation_range"],
    }

    monkeypatch.setattr(main, "list_legal_build_options", lambda **kwargs: [copy.deepcopy(option)])
    monkeypatch.setattr(main, "_perform_build_action_effect", lambda *args, **kwargs: None)

    actions = legal_actions(player, state=state, player_id=0)
    build_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "build" and int(action.value or 0) == 0
    ]

    assert [
        {
            "selection": (action.details or {}).get("selections"),
            "bonus_choices": (action.details or {}).get("build_card_bonus_choices"),
        }
        for action in build_actions
    ] == [
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "display", "display_index": 0}],
        },
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "display", "display_index": 1}],
        },
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "display", "display_index": 2}],
        },
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "deck"}],
        },
    ]


def test_build_card_bonus_uses_selected_display_choice():
    state = setup_game(seed=1463, player_names=["P1", "P2"])
    player = state.players[0]
    player.hand = []
    player.reputation = 4
    state.zoo_display = [
        AnimalCard("Display 1", 0, 1, 0, 0, number=711, instance_id="display-1"),
        AnimalCard("Display 2", 0, 1, 0, 0, number=712, instance_id="display-2"),
        AnimalCard("Display 3", 0, 1, 0, 0, number=713, instance_id="display-3"),
        AnimalCard("Display 4", 0, 1, 0, 0, number=714, instance_id="display-4"),
    ]
    state.zoo_deck = [
        AnimalCard("Deck Fill", 0, 1, 0, 0, number=799, instance_id="deck-fill-1"),
    ]

    main._apply_build_placement_bonus(
        state,
        player,
        "card_in_reputation_range",
        {"build_card_bonus_choices": [{"draw_source": "display", "display_index": 1}]},
        bonus_index=0,
        allow_interactive=False,
    )

    assert [card.instance_id for card in player.hand] == ["display-2"]
    assert [card.instance_id for card in state.zoo_display] == ["display-1", "display-3", "display-4", "deck-fill-1"]


def test_build_legal_actions_filter_invalid_reputation_range_bonus_choice(monkeypatch):
    state = setup_game(seed=1465, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.reputation = 4
    state.zoo_display = [
        AnimalCard(f"Display {idx}", 0, 1, 0, 0, number=720 + idx, instance_id=f"display-{idx}")
        for idx in range(1, 5)
    ]

    option = {
        "index": 1,
        "building_type": "SIZE_2",
        "building_label": "enclosure_2",
        "cells": [(4, 1), (5, 1)],
        "size": 2,
        "cost": 4,
        "placement_bonuses": ["card_in_reputation_range"],
    }

    monkeypatch.setattr(main, "list_legal_build_options", lambda **kwargs: [copy.deepcopy(option)])

    def _guard_invalid_bonus_choice(state, player, strength, player_id=None, details=None):
        queued_choices = list((details or {}).get("build_card_bonus_choices") or [])
        if queued_choices and int(queued_choices[0].get("display_index", -1)) == 2:
            raise ValueError("Chosen display card is outside reputation range.")
        return None

    monkeypatch.setattr(main, "_perform_build_action_effect", _guard_invalid_bonus_choice)

    actions = legal_actions(player, state=state, player_id=0)
    build_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "build" and int(action.value or 0) == 0
    ]

    assert [
        {
            "selection": (action.details or {}).get("selections"),
            "bonus_choices": (action.details or {}).get("build_card_bonus_choices"),
        }
        for action in build_actions
    ] == [
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "display", "display_index": 0}],
        },
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "display", "display_index": 1}],
        },
        {
            "selection": [{"building_type": "SIZE_2", "cells": [[4, 1], [5, 1]]}],
            "bonus_choices": [{"draw_source": "deck"}],
        },
    ]


def test_build_bonus_archaeologist_chain_keeps_non_interactive_mode(monkeypatch):
    state = setup_game(seed=1464, player_names=["P1", "P2"])
    player = state.players[0]
    player.reputation = 4
    player.action_upgraded = {action: False for action in MAIN_ACTION_CARDS}
    state.map_tile_bonuses = {
        (0, 0): "5coins",
        (1, 1): "reputation",
    }

    monkeypatch.setattr("main._player_has_sponsor", lambda _player, sponsor_number: sponsor_number == 221)
    monkeypatch.setattr("main._player_border_coords", lambda _player: {(0, 0)})
    monkeypatch.setattr("main._player_all_covered_cells", lambda _player: set())
    monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(AssertionError("input() must not be called")))

    main._apply_build_placement_bonus(
        state=state,
        player=player,
        bonus="5coins",
        details={},
        bonus_index=0,
        bonus_coord=(0, 0),
        allow_interactive=False,
    )

    assert player.reputation >= 5
    assert player.action_upgraded["animals"] is True


def test_animals_trade_with_sponsor_228_skips_stale_followup_instead_of_raising(monkeypatch):
    state = setup_game(seed=1465, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.money = 50
    player.reputation = 10
    player.enclosures = [
        Enclosure(size=1, occupied=False, origin=(0, 0), rotation="ROT_0"),
        Enclosure(size=1, occupied=False, origin=(1, -1), rotation="ROT_0"),
    ]
    player.hand = [
        AnimalCard(
            "Trader",
            3,
            1,
            1,
            0,
            card_type="animal",
            ability_title="Trade",
            number=970201,
            instance_id="970201",
        ),
        AnimalCard(
            "ExtraSmall",
            2,
            1,
            3,
            0,
            card_type="animal",
            number=970202,
            instance_id="970202",
        ),
    ]
    state.zoo_display = [
        AnimalCard(
            "DisplayCard",
            2,
            1,
            1,
            0,
            card_type="animal",
            number=970299,
            instance_id="970299",
        )
    ]

    monkeypatch.setattr("main._player_has_sponsor", lambda _player, sponsor_number: sponsor_number == 228)

    options = main.list_legal_animals_options(state=state, player_id=0, strength=2)
    trade_index = next(
        int(option["index"]) - 1
        for option in options
        if [str(play.get("card_instance_id") or "") for play in option.get("plays") or []] == ["970201"]
    )

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": trade_index},
        player_id=0,
    )

    assert any(str(entry).startswith("sponsor_228_extra_small_animal") for entry in state.effect_log)
    assert any(str(entry).startswith("animals_followup_skipped_missing_hand_card") for entry in state.effect_log)
    assert any(card.instance_id == "970299" for card in player.hand)


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

    monkeypatch.setattr("builtins.input", lambda _: "1")
    _resolve_manual_opening_drafts(state, {"P1"})

    assert p1.opening_draft_kept_indices == [0, 1, 2, 3]
    assert [card.name for card in p1.hand] == [card.name for card in p1.opening_draft_drawn[:4]]


def test_setup_manual_opening_draft_exposes_pending_keep_actions():
    state = setup_game(
        seed=78,
        player_names=["P1", "P2"],
        manual_opening_draft_player_names={"P1"},
    )
    p1 = state.players[0]

    assert state.pending_decision_kind == "opening_draft_keep"
    assert state.pending_decision_player_id == 0

    actions = legal_actions(p1, state=state, player_id=0)
    assert len(actions) == 70
    assert all(action.type == ActionType.PENDING_DECISION for action in actions)
    assert all((action.details or {}).get("pending_kind") == "opening_draft_keep" for action in actions)
    assert all(len((action.details or {}).get("keep_card_instance_ids") or []) == 4 for action in actions)
    assert (actions[0].details or {}).get("action_label") == "keep [1 2 3 4]"
    assert (actions[-1].details or {}).get("action_label") == "keep [5 6 7 8]"


def test_break_discard_pending_actions_use_index_combo_labels():
    state = setup_game(seed=79, player_names=["P1", "P2"])
    player = state.players[0]
    player.hand = [
        AnimalCard(f"Card {idx}", idx, 1, 0, 0, number=600 + idx, instance_id=f"break-{idx}")
        for idx in range(5)
    ]
    state.pending_decision_kind = "break_discard"
    state.pending_decision_player_id = 0
    state.pending_decision_payload = {"discard_target": 2}

    actions = legal_actions(player, state=state, player_id=0)

    assert len(actions) == 10
    assert (actions[0].details or {}).get("action_label") == "discard [1 2]"
    assert (actions[-1].details or {}).get("action_label") == "discard [4 5]"


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


def test_upgraded_cards_strength_one_does_not_generate_overdraw_display_choices():
    state = setup_game(seed=281, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]  # cards strength=1
    player.action_upgraded["cards"] = True
    player.reputation = 9

    actions = legal_actions(player, state=state, player_id=0)
    cards_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards" and int(action.value or 0) == 0
    ]

    assert cards_actions
    for action in cards_actions:
        details = dict(action.details or {})
        from_display = list(details.get("from_display_indices") or [])
        from_deck = int(details.get("from_deck_count", 0) or 0)
        assert len(from_display) + from_deck <= 1
        apply_action(copy.deepcopy(state), action)


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
        and (
            (action.details or {}).get("association_task_sequence")
            and (action.details or {})["association_task_sequence"][0].get("project_id") == "P101_SpeciesDiversity"
        )
    ]
    draw_actions = [
        action
        for action in project_actions
        if (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "draw_1_card_deck_or_reputation_range"
    ]
    free_enclosure_actions = [
        action
        for action in project_actions
        if (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "build_free_standard_enclosure_size_2"
    ]

    assert len(project_actions) > len(state.zoo_display)
    assert all(
        ((action.details or {}).get("association_task_sequence") or [])[0].get("project_level") == "right_level"
        for action in project_actions
    )
    assert len(draw_actions) == len(state.zoo_display)
    assert free_enclosure_actions
    assert any(
        (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "gain_5_coins"
        for action in project_actions
    )
    assert any(
        (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "play_1_sponsor_by_paying_cost"
        for action in project_actions
    )
    assert any(
        (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "gain_worker_1"
        for action in project_actions
    )
    assert any(
        (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "gain_12_coins"
        for action in project_actions
    )
    assert any(
        (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "gain_3_x_tokens"
        for action in project_actions
    )

    display_action = draw_actions[0]
    expected_display_card = state.zoo_display[0]
    hand_before = len(player.hand)
    apply_action(state, display_action)

    assert player.map_left_track_unlocked_count == 1
    assert player.map_left_track_unlocked_effects == ["draw_1_card_deck_or_reputation_range"]
    assert len(player.hand) == hand_before + 1
    assert any(card.instance_id == expected_display_card.instance_id for card in player.hand)


def test_state_legal_actions_project_display_unlock_sponsor_uses_post_project_money():
    state = setup_game(seed=2861, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "build", "animals", "sponsors", "association"]
    player.action_upgraded["association"] = True
    player.money = 4
    player.reputation = 2
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
    sponsor_238 = next(card for card in state.zoo_deck if card.number == 238)
    state.zoo_deck.remove(sponsor_238)
    player.hand = [sponsor_238]
    state.zoo_display[1] = AnimalCard(
        "SMALL ANIMALS",
        0,
        0,
        0,
        0,
        card_type="conservation_project",
        number=130,
        instance_id="cp-130",
    )

    actions = legal_actions(player, state=state, player_id=0)
    display_unlock_4_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "association"
        and (
            (action.details or {}).get("association_task_sequence")
            and (action.details or {})["association_task_sequence"][0].get("project_from_display_index") == 1
        )
        and (
            (
                ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice")
                or {}
            ).get("unlock_index")
            == 3
        )
    ]

    assert display_unlock_4_actions
    assert all(
        (
            ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice") or {}
        ).get("effect_code")
        == "play_1_sponsor_by_paying_cost"
        and not (
            (
                ((action.details or {}).get("association_task_sequence") or [])[0].get("map_left_track_choice")
                or {}
            ).get("sponsor_details")
        )
        for action in display_unlock_4_actions
    )


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


def test_state_legal_actions_expand_build_ii_to_concrete_actions(monkeypatch):
    state = setup_game(seed=284, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_upgraded["build"] = True
    player.action_order = ["animals", "cards", "association", "sponsors", "build"]

    option = {
        "index": 1,
        "building_type": "SIZE_1",
        "building_label": "enclosure_1",
        "cells": [(0, 0)],
        "size": 1,
        "cost": 2,
        "placement_bonuses": [],
    }

    monkeypatch.setattr(main, "list_legal_build_options", lambda **kwargs: [copy.deepcopy(option)])
    monkeypatch.setattr(main, "_perform_build_action_effect", lambda *args, **kwargs: None)

    actions = legal_actions(player, state=state, player_id=0)
    build_action = next(
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "build" and int(action.value or 0) == 0
    )

    assert build_action.details.get("concrete") is True
    assert build_action.details.get("selections") == [{"building_type": "SIZE_1", "cells": [[0, 0]]}]


def test_state_legal_actions_build_ii_sequences_stop_at_two_buildings(monkeypatch):
    state = setup_game(seed=384, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_upgraded["build"] = True
    player.action_order = ["animals", "cards", "association", "sponsors", "build"]

    options_by_type = {
        "SIZE_1": {
            "index": 1,
            "building_type": "SIZE_1",
            "building_label": "enclosure_1",
            "cells": [(0, 0)],
            "size": 1,
            "cost": 2,
            "placement_bonuses": [],
        },
        "KIOSK": {
            "index": 2,
            "building_type": "KIOSK",
            "building_label": "kiosk",
            "cells": [(1, 0)],
            "size": 1,
            "cost": 2,
            "placement_bonuses": [],
        },
        "PAVILION": {
            "index": 3,
            "building_type": "PAVILION",
            "building_label": "pavilion",
            "cells": [(2, 0)],
            "size": 1,
            "cost": 2,
            "placement_bonuses": [],
        },
    }

    def fake_list_legal_build_options(*, already_built_types=None, **_kwargs):
        built = already_built_types or set()
        return [
            copy.deepcopy(option)
            for type_name, option in options_by_type.items()
            if main.BuildingType[type_name] not in built
        ]

    monkeypatch.setattr(main, "list_legal_build_options", fake_list_legal_build_options)
    monkeypatch.setattr(main, "_perform_build_action_effect", lambda *args, **kwargs: None)

    actions = legal_actions(player, state=state, player_id=0)
    build_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "build"
    ]

    assert build_actions
    assert all(str(action).count(" ; then ") <= 1 for action in build_actions)
    assert any(str(action).count(" ; then ") == 1 for action in build_actions)


def test_build_ii_rejects_more_than_two_requested_selections():
    state = setup_game(seed=385, player_names=["P1", "P2"])
    player = state.players[0]
    player.action_upgraded["build"] = True

    with pytest.raises(ValueError, match="at most two"):
        main._perform_build_action_effect(
            state=state,
            player=player,
            strength=5,
            player_id=0,
            details={
                "selections": [
                    {"building_type": "SIZE_1", "cells": [[0, 0]]},
                    {"building_type": "KIOSK", "cells": [[1, 0]]},
                    {"building_type": "PAVILION", "cells": [[2, 0]]},
                ]
            },
        )


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

    responses = iter(["10", "1"])
    monkeypatch.setattr("builtins.input", lambda _: next(responses))
    _resolve_break(state, allow_interactive=True)

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
    assert all("discard [" in str(action) for action in pending_actions)
    assert (pending_actions[0].details or {}).get("action_label") == "discard [1]"

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

    monkeypatch.setattr("builtins.input", lambda _: "4")

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
    assert {str(action).split(" ; ", 1)[1] for action in animal_actions} == {
        "sell []",
        "sell [1]",
        "sell [2]",
        "sell [3]",
        "sell [1 2]",
        "sell [1 3]",
        "sell [2 3]",
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
    assert {str(action).split(" ; ", 1)[1] for action in animal_actions} == {
        "pouch []",
        "pouch [1]",
    }

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


def test_sell_hand_choices_are_clamped_to_runtime_limit_in_non_interactive_mode():
    state = setup_game(seed=2931, player_names=["P1", "P2"])
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
        AnimalCard("Keep B", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-b"),
    ]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    chosen = next(
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "Seller" in str((action.details or {}).get("action_label") or "")
    )

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            value=chosen.value,
            card_name=chosen.card_name,
            details={
                **dict(chosen.details or {}),
                "sell_hand_card_choices": [
                    {"card_instance_ids": ["keep-b", "missing-id", "keep-a"]},
                ],
            },
        ),
    )

    assert player.money == 9
    assert [card.instance_id for card in player.hand] == ["keep-a"]
    assert {card.instance_id for card in state.zoo_discard} >= {"keep-b"}


def test_pouch_choices_are_clamped_to_runtime_limit_in_non_interactive_mode():
    state = setup_game(seed=2932, player_names=["P1", "P2"])
    player = state.players[0]
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
        AnimalCard("Keep B", 0, 1, 0, 0, card_type="sponsor", instance_id="keep-b"),
    ]
    player.action_order = ["cards", "build", "animals", "association", "sponsors"]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    chosen = next(
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "Poucher" in str((action.details or {}).get("action_label") or "")
    )

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            value=chosen.value,
            card_name=chosen.card_name,
            details={
                **dict(chosen.details or {}),
                "pouch_hand_card_choices": [
                    {"card_instance_ids": ["keep-a", "missing-id", "keep-b"]},
                ],
            },
        ),
    )

    assert player.appeal == 2
    assert [card.instance_id for card in player.hand] == ["keep-b"]
    assert [card.instance_id for card in player.pouched_cards] == ["keep-a"]
    assert [card.instance_id for card in player.pouched_cards_by_host["poucher"]] == ["keep-a"]


def test_boost_action_card_effect_expands_animals_legal_actions_and_applies_slot_choice():
    state = setup_game(seed=2941, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.hand = [
        AnimalCard(
            "RACCOON",
            11,
            1,
            4,
            0,
            card_type="animal",
            badges=("Predator", "Bear", "America"),
            ability_title="Boost: Association",
            ability_text="After finishing this action, you may place your Association Action card",
            number=415,
            instance_id="raccoon",
        )
    ]
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.enclosures = [Enclosure(size=1, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "RACCOON" in str((action.details or {}).get("action_label") or "")
    ]

    assert len(animal_actions) == 3
    assert {
        tuple(choice.get("mode") for choice in list((action.details or {}).get("boost_action_choices") or []))
        for action in animal_actions
    } == {("skip",), ("slot1",), ("slot5",)}
    assert {str(action).split(" ; ", 1)[1] for action in animal_actions} == {
        "boost skip",
        "boost association->1",
        "boost association->5",
    }

    chosen = next(
        action
        for action in animal_actions
        if [choice.get("mode") for choice in list((action.details or {}).get("boost_action_choices") or [])] == ["slot5"]
    )
    apply_action(state, chosen)

    assert player.action_order == ["animals", "cards", "build", "sponsors", "association"]


def test_petting_zoo_animals_only_generate_petting_zoo_placements():
    state = setup_game(seed=2942, player_names=["P1", "P2"])
    player = state.players[0]
    state.current_player = 0
    player.money = 20
    player.hand = [
        AnimalCard(
            "HORSE",
            7,
            1,
            0,
            0,
            reputation_gain=1,
            card_type="animal",
            badges=("Pet",),
            ability_title="Petting Zoo Animal",
            ability_text="Gain 3",
            number=521,
            instance_id="horse",
        )
    ]
    player.action_order = ["cards", "animals", "build", "sponsors", "association"]
    player.enclosures = [
        Enclosure(size=2, origin=(0, 0), enclosure_type="standard"),
        Enclosure(size=3, origin=(1, 0), enclosure_type="petting_zoo", animal_capacity=3),
    ]

    actions = legal_actions(player, state=state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "HORSE" in str((action.details or {}).get("action_label") or "")
    ]

    assert animal_actions
    assert all("petting_zoo" in str(action) for action in animal_actions)
    assert all("E1" not in str(action) for action in animal_actions)


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


def test_animals_concrete_actions_prune_higher_x_duplicates_with_same_payload():
    state = setup_game(seed=2971, player_names=["P1", "P2"])
    player = state.players[0]
    player.x_tokens = 1
    player.money = 30
    player.action_order = ["build", "cards", "association", "animals", "sponsors"]
    player.action_upgraded["animals"] = False
    player.hand = [
        AnimalCard(
            "NEW ZEALAND SEA LION",
            17,
            3,
            6,
            0,
            card_type="animal",
            number=423,
            instance_id="sea-lion",
        )
    ]
    player.enclosures = [Enclosure(size=3, origin=(0, 0))]

    actions = legal_actions(player, state=state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and "NEW ZEALAND SEA LION" in str(action)
    ]

    assert len(animal_actions) == 1
    assert int(animal_actions[0].value or 0) == 0
    assert "strength=4" in str(animal_actions[0])
    assert "x=1" not in str(animal_actions[0])


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


def test_hypnosis_can_target_another_player_when_actor_is_tied_for_highest_appeal(monkeypatch):
    state = setup_game(seed=1881, player_names=["P1", "P2"])
    actor = state.players[0]
    tied_target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Hypnosis Beast",
            0,
            1,
            0,
            0,
            ability_title="Hypnosis 3",
            number=99031,
            instance_id="hyp-tie-1",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    actor.appeal = 5
    tied_target.appeal = 5
    tied_target.action_order = ["cards", "build", "animals", "association", "sponsors"]

    monkeypatch.setattr(main, "_prompt_build_action_details_for_human", lambda **kwargs: {"selections": []})
    monkeypatch.setattr("builtins.input", lambda _: "2")

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0, "_interactive": True},
    )

    assert tied_target.action_order[0] == "build"


def test_hypnosis_does_not_offer_unexecutable_association_action():
    state = setup_game(seed=1882, player_names=["P1", "P2"])
    actor = state.players[0]
    target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Hypnosis Scout",
            0,
            1,
            0,
            0,
            ability_title="Hypnosis 1",
            number=99032,
            instance_id="hyp-no-assoc-1",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.workers = 0
    target.action_order = ["association", "cards", "build", "animals", "sponsors"]
    order_before = list(target.action_order)

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert target.action_order == order_before
    assert any("effect[hypnosis] target=P2 no_action" in line for line in state.effect_log)


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


def test_pilfering_can_target_tied_highest_other_player_when_actor_is_also_tied_highest():
    state = setup_game(seed=1891, player_names=["P1", "P2"])
    actor = state.players[0]
    tied_target = state.players[1]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=99041,
            instance_id="pilfer-tie-1",
        )
    ]
    actor.money = 2
    actor.appeal = 5
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    tied_target.appeal = 5
    tied_target.money = 11
    tied_target.hand = []

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert actor.money == 7
    assert tied_target.money == 6


def test_pilfering_has_no_effect_when_actor_is_unique_highest():
    state = setup_game(seed=1892, player_names=["P1", "P2"])
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
            number=99042,
            instance_id="pilfer-high-1",
        )
    ]
    actor.money = 2
    actor.appeal = 6
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.money = 11
    money_before = actor.money

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert actor.money == money_before
    assert target.money == 11


def test_pilfering_noninteractive_defaults_to_money_loss_when_both_losses_are_legal():
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

    assert actor.money == 7
    assert target.money == 6


def test_pilfering_noninteractive_uses_explicit_card_loss_choice():
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


def test_pilfering_noninteractive_defaults_to_first_target_when_multiple_players_are_tied_highest():
    state = setup_game(seed=1901, player_names=["P1", "P2"])
    state.players.append(copy.deepcopy(state.players[1]))
    state.players[2].name = "P3"
    actor = state.players[0]
    target_a = state.players[1]
    target_b = state.players[2]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=99051,
            instance_id="pilfer-target-1",
        )
    ]
    actor.money = 2
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target_a.appeal = 5
    target_a.money = 11
    target_a.hand = []
    target_b.appeal = 5
    target_b.money = 8
    target_b.hand = []

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert actor.money == 7
    assert target_a.money == 6
    assert target_b.money == 8

    state = setup_game(seed=1901, player_names=["P1", "P2"])
    state.players.append(copy.deepcopy(state.players[1]))
    state.players[2].name = "P3"
    actor = state.players[0]
    target_a = state.players[1]
    target_b = state.players[2]
    actor.hand = [
        AnimalCard(
            "Pilfer Beast",
            0,
            1,
            0,
            0,
            ability_title="Pilfering 1",
            number=99051,
            instance_id="pilfer-target-1",
        )
    ]
    actor.money = 2
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target_a.appeal = 5
    target_a.money = 11
    target_a.hand = []
    target_b.appeal = 5
    target_b.money = 8
    target_b.hand = []

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={
            "animals_sequence_index": 0,
            "pilfering_choices": [{"target_player": "P3", "choice": "money"}],
        },
    )

    assert actor.money == 7
    assert target_a.money == 11
    assert target_b.money == 3


def test_pilfering_interactive_target_can_choose_card_loss_and_receive_random_card(monkeypatch):
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


def test_pilfering_only_card_loss_with_multiple_cards_transfers_random_card(monkeypatch):
    state = setup_game(seed=1921, player_names=["P1", "P2"])
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
            number=99071,
            instance_id="pilfer-card-choice-1",
        )
    ]
    actor.enclosures = [Enclosure(size=1, origin=(0, 0))]
    target.appeal = 5
    target.money = 4
    target.hand = [
        AnimalCard("Card A", 2, 1, 0, 0, number=9915, instance_id="card-a"),
        AnimalCard("Card B", 8, 3, 0, 0, number=9916, instance_id="card-b"),
    ]

    monkeypatch.setattr(main.random, "randrange", lambda upper: 1)

    _perform_animals_action_effect(
        state=state,
        player=actor,
        player_id=0,
        strength=3,
        details={"animals_sequence_index": 0},
    )

    assert [card.instance_id for card in actor.hand] == ["card-b"]
    assert [card.instance_id for card in target.hand] == ["card-a"]


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


def test_cards_draw_exceeding_remaining_deck_forces_immediate_game_end():
    state = setup_game(seed=194, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.action_order = ["cards", "animals", "build", "association", "sponsors"]
    p0.action_upgraded["cards"] = False
    state.current_player = 0
    state.zoo_deck = []
    state.zoo_display = []

    cards_action = next(
        action
        for action in legal_actions(p0, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION and action.card_name == "cards"
    )
    apply_action(state, cards_action)

    assert state.game_over()
    assert state.forced_game_over is True
    assert "cards_draw_exceeds_deck" in state.forced_game_over_reason
    assert state.pending_decision_kind == ""


def test_game_forces_end_after_exceeding_100_rounds():
    state = make_state(195)
    state.current_player = 0

    # 2 actions = 1 round for 2 players. End should trigger once rounds exceed 100.
    # Use the first fully materialized legal action each step.
    for _ in range(260):
        if state.game_over():
            break
        actor_id = (
            int(state.pending_decision_player_id)
            if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None
            else int(state.current_player)
        )
        apply_action(state, materialize_first_action(state, actor_id))

    assert state.game_over()
    assert state.forced_game_over is True
    reason = str(state.forced_game_over_reason or "")
    assert ("round_limit_exceeded" in reason) or ("cards_draw_exceeds_deck" in reason)
    if "round_limit_exceeded" in reason:
        assert main._completed_rounds(state) > 100


def test_increase_reputation_non_interactive_does_not_prompt_for_milestone_upgrade(monkeypatch):
    state = setup_game(seed=196, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.reputation = 4

    def _fail_input(_prompt: str) -> str:  # pragma: no cover - should not be called
        raise AssertionError("input() should not be called for non-interactive reputation gain")

    monkeypatch.setattr("builtins.input", _fail_input)

    main._increase_reputation(state=state, player=p0, amount=1)

    assert p0.reputation == 5
    assert p0.action_upgraded["animals"] is True


def test_rank_scores_assigns_same_rank_to_ties():
    ranking = _rank_scores({"P1": 100, "P2": 95, "P3": 95, "P4": 80})

    assert ranking == [
        (1, "P1", 100),
        (2, "P2", 95),
        (2, "P3", 95),
        (4, "P4", 80),
    ]

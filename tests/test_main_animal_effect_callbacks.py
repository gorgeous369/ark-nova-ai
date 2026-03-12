import main
import pytest

from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation
from main import (
    ActionType,
    AnimalCard,
    Enclosure,
    EnclosureObject,
    SetupCardRef,
    _ActionDetailExpansionRequired,
    _ensure_player_map_initialized,
    _perform_animals_action_effect,
    _resume_animals_followup_from_pending_payload,
)
from tests.helpers import (
    configure_player,
    configure_standard_enclosures,
    legal_actions_for_player,
    make_basic_animals_play_state,
    make_state,
    pending_actions,
    play_legal_main_action,
    play_pending_action,
    set_pending_decision,
)


def _prepare_glide_effect_state(seed: int = 614):
    state, player = make_basic_animals_play_state(seed=seed)
    player.hand = [
        AnimalCard(
            name="Glider",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Glide 2",
            card_type="animal",
            number=9201,
            instance_id="glide-animal",
        ),
        AnimalCard(
            name="Sea Discard A",
            cost=0,
            size=5,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("SeaAnimal",),
            number=9202,
            instance_id="sea-discard-a",
        ),
        AnimalCard(
            name="Sea Discard B",
            cost=0,
            size=5,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("SeaAnimal",),
            number=9203,
            instance_id="sea-discard-b",
        ),
        AnimalCard(
            name="Land Discard",
            cost=0,
            size=5,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Reptile",),
            number=9204,
            instance_id="land-discard",
        ),
    ]
    return state, player


def test_glide_effect_discards_only_cards_with_sea_animal_icons():
    state, player = _prepare_glide_effect_state(seed=614)

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={
            "glide_discard_card_choices": [["sea-discard-a"]],
            "glide_reward_choices": [{"reward": "reputation"}],
        },
        player_id=0,
    )

    assert player.reputation == 7
    assert {card.instance_id for card in player.hand} == {"sea-discard-b", "land-discard"}
    assert any(card.instance_id == "sea-discard-a" for card in state.zoo_discard)


def test_glide_effect_rejects_non_sea_animal_discard_choice():
    state, player = _prepare_glide_effect_state(seed=615)

    with pytest.raises(ValueError, match="glide_discard_card_choices selected card is not legal"):
        _perform_animals_action_effect(
            state=state,
            player=player,
            strength=2,
            details={"glide_discard_card_choices": [["land-discard"]]},
            player_id=0,
        )


def test_glide_effect_expands_only_sea_animal_discard_choices():
    state, player = _prepare_glide_effect_state(seed=616)

    with pytest.raises(_ActionDetailExpansionRequired) as excinfo:
        _perform_animals_action_effect(
            state=state,
            player=player,
            strength=2,
            details={"_expand_implicit_choices": True},
            player_id=0,
        )

    expanded_choices = {
        tuple(sorted((details["glide_discard_card_choices"][0].get("card_instance_ids") or [])))
        for details, _label in excinfo.value.variants
    }
    assert expanded_choices == {
        (),
        ("sea-discard-a",),
        ("sea-discard-b",),
        ("sea-discard-a", "sea-discard-b"),
    }


def test_mark_effect_marks_first_display_animal_in_range():
    state, player = make_basic_animals_play_state(seed=611)
    player.hand = [
        AnimalCard(
            name="Marker",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Mark",
            card_type="animal",
            number=9001,
            instance_id="test-mark",
        )
    ]
    state.zoo_display = [
        AnimalCard(
            name="Display Animal",
            cost=3,
            size=1,
            appeal=2,
            conservation=0,
            card_type="animal",
            number=9101,
            instance_id="disp-animal",
        ),
        AnimalCard(
            name="Display Sponsor",
            cost=3,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=9102,
            instance_id="disp-sponsor",
        ),
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert "disp-animal" in state.marked_display_card_ids
    assert any("effect[mark] marked=1" in line for line in state.effect_log)


def test_dominance_effect_takes_specific_unused_base_project():
    state, player = make_basic_animals_play_state(seed=612)
    state.unused_base_conservation_projects = [
        SetupCardRef(data_id="P108_Primates", title="PRIMATES"),
        SetupCardRef(data_id="P103_Africa", title="AFRICA"),
    ]
    player.hand = [
        AnimalCard(
            name="Dominance Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Dominance",
            card_type="animal",
            number=9002,
            instance_id="test-dominance",
        )
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert all(project.data_id != "P108_Primates" for project in state.unused_base_conservation_projects)
    assert any(card.card_type == "conservation_project" and "PRIMATES" in card.name for card in player.hand)
    assert any("effect[dominance] target=primates taken=1" in line for line in state.effect_log)


def test_assertion_effect_legal_actions_expand_skip_and_selected_project_choice():
    state, player = make_basic_animals_play_state(seed=6121)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    state.unused_base_conservation_projects = [
        SetupCardRef(data_id="P108_Primates", title="PRIMATES"),
        SetupCardRef(data_id="P103_Africa", title="AFRICA"),
    ]
    player.hand = [
        AnimalCard(
            name="Assertion Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Assertion",
            card_type="animal",
            number=90021,
            instance_id="test-assertion",
        )
    ]

    actions = legal_actions_for_player(state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals" and int(action.value or 0) == 0
    ]

    assert any((action.details or {}).get("unused_base_project_choices") == [{"skip": True}] for action in animal_actions)

    play_legal_main_action(
        state,
        player_id=0,
        card_name="animals",
        predicate=lambda action: (
            int(action.value or 0) == 0
            and (action.details or {}).get("unused_base_project_choices") == [{"project_data_id": "P103_Africa"}]
        ),
    )

    assert all(project.data_id != "P103_Africa" for project in state.unused_base_conservation_projects)
    assert any(card.card_type == "conservation_project" and "AFRICA" in card.name for card in player.hand)


def test_cut_down_effect_removes_one_empty_standard_enclosure_and_refunds():
    state, player = make_basic_animals_play_state(seed=613)
    _ensure_player_map_initialized(state, player)
    assert player.zoo_map is not None
    configure_standard_enclosures(player, sizes=(2, 1), origins=((0, 0), (2, 0)))
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, HexTile(0, 0), Rotation.ROT_0))
    player.zoo_map.add_building(Building(BuildingType.SIZE_1, HexTile(2, 0), Rotation.ROT_0))
    player.money = 10
    player.hand = [
        AnimalCard(
            name="Cutter",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Cut Down",
            card_type="animal",
            number=9003,
            instance_id="test-cut-down",
        )
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert len(player.enclosures) == 1
    assert player.money == 12
    assert any("effect[cut_down] removed=1 refunded=2" in line for line in state.effect_log)


def test_trade_effect_requires_explicit_choice_when_multiple_pairs_exist():
    state, player = make_basic_animals_play_state(seed=6131)
    player.reputation = 5
    player.hand = [
        AnimalCard(
            name="Trader",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Trade",
            card_type="animal",
            number=90130,
            instance_id="trade-animal",
        ),
        AnimalCard(
            name="Hand A",
            cost=1,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=90131,
            instance_id="trade-hand-a",
        ),
        AnimalCard(
            name="Hand B",
            cost=1,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=90132,
            instance_id="trade-hand-b",
        ),
    ]
    state.zoo_display = [
        AnimalCard("Display 1", 1, 1, 0, 0, card_type="animal", number=90133, instance_id="trade-display-1"),
        AnimalCard("Display 2", 1, 1, 0, 0, card_type="animal", number=90134, instance_id="trade-display-2"),
    ]

    with pytest.raises(ValueError, match="trade_choices entry is required"):
        _perform_animals_action_effect(
            state=state,
            player=player,
            strength=2,
            details={"animals_sequence_index": 0},
            player_id=0,
        )


def test_shark_attack_effect_discards_animals_from_display_and_gains_money():
    state, player = make_basic_animals_play_state(seed=614)
    player.reputation = 15
    player.money = 5
    player.hand = [
        AnimalCard(
            name="Shark",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Shark Attack 2",
            card_type="animal",
            number=9004,
            instance_id="test-shark",
        )
    ]
    state.zoo_display = [
        AnimalCard(
            name="A1",
            cost=3,
            size=1,
            appeal=5,
            conservation=0,
            card_type="animal",
            number=9201,
            instance_id="disp-a1",
        ),
        AnimalCard(
            name="A2",
            cost=4,
            size=1,
            appeal=3,
            conservation=0,
            card_type="animal",
            number=9202,
            instance_id="disp-a2",
        ),
        AnimalCard(
            name="S1",
            cost=4,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=9203,
            instance_id="disp-s1",
        ),
    ]
    state.zoo_deck = [
        AnimalCard(
            name="Top1",
            cost=3,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9301,
            instance_id="deck-1",
        ),
        AnimalCard(
            name="Top2",
            cost=3,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9302,
            instance_id="deck-2",
        ),
    ]
    discard_before = len(state.zoo_discard)

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert len(state.zoo_discard) == discard_before + 2
    assert player.money == 9
    assert any("effect[shark_attack] discarded=2 money=+4" in line for line in state.effect_log)


def test_extra_shift_requires_explicit_choice_when_multiple_tasks_can_recall_workers():
    state, player = make_basic_animals_play_state(seed=6141)
    player.workers = 0
    player.workers_on_association_board = 2
    player.association_workers_by_task["reputation"] = 1
    player.association_workers_by_task["university"] = 1
    player.hand = [
        AnimalCard(
            name="Shift Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Extra Shift",
            card_type="animal",
            number=90140,
            instance_id="shift-animal",
        )
    ]

    with pytest.raises(ValueError, match="return_association_worker_choices entry is required"):
        _perform_animals_action_effect(
            state=state,
            player=player,
            strength=2,
            details={"animals_sequence_index": 0},
            player_id=0,
        )


def test_hypnosis_legal_actions_expand_distinct_x_spend_choices():
    state, player = make_basic_animals_play_state(seed=6142)
    target = state.players[1]
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    player.x_tokens = 1
    player.hand = [
        AnimalCard(
            name="Hypnosis Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Hypnosis 1",
            card_type="animal",
            number=90141,
            instance_id="hypnosis-animal",
        )
    ]
    target.appeal = 5
    target.action_order = ["cards", "build", "animals", "association", "sponsors"]

    actions = legal_actions_for_player(state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals" and int(action.value or 0) == 0
    ]

    x_choices = {
        tuple((action.details or {}).get("hypnosis_x_spent_choices") or [])
        for action in animal_actions
        if "hypnosis_x_spent_choices" in (action.details or {})
    }

    assert x_choices == {(0,), (1,)}


def test_hypnosis_ignores_stale_x_spend_queue_values_that_are_no_longer_legal():
    state, player = make_basic_animals_play_state(seed=61421)
    target = state.players[1]
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    player.x_tokens = 0
    player.reputation = 0
    player.hand = [
        AnimalCard(
            name="Hypnosis Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Hypnosis 1",
            card_type="animal",
            number=90142,
            instance_id="hypnosis-stale-x",
        )
    ]
    target.action_order = ["cards", "build", "animals", "association", "sponsors"]
    state.zoo_display = []
    state.zoo_deck = [
        AnimalCard(
            name="DeckDraw",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            number=90143,
            instance_id="deck-draw",
        )
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={
            "animals_sequence_index": 0,
            "hypnosis_target_players": [1],
            "hypnosis_targets": ["cards"],
            "hypnosis_x_spent_choices": [1],
        },
        player_id=0,
    )

    assert any(
        "effect[hypnosis] target=P2 action=cards x=0" in str(entry)
        for entry in state.effect_log
    )


def test_resistance_effect_interactive_prompts_for_final_scoring_choice(monkeypatch):
    state, player = make_basic_animals_play_state(seed=615)
    player.hand = [
        AnimalCard(
            name="Resister",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Resistance",
            ability_text="Draw 2 Final Scoring cards. Keep 1 and discard the other.",
            card_type="animal",
            number=9005,
            instance_id="test-resistance",
        )
    ]
    player.final_scoring_cards = []
    state.final_scoring_deck = [
        SetupCardRef(data_id="F001", title="FIRST"),
        SetupCardRef(data_id="F002", title="SECOND"),
        SetupCardRef(data_id="F003", title="THIRD"),
    ]

    monkeypatch.setattr("builtins.input", lambda _: "2")

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0, "_interactive": True},
        player_id=0,
    )

    assert [card.data_id for card in player.final_scoring_cards] == ["F002"]
    assert [card.data_id for card in state.final_scoring_discard] == ["F001"]
    assert [card.data_id for card in state.final_scoring_deck] == ["F003"]


def test_resistance_effect_reveals_final_scoring_then_uses_pending_choice():
    state, player = make_basic_animals_play_state(seed=616)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    player.hand = [
        AnimalCard(
            name="Resister",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Resistance",
            ability_text="Draw 2 Final Scoring cards. Keep 1 and discard the other.",
            card_type="animal",
            number=9006,
            instance_id="test-resistance-expand",
        )
    ]
    player.final_scoring_cards = []
    state.final_scoring_deck = [
        SetupCardRef(data_id="F001", title="FIRST"),
        SetupCardRef(data_id="F002", title="SECOND"),
        SetupCardRef(data_id="F003", title="THIRD"),
    ]

    actions = legal_actions_for_player(state, player_id=0)
    animal_actions = [
        action
        for action in actions
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
    ]
    assert animal_actions
    assert all(not list((action.details or {}).get("final_scoring_keep_choices") or []) for action in animal_actions)

    play_legal_main_action(state, player_id=0, card_name="animals")

    assert state.pending_decision_kind == "revealed_final_scoring_keep"
    assert len(pending_actions(state)) == 2

    play_pending_action(state, keep_final_scoring_refs=["F002"])

    assert [card.data_id for card in player.final_scoring_cards] == ["F002"]
    assert [card.data_id for card in state.final_scoring_discard] == ["F001"]
    assert [card.data_id for card in state.final_scoring_deck] == ["F003"]
    assert state.pending_decision_kind == ""


def test_final_scoring_discard_pending_action_discards_selected_ref():
    state, player = make_basic_animals_play_state(seed=6161)
    player.final_scoring_cards = [
        SetupCardRef(data_id="F001", title="FIRST"),
        SetupCardRef(data_id="F002", title="SECOND"),
    ]
    set_pending_decision(
        state,
        player_id=0,
        kind="final_scoring_discard",
        payload={"discard_target": 1},
    )

    play_pending_action(state, discard_final_scoring_refs=["F002"])

    assert [card.data_id for card in player.final_scoring_cards] == ["F001"]
    assert [card.data_id for card in state.final_scoring_discard] == ["F002"]
    assert state.pending_decision_kind == ""


def test_hunter_effect_uses_pending_revealed_animal_choice():
    state, player = make_basic_animals_play_state(seed=617)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    player.hand = [
        AnimalCard(
            name="Hunter",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Hunter 3",
            card_type="animal",
            number=9007,
            instance_id="test-hunter",
        )
    ]
    state.zoo_deck = [
        AnimalCard(
            name="A1",
            cost=3,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9301,
            instance_id="deck-hunter-a1",
        ),
        AnimalCard(
            name="S1",
            cost=3,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=9302,
            instance_id="deck-hunter-s1",
        ),
        AnimalCard(
            name="A2",
            cost=3,
            size=1,
            appeal=2,
            conservation=0,
            card_type="animal",
            number=9303,
            instance_id="deck-hunter-a2",
        ),
    ]

    actions = legal_actions_for_player(state, player_id=0)
    animal_actions = [action for action in actions if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"]
    assert animal_actions
    assert all(not list((action.details or {}).get("deck_keep_card_choices") or []) for action in animal_actions)

    play_legal_main_action(state, player_id=0, card_name="animals")

    assert state.pending_decision_kind == "revealed_cards_keep"
    assert len(pending_actions(state)) == 2

    play_pending_action(state, keep_card_instance_ids=["deck-hunter-a2"])

    assert any(card.instance_id == "deck-hunter-a2" for card in player.hand)
    assert all(card.instance_id != "deck-hunter-a1" for card in player.hand)
    assert {card.instance_id for card in state.zoo_discard} >= {"deck-hunter-a1", "deck-hunter-s1"}
    assert state.pending_decision_kind == ""


def test_digging_effect_uses_pending_choices_after_each_replenish():
    state, player = make_basic_animals_play_state(seed=618)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    player.hand = [
        AnimalCard(
            name="Digger",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Digging 2",
            card_type="animal",
            number=9008,
            instance_id="test-digging",
        ),
        AnimalCard(
            name="Hand Choice",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9306,
            instance_id="hand-digging",
        ),
    ]
    state.zoo_display = [
        AnimalCard(
            name="Display 1",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9304,
            instance_id="disp-digging-1",
        ),
    ]
    state.zoo_deck = [
        AnimalCard(
            name="Refill 1",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9305,
            instance_id="deck-digging-1",
        ),
        AnimalCard(
            name="Refill 2",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9307,
            instance_id="deck-digging-2",
        ),
    ]

    actions = legal_actions_for_player(state, player_id=0)
    animal_actions = [action for action in actions if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"]
    assert animal_actions
    assert all(not list((action.details or {}).get("digging_choices") or []) for action in animal_actions)

    play_legal_main_action(state, player_id=0, card_name="animals")

    assert state.pending_decision_kind == "digging_choice"
    play_pending_action(state, digging_choice_mode="display", display_index=0)

    assert state.pending_decision_kind == "digging_choice"
    assert state.zoo_display[0].instance_id == "deck-digging-1"
    next_pending_actions = pending_actions(state)
    assert any(
        (action.details or {}).get("digging_choice_mode") == "display"
        and (action.details or {}).get("display_index") == 0
        and (action.details or {}).get("display_card_number") == 9305
        for action in next_pending_actions
    )
    play_pending_action(state, digging_choice_mode="skip")
    assert state.pending_decision_kind == ""


def test_digging_pending_actions_expand_followup_clever_targets_and_resume_cleanly():
    state, player = make_basic_animals_play_state(seed=619)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    player.hand = [
        AnimalCard(
            name="Digger",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Digging 1",
            card_type="animal",
            number=9309,
            instance_id="digger-followup",
        ),
        AnimalCard(
            name="Clever Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Clever",
            card_type="animal",
            number=9310,
            instance_id="clever-followup",
        ),
    ]
    configure_standard_enclosures(player, sizes=(1, 1), origins=((0, 0), (1, 0)))
    state.zoo_display = [
        AnimalCard(
            name="Display Digging",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9311,
            instance_id="disp-digging-followup",
        ),
    ]
    state.zoo_deck = [
        AnimalCard(
            name="Refill Digging",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9312,
            instance_id="deck-digging-followup",
        ),
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={
            "_selected_plays_override": [
                {"card_instance_id": "digger-followup", "enclosure_index": 0, "card_cost": 0},
                {"card_instance_id": "clever-followup", "enclosure_index": 1, "card_cost": 0},
            ]
        },
        player_id=0,
    )

    assert state.pending_decision_kind == "digging_choice"

    current_pending_actions = pending_actions(state)
    assert any(
        (action.details or {}).get("digging_choice_mode") == "display"
        and (action.details or {}).get("display_index") == 0
        and (action.details or {}).get("clever_targets") == ["sponsors"]
        for action in current_pending_actions
    )

    play_pending_action(
        state,
        digging_choice_mode="display",
        display_index=0,
        clever_targets=["sponsors"],
    )

    assert state.pending_decision_kind == ""
    assert player.action_order[0] == "sponsors"
    assert any(card.instance_id == "clever-followup" for card in player.zoo_cards)
    assert all(card.instance_id != "clever-followup" for card in player.hand)


def test_posturing_effect_uses_pending_free_build_choices_after_each_placement():
    state, player = make_basic_animals_play_state(seed=6191)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    _ensure_player_map_initialized(state, player)
    player.hand = [
        AnimalCard(
            name="Posturer",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Posturing 2",
            card_type="animal",
            number=9313,
            instance_id="posturing-animal",
        )
    ]

    actions = legal_actions_for_player(state, player_id=0)
    animal_actions = [action for action in actions if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"]
    assert animal_actions
    assert all(not list((action.details or {}).get("free_building_placement_choices") or []) for action in animal_actions)

    play_legal_main_action(state, player_id=0, card_name="animals")

    assert state.pending_decision_kind == "animals_free_building_placement_choice"
    play_pending_action(
        state,
        predicate=lambda action: (action.details or {}).get("animals_free_build_selection") is not None,
    )

    assert state.pending_decision_kind == "animals_free_building_placement_choice"
    next_pending_actions = pending_actions(state)
    assert any(bool((action.details or {}).get("animals_free_build_skip")) for action in next_pending_actions)
    play_pending_action(state, animals_free_build_skip=True)

    assert state.pending_decision_kind == ""


def test_posturing_pending_actions_expand_followup_clever_targets_and_resume_cleanly():
    state, player = make_basic_animals_play_state(seed=6192)
    configure_player(
        state,
        action_order=("cards", "animals", "build", "association", "sponsors"),
    )
    configure_standard_enclosures(player, sizes=(1, 1), origins=((0, 0), (1, 0)))
    _ensure_player_map_initialized(state, player)
    player.hand = [
        AnimalCard(
            name="Posturer",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Posturing 2",
            card_type="animal",
            number=9314,
            instance_id="posturing-followup",
        ),
        AnimalCard(
            name="Clever Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Clever",
            card_type="animal",
            number=9315,
            instance_id="clever-posturing-followup",
        ),
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={
            "_selected_plays_override": [
                {"card_instance_id": "posturing-followup", "enclosure_index": 0, "card_cost": 0},
                {"card_instance_id": "clever-posturing-followup", "enclosure_index": 1, "card_cost": 0},
            ]
        },
        player_id=0,
    )

    assert state.pending_decision_kind == "animals_free_building_placement_choice"

    play_pending_action(
        state,
        predicate=lambda action: (action.details or {}).get("animals_free_build_selection") is not None,
    )

    assert state.pending_decision_kind == "animals_free_building_placement_choice"
    second_pending_actions = pending_actions(state)
    assert any(
        (action.details or {}).get("animals_free_build_selection") is not None
        and (action.details or {}).get("clever_targets") == ["sponsors"]
        for action in second_pending_actions
    )

    play_pending_action(
        state,
        predicate=lambda action: (
            (action.details or {}).get("animals_free_build_selection") is not None
            and (action.details or {}).get("clever_targets") == ["sponsors"]
        ),
    )

    assert state.pending_decision_kind == ""
    assert player.action_order[0] == "sponsors"
    assert any(card.instance_id == "clever-posturing-followup" for card in player.zoo_cards)
    assert all(card.instance_id != "clever-posturing-followup" for card in player.hand)


def test_digging_pending_actions_fall_back_to_base_choices_when_followup_simulation_fails(monkeypatch):
    state, player = make_basic_animals_play_state(seed=6193)
    set_pending_decision(
        state,
        player_id=0,
        kind="digging_choice",
        payload={"remaining_loops": 1},
    )
    state.zoo_display = [
        AnimalCard(
            name="Dig Display",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=9316,
            instance_id="dig-display",
        )
    ]
    hand_card = AnimalCard(
        name="Dig Hand",
        cost=0,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=9317,
        instance_id="dig-hand",
    )
    player.hand = [hand_card]

    monkeypatch.setattr(main, "_resolve_pending_action_variants_by_simulation", lambda **kwargs: [])

    current_pending_actions = pending_actions(state)

    assert any((action.details or {}).get("digging_choice_mode") == "skip" for action in current_pending_actions)
    assert any(
        (action.details or {}).get("digging_choice_mode") == "display"
        and (action.details or {}).get("display_index") == 0
        for action in current_pending_actions
    )
    assert any(
        (action.details or {}).get("digging_choice_mode") == "hand"
        and (action.details or {}).get("hand_card_instance_id") == hand_card.instance_id
        for action in current_pending_actions
    )


def test_animals_free_build_pending_actions_fall_back_to_base_choices_when_followup_simulation_fails(monkeypatch):
    state, player = make_basic_animals_play_state(seed=6194)
    _ensure_player_map_initialized(state, player)
    set_pending_decision(
        state,
        player_id=0,
        kind="animals_free_building_placement_choice",
        payload={
            "remaining_loops": 1,
            "animals_free_build_effect": "posturing",
        },
    )

    monkeypatch.setattr(main, "_resolve_pending_action_variants_by_simulation", lambda **kwargs: [])

    current_pending_actions = pending_actions(state)

    assert any(bool((action.details or {}).get("animals_free_build_skip")) for action in current_pending_actions)
    assert any((action.details or {}).get("animals_free_build_selection") is not None for action in current_pending_actions)


def test_resume_animals_followup_rebinds_stale_enclosure_choice():
    state = make_state(9308)
    player = configure_player(
        state,
        money=10,
        hand=[
            AnimalCard(
                name="Resume Mammal",
                cost=0,
                size=2,
                appeal=2,
                conservation=0,
                card_type="animal",
                number=9308,
                instance_id="resume-animal",
            )
        ],
    )
    player.enclosures = [
        Enclosure(size=2, origin=(0, 0), enclosure_type="reptile_house", animal_capacity=2),
        Enclosure(size=2, origin=(1, 0), enclosure_type="standard", animal_capacity=1),
    ]
    player.enclosure_objects = [
        EnclosureObject(
            size=2,
            enclosure_type="reptile_house",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(0, 0),
            rotation="ROT_0",
        ),
        EnclosureObject(
            size=2,
            enclosure_type="enclosure_2",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(1, 0),
            rotation="ROT_0",
        ),
    ]

    _resume_animals_followup_from_pending_payload(
        state,
        player_id=0,
        payload={
            "resume_animals_plays": [
                {
                    "card_instance_id": "resume-animal",
                    "card_hand_index": 0,
                    "enclosure_index": 0,
                    "card_cost": 0,
                    "spaces_used": 1,
                }
            ]
        },
    )

    assert not player.hand
    assert player.enclosures[0].occupied is False
    assert player.enclosures[1].occupied is True
    assert any("animals_followup_rebound requested=1 rebound=1" in line for line in state.effect_log)

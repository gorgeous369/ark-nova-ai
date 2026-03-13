from __future__ import annotations

import copy
from types import SimpleNamespace

import main


class _FakeCard:
    def __init__(self, *, name: str, instance_id: str, card_type: str = "animal"):
        self.name = name
        self.instance_id = instance_id
        self.card_type = card_type
        self.size = 2
        self.number = 1
        self.appeal = 3
        self.reputation_gain = 0
        self.conservation = 1


class _FakeEnclosure:
    def __init__(self, *, size: int):
        self.size = size
        self.origin = (0, 0)
        self.rotation = "ROT_0"
        self.enclosure_type = "standard"
        self.animal_capacity = 1


class _FakePlayer:
    def __init__(self):
        self.action_upgraded = {"animals": True}
        self.action_order = ["cards", "build", "animals", "association", "sponsors"]
        self.hand = [
            _FakeCard(name="Tiger", instance_id="card-tiger"),
            _FakeCard(name="Bear", instance_id="card-bear"),
        ]
        self.money = 10
        self.reputation = 5
        self.enclosures = [
            _FakeEnclosure(size=2),
            _FakeEnclosure(size=3),
        ]


class _FakeState:
    def __init__(self):
        self.players = [_FakePlayer()]
        self.effect_log = []
        self.pending_decision_kind = ""
        self.pending_decision_payload = {}
        self.current_player = 0
        self.zoo_display = []
        self.zoo_deck = []


def test_list_legal_animals_options_emits_animals_profile(monkeypatch):
    state = _FakeState()
    snapshots = []

    def _fake_serialize_animal_play_step(
        *,
        player,
        card,
        hand_index,
        enclosure,
        enclosure_index,
        resolved_cost,
    ):
        del player
        del enclosure
        return {
            "card_instance_id": card.instance_id,
            "card_name": card.name,
            "card_hand_index": int(hand_index),
            "enclosure_index": int(enclosure_index),
            "card_cost": int(resolved_cost),
            "card_appeal": int(card.appeal),
            "card_reputation": int(card.reputation_gain),
            "card_conservation": int(card.conservation),
            "spaces_used": 1,
        }

    monkeypatch.setattr(main, "_animals_play_limit", lambda strength, upgraded: 2)
    monkeypatch.setattr(main, "_animal_size_restriction_met", lambda player, card: True)
    monkeypatch.setattr(main, "_animal_condition_ignore_capacity", lambda player, card: 0)
    monkeypatch.setattr(main, "_animal_icon_missing_conditions", lambda player, card: 0)
    monkeypatch.setattr(main, "_animal_play_cost", lambda player, card: 1)
    monkeypatch.setattr(
        main,
        "_enclosure_can_host_animal_with_ignores",
        lambda **kwargs: True,
    )
    monkeypatch.setattr(
        main,
        "_serialize_animal_play_step",
        _fake_serialize_animal_play_step,
    )

    state_token = main._LEGAL_ACTIONS_PROFILE_STATE.set(
        {
            "phase": "list_legal_animals_options",
            "player_id": 0,
            "concrete": True,
            "pending": "",
            "action_order": [],
            "branch_metrics": {},
            "abstract_action_count": 0,
            "annotated_action_count": 0,
            "pruned_action_count": 0,
            "concrete_action_count": 0,
            "current_branch": "animals",
            "last_subfunction": "",
            "last_completed_subfunction": "",
        }
    )
    try:
        with main.legal_actions_profiling(lambda snapshot: snapshots.append(snapshot)):
            options = main.list_legal_animals_options(state=state, player_id=0, strength=5)
    finally:
        main._LEGAL_ACTIONS_PROFILE_STATE.reset(state_token)

    assert len(options) == 12
    assert snapshots
    assert any(
        snapshot.get("animals_profile", {}).get("mode") == "scan_second_card"
        and snapshot.get("animals_profile", {}).get("second_card", {}).get("card_name") == "Tiger"
        for snapshot in snapshots
    )

    final_snapshot = snapshots[-1]
    assert final_snapshot["current_branch"] == "animals"
    assert final_snapshot["phase"] == "list_legal_animals_options"
    assert final_snapshot["animals_profile"]["mode"] == "completed"
    assert final_snapshot["animals_profile"]["single_step_count"] == 4
    assert final_snapshot["animals_profile"]["option_count"] == 12


def test_enumerate_concrete_animals_actions_profiles_current_resolve_option(monkeypatch):
    player = _FakePlayer()
    player.sponsor_tokens_by_number = {}
    state = _FakeState()
    snapshots = []
    resolve_snapshots = []
    option = {
        "index": 7,
        "plays": [
            {
                "card_instance_id": "card-tiger",
                "card_name": "Tiger",
                "card_hand_index": 0,
                "enclosure_index": 1,
            },
            {
                "card_instance_id": "card-bear",
                "card_name": "Bear",
                "card_hand_index": 1,
                "enclosure_index": 0,
            },
        ],
    }

    monkeypatch.setattr(main, "list_legal_animals_options", lambda **kwargs: [copy.deepcopy(option)])
    monkeypatch.setattr(
        main,
        "_format_animals_play_step_for_human",
        lambda step, player: f"{step['card_name']}@E{step['enclosure_index']}",
    )
    monkeypatch.setattr(main, "_enumerate_animals_effect_choice_variants", lambda *args, **kwargs: [({}, "")])
    monkeypatch.setattr(main, "_player_has_sponsor", lambda *args, **kwargs: False)
    monkeypatch.setattr(main, "resolve_card_effect", lambda card: SimpleNamespace(code=""))

    def _fake_resolve_action_detail_variants_by_simulation(**kwargs):
        del kwargs
        profile_state = copy.deepcopy(main._LEGAL_ACTIONS_PROFILE_STATE.get())
        resolve_snapshots.append(profile_state)
        return [({"animals_sequence_index": 6}, "")]

    monkeypatch.setattr(
        main,
        "_resolve_action_detail_variants_by_simulation",
        _fake_resolve_action_detail_variants_by_simulation,
    )

    state_token = main._LEGAL_ACTIONS_PROFILE_STATE.set(
        {
            "phase": "enumerate_concrete_animals_actions",
            "player_id": 0,
            "concrete": True,
            "pending": "",
            "action_order": [],
            "branch_metrics": {},
            "abstract_action_count": 0,
            "annotated_action_count": 0,
            "pruned_action_count": 0,
            "concrete_action_count": 0,
            "current_branch": "animals",
            "last_subfunction": "",
            "last_completed_subfunction": "",
        }
    )
    try:
        with main.legal_actions_profiling(lambda snapshot: snapshots.append(snapshot)):
            actions = main._enumerate_concrete_animals_actions(
                state=state,
                player=player,
                player_id=0,
                template_action=main.Action(
                    main.ActionType.MAIN_ACTION,
                    value=0,
                    card_name="animals",
                    details={"effective_strength": 5},
                ),
            )
    finally:
        main._LEGAL_ACTIONS_PROFILE_STATE.reset(state_token)

    assert len(actions) == 1
    assert resolve_snapshots
    resolve_profile = resolve_snapshots[0]["animals_profile"]
    assert resolve_profile["mode"] == "resolve_option_details"
    assert resolve_profile["current_option"]["rendered"] == "Tiger@E1 ; then Bear@E0"
    assert resolve_profile["current_option"]["index"] == 7
    assert resolve_profile["current_option_label"] == "Tiger@E1 ; then Bear@E0"
    assert resolve_profile["first_play"]["card_name"] == "Tiger"
    assert resolve_profile["second_card"]["card_name"] == "Bear"


def test_list_legal_animals_options_clears_resolve_fields_during_scan_second_card(monkeypatch):
    state = _FakeState()
    snapshots = []

    def _fake_serialize_animal_play_step(
        *,
        player,
        card,
        hand_index,
        enclosure,
        enclosure_index,
        resolved_cost,
    ):
        del player
        del enclosure
        del resolved_cost
        return {
            "card_instance_id": card.instance_id,
            "card_name": card.name,
            "card_hand_index": int(hand_index),
            "enclosure_index": int(enclosure_index),
            "card_cost": 1,
            "card_appeal": int(card.appeal),
            "card_reputation": int(card.reputation_gain),
            "card_conservation": int(card.conservation),
            "spaces_used": 1,
        }

    monkeypatch.setattr(main, "_animals_play_limit", lambda strength, upgraded: 2)
    monkeypatch.setattr(main, "_animal_size_restriction_met", lambda player, card: True)
    monkeypatch.setattr(main, "_animal_condition_ignore_capacity", lambda player, card: 0)
    monkeypatch.setattr(main, "_animal_icon_missing_conditions", lambda player, card: 0)
    monkeypatch.setattr(main, "_animal_play_cost", lambda player, card: 1)
    monkeypatch.setattr(main, "_enclosure_can_host_animal_with_ignores", lambda **kwargs: True)
    monkeypatch.setattr(main, "_serialize_animal_play_step", _fake_serialize_animal_play_step)

    state_token = main._LEGAL_ACTIONS_PROFILE_STATE.set(
        {
            "phase": "list_legal_animals_options",
            "player_id": 0,
            "concrete": True,
            "pending": "",
            "action_order": [],
            "branch_metrics": {},
            "abstract_action_count": 0,
            "annotated_action_count": 0,
            "pruned_action_count": 0,
            "concrete_action_count": 0,
            "current_branch": "animals",
            "last_subfunction": "",
            "last_completed_subfunction": "",
        }
    )
    try:
        with main.legal_actions_profiling(lambda snapshot: snapshots.append(snapshot)):
            main.list_legal_animals_options(state=state, player_id=0, strength=5)
    finally:
        main._LEGAL_ACTIONS_PROFILE_STATE.reset(state_token)

    scan_snapshots = [
        snapshot["animals_profile"]
        for snapshot in snapshots
        if snapshot.get("animals_profile", {}).get("mode") == "scan_second_card"
    ]
    assert scan_snapshots
    assert all(item.get("current_option") is None for item in scan_snapshots)
    assert all(item.get("current_option_label") == "" for item in scan_snapshots)


def test_list_legal_animals_options_reuses_single_step_host_checks(monkeypatch):
    state = _FakeState()
    host_check_counter = {"count": 0}

    def _fake_serialize_animal_play_step(
        *,
        player,
        card,
        hand_index,
        enclosure,
        enclosure_index,
        resolved_cost,
    ):
        del player
        del enclosure
        return {
            "card_instance_id": card.instance_id,
            "card_name": card.name,
            "card_hand_index": int(hand_index),
            "enclosure_index": int(enclosure_index),
            "card_cost": int(resolved_cost),
            "card_appeal": int(card.appeal),
            "card_reputation": int(card.reputation_gain),
            "card_conservation": int(card.conservation),
            "spaces_used": 1,
        }

    def _fake_can_host(**kwargs):
        del kwargs
        host_check_counter["count"] += 1
        return True

    monkeypatch.setattr(main, "_animals_play_limit", lambda strength, upgraded: 2)
    monkeypatch.setattr(main, "_animal_size_restriction_met", lambda player, card: True)
    monkeypatch.setattr(main, "_animal_condition_ignore_capacity", lambda player, card: 0)
    monkeypatch.setattr(main, "_animal_icon_missing_conditions", lambda player, card: 0)
    monkeypatch.setattr(main, "_animal_play_cost", lambda player, card: 1)
    monkeypatch.setattr(main, "_enclosure_can_host_animal_with_ignores", _fake_can_host)
    monkeypatch.setattr(main, "_serialize_animal_play_step", _fake_serialize_animal_play_step)

    options = main.list_legal_animals_options(state=state, player_id=0, strength=5)

    assert len(options) == 12
    assert host_check_counter["count"] == 8


def test_enumerate_concrete_animals_actions_dedupes_equivalent_resolution_signatures(monkeypatch):
    player = _FakePlayer()
    player.sponsor_tokens_by_number = {}
    state = _FakeState()
    state.players = [player]
    options = [
        {
            "index": 1,
            "plays": [
                {
                    "card_instance_id": "card-tiger",
                    "card_name": "Tiger",
                    "card_hand_index": 0,
                    "enclosure_index": 1,
                },
                {
                    "card_instance_id": "card-bear",
                    "card_name": "Bear",
                    "card_hand_index": 1,
                    "enclosure_index": 0,
                },
            ],
        },
        {
            "index": 2,
            "plays": [
                {
                    "card_instance_id": "card-bear",
                    "card_name": "Bear",
                    "card_hand_index": 1,
                    "enclosure_index": 0,
                },
                {
                    "card_instance_id": "card-tiger",
                    "card_name": "Tiger",
                    "card_hand_index": 0,
                    "enclosure_index": 1,
                },
            ],
        },
    ]

    monkeypatch.setattr(main, "list_legal_animals_options", lambda **kwargs: copy.deepcopy(options))
    monkeypatch.setattr(
        main,
        "_format_animals_play_step_for_human",
        lambda step, player: f"{step['card_name']}@E{step['enclosure_index']}",
    )
    monkeypatch.setattr(main, "_enumerate_animals_effect_choice_variants", lambda *args, **kwargs: [({}, "")])
    monkeypatch.setattr(main, "_player_has_sponsor", lambda *args, **kwargs: False)
    monkeypatch.setattr(main, "resolve_card_effect", lambda card: SimpleNamespace(code=""))
    monkeypatch.setattr(
        main,
        "_perform_animals_action_effect",
        lambda state, player, strength, details=None, player_id=0: None,
    )
    monkeypatch.setattr(
        main,
        "_animals_resolution_result_signature",
        lambda state, viewer_player_id: "same-outcome",
    )

    actions = main._enumerate_concrete_animals_actions(
        state=state,
        player=player,
        player_id=0,
        template_action=main.Action(
            main.ActionType.MAIN_ACTION,
            value=0,
            card_name="animals",
            details={"effective_strength": 5},
        ),
    )

    assert len(actions) == 1
    assert str(actions[0]) == "animals(strength=5) | Tiger@E1 ; then Bear@E0"


def test_enumerate_build_bonus_choice_variants_emits_build_profile():
    state = _FakeState()
    state.zoo_display = [_FakeCard(name="Display", instance_id="display-1")]
    state.zoo_deck = [_FakeCard(name="Deck", instance_id="deck-1")]
    snapshots = []
    option = {
        "index": 2,
        "building_type": "SIZE_3",
        "building_label": "enclosure_3",
        "cells": [[0, 0], [1, 0], [2, 0]],
        "size": 3,
        "cost": 6,
        "placement_bonuses": ["action_to_slot_1", "card_in_reputation_range"],
    }

    state_token = main._LEGAL_ACTIONS_PROFILE_STATE.set(
        {
            "phase": "enumerate_build_bonus_choice_variants",
            "player_id": 0,
            "concrete": True,
            "pending": "",
            "action_order": [],
            "branch_metrics": {},
            "abstract_action_count": 0,
            "annotated_action_count": 0,
            "pruned_action_count": 0,
            "concrete_action_count": 0,
            "current_branch": "build",
            "last_subfunction": "",
            "last_completed_subfunction": "",
        }
    )
    try:
        with main.legal_actions_profiling(lambda snapshot: snapshots.append(snapshot)):
            variants = main._enumerate_build_bonus_choice_variants(
                state=state,
                player_id=0,
                option=copy.deepcopy(option),
            )
    finally:
        main._LEGAL_ACTIONS_PROFILE_STATE.reset(state_token)

    assert len(variants) == 10
    assert any(
        snapshot.get("build_profile", {}).get("mode") == "expand_bonus"
        and snapshot.get("build_profile", {}).get("current_bonus") == "card_in_reputation_range"
        and snapshot.get("build_profile", {}).get("variant_count_before") == 5
        for snapshot in snapshots
    )

    final_snapshot = snapshots[-1]
    build_profile = final_snapshot["build_profile"]
    assert final_snapshot["current_branch"] == "build"
    assert final_snapshot["phase"] == "enumerate_build_bonus_choice_variants"
    assert build_profile["mode"] == "completed_bonus_variants"
    assert build_profile["current_option"]["rendered"] == "enclosure_3 cells=[(0,0),(1,0),(2,0)]"
    assert build_profile["bonus_count"] == 2
    assert build_profile["variant_count_before"] == 10
    assert build_profile["deduped_variant_count"] == 10

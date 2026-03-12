import main

from tests.helpers import make_state


def test_attach_resume_context_defaults_populates_missing_fields_without_overwriting_existing_values():
    state = make_state(9401)
    state.pending_decision_payload = {
        "resume_sponsor_strength": 9,
        "resume_animals_player_id": 1,
    }
    source_payload = {
        "resume_kind": "sponsors_followup",
        "resume_sponsors_player_id": 0,
        "resume_sponsor_selections": [{"card_instance_id": "sponsor-a"}],
        "resume_sponsor_details": {"existing": "sponsor"},
        "resume_sponsor_strength": 4,
        "resume_sponsor_took_from_display": True,
        "resume_animals_player_id": 0,
        "resume_animals_plays": [{"card_instance_id": "animal-a"}],
        "resume_animals_details": {"existing": "animal"},
        "resume_animals_sponsor_228_postplay": True,
        "break_triggered": True,
        "consumed_venom": True,
    }

    main._attach_resume_context_defaults_to_pending_decision(state, payload=source_payload)
    source_payload["resume_sponsor_selections"][0]["card_instance_id"] = "mutated"
    source_payload["resume_animals_details"]["existing"] = "mutated"

    assert state.pending_decision_payload["resume_kind"] == "sponsors_followup"
    assert state.pending_decision_payload["resume_sponsors_player_id"] == 0
    assert state.pending_decision_payload["resume_sponsor_selections"] == [{"card_instance_id": "sponsor-a"}]
    assert state.pending_decision_payload["resume_sponsor_details"] == {"existing": "sponsor"}
    assert state.pending_decision_payload["resume_sponsor_strength"] == 9
    assert state.pending_decision_payload["resume_sponsor_took_from_display"] is True
    assert state.pending_decision_payload["resume_animals_player_id"] == 1
    assert state.pending_decision_payload["resume_animals_plays"] == [{"card_instance_id": "animal-a"}]
    assert state.pending_decision_payload["resume_animals_details"] == {"existing": "animal"}
    assert state.pending_decision_payload["resume_animals_sponsor_228_postplay"] is True
    assert state.pending_decision_payload["break_triggered"] is True
    assert state.pending_decision_payload["consumed_venom"] is True


def test_merge_pending_followup_details_into_payload_updates_resume_detail_buckets_consistently():
    payload = {
        "resume_kind": "sponsors_followup",
        "resume_animals_details": {"animals_only": True},
        "resume_sponsor_details": {"sponsors_only": True},
        "resume_sponsor_passive_effect": "sponsor_214",
        "resume_sponsor_passive_details": {"passive_only": True},
    }

    merged = main._merge_pending_followup_details_into_payload(
        payload,
        pending_kind="revealed_cards_keep",
        action_details={
            "pending_kind": "revealed_cards_keep",
            "concrete": True,
            "action_label": "keep [A]",
            "keep_card_instance_ids": ["animal-a"],
            "keep_card_numbers": [9402],
            "revealed_source": "deck",
            "clever_targets": ["sponsors"],
            "effective_strength": 4,
        },
    )

    assert payload == {
        "resume_kind": "sponsors_followup",
        "resume_animals_details": {"animals_only": True},
        "resume_sponsor_details": {"sponsors_only": True},
        "resume_sponsor_passive_effect": "sponsor_214",
        "resume_sponsor_passive_details": {"passive_only": True},
    }
    assert merged["resume_animals_details"] == {
        "animals_only": True,
        "clever_targets": ["sponsors"],
        "effective_strength": 4,
    }
    assert merged["resume_sponsor_details"] == {
        "sponsors_only": True,
        "clever_targets": ["sponsors"],
        "effective_strength": 4,
    }
    assert merged["resume_sponsor_passive_details"] == {
        "passive_only": True,
        "clever_targets": ["sponsors"],
        "effective_strength": 4,
    }
    assert merged["resume_sponsor_strength"] == 4
    assert "keep_card_instance_ids" not in merged["resume_animals_details"]
    assert "keep_card_numbers" not in merged["resume_sponsor_details"]


def test_split_deferred_sponsor_pending_followup_fragment_uses_registry_metadata():
    general_fragment, sponsor_fragment = main._split_deferred_sponsor_pending_followup_fragment(
        "sponsor_214_action_to_slot_1_choice",
        {
            "sponsor_action_to_slot_1_choices": [{"action_card": "cards"}],
            "effective_strength": 5,
            "clever_targets": ["animals"],
        },
    )

    assert sponsor_fragment == {
        "sponsor_action_to_slot_1_choices": [{"action_card": "cards"}],
    }
    assert general_fragment == {
        "effective_strength": 5,
        "clever_targets": ["animals"],
    }


def test_serialize_public_pending_payload_uses_registry_keys_and_extra_serializer():
    serialized = main._serialize_public_pending_payload(
        "opening_draft_keep",
        {
            "keep_target": 2,
            "draft_card_instance_ids": ["draft-a", "draft-b", "draft-c"],
            "ignored_field": "hidden",
        },
    )

    assert serialized == {
        "keep_target": 2,
        "draft_card_count": 3,
    }

import copy

import numpy as np

import main

from arknova_rl.encoding import ActionFeatureEncoder, ObservationEncoder
from main import Action, ActionType, AnimalCard, SetupCardRef, legal_actions
from tests.helpers import make_state


def test_action_feature_encoder_is_deterministic():
    state = make_state(701)
    actor = state.players[0]
    actions = legal_actions(actor, state=state, player_id=0)
    assert actions

    encoder = ActionFeatureEncoder()
    vec_a = encoder.encode(actions[0])
    vec_b = encoder.encode(actions[0])

    assert vec_a.shape[0] == encoder.feature_dim
    assert np.array_equal(vec_a, vec_b)


def test_action_feature_encoder_ignores_ui_labels_and_opaque_instance_ids():
    encoder = ActionFeatureEncoder()
    action_a = Action(
        ActionType.MAIN_ACTION,
        value=1,
        card_name="animals",
        details={
            "effective_strength": 4,
            "concrete": True,
            "action_label": "play slot A",
            "card_instance_id": "animal-a",
            "discard_card_instance_ids": ["d-a"],
        },
    )
    action_b = Action(
        ActionType.MAIN_ACTION,
        value=1,
        card_name="animals",
        details={
            "effective_strength": 4,
            "concrete": True,
            "action_label": "play slot B",
            "card_instance_id": "animal-b",
            "discard_card_instance_ids": ["d-b"],
        },
    )

    vec_a = encoder.encode(action_a)
    vec_b = encoder.encode(action_b)

    assert np.array_equal(vec_a, vec_b)


def test_observation_encoder_public_part_ignores_opponent_hidden_faces():
    encoder = ObservationEncoder()
    state_a = make_state(702)
    state_b = copy.deepcopy(state_a)

    opp_a = state_a.players[1]
    opp_b = state_b.players[1]
    opp_a.hand = [
        AnimalCard("OPP_A_CARD_1", 2, 1, 0, 0, number=99701, instance_id="opp-a-1"),
        AnimalCard("OPP_A_CARD_2", 3, 1, 0, 0, number=99702, instance_id="opp-a-2"),
    ]
    opp_b.hand = [
        AnimalCard("OPP_B_CARD_1", 9, 4, 0, 0, number=99801, instance_id="opp-b-1"),
        AnimalCard("OPP_B_CARD_2", 8, 4, 0, 0, number=99802, instance_id="opp-b-2"),
    ]
    opp_a.final_scoring_cards = [SetupCardRef(data_id="F-A", title="FINAL_A")]
    opp_b.final_scoring_cards = [SetupCardRef(data_id="F-B", title="FINAL_B")]

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))

    assert np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_is_viewer_invariant():
    encoder = ObservationEncoder()
    state = make_state(7021)

    public_0 = encoder.encode_public_observation(main.build_public_observation(state, viewer_player_id=0))
    public_1 = encoder.encode_public_observation(main.build_public_observation(state, viewer_player_id=1))

    assert np.array_equal(public_0, public_1)


def test_observation_encoder_can_skip_global_encoding():
    encoder = ObservationEncoder()
    state = make_state(7022)

    local_vec, global_vec = encoder.encode_from_state(state, 0, include_global=False)

    assert local_vec.size > 0
    assert global_vec.shape == (0,)


def test_observation_encoder_private_part_changes_with_own_hidden_faces():
    encoder = ObservationEncoder()
    state_a = make_state(703)
    state_b = copy.deepcopy(state_a)

    me_a = state_a.players[0]
    me_b = state_b.players[0]
    me_a.hand = [AnimalCard("SELF_A", 2, 1, 0, 0, number=99601, instance_id="self-a")]
    me_b.hand = [AnimalCard("SELF_B", 9, 5, 0, 0, number=99602, instance_id="self-b")]
    me_a.final_scoring_cards = [SetupCardRef(data_id="FS_A", title="SELF_FS_A")]
    me_b.final_scoring_cards = [SetupCardRef(data_id="FS_B", title="SELF_FS_B")]

    private_a = encoder.encode_private_observation(main.build_private_observation(state_a, viewer_player_id=0))
    private_b = encoder.encode_private_observation(main.build_private_observation(state_b, viewer_player_id=0))

    assert private_a.shape == private_b.shape
    assert not np.array_equal(private_a, private_b)


def test_observation_encoder_public_part_changes_with_map_layout():
    encoder = ObservationEncoder()
    state_a = make_state(704)
    state_b = copy.deepcopy(state_a)

    opp_a = state_a.players[1]
    opp_b = state_b.players[1]
    main._ensure_player_map_initialized(state_a, opp_a)
    main._ensure_player_map_initialized(state_b, opp_b)
    assert opp_b.zoo_map is not None
    opp_b.zoo_map.add_building(
        main.Building(main.BuildingType.KIOSK, main.HexTile(0, 0), main.Rotation.ROT_0)
    )

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))

    assert public_a.shape == public_b.shape
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_display_card_faces():
    encoder = ObservationEncoder()
    state_a = make_state(705)
    state_b = copy.deepcopy(state_a)

    state_a.zoo_display[0] = AnimalCard(
        "DISPLAY_A",
        4,
        2,
        1,
        0,
        card_type="animal",
        number=99501,
        instance_id="display-a",
    )
    state_b.zoo_display[0] = AnimalCard(
        "DISPLAY_B",
        4,
        2,
        1,
        0,
        card_type="animal",
        number=99502,
        instance_id="display-b",
    )

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_conservation_slot_ownership():
    encoder = ObservationEncoder()
    state_a = make_state(706)
    state_b = copy.deepcopy(state_a)

    state_a.conservation_project_slots = {"P900_Custom": {"2": 0, "5": None}}
    state_b.conservation_project_slots = {"P900_Custom": {"2": 1, "5": None}}

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_public_resource_identities():
    encoder = ObservationEncoder()
    state_a = make_state(707)
    state_b = copy.deepcopy(state_a)

    state_a.available_partner_zoos = {"America"}
    state_b.available_partner_zoos = {"Asia"}
    state_a.available_universities = {"University A"}
    state_b.available_universities = {"University B"}

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_public_player_detail_layout():
    encoder = ObservationEncoder()
    state_a = make_state(708)
    state_b = copy.deepcopy(state_a)
    p_a = state_a.players[0]
    p_b = state_b.players[0]

    p_a.zoo_cards = [
        AnimalCard("ZOO_A", 3, 2, 1, 0, card_type="animal", number=99401, instance_id="zoo-a")
    ]
    p_b.zoo_cards = [
        AnimalCard("ZOO_B", 3, 2, 1, 0, card_type="animal", number=99402, instance_id="zoo-b")
    ]
    p_a.enclosures = [
        main.Enclosure(
            size=3,
            occupied=True,
            origin=(0, 0),
            rotation="ROT_0",
            enclosure_type="standard",
            used_capacity=1,
            animal_capacity=3,
        )
    ]
    p_b.enclosures = [
        main.Enclosure(
            size=3,
            occupied=True,
            origin=(0, 0),
            rotation="ROT_0",
            enclosure_type="standard",
            used_capacity=2,
            animal_capacity=3,
        )
    ]
    p_a.enclosure_objects = [
        main.EnclosureObject(
            size=3,
            enclosure_type="standard",
            adjacent_rock=1,
            adjacent_water=0,
            animals_inside=1,
            origin=(0, 0),
            rotation="ROT_0",
        )
    ]
    p_b.enclosure_objects = [
        main.EnclosureObject(
            size=3,
            enclosure_type="standard",
            adjacent_rock=0,
            adjacent_water=1,
            animals_inside=1,
            origin=(0, 0),
            rotation="ROT_0",
        )
    ]

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_pending_payload_scalars():
    encoder = ObservationEncoder()
    state_a = make_state(709)
    state_b = copy.deepcopy(state_a)

    state_a.pending_decision_kind = "cards_discard"
    state_a.pending_decision_player_id = 0
    state_a.pending_decision_payload = {"discard_target": 1}
    state_b.pending_decision_kind = "cards_discard"
    state_b.pending_decision_player_id = 0
    state_b.pending_decision_payload = {"discard_target": 2}

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_hides_pending_private_instance_ids():
    encoder = ObservationEncoder()
    state_a = make_state(710)
    state_b = copy.deepcopy(state_a)

    state_a.pending_decision_kind = "opening_draft_keep"
    state_a.pending_decision_player_id = 0
    state_a.pending_decision_payload = {
        "keep_target": 4,
        "draft_card_instance_ids": ["draft-private-a", "draft-private-b"],
    }
    state_b.pending_decision_kind = "opening_draft_keep"
    state_b.pending_decision_player_id = 0
    state_b.pending_decision_payload = {
        "keep_target": 4,
        "draft_card_instance_ids": ["draft-private-x", "draft-private-y"],
    }

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert np.array_equal(public_a, public_b)


def test_observation_encoder_private_part_ignores_private_instance_ids_when_faces_match():
    encoder = ObservationEncoder()
    state_a = make_state(711)
    state_b = copy.deepcopy(state_a)

    card_common = dict(
        name="SELF_PRIVATE",
        cost=4,
        size=2,
        appeal=1,
        conservation=0,
        card_type="animal",
        number=99311,
    )
    state_a.players[0].hand = [AnimalCard(instance_id="private-a", **card_common)]
    state_b.players[0].hand = [AnimalCard(instance_id="private-b", **card_common)]

    private_a = encoder.encode_private_observation(main.build_private_observation(state_a, viewer_player_id=0))
    private_b = encoder.encode_private_observation(main.build_private_observation(state_b, viewer_player_id=0))
    assert np.array_equal(private_a, private_b)


def test_observation_encoder_public_part_changes_with_action_token_state_details():
    encoder = ObservationEncoder()
    state_a = make_state(712)
    state_b = copy.deepcopy(state_a)

    p_a = state_a.players[0]
    p_b = state_b.players[0]
    p_a.multiplier_tokens_on_actions["cards"] = 1
    p_a.venom_tokens_on_actions["build"] = 1
    p_a.constriction_tokens_on_actions["association"] = 1
    p_a.extra_actions_granted["sponsors"] = 2
    p_a.extra_any_actions = 1
    p_a.extra_strength_actions[3] = 1
    p_a.camouflage_condition_ignores = 1
    p_a.sponsor_tokens_by_number[253] = 2
    p_a.sponsor_waza_assignment_mode = "large"
    p_a.sponsor_ignore_large_condition_charges = 1

    p_b.multiplier_tokens_on_actions["animals"] = 1
    p_b.venom_tokens_on_actions["cards"] = 1
    p_b.constriction_tokens_on_actions["sponsors"] = 1
    p_b.extra_actions_granted["build"] = 2
    p_b.extra_any_actions = 0
    p_b.extra_strength_actions[4] = 1
    p_b.camouflage_condition_ignores = 0
    p_b.sponsor_tokens_by_number[227] = 2
    p_b.sponsor_waza_assignment_mode = "small"
    p_b.sponsor_ignore_large_condition_charges = 0

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_trigger_player_identity():
    encoder = ObservationEncoder()
    state_a = make_state(713)
    state_b = copy.deepcopy(state_a)

    state_a.endgame_trigger_player = 0
    state_b.endgame_trigger_player = 1
    state_a.break_trigger_player = 1
    state_b.break_trigger_player = 0
    state_a.endgame_trigger_turn_index = 25
    state_b.endgame_trigger_turn_index = 25

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_opening_draft_public_card_count():
    encoder = ObservationEncoder()
    state_a = make_state(714)
    state_b = copy.deepcopy(state_a)

    state_a.pending_decision_kind = "opening_draft_keep"
    state_a.pending_decision_player_id = 0
    state_a.pending_decision_payload = {
        "keep_target": 4,
        "draft_card_instance_ids": ["draft-a", "draft-b"],
    }
    state_b.pending_decision_kind = "opening_draft_keep"
    state_b.pending_decision_player_id = 0
    state_b.pending_decision_payload = {
        "keep_target": 4,
        "draft_card_instance_ids": ["draft-x", "draft-y", "draft-z"],
    }

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_private_part_changes_with_pouched_host_card_mapping():
    encoder = ObservationEncoder()
    state_a = make_state(715)
    state_b = copy.deepcopy(state_a)

    host_a_1 = AnimalCard("HOST_1", 4, 2, 0, 0, number=99241, instance_id="host-a-1")
    host_a_2 = AnimalCard("HOST_2", 5, 2, 0, 0, number=99242, instance_id="host-a-2")
    host_b_1 = AnimalCard("HOST_1", 4, 2, 0, 0, number=99241, instance_id="host-b-1")
    host_b_2 = AnimalCard("HOST_2", 5, 2, 0, 0, number=99242, instance_id="host-b-2")
    a_card_1 = AnimalCard("POUCH_A", 2, 1, 0, 0, number=99251, instance_id="pouch-a-1")
    a_card_2 = AnimalCard("POUCH_B", 3, 1, 0, 0, number=99252, instance_id="pouch-b-1")
    b_card_1 = AnimalCard("POUCH_A", 2, 1, 0, 0, number=99251, instance_id="pouch-a-1")
    b_card_2 = AnimalCard("POUCH_B", 3, 1, 0, 0, number=99252, instance_id="pouch-b-1")

    me_a = state_a.players[0]
    me_b = state_b.players[0]
    me_a.zoo_cards = [host_a_1, host_a_2]
    me_b.zoo_cards = [host_b_1, host_b_2]
    me_a.pouched_cards = [a_card_1, a_card_2]
    me_b.pouched_cards = [b_card_1, b_card_2]
    me_a.pouched_cards_by_host = {host_a_1.instance_id: [a_card_1], host_a_2.instance_id: [a_card_2]}
    me_b.pouched_cards_by_host = {host_b_1.instance_id: [b_card_2], host_b_2.instance_id: [b_card_1]}

    private_a = encoder.encode_private_observation(main.build_private_observation(state_a, viewer_player_id=0))
    private_b = encoder.encode_private_observation(main.build_private_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(private_a, private_b)


def test_observation_encoder_local_part_marks_viewer_slot_but_global_part_stays_shared():
    encoder = ObservationEncoder()
    state = make_state(716)
    for player in state.players:
        player.hand = []
        player.final_scoring_cards = []
        player.opening_draft_drawn = []
        player.opening_draft_kept_indices = []
        player.pouched_cards = []
        player.pouched_cards_by_host = {}

    local_0, global_0 = encoder.encode_from_state(state, viewer_player_id=0)
    local_1, global_1 = encoder.encode_from_state(state, viewer_player_id=1)

    assert not np.array_equal(local_0, local_1)
    assert np.array_equal(global_0, global_1)


def test_observation_encoder_global_part_sees_opponent_hidden_faces_but_local_part_does_not():
    encoder = ObservationEncoder()
    state_a = make_state(717)
    state_b = copy.deepcopy(state_a)

    state_a.players[1].hand = [AnimalCard("OPP_A", 2, 1, 0, 0, number=99261, instance_id="opp-a")]
    state_b.players[1].hand = [AnimalCard("OPP_B", 7, 4, 0, 0, number=99262, instance_id="opp-b")]

    local_a, global_a = encoder.encode_from_state(state_a, viewer_player_id=0)
    local_b, global_b = encoder.encode_from_state(state_b, viewer_player_id=0)

    assert np.array_equal(local_a, local_b)
    assert not np.array_equal(global_a, global_b)

import copy

import numpy as np

import main

from arknova_rl.encoding import ActionFeatureEncoder, ObservationEncoder
from main import AnimalCard, SetupCardRef, legal_actions, setup_game


def test_action_feature_encoder_is_deterministic():
    state = setup_game(seed=701, player_names=["P1", "P2"])
    actor = state.players[0]
    actions = legal_actions(actor, state=state, player_id=0)
    assert actions

    encoder = ActionFeatureEncoder()
    vec_a = encoder.encode(actions[0])
    vec_b = encoder.encode(actions[0])

    assert vec_a.shape[0] == encoder.feature_dim
    assert np.array_equal(vec_a, vec_b)


def test_observation_encoder_public_part_ignores_opponent_hidden_faces():
    encoder = ObservationEncoder()
    state_a = setup_game(seed=702, player_names=["P1", "P2"])
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


def test_observation_encoder_private_part_changes_with_own_hidden_faces():
    encoder = ObservationEncoder()
    state_a = setup_game(seed=703, player_names=["P1", "P2"])
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
    state_a = setup_game(seed=704, player_names=["P1", "P2"])
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
    state_a = setup_game(seed=705, player_names=["P1", "P2"])
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
    state_a = setup_game(seed=706, player_names=["P1", "P2"])
    state_b = copy.deepcopy(state_a)

    state_a.conservation_project_slots = {"P900_Custom": {"2": 0, "5": None}}
    state_b.conservation_project_slots = {"P900_Custom": {"2": 1, "5": None}}

    public_a = encoder.encode_public_observation(main.build_public_observation(state_a, viewer_player_id=0))
    public_b = encoder.encode_public_observation(main.build_public_observation(state_b, viewer_player_id=0))
    assert not np.array_equal(public_a, public_b)


def test_observation_encoder_public_part_changes_with_public_resource_identities():
    encoder = ObservationEncoder()
    state_a = setup_game(seed=707, player_names=["P1", "P2"])
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
    state_a = setup_game(seed=708, player_names=["P1", "P2"])
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
    state_a = setup_game(seed=709, player_names=["P1", "P2"])
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
    state_a = setup_game(seed=710, player_names=["P1", "P2"])
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


def test_observation_encoder_private_part_changes_with_private_instance_ids():
    encoder = ObservationEncoder()
    state_a = setup_game(seed=711, player_names=["P1", "P2"])
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
    assert not np.array_equal(private_a, private_b)

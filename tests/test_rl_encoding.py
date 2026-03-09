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


import copy
import json

import main

from main import (
    AnimalCard,
    SetupCardRef,
    build_player_observation,
    build_private_observation,
    build_public_observation,
)
from tests.helpers import make_state


def test_public_observation_hides_private_card_faces_and_keeps_counts():
    state = make_state(601)
    p0 = state.players[0]
    p1 = state.players[1]

    p0.hand = [
        AnimalCard("P0_SECRET_HAND_ALPHA", 2, 1, 0, 0, number=99001, instance_id="p0-secret-hand-1"),
        AnimalCard("P0_SECRET_HAND_BETA", 3, 1, 0, 0, number=99002, instance_id="p0-secret-hand-2"),
    ]
    p1.hand = [
        AnimalCard("P1_SECRET_HAND_ALPHA", 4, 2, 0, 0, number=99003, instance_id="p1-secret-hand-1"),
    ]
    p0.final_scoring_cards = [
        SetupCardRef(data_id="P0_SECRET_FS_A", title="P0_SECRET_FINAL_ALPHA"),
        SetupCardRef(data_id="P0_SECRET_FS_B", title="P0_SECRET_FINAL_BETA"),
    ]
    p1.final_scoring_cards = [
        SetupCardRef(data_id="P1_SECRET_FS_A", title="P1_SECRET_FINAL_ALPHA"),
    ]

    public_obs = build_public_observation(state, viewer_player_id=0)
    dumped = json.dumps(public_obs, sort_keys=True, ensure_ascii=True)

    assert "P0_SECRET_HAND_ALPHA" not in dumped
    assert "P1_SECRET_HAND_ALPHA" not in dumped
    assert "P0_SECRET_FINAL_ALPHA" not in dumped
    assert "P1_SECRET_FINAL_ALPHA" not in dumped
    assert public_obs["players"][0]["hand_count"] == 2
    assert public_obs["players"][1]["hand_count"] == 1
    assert public_obs["players"][0]["final_scoring_count"] == 2
    assert public_obs["players"][1]["final_scoring_count"] == 1


def test_public_observation_is_invariant_when_opponent_private_faces_change():
    state_a = make_state(602)
    state_b = copy.deepcopy(state_a)

    opponent_a = state_a.players[1]
    opponent_b = state_b.players[1]
    opponent_a.hand = [
        AnimalCard("OPP_PRIVATE_A_1", 2, 1, 0, 0, number=99101, instance_id="opp-a-1"),
        AnimalCard("OPP_PRIVATE_A_2", 3, 1, 0, 0, number=99102, instance_id="opp-a-2"),
    ]
    opponent_b.hand = [
        AnimalCard("OPP_PRIVATE_B_1", 8, 3, 0, 0, number=99201, instance_id="opp-b-1"),
        AnimalCard("OPP_PRIVATE_B_2", 9, 4, 0, 0, number=99202, instance_id="opp-b-2"),
    ]
    opponent_a.final_scoring_cards = [SetupCardRef(data_id="F-A", title="FS_A")]
    opponent_b.final_scoring_cards = [SetupCardRef(data_id="F-B", title="FS_B")]

    public_a = build_public_observation(state_a, viewer_player_id=0)
    public_b = build_public_observation(state_b, viewer_player_id=0)

    assert public_a == public_b


def test_private_observation_contains_only_viewer_hidden_faces():
    state = make_state(603)
    p0 = state.players[0]
    p1 = state.players[1]

    p0.hand = [
        AnimalCard("P0_PRIVATE_HAND", 1, 1, 0, 0, number=99301, instance_id="p0-private-1"),
    ]
    p1.hand = [
        AnimalCard("P1_PRIVATE_HAND", 1, 1, 0, 0, number=99302, instance_id="p1-private-1"),
    ]
    p0.final_scoring_cards = [SetupCardRef(data_id="P0_FS", title="P0_PRIVATE_FINAL")]
    p1.final_scoring_cards = [SetupCardRef(data_id="P1_FS", title="P1_PRIVATE_FINAL")]

    private_obs = build_private_observation(state, viewer_player_id=0)
    dumped = json.dumps(private_obs, sort_keys=True, ensure_ascii=True)

    assert "P0_PRIVATE_HAND" in dumped
    assert "P0_PRIVATE_FINAL" in dumped
    assert "P1_PRIVATE_HAND" not in dumped
    assert "P1_PRIVATE_FINAL" not in dumped
    assert private_obs["player"]["hand_count"] == 1
    assert private_obs["player"]["final_scoring_count"] == 1


def test_player_observation_contains_public_and_private_sections():
    state = make_state(604)

    observation = build_player_observation(state, viewer_player_id=0)

    assert set(observation.keys()) == {"public", "private"}
    assert observation["public"]["viewer_player_id"] == 0
    assert observation["private"]["viewer_player_id"] == 0

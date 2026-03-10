from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import torch

import main

from arknova_rl.config import PPOTrainConfig
from arknova_rl.evaluator import (
    _build_model_and_encoders,
    evaluate_policy_matchup,
    load_policy_bundle_from_checkpoint,
)


class _FakeModel:
    def __init__(self, *, state_dim: int = 4, action_dim: int = 3, global_state_dim: int = 0) -> None:
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.global_state_dim = int(global_state_dim)
        self.use_lstm = False
        self.use_centralized_value = False
        self.training = False
        self.loaded_state_dict = None

    def load_state_dict(self, state_dict):
        self.loaded_state_dict = dict(state_dict)

    def eval(self):
        self.training = False
        return self

    def train(self, mode=True):
        self.training = bool(mode)
        return self

    def forward_step(self, *, state_vec, action_features, action_mask, hidden=None, global_state_vec=None):
        del action_mask, hidden, global_state_vec
        batch_size = int(state_vec.shape[0])
        action_count = int(action_features.shape[1])
        logits = torch.zeros((batch_size, action_count), dtype=torch.float32, device=state_vec.device)
        values = torch.zeros((batch_size,), dtype=torch.float32, device=state_vec.device)
        return logits, values, None


def _write_checkpoint(path):
    config = PPOTrainConfig(seed=77)
    config.resolve_algo_flags()
    torch.save(
        {
            "update": 0,
            "model_state_dict": {"fake_weight": torch.tensor([1.0])},
            "optimizer_state_dict": {},
            "config": asdict(config),
            "metrics": [],
        },
        path,
    )


def test_evaluate_policy_matchup_from_checkpoint_is_well_formed(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "checkpoint_0000.pt"
    _write_checkpoint(checkpoint_path)

    fake_models = []

    def _fake_build_model_and_encoders(*, config, device):
        del config, device
        model = _FakeModel()
        fake_models.append(model)
        return model, SimpleNamespace(), SimpleNamespace()

    monkeypatch.setattr("arknova_rl.evaluator._build_model_and_encoders", _fake_build_model_and_encoders)

    device = torch.device("cpu")
    bundle_a = load_policy_bundle_from_checkpoint(checkpoint_path, device=device, name="A")
    bundle_b = load_policy_bundle_from_checkpoint(checkpoint_path, device=device, name="B")

    class _FakeState:
        def __init__(self):
            self.pending_decision_kind = ""
            self.pending_decision_player_id = None
            self.current_player = 0
            self.players = ["P1", "P2"]
            self.forced_game_over_reason = "fake_terminal_reason"
            self.endgame_trigger_player = None
            self._move_count = 0

        def game_over(self):
            return self._move_count >= 2

    def _fake_setup_game(*, seed, player_names):
        del seed, player_names
        return _FakeState()

    def _fake_legal_actions(player, *, state, player_id):
        del player, state, player_id
        return [main.Action(main.ActionType.MAIN_ACTION, card_name="cards", details={"concrete": True})]

    def _fake_apply_action(state, action):
        del action
        state._move_count += 1
        state.current_player = min(1, state._move_count)

    def _fake_final_score_points(state, player):
        del state
        return 10 if player == "P1" else 5

    monkeypatch.setattr(main, "setup_game", _fake_setup_game)
    monkeypatch.setattr(main, "legal_actions", _fake_legal_actions)
    monkeypatch.setattr(main, "apply_action", _fake_apply_action)
    monkeypatch.setattr(main, "_final_score_points", _fake_final_score_points)
    monkeypatch.setattr(main, "_completed_rounds", lambda state: 1)

    state_dim = int(bundle_a.model.state_dim)
    global_dim = int(bundle_a.model.global_state_dim)
    action_dim = int(bundle_a.model.action_dim)
    zero_local = np.zeros((state_dim,), dtype=np.float32)
    zero_global = np.zeros((global_dim,), dtype=np.float32)

    for bundle in (bundle_a, bundle_b):
        bundle.obs_encoder.encode_from_state = (
            lambda state, actor_id, include_global=True, _local=zero_local, _global=zero_global: (
                _local,
                _global if include_global else np.zeros((0,), dtype=np.float32),
            )
        )
        bundle.action_encoder.encode_many = (
            lambda legal, _action_dim=action_dim: np.zeros((len(legal), _action_dim), dtype=np.float32)
        )

    metrics = evaluate_policy_matchup(
        policy_a=bundle_a,
        policy_b=bundle_b,
        episodes=1,
        seed=123,
        device=device,
        deterministic=True,
    )

    assert metrics.episodes == 1
    assert metrics.wins_a + metrics.wins_b + metrics.draws == 1
    assert metrics.avg_completed_rounds > 0.0
    assert metrics.terminal_reason_counts
    assert 0.0 <= metrics.win_rate_a <= 1.0
    assert len(fake_models) == 2
    assert all(model.loaded_state_dict is not None for model in fake_models)
    assert all(float(model.loaded_state_dict["fake_weight"].item()) == 1.0 for model in fake_models)

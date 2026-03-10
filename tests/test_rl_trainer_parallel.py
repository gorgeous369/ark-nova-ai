from dataclasses import asdict
from types import SimpleNamespace

import torch

from arknova_rl.config import PPOTrainConfig
from arknova_rl.trainer import (
    EpisodeRolloutResult,
    RolloutStep,
    _build_model_and_encoders,
    _collect_rollout_chunk_worker,
    _state_dict_to_cpu,
)


def test_collect_rollout_chunk_worker_returns_episode_results(monkeypatch):
    config = PPOTrainConfig(seed=91, episodes_per_update=2, rollout_workers=2)
    config.resolve_algo_flags()
    observed_specs = []
    loaded_state_dicts = []

    class _FakeModel:
        def load_state_dict(self, state_dict):
            loaded_state_dicts.append(dict(state_dict))

        def eval(self):
            return self

    def _fake_collect_episode_rollout(**kwargs):
        observed_specs.append((int(kwargs["episode_index"]), int(kwargs["episode_seed"])))
        return EpisodeRolloutResult(
            episode_index=int(kwargs["episode_index"]),
            rollout_steps=[
                RolloutStep(
                    sequence_key=(int(kwargs["episode_index"]), 0),
                    actor_id=0,
                    state_vec=torch.zeros(1, dtype=torch.float32).numpy(),
                    global_state_vec=torch.zeros(1, dtype=torch.float32).numpy(),
                    action_features=torch.zeros((1, 1), dtype=torch.float32).numpy(),
                    action_mask=torch.ones(1, dtype=torch.float32).numpy(),
                    action_index=0,
                    old_logprob=0.0,
                    old_value=0.0,
                )
            ],
            completed_rounds=1,
            terminal_abs_diffs=[0.0],
            terminal_reason="fake_terminal_reason",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(
        "arknova_rl.trainer._collect_episode_rollout",
        _fake_collect_episode_rollout,
    )
    monkeypatch.setattr(
        "arknova_rl.trainer._build_model_and_encoders",
        lambda **kwargs: (_FakeModel(), SimpleNamespace(), SimpleNamespace()),
    )

    results = _collect_rollout_chunk_worker(
        {
            "config": asdict(config),
            "model_state_dict": {"fake_weight": torch.tensor([1.0])},
            "episode_specs": [(0, 111), (1, 222)],
            "task_seed": 333,
        }
    )

    assert len(results) == 2
    assert {result.episode_index for result in results} == {0, 1}
    assert observed_specs == [(0, 111), (1, 222)]
    assert all(result.rollout_steps for result in results)
    assert all(result.completed_rounds > 0 for result in results)
    assert all(result.elapsed_seconds >= 0.0 for result in results)
    assert len(loaded_state_dicts) == 1
    assert float(loaded_state_dicts[0]["fake_weight"].item()) == 1.0

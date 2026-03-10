from dataclasses import asdict

import torch

from arknova_rl.config import PPOTrainConfig
from arknova_rl.trainer import (
    _build_model_and_encoders,
    _collect_rollout_chunk_worker,
    _state_dict_to_cpu,
)


def test_collect_rollout_chunk_worker_returns_episode_results():
    config = PPOTrainConfig(seed=91, episodes_per_update=2, rollout_workers=2)
    config.resolve_algo_flags()
    device = torch.device("cpu")
    model, _obs_encoder, _action_encoder = _build_model_and_encoders(config=config, device=device)

    results = _collect_rollout_chunk_worker(
        {
            "config": asdict(config),
            "model_state_dict": _state_dict_to_cpu(model.state_dict()),
            "episode_specs": [(0, 111), (1, 222)],
            "task_seed": 333,
        }
    )

    assert len(results) == 2
    assert {result.episode_index for result in results} == {0, 1}
    assert all(result.rollout_steps for result in results)
    assert all(result.completed_rounds > 0 for result in results)
    assert all(result.elapsed_seconds >= 0.0 for result in results)

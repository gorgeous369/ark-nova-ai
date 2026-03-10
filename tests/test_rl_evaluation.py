from dataclasses import asdict

import torch

from arknova_rl.config import PPOTrainConfig
from arknova_rl.evaluator import (
    _build_model_and_encoders,
    evaluate_policy_matchup,
    load_policy_bundle_from_checkpoint,
)


def _write_checkpoint(path):
    config = PPOTrainConfig(seed=77)
    config.resolve_algo_flags()
    device = torch.device("cpu")
    model, _obs_encoder, _action_encoder = _build_model_and_encoders(config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    torch.save(
        {
            "update": 0,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "metrics": [],
        },
        path,
    )


def test_evaluate_policy_matchup_from_checkpoint_is_well_formed(tmp_path):
    checkpoint_path = tmp_path / "checkpoint_0000.pt"
    _write_checkpoint(checkpoint_path)

    device = torch.device("cpu")
    bundle_a = load_policy_bundle_from_checkpoint(checkpoint_path, device=device, name="A")
    bundle_b = load_policy_bundle_from_checkpoint(checkpoint_path, device=device, name="B")
    metrics = evaluate_policy_matchup(
        policy_a=bundle_a,
        policy_b=bundle_b,
        episodes=2,
        seed=123,
        device=device,
        deterministic=True,
    )

    assert metrics.episodes == 2
    assert metrics.wins_a + metrics.wins_b + metrics.draws == 2
    assert metrics.avg_completed_rounds > 0.0
    assert metrics.terminal_reason_counts
    assert 0.0 <= metrics.win_rate_a <= 1.0

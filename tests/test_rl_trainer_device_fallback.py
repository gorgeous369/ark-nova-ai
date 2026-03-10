import torch

from arknova_rl.trainer import _parallel_rollout_unavailable_reason


def test_parallel_rollout_disabled_on_mps():
    reason = _parallel_rollout_unavailable_reason(
        device=torch.device("mps"),
        requested_workers=2,
    )
    assert "MPS backend" in reason


def test_parallel_rollout_allowed_on_cpu():
    reason = _parallel_rollout_unavailable_reason(
        device=torch.device("cpu"),
        requested_workers=2,
    )
    assert reason == ""

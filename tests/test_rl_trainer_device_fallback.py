import arknova_rl.trainer as trainer
import pytest
import torch

from arknova_rl.trainer import (
    _inference_device_for_training_device,
    _parallel_rollout_unavailable_reason,
    _resolve_training_device,
)


def test_parallel_rollout_disabled_on_non_cpu_inference():
    reason = _parallel_rollout_unavailable_reason(
        device=torch.device("cuda"),
        requested_workers=2,
    )
    assert "cpu inference workers" in reason


def test_parallel_rollout_allowed_on_cpu():
    reason = _parallel_rollout_unavailable_reason(
        device=torch.device("cpu"),
        requested_workers=2,
    )
    assert reason == ""


def test_cuda_training_uses_cpu_for_inference():
    inference_device = _inference_device_for_training_device(torch.device("cuda"))
    assert inference_device.type == "cpu"


def test_auto_training_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    device = _resolve_training_device("auto")
    assert device.type == "cuda"


def test_auto_training_device_defaults_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    device = _resolve_training_device("auto")
    assert device.type == "cpu"


def test_unsupported_training_device_raises_value_error():
    with pytest.raises(ValueError, match="unsupported torch device"):
        _resolve_training_device("tpu")

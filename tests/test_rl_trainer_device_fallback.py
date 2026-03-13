import arknova_rl.trainer as trainer
import torch

from arknova_rl.trainer import (
    _inference_device_for_training_device,
    _parallel_rollout_unavailable_reason,
    _resolve_training_device,
)


def test_parallel_rollout_disabled_on_non_cpu_inference():
    reason = _parallel_rollout_unavailable_reason(
        device=torch.device("mps"),
        requested_workers=2,
    )
    assert "cpu inference workers" in reason


def test_parallel_rollout_allowed_on_cpu():
    reason = _parallel_rollout_unavailable_reason(
        device=torch.device("cpu"),
        requested_workers=2,
    )
    assert reason == ""


def test_mps_training_uses_cpu_for_inference():
    inference_device = _inference_device_for_training_device(torch.device("mps"))
    assert inference_device.type == "cpu"


def test_auto_training_device_prefers_mps_when_available(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "_mps_training_available", lambda: True)
    device = _resolve_training_device("auto")
    assert device.type == "mps"


def test_auto_training_device_defaults_to_cpu_when_no_accelerator(monkeypatch):
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer, "_mps_training_available", lambda: False)
    device = _resolve_training_device("auto")
    assert device.type == "cpu"


def test_mps_requested_training_uses_mps_when_available(monkeypatch):
    monkeypatch.setattr(trainer, "_mps_training_available", lambda: True)
    device = _resolve_training_device("mps")
    assert device.type == "mps"


def test_mps_requested_training_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(trainer, "_mps_training_available", lambda: False)
    device = _resolve_training_device("mps")
    assert device.type == "cpu"

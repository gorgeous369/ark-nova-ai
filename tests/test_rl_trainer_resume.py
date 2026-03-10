import torch

from arknova_rl.trainer import _restore_torch_rng_states


def test_restore_torch_rng_states_coerces_loaded_tensor_to_cpu_bytetensor():
    current_state = torch.get_rng_state()
    fake_loaded_state = current_state.to(dtype=torch.float32)

    _restore_torch_rng_states(
        {"torch_rng_state": fake_loaded_state},
        device=torch.device("cpu"),
    )

    restored = torch.get_rng_state()
    assert restored.dtype == torch.uint8
    assert restored.device.type == "cpu"
    assert restored.shape == current_state.shape

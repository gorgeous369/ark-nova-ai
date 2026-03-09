"""Training config for Ark Nova self-play PPO variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOTrainConfig:
    algo: str = "masked_ppo"  # masked_ppo | recurrent_ppo | mappo
    seed: int = 42
    device: str = "cpu"

    total_updates: int = 200
    episodes_per_update: int = 8
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_epsilon: float = 0.2
    update_epochs: int = 3
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5

    hidden_size: int = 256
    lstm_size: int = 128
    action_hidden_size: int = 128
    use_lstm: bool = True
    use_centralized_value: bool = False

    step_reward_scale: float = 0.2
    terminal_reward_scale: float = 1.0

    checkpoint_interval: int = 20
    log_interval: int = 1

    def resolve_algo_flags(self) -> None:
        algo_name = str(self.algo).strip().lower()
        if algo_name not in {"masked_ppo", "recurrent_ppo", "mappo"}:
            raise ValueError(f"Unsupported algo: {self.algo}")
        self.algo = algo_name
        if algo_name == "masked_ppo":
            self.use_lstm = True
            self.use_centralized_value = False
            return
        if algo_name == "recurrent_ppo":
            self.use_lstm = True
            self.use_centralized_value = False
            return
        self.use_lstm = True
        self.use_centralized_value = True


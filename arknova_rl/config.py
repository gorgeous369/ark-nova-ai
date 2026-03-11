"""Training config for Ark Nova self-play PPO variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOTrainConfig:
    algo: str = "masked_ppo"  # masked_ppo | recurrent_ppo
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

    hidden_size: int = 512
    lstm_size: int = 512
    action_hidden_size: int = 256
    use_lstm: bool = True

    step_reward_scale: float = 0.2
    terminal_reward_scale: float = 1.0
    endgame_trigger_reward: float = 2.0
    endgame_speed_bonus: float = 2.0
    terminal_win_bonus: float = 3.0
    terminal_loss_penalty: float = 3.0

    checkpoint_interval: int = 20
    log_interval: int = 1
    rollout_workers: int = 1
    fixed_eval_interval: int = 20
    fixed_eval_episodes: int = 8
    fixed_eval_deterministic: bool = True
    fixed_eval_opponent: str = ""

    def resolve_algo_flags(self) -> None:
        algo_name = str(self.algo).strip().lower()
        if algo_name not in {"masked_ppo", "recurrent_ppo"}:
            raise ValueError(f"Unsupported algo: {self.algo}")
        self.algo = algo_name
        self.use_lstm = True

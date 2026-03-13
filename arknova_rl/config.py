"""Training config for Ark Nova self-play PPO variants."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PPOTrainConfig:
    seed: int = 42
    device: str = "auto"

    total_updates: int = 200
    episodes_per_update: int = 16
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

    step_reward_scale: float = 0.8
    terminal_reward_scale: float = 1.0
    endgame_trigger_reward: float = 1.0
    endgame_speed_bonus: float = 8.0
    terminal_win_bonus: float = 1.0
    terminal_loss_penalty: float = 1.0

    checkpoint_interval: int = 5
    log_interval: int = 1
    rollout_workers: int = 8
    slow_episode_trace_enabled: bool = False
    slow_episode_trace_start_seconds: float = 300.0
    slow_episode_trace_stop_seconds: float = 480.0
    fixed_eval_interval: int = 10
    fixed_eval_episodes: int = 10
    fixed_eval_deterministic: bool = True
    fixed_eval_opponent: str = ""

    def resolve_algo_flags(self) -> None:
        self.slow_episode_trace_enabled = bool(
            getattr(self, "slow_episode_trace_enabled", False)
        )
        self.slow_episode_trace_start_seconds = max(
            0.0,
            float(getattr(self, "slow_episode_trace_start_seconds", 0.0) or 0.0),
        )
        self.slow_episode_trace_stop_seconds = float(
            getattr(self, "slow_episode_trace_stop_seconds", 0.0) or 0.0
        )
        if self.slow_episode_trace_stop_seconds < 0.0:
            self.slow_episode_trace_stop_seconds = 0.0
        if (
            self.slow_episode_trace_start_seconds > 0.0
            and self.slow_episode_trace_stop_seconds > 0.0
            and self.slow_episode_trace_stop_seconds < self.slow_episode_trace_start_seconds
        ):
            raise ValueError(
                "slow_episode_trace_stop_seconds must be >= slow_episode_trace_start_seconds."
            )

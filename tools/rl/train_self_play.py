"""Train Ark Nova self-play agents with recurrent PPO.

Example:
    .venv/bin/python tools/rl/train_self_play.py \
      --updates 100 \
      --episodes-per-update 8 \
      --output-dir runs/ppo_masked
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arknova_rl.config import PPOTrainConfig, SUPPORTED_TRAINING_DEVICE_HELP, normalize_torch_device_spec


def _parse_training_device_arg(raw_value: str) -> str:
    try:
        return normalize_torch_device_spec(raw_value, allow_auto=True)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    defaults = PPOTrainConfig()
    parser = argparse.ArgumentParser(description="Ark Nova self-play RL trainer")
    parser.add_argument("--seed", type=int, default=defaults.seed, help="Random seed")
    parser.add_argument(
        "--device",
        type=_parse_training_device_arg,
        default=defaults.device,
        help=(
            "Training device for model updates: "
            f"{SUPPORTED_TRAINING_DEVICE_HELP} "
            "(rollout/fixed-eval inference stays on cpu)"
        ),
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=defaults.rollout_workers,
        help="Parallel worker processes used to collect rollout episodes",
    )
    parser.add_argument("--updates", type=int, default=defaults.total_updates, help="PPO update iterations")
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=defaults.episodes_per_update,
        help="Episodes collected per PPO update",
    )
    parser.add_argument("--lr", type=float, default=defaults.learning_rate, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=defaults.gamma, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=defaults.gae_lambda, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=defaults.clip_epsilon, help="PPO clip epsilon")
    parser.add_argument("--epochs", type=int, default=defaults.update_epochs, help="PPO epochs per update")
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size, help="MLP hidden width")
    parser.add_argument("--lstm-size", type=int, default=defaults.lstm_size, help="LSTM hidden size")
    parser.add_argument(
        "--action-hidden-size",
        type=int,
        default=defaults.action_hidden_size,
        help="Action encoder hidden size",
    )
    parser.add_argument(
        "--step-reward-scale",
        type=float,
        default=defaults.step_reward_scale,
        help="Scale for per-step progress-delta reward",
    )
    parser.add_argument(
        "--terminal-reward-scale",
        type=float,
        default=defaults.terminal_reward_scale,
        help="Scale for terminal score-diff reward",
    )
    parser.add_argument(
        "--endgame-trigger-reward",
        type=float,
        default=defaults.endgame_trigger_reward,
        help="Reward added when the acting player triggers endgame (reaches score threshold)",
    )
    parser.add_argument(
        "--endgame-speed-bonus",
        type=float,
        default=defaults.endgame_speed_bonus,
        help="Additional bonus scaled by how early endgame is triggered",
    )
    parser.add_argument(
        "--terminal-win-bonus",
        type=float,
        default=defaults.terminal_win_bonus,
        help="Extra terminal reward for finishing above all opponents",
    )
    parser.add_argument(
        "--terminal-loss-penalty",
        type=float,
        default=defaults.terminal_loss_penalty,
        help="Extra terminal penalty for finishing below the top opponent score",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=defaults.checkpoint_interval,
        help="Checkpoint save frequency (updates)",
    )
    parser.add_argument(
        "--slow-episode-trace",
        action="store_true",
        help=(
            "Enable slow episode trace recording "
            f"(default window: {defaults.slow_episode_trace_start_seconds:.0f}s -> "
            f"{defaults.slow_episode_trace_stop_seconds:.0f}s)"
        ),
    )
    parser.add_argument(
        "--slow-episode-trace-start-seconds",
        type=float,
        default=None,
        help="Trace start threshold in seconds; providing this implicitly enables slow episode tracing",
    )
    parser.add_argument(
        "--slow-episode-trace-stop-seconds",
        type=float,
        default=None,
        help="Trace stop threshold in seconds; providing this implicitly enables slow episode tracing",
    )
    parser.add_argument(
        "--slow-episode-trace-seconds",
        dest="slow_episode_trace_start_seconds",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fixed-eval-interval",
        type=int,
        default=defaults.fixed_eval_interval,
        help="Run fixed-checkpoint evaluation every N updates (0 disables)",
    )
    parser.add_argument(
        "--fixed-eval-episodes",
        type=int,
        default=defaults.fixed_eval_episodes,
        help="Episodes per fixed evaluation run",
    )
    parser.add_argument(
        "--fixed-eval-opponent",
        type=str,
        default=defaults.fixed_eval_opponent,
        help="Optional fixed opponent checkpoint path; defaults to checkpoint_0000.pt or earliest checkpoint in output dir",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/self_play",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Checkpoint path to resume from",
    )
    return parser.parse_args(argv)


def _resolve_slow_episode_trace_settings(
    args: argparse.Namespace,
    *,
    defaults: PPOTrainConfig | None = None,
) -> tuple[bool, float, float]:
    resolved_defaults = defaults or PPOTrainConfig()
    explicit_start = getattr(args, "slow_episode_trace_start_seconds", None)
    explicit_stop = getattr(args, "slow_episode_trace_stop_seconds", None)
    trace_requested = (
        bool(getattr(args, "slow_episode_trace", False))
        or explicit_start is not None
        or explicit_stop is not None
    )
    start_seconds = (
        float(resolved_defaults.slow_episode_trace_start_seconds)
        if explicit_start is None
        else float(explicit_start)
    )
    stop_seconds = (
        float(resolved_defaults.slow_episode_trace_stop_seconds)
        if explicit_stop is None
        else float(explicit_stop)
    )
    return bool(trace_requested), float(start_seconds), float(stop_seconds)


def main_cli() -> None:
    args = parse_args()
    try:
        from arknova_rl.trainer import format_log_timestamp, train_self_play
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "")) == "torch":
            raise SystemExit(
                "PyTorch is required for RL training. Install torch first, then rerun."
            ) from exc
        raise

    default_config = PPOTrainConfig()
    (
        slow_episode_trace_enabled,
        slow_episode_trace_start_seconds,
        slow_episode_trace_stop_seconds,
    ) = _resolve_slow_episode_trace_settings(args, defaults=default_config)

    config = PPOTrainConfig(
        seed=args.seed,
        device=args.device,
        rollout_workers=args.rollout_workers,
        total_updates=args.updates,
        episodes_per_update=args.episodes_per_update,
        learning_rate=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_epsilon=args.clip_eps,
        update_epochs=args.epochs,
        hidden_size=args.hidden_size,
        lstm_size=args.lstm_size,
        action_hidden_size=args.action_hidden_size,
        step_reward_scale=args.step_reward_scale,
        terminal_reward_scale=args.terminal_reward_scale,
        endgame_trigger_reward=args.endgame_trigger_reward,
        endgame_speed_bonus=args.endgame_speed_bonus,
        terminal_win_bonus=args.terminal_win_bonus,
        terminal_loss_penalty=args.terminal_loss_penalty,
        checkpoint_interval=args.checkpoint_interval,
        slow_episode_trace_enabled=slow_episode_trace_enabled,
        slow_episode_trace_start_seconds=slow_episode_trace_start_seconds,
        slow_episode_trace_stop_seconds=slow_episode_trace_stop_seconds,
        fixed_eval_interval=args.fixed_eval_interval,
        fixed_eval_episodes=args.fixed_eval_episodes,
        fixed_eval_opponent=args.fixed_eval_opponent,
    )
    output_dir = Path(args.output_dir)
    resume_from = Path(args.resume_from) if str(args.resume_from).strip() else None
    print(
        f"[{format_log_timestamp()}] starting training: "
        f"updates={config.total_updates} "
        f"episodes_per_update={config.episodes_per_update} "
        f"device={config.device} rollout_workers={config.rollout_workers} "
        f"output_dir={output_dir} "
        f"resume_from={resume_from if resume_from is not None else '-'}"
    )
    train_self_play(
        config=config,
        output_dir=output_dir,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    main_cli()

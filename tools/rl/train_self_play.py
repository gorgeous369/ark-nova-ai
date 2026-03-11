"""Train Ark Nova self-play agents with Masked PPO/Recurrent PPO.

Example:
    .venv/bin/python tools/rl/train_self_play.py \
      --algo masked_ppo \
      --updates 100 \
      --episodes-per-update 8 \
      --output-dir runs/ppo_masked
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arknova_rl.config import PPOTrainConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ark Nova self-play RL trainer")
    parser.add_argument(
        "--algo",
        type=str,
        default="masked_ppo",
        choices=["masked_ppo", "recurrent_ppo"],
        help="Training variant",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Training device for model updates: cpu | cuda[:index] (mps requests fall back to cpu)",
    )
    parser.add_argument(
        "--rollout-workers",
        type=int,
        default=1,
        help="Parallel worker processes used to collect rollout episodes",
    )
    parser.add_argument("--updates", type=int, default=200, help="PPO update iterations")
    parser.add_argument(
        "--episodes-per-update",
        type=int,
        default=8,
        help="Episodes collected per PPO update",
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon")
    parser.add_argument("--epochs", type=int, default=3, help="PPO epochs per update")
    parser.add_argument("--hidden-size", type=int, default=256, help="MLP hidden width")
    parser.add_argument("--lstm-size", type=int, default=128, help="LSTM hidden size")
    parser.add_argument(
        "--action-hidden-size",
        type=int,
        default=128,
        help="Action encoder hidden size",
    )
    parser.add_argument(
        "--step-reward-scale",
        type=float,
        default=0.2,
        help="Scale for per-step progress-delta reward",
    )
    parser.add_argument(
        "--terminal-reward-scale",
        type=float,
        default=0.2,
        help="Scale for terminal score-diff reward",
    )
    parser.add_argument(
        "--endgame-trigger-reward",
        type=float,
        default=1.0,
        help="Reward added when the acting player triggers endgame (reaches score threshold)",
    )
    parser.add_argument(
        "--endgame-speed-bonus",
        type=float,
        default=6.0,
        help="Additional bonus scaled by how early endgame is triggered",
    )
    parser.add_argument(
        "--terminal-win-bonus",
        type=float,
        default=2.0,
        help="Extra terminal reward for finishing above all opponents",
    )
    parser.add_argument(
        "--terminal-loss-penalty",
        type=float,
        default=2.0,
        help="Extra terminal penalty for finishing below the top opponent score",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=20,
        help="Checkpoint save frequency (updates)",
    )
    parser.add_argument(
        "--fixed-eval-interval",
        type=int,
        default=20,
        help="Run fixed-checkpoint evaluation every N updates (0 disables)",
    )
    parser.add_argument(
        "--fixed-eval-episodes",
        type=int,
        default=8,
        help="Episodes per fixed evaluation run",
    )
    parser.add_argument(
        "--fixed-eval-opponent",
        type=str,
        default="",
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
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    try:
        from arknova_rl.trainer import train_self_play
    except ModuleNotFoundError as exc:
        if str(getattr(exc, "name", "")) == "torch":
            raise SystemExit(
                "PyTorch is required for RL training. Install torch first, then rerun."
            ) from exc
        raise

    config = PPOTrainConfig(
        algo=args.algo,
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
        fixed_eval_interval=args.fixed_eval_interval,
        fixed_eval_episodes=args.fixed_eval_episodes,
        fixed_eval_opponent=args.fixed_eval_opponent,
    )
    output_dir = Path(args.output_dir)
    resume_from = Path(args.resume_from) if str(args.resume_from).strip() else None
    print(
        "starting training: "
        f"algo={config.algo} updates={config.total_updates} "
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

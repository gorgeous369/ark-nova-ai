"""Evaluate two self-play checkpoints against each other."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arknova_rl.evaluator import evaluate_policy_matchup, load_policy_bundle_from_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate two Ark Nova self-play checkpoints")
    parser.add_argument("--checkpoint-a", type=str, required=True, help="Checkpoint path for player A")
    parser.add_argument("--checkpoint-b", type=str, required=True, help="Checkpoint path for player B")
    parser.add_argument("--episodes", type=int, default=16, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=12345, help="Evaluation seed")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device")
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of using greedy argmax",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON output path",
    )
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    device = torch.device(args.device)
    bundle_a = load_policy_bundle_from_checkpoint(
        Path(args.checkpoint_a),
        device=device,
        name="A",
    )
    bundle_b = load_policy_bundle_from_checkpoint(
        Path(args.checkpoint_b),
        device=device,
        name="B",
    )
    metrics = evaluate_policy_matchup(
        policy_a=bundle_a,
        policy_b=bundle_b,
        episodes=int(args.episodes),
        seed=int(args.seed),
        device=device,
        deterministic=not bool(args.stochastic),
    )
    reason_text = ", ".join(
        f"{key}:{value}"
        for key, value in sorted(metrics.terminal_reason_counts.items())
    ) or "-"
    print(
        f"[eval] episodes={metrics.episodes} "
        f"win_rate_a={metrics.win_rate_a:.3f} "
        f"wins_a={metrics.wins_a} wins_b={metrics.wins_b} draws={metrics.draws} "
        f"avg_score_a={metrics.avg_score_a:.2f} avg_score_b={metrics.avg_score_b:.2f} "
        f"avg_diff={metrics.avg_score_diff_a_minus_b:.2f} "
        f"rounds_avg={metrics.avg_completed_rounds:.2f} "
        f"reasons={reason_text}"
    )
    if str(args.output_json).strip():
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(metrics.to_dict(), ensure_ascii=True, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"saved evaluation: {output_path}")


if __name__ == "__main__":
    main_cli()


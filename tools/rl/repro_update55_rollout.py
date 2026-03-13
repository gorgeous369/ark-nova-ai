from __future__ import annotations

import multiprocessing as mp
import random
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from arknova_rl import trainer
from arknova_rl.runtime import build_model_and_encoders, load_torch_checkpoint, restore_config


def main() -> None:
    checkpoint_path = Path("runs/self_play_masked/checkpoint_0054.pt")
    device = torch.device("cpu")
    checkpoint = load_torch_checkpoint(checkpoint_path, device=device)
    config = restore_config(checkpoint.get("config"))
    config.device = "cpu"
    config.rollout_workers = 8
    config.episodes_per_update = 16
    config.slow_episode_trace_start_seconds = 30.0
    config.slow_episode_trace_stop_seconds = 120.0

    model, obs_encoder, action_encoder = build_model_and_encoders(config=config, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rng = random.Random()
    python_rng_state = checkpoint.get("python_rng_state")
    if python_rng_state is not None:
        rng.setstate(python_rng_state)

    start = time.perf_counter()
    with ProcessPoolExecutor(
        max_workers=int(config.rollout_workers),
        mp_context=mp.get_context("spawn"),
    ) as executor:
        sequences, stats = trainer._collect_rollout(
            model=model,
            obs_encoder=obs_encoder,
            action_encoder=action_encoder,
            config=config,
            rng=rng,
            device=device,
            update_index=55,
            trace_dir=Path("runs/self_play_masked/slow_episode_traces_repro_update55"),
            rollout_executor=executor,
        )
    elapsed = time.perf_counter() - start
    print(
        {
            "elapsed_seconds": round(elapsed, 3),
            "sequence_count": len(sequences),
            "episode_count": int(stats.get("episodes", 0)),
            "episode_avg_seconds": round(float(stats.get("episode_avg_seconds", 0.0)), 3),
            "rounds_avg": round(float(stats.get("rounds_avg", 0.0)), 3),
            "reasons": dict(stats.get("reasons", {})),
        }
    )


if __name__ == "__main__":
    main()

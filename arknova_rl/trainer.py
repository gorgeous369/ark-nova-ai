"""Self-play training loop for Masked PPO / Recurrent PPO / MAPPO."""

from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import main

from .config import PPOTrainConfig
from .encoding import ActionFeatureEncoder, ObservationEncoder
from .evaluator import (
    PolicyBundle,
    evaluate_policy_matchup,
    load_policy_bundle_from_checkpoint,
)
from .model import MaskedActorCritic
from .runtime import (
    build_model_and_encoders as _build_model_and_encoders,
    current_actor_id as _current_actor_id,
    load_torch_checkpoint as _load_torch_checkpoint,
)


@dataclass
class RolloutStep:
    sequence_key: Tuple[int, int]
    actor_id: int
    state_vec: np.ndarray
    global_state_vec: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action_index: int
    old_logprob: float
    old_value: float
    reward: float = 0.0
    advantage: float = 0.0
    return_: float = 0.0


@dataclass
class TrainUpdateMetrics:
    step_count: int
    episode_count: int
    avg_completed_rounds: float
    avg_terminal_score_diff_abs: float
    episode_time_avg_sec: float
    episode_time_min_sec: float
    episode_time_max_sec: float
    update_time_sec: float
    model_update_time_sec: float
    terminal_reason_counts: Dict[str, int]
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


@dataclass
class EpisodeRolloutResult:
    episode_index: int
    rollout_steps: List[RolloutStep]
    completed_rounds: int
    terminal_abs_diffs: List[float]
    terminal_reason: str
    elapsed_seconds: float

def _progress_score_diff(state: main.GameState, player_id: int) -> float:
    actor_score = float(main._progress_score(state.players[player_id]))
    other_scores = [
        float(main._progress_score(player))
        for idx, player in enumerate(state.players)
        if idx != int(player_id)
    ]
    if not other_scores:
        return actor_score
    return actor_score - float(np.mean(other_scores))


def _terminal_score_diff(state: main.GameState, player_id: int) -> float:
    actor_score = float(main._final_score_points(state, state.players[player_id]))
    other_scores = [
        float(main._final_score_points(state, player))
        for idx, player in enumerate(state.players)
        if idx != int(player_id)
    ]
    if not other_scores:
        return actor_score
    return actor_score - float(np.mean(other_scores))


def _endgame_speed_bonus(*, state: main.GameState, config: PPOTrainConfig) -> float:
    round_limit = int(getattr(state, "max_rounds", 0) or 0)
    if round_limit <= 0:
        return 0.0
    completed_rounds = int(main._completed_rounds(state))
    speed_ratio = max(0.0, 1.0 - (float(completed_rounds) / float(round_limit)))
    return float(config.endgame_speed_bonus) * speed_ratio


def _terminal_outcome(state: main.GameState, player_id: int) -> int:
    actor_score = int(main._final_score_points(state, state.players[player_id]))
    other_scores = [
        int(main._final_score_points(state, player))
        for idx, player in enumerate(state.players)
        if idx != int(player_id)
    ]
    if not other_scores:
        return 0
    best_other = max(other_scores)
    if actor_score > best_other:
        return 1
    if actor_score < best_other:
        return -1
    return 0


def _to_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _collect_episode_rollout(
    *,
    model: MaskedActorCritic,
    obs_encoder: ObservationEncoder,
    action_encoder: ActionFeatureEncoder,
    config: PPOTrainConfig,
    device: torch.device,
    episode_index: int,
    episode_seed: int,
) -> EpisodeRolloutResult:
    episode_started_at = time.perf_counter()
    state = main.setup_game(seed=episode_seed, player_names=["P1", "P2"])
    hidden_by_actor: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
    if model.use_lstm:
        for player_id in range(len(state.players)):
            hidden_by_actor[player_id] = model.init_hidden(1, device=device)

    rollout_steps: List[RolloutStep] = []
    sequence_indices: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    terminal_abs_diffs: List[float] = []

    while str(state.pending_decision_kind or "").strip() or not state.game_over():
        actor_id = _current_actor_id(state)
        actor = state.players[actor_id]
        legal = main.legal_actions(actor, state=state, player_id=actor_id)
        if not legal:
            raise RuntimeError(
                f"No legal actions for actor={actor_id} pending={state.pending_decision_kind!r}."
            )

        state_vec, global_vec = obs_encoder.encode_from_state(
            state,
            actor_id,
            include_global=bool(model.use_centralized_value),
        )
        action_features = action_encoder.encode_many(legal)
        action_mask = np.ones((len(legal),), dtype=np.float32)

        state_t = _to_tensor(state_vec, device=device).unsqueeze(0)
        global_t = _to_tensor(global_vec, device=device).unsqueeze(0)
        action_t = _to_tensor(action_features, device=device).unsqueeze(0)
        mask_t = _to_tensor(action_mask, device=device).unsqueeze(0)

        hidden = hidden_by_actor.get(actor_id) if model.use_lstm else None
        with torch.no_grad():
            logits, value, next_hidden = model.forward_step(
                state_vec=state_t,
                action_features=action_t,
                action_mask=mask_t,
                hidden=hidden,
                global_state_vec=global_t if model.use_centralized_value else None,
            )
            dist = Categorical(logits=logits[0])
            sampled_index = int(dist.sample().item())
            logprob = float(dist.log_prob(torch.tensor(sampled_index, device=device)).item())
            predicted_value = float(value.item())
        if model.use_lstm:
            hidden_by_actor[actor_id] = (
                next_hidden[0].detach(),
                next_hidden[1].detach(),
            ) if next_hidden is not None else None

        before_diff = _progress_score_diff(state, actor_id)
        endgame_was_triggered = state.endgame_trigger_player is not None
        chosen_action = legal[sampled_index]
        main.apply_action(state, chosen_action)
        after_diff = _progress_score_diff(state, actor_id)
        reward = (after_diff - before_diff) * float(config.step_reward_scale)
        if (
            not endgame_was_triggered
            and state.endgame_trigger_player is not None
            and int(state.endgame_trigger_player) == int(actor_id)
        ):
            reward += float(config.endgame_trigger_reward)
            reward += _endgame_speed_bonus(state=state, config=config)

        step = RolloutStep(
            sequence_key=(episode_index, actor_id),
            actor_id=actor_id,
            state_vec=state_vec,
            global_state_vec=global_vec,
            action_features=action_features,
            action_mask=action_mask,
            action_index=sampled_index,
            old_logprob=logprob,
            old_value=predicted_value,
            reward=float(reward),
        )
        step_index = len(rollout_steps)
        rollout_steps.append(step)
        sequence_indices[step.sequence_key].append(step_index)

    for player_id in range(len(state.players)):
        seq_key = (episode_index, player_id)
        if seq_key not in sequence_indices or not sequence_indices[seq_key]:
            continue
        terminal_diff = _terminal_score_diff(state, player_id)
        terminal_abs_diffs.append(abs(float(terminal_diff)))
        last_step_idx = sequence_indices[seq_key][-1]
        terminal_reward = float(config.terminal_reward_scale) * float(terminal_diff)
        outcome = _terminal_outcome(state, player_id)
        if outcome > 0:
            terminal_reward += float(config.terminal_win_bonus)
        elif outcome < 0:
            terminal_reward -= float(config.terminal_loss_penalty)
        rollout_steps[last_step_idx].reward += terminal_reward

    terminal_reason = str(state.forced_game_over_reason or "")
    if not terminal_reason and state.endgame_trigger_player is not None:
        terminal_reason = "score_threshold_endgame"
    if not terminal_reason:
        terminal_reason = "unknown_terminal_reason"

    return EpisodeRolloutResult(
        episode_index=int(episode_index),
        rollout_steps=rollout_steps,
        completed_rounds=int(main._completed_rounds(state)),
        terminal_abs_diffs=terminal_abs_diffs,
        terminal_reason=terminal_reason,
        elapsed_seconds=float(time.perf_counter() - episode_started_at),
    )


def _build_sequence_indices(
    rollout_steps: Sequence[RolloutStep],
) -> DefaultDict[Tuple[int, int], List[int]]:
    sequence_indices: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    for step_index, step in enumerate(rollout_steps):
        sequence_indices[step.sequence_key].append(step_index)
    return sequence_indices


def _finalize_rollout_advantages(
    *,
    rollout_steps: Sequence[RolloutStep],
    sequence_indices: Dict[Tuple[int, int], List[int]],
    config: PPOTrainConfig,
) -> None:
    for indices in sequence_indices.values():
        gae = 0.0
        for local_pos in range(len(indices) - 1, -1, -1):
            step_idx = indices[local_pos]
            step = rollout_steps[step_idx]
            if local_pos == len(indices) - 1:
                next_value = 0.0
                nonterminal = 0.0
            else:
                next_step = rollout_steps[indices[local_pos + 1]]
                next_value = float(next_step.old_value)
                nonterminal = 1.0
            delta = float(step.reward) + float(config.gamma) * next_value * nonterminal - float(step.old_value)
            gae = delta + float(config.gamma) * float(config.gae_lambda) * nonterminal * gae
            step.advantage = float(gae)
            step.return_ = float(step.old_value + step.advantage)


def _state_dict_to_cpu(model_state_dict: Dict[str, Any]) -> Dict[str, Any]:
    cpu_state: Dict[str, Any] = {}
    for key, value in model_state_dict.items():
        if isinstance(value, torch.Tensor):
            cpu_state[str(key)] = value.detach().to(device="cpu")
        else:
            cpu_state[str(key)] = value
    return cpu_state


def _chunk_episode_specs(
    episode_specs: Sequence[Tuple[int, int]],
    chunk_count: int,
) -> List[List[Tuple[int, int]]]:
    chunks: List[List[Tuple[int, int]]] = [[] for _ in range(max(1, int(chunk_count)))]
    for index, spec in enumerate(episode_specs):
        chunks[index % len(chunks)].append(spec)
    return [chunk for chunk in chunks if chunk]


def _collect_rollout_chunk_worker(payload: Dict[str, Any]) -> List[EpisodeRolloutResult]:
    torch.set_num_threads(1)
    task_seed = int(payload.get("task_seed", 0))
    random.seed(task_seed)
    np.random.seed(task_seed)
    torch.manual_seed(task_seed)

    config = PPOTrainConfig(**dict(payload.get("config") or {}))
    config.resolve_algo_flags()
    config.device = "cpu"
    device = torch.device("cpu")
    model, obs_encoder, action_encoder = _build_model_and_encoders(config=config, device=device)
    model.load_state_dict(dict(payload.get("model_state_dict") or {}))
    model.eval()

    episode_results: List[EpisodeRolloutResult] = []
    for episode_index, episode_seed in list(payload.get("episode_specs") or []):
        episode_results.append(
            _collect_episode_rollout(
                model=model,
                obs_encoder=obs_encoder,
                action_encoder=action_encoder,
                config=config,
                device=device,
                episode_index=int(episode_index),
                episode_seed=int(episode_seed),
            )
        )
    return episode_results


def _collect_rollout(
    *,
    model: MaskedActorCritic,
    obs_encoder: ObservationEncoder,
    action_encoder: ActionFeatureEncoder,
    config: PPOTrainConfig,
    rng: random.Random,
    device: torch.device,
    update_index: int,
    rollout_executor: Optional[Executor] = None,
) -> Tuple[List[RolloutStep], Dict[str, Any]]:
    del update_index
    rollout_steps: List[RolloutStep] = []
    episode_rounds: List[int] = []
    episode_elapsed_seconds: List[float] = []
    terminal_abs_diffs: List[float] = []
    terminal_reason_counter: Counter[str] = Counter()
    episode_specs = [
        (episode_idx, rng.randint(0, 10_000_000))
        for episode_idx in range(int(config.episodes_per_update))
    ]
    worker_count = max(1, int(getattr(config, "rollout_workers", 1) or 1))

    episode_results: List[EpisodeRolloutResult] = []
    if rollout_executor is None or worker_count <= 1 or len(episode_specs) <= 1:
        for episode_idx, episode_seed in episode_specs:
            episode_results.append(
                _collect_episode_rollout(
                    model=model,
                    obs_encoder=obs_encoder,
                    action_encoder=action_encoder,
                    config=config,
                    device=device,
                    episode_index=int(episode_idx),
                    episode_seed=int(episode_seed),
                )
            )
    else:
        cpu_model_state = _state_dict_to_cpu(model.state_dict())
        chunked_specs = _chunk_episode_specs(episode_specs, min(worker_count, len(episode_specs)))
        futures = [
            rollout_executor.submit(
                _collect_rollout_chunk_worker,
                {
                    "config": asdict(config),
                    "model_state_dict": cpu_model_state,
                    "episode_specs": chunk,
                    "task_seed": rng.randint(0, 10_000_000),
                },
            )
            for chunk in chunked_specs
        ]
        for future in futures:
            episode_results.extend(list(future.result()))

    episode_results.sort(key=lambda item: int(item.episode_index))
    for episode_result in episode_results:
        rollout_steps.extend(episode_result.rollout_steps)
        episode_rounds.append(int(episode_result.completed_rounds))
        episode_elapsed_seconds.append(float(episode_result.elapsed_seconds))
        terminal_abs_diffs.extend(float(value) for value in episode_result.terminal_abs_diffs)
        terminal_reason_counter[str(episode_result.terminal_reason)] += 1

    sequence_indices = _build_sequence_indices(rollout_steps)
    _finalize_rollout_advantages(
        rollout_steps=rollout_steps,
        sequence_indices=sequence_indices,
        config=config,
    )

    rollout_stats = {
        "episode_count": len(episode_results),
        "step_count": len(rollout_steps),
        "avg_completed_rounds": float(np.mean(episode_rounds)) if episode_rounds else 0.0,
        "avg_terminal_score_diff_abs": (
            float(np.mean(terminal_abs_diffs))
            if terminal_abs_diffs
            else 0.0
        ),
        "episode_elapsed_seconds": list(episode_elapsed_seconds),
        "avg_episode_time_sec": (
            float(np.mean(episode_elapsed_seconds)) if episode_elapsed_seconds else 0.0
        ),
        "min_episode_time_sec": (
            float(np.min(episode_elapsed_seconds)) if episode_elapsed_seconds else 0.0
        ),
        "max_episode_time_sec": (
            float(np.max(episode_elapsed_seconds)) if episode_elapsed_seconds else 0.0
        ),
        "terminal_reason_counts": dict(terminal_reason_counter),
        "sequence_indices": sequence_indices,
    }
    return rollout_steps, rollout_stats


def _zero_loss_metrics() -> Dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }


def _build_feedforward_training_batch(
    rollout_steps: Sequence[RolloutStep],
    normalized_advantages: np.ndarray,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    step_count = len(rollout_steps)
    max_action_count = max(int(step.action_features.shape[0]) for step in rollout_steps)
    action_dim = int(rollout_steps[0].action_features.shape[1])

    state_array = np.stack([step.state_vec for step in rollout_steps], axis=0).astype(np.float32)
    global_array = np.stack([step.global_state_vec for step in rollout_steps], axis=0).astype(np.float32)
    action_array = np.zeros((step_count, max_action_count, action_dim), dtype=np.float32)
    mask_array = np.zeros((step_count, max_action_count), dtype=np.float32)
    action_indices = np.zeros((step_count,), dtype=np.int64)
    old_logprobs = np.zeros((step_count,), dtype=np.float32)
    returns = np.zeros((step_count,), dtype=np.float32)

    for idx, step in enumerate(rollout_steps):
        action_count = int(step.action_features.shape[0])
        action_array[idx, :action_count] = step.action_features
        mask_array[idx, :action_count] = step.action_mask
        action_indices[idx] = int(step.action_index)
        old_logprobs[idx] = float(step.old_logprob)
        returns[idx] = float(step.return_)

    return {
        "state_vec": _to_tensor(state_array, device=device),
        "global_state_vec": _to_tensor(global_array, device=device),
        "action_features": _to_tensor(action_array, device=device),
        "action_mask": _to_tensor(mask_array, device=device),
        "action_index": torch.as_tensor(action_indices, dtype=torch.long, device=device),
        "old_logprob": _to_tensor(old_logprobs, device=device),
        "target_return": _to_tensor(returns, device=device),
        "advantage": _to_tensor(normalized_advantages, device=device),
    }


def _build_recurrent_training_sequences(
    rollout_steps: Sequence[RolloutStep],
    ordered_sequences: Sequence[Tuple[Tuple[int, int], List[int]]],
    normalized_advantages: np.ndarray,
    *,
    device: torch.device,
) -> List[Dict[str, torch.Tensor]]:
    sequence_batches: List[Dict[str, torch.Tensor]] = []
    for _sequence_key, indices in ordered_sequences:
        if not indices:
            continue
        max_action_count = max(int(rollout_steps[step_idx].action_features.shape[0]) for step_idx in indices)
        action_dim = int(rollout_steps[indices[0]].action_features.shape[1])
        step_count = len(indices)

        state_array = np.stack([rollout_steps[step_idx].state_vec for step_idx in indices], axis=0).astype(np.float32)
        global_array = np.stack(
            [rollout_steps[step_idx].global_state_vec for step_idx in indices],
            axis=0,
        ).astype(np.float32)
        action_array = np.zeros((step_count, max_action_count, action_dim), dtype=np.float32)
        mask_array = np.zeros((step_count, max_action_count), dtype=np.float32)
        action_indices = np.zeros((step_count,), dtype=np.int64)
        old_logprobs = np.zeros((step_count,), dtype=np.float32)
        returns = np.zeros((step_count,), dtype=np.float32)
        advantages = np.zeros((step_count,), dtype=np.float32)

        for seq_idx, step_idx in enumerate(indices):
            step = rollout_steps[step_idx]
            action_count = int(step.action_features.shape[0])
            action_array[seq_idx, :action_count] = step.action_features
            mask_array[seq_idx, :action_count] = step.action_mask
            action_indices[seq_idx] = int(step.action_index)
            old_logprobs[seq_idx] = float(step.old_logprob)
            returns[seq_idx] = float(step.return_)
            advantages[seq_idx] = float(normalized_advantages[step_idx])

        sequence_batches.append(
            {
                "state_vec": _to_tensor(state_array, device=device),
                "global_state_vec": _to_tensor(global_array, device=device),
                "action_features": _to_tensor(action_array, device=device),
                "action_mask": _to_tensor(mask_array, device=device),
                "action_index": torch.as_tensor(action_indices, dtype=torch.long, device=device),
                "old_logprob": _to_tensor(old_logprobs, device=device),
                "target_return": _to_tensor(returns, device=device),
                "advantage": _to_tensor(advantages, device=device),
            }
        )
    return sequence_batches


def _update_model(
    *,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_steps: Sequence[RolloutStep],
    sequence_indices: Dict[Tuple[int, int], List[int]],
    config: PPOTrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    if not rollout_steps:
        return _zero_loss_metrics()

    advantages = np.asarray([float(step.advantage) for step in rollout_steps], dtype=np.float32)
    adv_mean = float(np.mean(advantages))
    adv_std = float(np.std(advantages))
    if adv_std < 1e-8:
        adv_std = 1.0
    normalized_advantages = (advantages - adv_mean) / adv_std

    epoch_policy_losses: List[float] = []
    epoch_value_losses: List[float] = []
    epoch_entropies: List[float] = []
    epoch_total_losses: List[float] = []

    ordered_sequences = sorted(sequence_indices.items(), key=lambda item: item[0])
    feedforward_batch: Optional[Dict[str, torch.Tensor]] = None
    recurrent_batches: Optional[List[Dict[str, torch.Tensor]]] = None
    if not model.use_lstm:
        feedforward_batch = _build_feedforward_training_batch(
            rollout_steps,
            normalized_advantages,
            device=device,
        )
    else:
        recurrent_batches = _build_recurrent_training_sequences(
            rollout_steps,
            ordered_sequences,
            normalized_advantages,
            device=device,
        )
    for _ in range(config.update_epochs):
        if feedforward_batch is not None:
            logits, values, _ = model.forward_step(
                state_vec=feedforward_batch["state_vec"],
                action_features=feedforward_batch["action_features"],
                action_mask=feedforward_batch["action_mask"],
                hidden=None,
                global_state_vec=(
                    feedforward_batch["global_state_vec"]
                    if model.use_centralized_value
                    else None
                ),
            )
            step_log_probs = torch.log_softmax(logits, dim=-1)
            step_probs = torch.softmax(logits, dim=-1)
            chosen_indices = feedforward_batch["action_index"].unsqueeze(1)
            new_logprob = step_log_probs.gather(1, chosen_indices).squeeze(1)
            entropy = -(step_probs * step_log_probs).sum(dim=-1)
            old_logprob = feedforward_batch["old_logprob"]
            target_return = feedforward_batch["target_return"]
            advantage = feedforward_batch["advantage"]
            ratio = torch.exp(new_logprob - old_logprob)
            clipped_ratio = torch.clamp(
                ratio,
                1.0 - float(config.clip_epsilon),
                1.0 + float(config.clip_epsilon),
            )
            surrogate_1 = ratio * advantage
            surrogate_2 = clipped_ratio * advantage
            policy_loss = -torch.min(surrogate_1, surrogate_2).mean()
            value_loss = ((target_return - values) ** 2).mean()
            entropy_mean = entropy.mean()
        else:
            policy_losses_t: List[torch.Tensor] = []
            value_losses_t: List[torch.Tensor] = []
            entropies_t: List[torch.Tensor] = []

            for sequence_batch in recurrent_batches or []:
                hidden = model.init_hidden(1, device=device)
                step_count = int(sequence_batch["state_vec"].shape[0])
                for step_idx in range(step_count):
                    state_t = sequence_batch["state_vec"][step_idx : step_idx + 1]
                    global_t = sequence_batch["global_state_vec"][step_idx : step_idx + 1]
                    action_t = sequence_batch["action_features"][step_idx : step_idx + 1]
                    mask_t = sequence_batch["action_mask"][step_idx : step_idx + 1]
                    logits, values, hidden = model.forward_step(
                        state_vec=state_t,
                        action_features=action_t,
                        action_mask=mask_t,
                        hidden=hidden,
                        global_state_vec=global_t if model.use_centralized_value else None,
                    )
                    step_log_probs = torch.log_softmax(logits[0], dim=-1)
                    step_probs = torch.softmax(logits[0], dim=-1)
                    action_index_t = sequence_batch["action_index"][step_idx]
                    new_logprob = step_log_probs[action_index_t]
                    entropy = -(step_probs * step_log_probs).sum()
                    new_value = values.squeeze(0)

                    old_logprob = sequence_batch["old_logprob"][step_idx]
                    target_return = sequence_batch["target_return"][step_idx]
                    advantage = sequence_batch["advantage"][step_idx]

                    ratio = torch.exp(new_logprob - old_logprob)
                    clipped_ratio = torch.clamp(
                        ratio,
                        1.0 - float(config.clip_epsilon),
                        1.0 + float(config.clip_epsilon),
                    )
                    surrogate_1 = ratio * advantage
                    surrogate_2 = clipped_ratio * advantage
                    policy_losses_t.append(-torch.min(surrogate_1, surrogate_2))
                    value_losses_t.append((target_return - new_value) ** 2)
                    entropies_t.append(entropy)

            if not policy_losses_t:
                break

            policy_loss = torch.stack(policy_losses_t).mean()
            value_loss = torch.stack(value_losses_t).mean()
            entropy_mean = torch.stack(entropies_t).mean()

        if policy_loss.numel() == 0:
            break
        total_loss = (
            policy_loss
            + float(config.value_coef) * value_loss
            - float(config.entropy_coef) * entropy_mean
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()

        epoch_policy_losses.append(float(policy_loss.detach().cpu().item()))
        epoch_value_losses.append(float(value_loss.detach().cpu().item()))
        epoch_entropies.append(float(entropy_mean.detach().cpu().item()))
        epoch_total_losses.append(float(total_loss.detach().cpu().item()))

    if not epoch_policy_losses:
        return _zero_loss_metrics()
    return {
        "policy_loss": float(np.mean(epoch_policy_losses)),
        "value_loss": float(np.mean(epoch_value_losses)),
        "entropy": float(np.mean(epoch_entropies)),
        "total_loss": float(np.mean(epoch_total_losses)),
    }

def _coerce_rng_state_tensor(raw_state: Any) -> Optional[torch.Tensor]:
    if not isinstance(raw_state, torch.Tensor):
        return None
    return raw_state.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _mps_rng_api_available() -> bool:
    return bool(
        hasattr(torch, "mps")
        and hasattr(torch, "backends")
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
        and hasattr(torch.mps, "get_rng_state")
        and hasattr(torch.mps, "set_rng_state")
    )


def _parallel_rollout_unavailable_reason(
    *,
    device: torch.device,
    requested_workers: int,
) -> str:
    if int(requested_workers) <= 1:
        return ""
    if device.type != "cpu":
        return f"parallel rollout requires cpu inference workers, got {device.type}"
    return ""


def _resolve_training_device(requested_device: str) -> torch.device:
    requested = str(requested_device or "cpu").strip().lower()
    if requested in {"", "auto", "cpu", "mps"}:
        return torch.device("cpu")
    return torch.device(requested)


def _inference_device_for_training_device(device: torch.device) -> torch.device:
    del device
    return torch.device("cpu")


def _build_policy_bundle_from_model_snapshot(
    *,
    name: str,
    source_model: MaskedActorCritic,
    config: PPOTrainConfig,
    device: torch.device,
) -> PolicyBundle:
    snapshot_model, snapshot_obs_encoder, snapshot_action_encoder = _build_model_and_encoders(
        config=config,
        device=device,
    )
    if device.type == "cpu":
        snapshot_state_dict = _state_dict_to_cpu(source_model.state_dict())
    else:
        snapshot_state_dict = source_model.state_dict()
    snapshot_model.load_state_dict(snapshot_state_dict)
    snapshot_model.eval()
    return PolicyBundle(
        name=name,
        model=snapshot_model,
        obs_encoder=snapshot_obs_encoder,
        action_encoder=snapshot_action_encoder,
    )


def _restore_torch_rng_states(
    checkpoint: Dict[str, Any],
    *,
    device: torch.device,
) -> None:
    cpu_rng_state = _coerce_rng_state_tensor(checkpoint.get("torch_rng_state"))
    if cpu_rng_state is not None:
        try:
            torch.set_rng_state(cpu_rng_state)
        except Exception:
            pass

    if device.type == "cuda" and torch.cuda.is_available():
        raw_cuda_states = checkpoint.get("torch_cuda_rng_state_all")
        if isinstance(raw_cuda_states, (list, tuple)):
            cuda_states = [
                state
                for state in (_coerce_rng_state_tensor(item) for item in raw_cuda_states)
                if state is not None
            ]
            if cuda_states:
                try:
                    torch.cuda.set_rng_state_all(cuda_states)
                except Exception:
                    pass

    if device.type == "mps" and _mps_rng_api_available():
        mps_rng_state = _coerce_rng_state_tensor(checkpoint.get("torch_mps_rng_state"))
        if mps_rng_state is not None:
            try:
                torch.mps.set_rng_state(mps_rng_state)
            except Exception:
                pass


def _save_checkpoint(
    *,
    path: Path,
    update_idx: int,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    config: PPOTrainConfig,
    metrics: Sequence[TrainUpdateMetrics],
    rng: random.Random,
) -> None:
    payload: Dict[str, Any] = {
        "update": int(update_idx),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(config),
        "metrics": [asdict(item) for item in metrics],
        "python_rng_state": rng.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": _coerce_rng_state_tensor(torch.get_rng_state()),
    }
    if torch.cuda.is_available():
        payload["torch_cuda_rng_state_all"] = [
            _coerce_rng_state_tensor(state)
            for state in torch.cuda.get_rng_state_all()
        ]
    if _mps_rng_api_available():
        payload["torch_mps_rng_state"] = _coerce_rng_state_tensor(torch.mps.get_rng_state())
    torch.save(
        payload,
        path,
    )


def _resolve_fixed_eval_opponent_path(
    *,
    output_path: Path,
    config: PPOTrainConfig,
    loaded_update: int,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    metrics: Sequence[TrainUpdateMetrics],
    rng: random.Random,
) -> Optional[Path]:
    explicit = str(getattr(config, "fixed_eval_opponent", "") or "").strip()
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.exists() else None

    baseline_path = output_path / "checkpoint_0000.pt"
    if baseline_path.exists():
        return baseline_path

    existing = sorted(output_path.glob("checkpoint_*.pt"))
    if existing:
        return existing[0]

    if loaded_update == 0:
        _save_checkpoint(
            path=baseline_path,
            update_idx=0,
            model=model,
            optimizer=optimizer,
            config=config,
            metrics=metrics,
            rng=rng,
        )
        print(f"saved fixed-eval baseline: {baseline_path}")
        return baseline_path

    return None


def _restore_metrics(raw_metrics: Any) -> List[TrainUpdateMetrics]:
    restored: List[TrainUpdateMetrics] = []
    if not isinstance(raw_metrics, list):
        return restored
    for item in raw_metrics:
        if not isinstance(item, dict):
            continue
        reason_counts_raw = item.get("terminal_reason_counts")
        reason_counts: Dict[str, int] = {}
        if isinstance(reason_counts_raw, dict):
            reason_counts = {
                str(key): int(value)
                for key, value in reason_counts_raw.items()
            }
        restored.append(
            TrainUpdateMetrics(
                step_count=int(item.get("step_count", 0)),
                episode_count=int(item.get("episode_count", 0)),
                avg_completed_rounds=float(item.get("avg_completed_rounds", 0.0)),
                avg_terminal_score_diff_abs=float(item.get("avg_terminal_score_diff_abs", 0.0)),
                episode_time_avg_sec=float(item.get("episode_time_avg_sec", 0.0)),
                episode_time_min_sec=float(item.get("episode_time_min_sec", 0.0)),
                episode_time_max_sec=float(item.get("episode_time_max_sec", 0.0)),
                update_time_sec=float(item.get("update_time_sec", 0.0)),
                model_update_time_sec=float(item.get("model_update_time_sec", 0.0)),
                terminal_reason_counts=reason_counts,
                policy_loss=float(item.get("policy_loss", 0.0)),
                value_loss=float(item.get("value_loss", 0.0)),
                entropy=float(item.get("entropy", 0.0)),
                total_loss=float(item.get("total_loss", 0.0)),
            )
        )
    return restored


def train_self_play(
    *,
    config: PPOTrainConfig,
    output_dir: Path,
    resume_from: Optional[Path] = None,
) -> List[TrainUpdateMetrics]:
    config.resolve_algo_flags()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    requested_device = str(config.device or "auto")
    device = _resolve_training_device(requested_device)
    inference_device = _inference_device_for_training_device(device)
    model, obs_encoder, action_encoder = _build_model_and_encoders(config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))
    requested_device_lower = requested_device.strip().lower()
    if requested_device_lower in {"", "auto", "mps"}:
        print(
            "device policy: "
            f"requested={requested_device_lower or 'cpu'}; using cpu for training and inference."
        )
    print(
        "resolved devices: "
        f"requested={requested_device} training={device.type} inference={inference_device.type}"
    )
    if inference_device != device:
        print(
            "inference device fallback: "
            f"training stays on {device.type}, rollout/fixed-eval inference uses {inference_device.type}."
        )

    loaded_update = 0
    all_metrics: List[TrainUpdateMetrics] = []
    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
        checkpoint = _load_torch_checkpoint(checkpoint_path, device=device)
        model_state = checkpoint.get("model_state_dict")
        if not isinstance(model_state, dict):
            raise ValueError("Checkpoint is missing model_state_dict.")
        try:
            model.load_state_dict(model_state)
        except RuntimeError as exc:
            raise ValueError(
                "Checkpoint model_state_dict is incompatible with current model/observation dimensions."
            ) from exc
        optimizer_state = checkpoint.get("optimizer_state_dict")
        if isinstance(optimizer_state, dict):
            optimizer.load_state_dict(optimizer_state)
        loaded_update = int(checkpoint.get("update", 0) or 0)
        if loaded_update < 0:
            loaded_update = 0
        all_metrics = _restore_metrics(checkpoint.get("metrics"))
        python_rng_state = checkpoint.get("python_rng_state")
        if python_rng_state is not None:
            try:
                rng.setstate(python_rng_state)
            except Exception:
                pass
        numpy_rng_state = checkpoint.get("numpy_rng_state")
        if isinstance(numpy_rng_state, tuple) and len(numpy_rng_state) == 5:
            try:
                np.random.set_state(numpy_rng_state)
            except Exception:
                pass
        _restore_torch_rng_states(checkpoint, device=device)
        print(f"resuming from checkpoint: {checkpoint_path} (loaded_update={loaded_update})")

    fixed_eval_interval = max(0, int(getattr(config, "fixed_eval_interval", 0) or 0))
    fixed_eval_episodes = max(0, int(getattr(config, "fixed_eval_episodes", 0) or 0))
    fixed_eval_opponent_path: Optional[Path] = None
    fixed_eval_bundle: Optional[PolicyBundle] = None
    fixed_eval_log_path = output_path / "fixed_eval.jsonl"
    timing_log_path = output_path / "update_timings.jsonl"
    if fixed_eval_interval > 0 and fixed_eval_episodes > 0:
        fixed_eval_opponent_path = _resolve_fixed_eval_opponent_path(
            output_path=output_path,
            config=config,
            loaded_update=loaded_update,
            model=model,
            optimizer=optimizer,
            metrics=all_metrics,
            rng=rng,
        )
        if fixed_eval_opponent_path is None:
            print("fixed evaluation disabled: no fixed opponent checkpoint available.")
        else:
            fixed_eval_bundle = load_policy_bundle_from_checkpoint(
                fixed_eval_opponent_path,
                device=inference_device,
                name=f"fixed:{fixed_eval_opponent_path.stem}",
            )
            print(f"fixed evaluation opponent: {fixed_eval_opponent_path}")

    start_update = loaded_update + 1
    end_update = loaded_update + int(config.total_updates)
    if start_update > end_update:
        return all_metrics

    rollout_executor: Optional[Executor] = None
    requested_rollout_workers = int(getattr(config, "rollout_workers", 1) or 1)
    parallel_rollout_reason = _parallel_rollout_unavailable_reason(
        device=inference_device,
        requested_workers=requested_rollout_workers,
    )
    if parallel_rollout_reason:
        print(
            "parallel rollout fallback: "
            f"{parallel_rollout_reason}; using single-process rollout."
        )
    elif requested_rollout_workers > 1:
        worker_count = requested_rollout_workers
        try:
            rollout_executor = ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=mp.get_context("spawn"),
            )
        except (OSError, PermissionError) as exc:
            print(
                "parallel rollout fallback: "
                f"ProcessPool unavailable ({exc.__class__.__name__}: {exc}); using single-process rollout."
            )
            rollout_executor = None
    try:
        for update_idx in range(start_update, end_update + 1):
            rollout_model = model
            rollout_obs_encoder = obs_encoder
            rollout_action_encoder = action_encoder
            rollout_device = device
            if inference_device != device:
                rollout_bundle = _build_policy_bundle_from_model_snapshot(
                    name=f"rollout:{update_idx:04d}",
                    source_model=model,
                    config=config,
                    device=inference_device,
                )
                rollout_model = rollout_bundle.model
                rollout_obs_encoder = rollout_bundle.obs_encoder
                rollout_action_encoder = rollout_bundle.action_encoder
                rollout_device = inference_device
            rollout_started_at = time.perf_counter()
            rollout_steps, rollout_stats = _collect_rollout(
                model=rollout_model,
                obs_encoder=rollout_obs_encoder,
                action_encoder=rollout_action_encoder,
                config=config,
                rng=rng,
                device=rollout_device,
                update_index=update_idx,
                rollout_executor=rollout_executor,
            )
            update_time_sec = float(time.perf_counter() - rollout_started_at)
            model_update_started_at = time.perf_counter()
            update_stats = _update_model(
                model=model,
                optimizer=optimizer,
                rollout_steps=rollout_steps,
                sequence_indices=rollout_stats["sequence_indices"],
                config=config,
                device=device,
            )
            model_update_time_sec = float(time.perf_counter() - model_update_started_at)
            metrics = TrainUpdateMetrics(
                step_count=int(rollout_stats["step_count"]),
                episode_count=int(rollout_stats["episode_count"]),
                avg_completed_rounds=float(rollout_stats["avg_completed_rounds"]),
                avg_terminal_score_diff_abs=float(rollout_stats["avg_terminal_score_diff_abs"]),
                episode_time_avg_sec=float(rollout_stats["avg_episode_time_sec"]),
                episode_time_min_sec=float(rollout_stats["min_episode_time_sec"]),
                episode_time_max_sec=float(rollout_stats["max_episode_time_sec"]),
                update_time_sec=update_time_sec,
                model_update_time_sec=model_update_time_sec,
                terminal_reason_counts=dict(rollout_stats["terminal_reason_counts"]),
                policy_loss=float(update_stats["policy_loss"]),
                value_loss=float(update_stats["value_loss"]),
                entropy=float(update_stats["entropy"]),
                total_loss=float(update_stats["total_loss"]),
            )
            all_metrics.append(metrics)
            with timing_log_path.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "update": int(update_idx),
                            "episode_time_sec": [
                                float(value)
                                for value in list(rollout_stats["episode_elapsed_seconds"])
                            ],
                            "episode_time_avg_sec": float(metrics.episode_time_avg_sec),
                            "episode_time_min_sec": float(metrics.episode_time_min_sec),
                            "episode_time_max_sec": float(metrics.episode_time_max_sec),
                            "update_time_sec": float(metrics.update_time_sec),
                            "model_update_time_sec": float(metrics.model_update_time_sec),
                        },
                        ensure_ascii=True,
                        sort_keys=True,
                    )
                    + "\n"
                )

            if update_idx % int(config.log_interval) == 0:
                reason_text = ", ".join(
                    f"{key}:{value}"
                    for key, value in sorted(metrics.terminal_reason_counts.items())
                ) or "-"
                print(
                    f"[update {update_idx:04d}] "
                    f"steps={metrics.step_count} episodes={metrics.episode_count} "
                    f"rounds_avg={metrics.avg_completed_rounds:.2f} "
                    f"term_diff_abs={metrics.avg_terminal_score_diff_abs:.2f} "
                    f"episode_avg={metrics.episode_time_avg_sec:.2f}s "
                    f"episode_min={metrics.episode_time_min_sec:.2f}s "
                    f"episode_max={metrics.episode_time_max_sec:.2f}s "
                    f"update_time={metrics.update_time_sec:.2f}s "
                    f"model_update_time={metrics.model_update_time_sec:.2f}s "
                    f"policy={metrics.policy_loss:.4f} value={metrics.value_loss:.4f} "
                    f"entropy={metrics.entropy:.4f} total={metrics.total_loss:.4f} "
                    f"reasons={reason_text}"
                )

            should_run_fixed_eval = (
                fixed_eval_bundle is not None
                and fixed_eval_interval > 0
                and (
                    update_idx % fixed_eval_interval == 0
                    or update_idx == end_update
                )
            )
            if should_run_fixed_eval:
                current_bundle = _build_policy_bundle_from_model_snapshot(
                    name=f"current:{update_idx:04d}",
                    source_model=model,
                    config=config,
                    device=inference_device,
                )
                fixed_eval_metrics = evaluate_policy_matchup(
                    policy_a=current_bundle,
                    policy_b=fixed_eval_bundle,
                    episodes=fixed_eval_episodes,
                    seed=int(config.seed) + int(update_idx) * 10_000,
                    device=inference_device,
                    deterministic=bool(getattr(config, "fixed_eval_deterministic", True)),
                )
                fixed_reason_text = ", ".join(
                    f"{key}:{value}"
                    for key, value in sorted(fixed_eval_metrics.terminal_reason_counts.items())
                ) or "-"
                print(
                    f"[fixed_eval {update_idx:04d}] "
                    f"vs={fixed_eval_bundle.name} episodes={fixed_eval_metrics.episodes} "
                    f"win_rate={fixed_eval_metrics.win_rate_a:.3f} "
                    f"wins={fixed_eval_metrics.wins_a}/{fixed_eval_metrics.wins_b} draws={fixed_eval_metrics.draws} "
                    f"score_diff={fixed_eval_metrics.avg_score_diff_a_minus_b:.2f} "
                    f"rounds_avg={fixed_eval_metrics.avg_completed_rounds:.2f} "
                    f"reasons={fixed_reason_text}"
                )
                with fixed_eval_log_path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            {
                                "update": int(update_idx),
                                "opponent_checkpoint": (
                                    str(fixed_eval_opponent_path)
                                    if fixed_eval_opponent_path is not None
                                    else ""
                                ),
                                **fixed_eval_metrics.to_dict(),
                            },
                            ensure_ascii=True,
                            sort_keys=True,
                        )
                        + "\n"
                    )

            if update_idx % int(config.checkpoint_interval) == 0 or update_idx == end_update:
                checkpoint_path = output_path / f"checkpoint_{update_idx:04d}.pt"
                _save_checkpoint(
                    path=checkpoint_path,
                    update_idx=update_idx,
                    model=model,
                    optimizer=optimizer,
                    config=config,
                    metrics=all_metrics,
                    rng=rng,
                )
                print(f"saved checkpoint: {checkpoint_path}")
    finally:
        if rollout_executor is not None:
            rollout_executor.shutdown(wait=True, cancel_futures=True)

    return all_metrics

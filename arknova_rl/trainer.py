"""Self-play training loop for Masked PPO / Recurrent PPO."""

from __future__ import annotations

from collections import Counter
from concurrent.futures import Executor, ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
from typing import Any, Dict, Iterator, List, Optional, Sequence, TextIO, Tuple

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
    restore_config as _restore_config,
)


@dataclass
class RolloutSequence:
    sequence_key: Tuple[int, int]
    actor_id: int
    state_vec: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    action_count: np.ndarray
    action_index: np.ndarray
    old_logprob: np.ndarray
    old_value: np.ndarray
    reward: np.ndarray
    advantage: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))
    return_: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float32))

    @property
    def step_count(self) -> int:
        return int(self.action_index.shape[0])


@dataclass
class _RolloutSequenceBuilder:
    sequence_key: Tuple[int, int]
    actor_id: int
    state_vecs: List[np.ndarray] = field(default_factory=list)
    action_feature_steps: List[np.ndarray] = field(default_factory=list)
    action_mask_steps: List[np.ndarray] = field(default_factory=list)
    action_indices: List[int] = field(default_factory=list)
    old_logprobs: List[float] = field(default_factory=list)
    old_values: List[float] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)

    @property
    def step_count(self) -> int:
        return len(self.action_indices)

    def append(
        self,
        *,
        state_vec: np.ndarray,
        action_features: np.ndarray,
        action_mask: np.ndarray,
        action_index: int,
        old_logprob: float,
        old_value: float,
        reward: float,
    ) -> None:
        self.state_vecs.append(np.asarray(state_vec, dtype=np.float32))
        self.action_feature_steps.append(np.asarray(action_features, dtype=np.float32))
        self.action_mask_steps.append(np.asarray(action_mask, dtype=np.bool_))
        self.action_indices.append(int(action_index))
        self.old_logprobs.append(float(old_logprob))
        self.old_values.append(float(old_value))
        self.rewards.append(float(reward))

    def add_terminal_reward(self, reward_delta: float) -> None:
        if not self.rewards:
            return
        self.rewards[-1] = float(self.rewards[-1] + float(reward_delta))

    def finalize(self) -> RolloutSequence:
        if not self.state_vecs:
            raise ValueError("Cannot finalize an empty rollout sequence.")

        step_count = len(self.state_vecs)
        action_dim = int(self.action_feature_steps[0].shape[1]) if self.action_feature_steps else 0
        action_count = np.asarray(
            [int(step_features.shape[0]) for step_features in self.action_feature_steps],
            dtype=np.int32,
        )
        max_action_count = int(np.max(action_count)) if action_count.size > 0 else 0

        state_array = np.stack(self.state_vecs, axis=0).astype(np.float32, copy=False)
        action_array = np.zeros((step_count, max_action_count, action_dim), dtype=np.float32)
        mask_array = np.zeros((step_count, max_action_count), dtype=np.bool_)

        for idx, (step_features, step_mask) in enumerate(zip(self.action_feature_steps, self.action_mask_steps)):
            legal_count = int(action_count[idx])
            if legal_count <= 0:
                continue
            action_array[idx, :legal_count] = step_features[:legal_count]
            mask_array[idx, :legal_count] = step_mask[:legal_count]

        scalar_shape = (step_count,)
        return RolloutSequence(
            sequence_key=self.sequence_key,
            actor_id=int(self.actor_id),
            state_vec=state_array,
            action_features=action_array,
            action_mask=mask_array,
            action_count=action_count,
            action_index=np.asarray(self.action_indices, dtype=np.int64),
            old_logprob=np.asarray(self.old_logprobs, dtype=np.float32),
            old_value=np.asarray(self.old_values, dtype=np.float32),
            reward=np.asarray(self.rewards, dtype=np.float32),
            advantage=np.zeros(scalar_shape, dtype=np.float32),
            return_=np.zeros(scalar_shape, dtype=np.float32),
        )


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
    episode_seed: int
    rollout_sequences: List[RolloutSequence]
    step_count: int
    completed_rounds: int
    terminal_abs_diffs: List[float]
    terminal_reason: str
    elapsed_seconds: float
    trace_path: str = ""


def _json_safe_trace_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_trace_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe_trace_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return repr(value)


@dataclass
class _SlowEpisodeTraceWriter:
    trace_dir: Optional[Path]
    update_index: int
    episode_index: int
    episode_seed: int
    start_after_seconds: float
    stop_after_seconds: float
    path: Optional[Path] = None
    handle: Optional[TextIO] = None
    stopped: bool = False

    def _should_start(self, *, elapsed_seconds: float) -> bool:
        if self.trace_dir is None:
            return False
        return self.start_after_seconds > 0.0 and elapsed_seconds >= self.start_after_seconds

    def _should_stop(self, *, elapsed_seconds: float) -> bool:
        return self.stop_after_seconds > 0.0 and elapsed_seconds > self.stop_after_seconds

    def _write_record(self, record: Dict[str, Any]) -> None:
        if self.handle is None:
            return
        self.handle.write(
            json.dumps(_json_safe_trace_value(record), ensure_ascii=True, sort_keys=True) + "\n"
        )
        self.handle.flush()

    def _stop(self, *, elapsed_seconds: float, action_count: int, reason: str) -> None:
        if self.handle is None:
            return
        self._write_record(
            {
                "kind": "trace_stopped",
                "update_index": int(self.update_index),
                "episode_index": int(self.episode_index),
                "episode_seed": int(self.episode_seed),
                "elapsed_seconds": float(elapsed_seconds),
                "action_count": int(action_count),
                "reason": str(reason),
            }
        )
        self.handle.close()
        self.handle = None
        self.stopped = True

    def stop(self, *, elapsed_seconds: float, action_count: int, reason: str) -> None:
        if (
            self.handle is None
            and not self.stopped
            and self.trace_dir is not None
            and self._should_start(elapsed_seconds=elapsed_seconds)
        ):
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.trace_dir / (
                f"update_{int(self.update_index):04d}_episode_{int(self.episode_index):03d}_seed_{int(self.episode_seed)}.jsonl"
            )
            self.handle = self.path.open("w", encoding="utf-8")
            self._write_record(
                {
                    "kind": "trace_started",
                    "update_index": int(self.update_index),
                    "episode_index": int(self.episode_index),
                    "episode_seed": int(self.episode_seed),
                    "trace_start_seconds": float(self.start_after_seconds),
                    "trace_stop_seconds": float(self.stop_after_seconds),
                    "trigger_elapsed_seconds": float(elapsed_seconds),
                    "trigger_action_count": int(action_count),
                }
            )
        self._stop(
            elapsed_seconds=elapsed_seconds,
            action_count=action_count,
            reason=reason,
        )

    def record_timeout(
        self,
        record: Dict[str, Any],
        *,
        elapsed_seconds: float,
        action_count: int,
        reason: str,
    ) -> None:
        if self.trace_dir is None or self.stopped or not self._should_start(elapsed_seconds=elapsed_seconds):
            return
        if self.handle is None:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.trace_dir / (
                f"update_{int(self.update_index):04d}_episode_{int(self.episode_index):03d}_seed_{int(self.episode_seed)}.jsonl"
            )
            self.handle = self.path.open("w", encoding="utf-8")
            self._write_record(
                {
                    "kind": "trace_started",
                    "update_index": int(self.update_index),
                    "episode_index": int(self.episode_index),
                    "episode_seed": int(self.episode_seed),
                    "trace_start_seconds": float(self.start_after_seconds),
                    "trace_stop_seconds": float(self.stop_after_seconds),
                    "trigger_elapsed_seconds": float(elapsed_seconds),
                    "trigger_action_count": int(action_count),
                }
            )
        self._write_record(record)
        self._stop(
            elapsed_seconds=elapsed_seconds,
            action_count=action_count,
            reason=reason,
        )

    def append(
        self,
        record: Dict[str, Any],
        *,
        elapsed_seconds: float,
        action_count: int,
    ) -> None:
        if self.trace_dir is None or self.stopped:
            return
        if self.handle is None:
            if not self._should_start(elapsed_seconds=elapsed_seconds):
                return
            if self._should_stop(elapsed_seconds=elapsed_seconds):
                return
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            self.path = self.trace_dir / (
                f"update_{int(self.update_index):04d}_episode_{int(self.episode_index):03d}_seed_{int(self.episode_seed)}.jsonl"
            )
            self.handle = self.path.open("w", encoding="utf-8")
            self._write_record(
                {
                    "kind": "trace_started",
                    "update_index": int(self.update_index),
                    "episode_index": int(self.episode_index),
                    "episode_seed": int(self.episode_seed),
                    "trace_start_seconds": float(self.start_after_seconds),
                    "trace_stop_seconds": float(self.stop_after_seconds),
                    "trigger_elapsed_seconds": float(elapsed_seconds),
                    "trigger_action_count": int(action_count),
                }
            )
        if self._should_stop(elapsed_seconds=elapsed_seconds):
            self._stop(
                elapsed_seconds=elapsed_seconds,
                action_count=action_count,
                reason="elapsed_limit",
            )
            return
        self._write_record(record)

    def finalize(self, record: Dict[str, Any]) -> str:
        if self.handle is not None:
            self._write_record(record)
            self.handle.close()
            self.handle = None
        return str(self.path) if self.path is not None else ""


_FEEDFORWARD_STEP_BATCH_SIZE = 256

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


def _to_tensor(
    array: np.ndarray,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


def _collect_episode_rollout(
    *,
    model: MaskedActorCritic,
    obs_encoder: ObservationEncoder,
    action_encoder: ActionFeatureEncoder,
    config: PPOTrainConfig,
    device: torch.device,
    episode_index: int,
    episode_seed: int,
    update_index: int = 0,
    trace_dir: Optional[Path] = None,
) -> EpisodeRolloutResult:
    episode_started_at = time.perf_counter()
    state = main.setup_game(seed=episode_seed, player_names=["P1", "P2"])
    hidden_by_actor: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
    if model.use_lstm:
        for player_id in range(len(state.players)):
            hidden_by_actor[player_id] = model.init_hidden(1, device=device)

    sequence_builders: Dict[Tuple[int, int], _RolloutSequenceBuilder] = {}
    terminal_abs_diffs: List[float] = []
    action_count = 0
    trace_writer = _SlowEpisodeTraceWriter(
        trace_dir=trace_dir,
        update_index=int(update_index),
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        start_after_seconds=float(getattr(config, "slow_episode_trace_start_seconds", 0.0) or 0.0),
        stop_after_seconds=float(getattr(config, "slow_episode_trace_stop_seconds", 0.0) or 0.0),
    )

    while str(state.pending_decision_kind or "").strip() or not state.game_over():
        step_started_at = time.perf_counter()
        actor_id = _current_actor_id(state)
        actor = state.players[actor_id]
        legal = main.legal_actions(actor, state=state, player_id=actor_id)
        after_legal_at = time.perf_counter()
        if not legal:
            raise RuntimeError(
                f"No legal actions for actor={actor_id} pending={state.pending_decision_kind!r}."
            )

        state_vec = obs_encoder.encode_from_state(state, actor_id)
        action_features = action_encoder.encode_many(legal)
        action_mask = np.ones((len(legal),), dtype=np.bool_)
        after_encode_at = time.perf_counter()

        state_t = _to_tensor(state_vec, device=device).unsqueeze(0)
        action_t = _to_tensor(action_features, device=device).unsqueeze(0)
        mask_t = _to_tensor(action_mask, device=device, dtype=torch.bool).unsqueeze(0)

        hidden = hidden_by_actor.get(actor_id) if model.use_lstm else None
        with torch.no_grad():
            logits, value, next_hidden = model.forward_step(
                state_vec=state_t,
                action_features=action_t,
                action_mask=mask_t,
                hidden=hidden,
            )
            dist = Categorical(logits=logits[0])
            sampled_index = int(dist.sample().item())
            logprob = float(dist.log_prob(torch.tensor(sampled_index, device=device)).item())
            predicted_value = float(value.item())
        after_forward_at = time.perf_counter()
        if model.use_lstm:
            hidden_by_actor[actor_id] = (
                next_hidden[0].detach(),
                next_hidden[1].detach(),
            ) if next_hidden is not None else None

        before_diff = _progress_score_diff(state, actor_id)
        endgame_was_triggered = state.endgame_trigger_player is not None
        chosen_action = legal[sampled_index]
        effect_log_cursor = len(state.effect_log)
        pending_before = str(state.pending_decision_kind or "")
        turn_index_before = int(state.turn_index)
        main.apply_action(state, chosen_action)
        after_apply_at = time.perf_counter()
        after_diff = _progress_score_diff(state, actor_id)
        reward = (after_diff - before_diff) * float(config.step_reward_scale)
        if (
            not endgame_was_triggered
            and state.endgame_trigger_player is not None
            and int(state.endgame_trigger_player) == int(actor_id)
        ):
            reward += float(config.endgame_trigger_reward)
            reward += _endgame_speed_bonus(state=state, config=config)

        sequence_key = (episode_index, actor_id)
        builder = sequence_builders.get(sequence_key)
        if builder is None:
            builder = _RolloutSequenceBuilder(
                sequence_key=sequence_key,
                actor_id=actor_id,
            )
            sequence_builders[sequence_key] = builder
        builder.append(
            state_vec=state_vec,
            action_features=action_features,
            action_mask=action_mask,
            action_index=sampled_index,
            old_logprob=logprob,
            old_value=predicted_value,
            reward=float(reward),
        )
        action_count += 1
        elapsed_seconds = float(after_apply_at - episode_started_at)
        stage_timings = {
            "legal_actions": float(after_legal_at - step_started_at),
            "encode": float(after_encode_at - after_legal_at),
            "forward": float(after_forward_at - after_encode_at),
            "apply_action": float(after_apply_at - after_forward_at),
            "step_total": float(after_apply_at - step_started_at),
        }
        action_trace_record = {
            "kind": "action",
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "update_index": int(update_index),
            "action_count": int(action_count),
            "elapsed_seconds": float(elapsed_seconds),
            "actor_id": int(actor_id),
            "actor_name": str(actor.name),
            "turn_index_before": int(turn_index_before),
            "turn_index_after": int(state.turn_index),
            "completed_rounds": int(main._completed_rounds(state)),
            "pending_before": pending_before,
            "pending_after": str(state.pending_decision_kind or ""),
            "legal_action_count": int(len(legal)),
            "action": str(chosen_action),
            "action_details": _json_safe_trace_value(dict(chosen_action.details or {})),
            "timing_seconds": stage_timings,
            "new_effect_log": [str(entry) for entry in state.effect_log[effect_log_cursor:]],
            "players": [
                {
                    "player_id": int(idx),
                    "name": str(player_state.name),
                    "money": int(player_state.money),
                    "appeal": int(player_state.appeal),
                    "conservation": int(player_state.conservation),
                    "reputation": int(player_state.reputation),
                    "x_tokens": int(player_state.x_tokens),
                    "progress_score": float(main._progress_score(player_state)),
                }
                for idx, player_state in enumerate(state.players)
            ],
        }
        trace_writer.append(
            action_trace_record,
            elapsed_seconds=elapsed_seconds,
            action_count=action_count,
        )
        trace_stop_seconds = float(getattr(config, "slow_episode_trace_stop_seconds", 0.0) or 0.0)
        if trace_stop_seconds > 0.0 and elapsed_seconds > trace_stop_seconds:
            trace_writer.record_timeout(
                action_trace_record,
                elapsed_seconds=elapsed_seconds,
                action_count=action_count,
                reason="episode_timeout",
            )
            raise TimeoutError(
                "Episode rollout exceeded slow_episode_trace_stop_seconds "
                f"({trace_stop_seconds:.2f}s) before reaching terminal state: "
                f"update={int(update_index)} episode_index={int(episode_index)} "
                f"episode_seed={int(episode_seed)} action_count={int(action_count)} "
                f"pending={state.pending_decision_kind!r} completed_rounds={int(main._completed_rounds(state))} "
                f"action={str(chosen_action)!r} timings={stage_timings}."
            )

    for player_id in range(len(state.players)):
        seq_key = (episode_index, player_id)
        builder = sequence_builders.get(seq_key)
        if builder is None or builder.step_count <= 0:
            continue
        terminal_diff = _terminal_score_diff(state, player_id)
        terminal_abs_diffs.append(abs(float(terminal_diff)))
        terminal_reward = float(config.terminal_reward_scale) * float(terminal_diff)
        outcome = _terminal_outcome(state, player_id)
        if outcome > 0:
            terminal_reward += float(config.terminal_win_bonus)
        elif outcome < 0:
            terminal_reward -= float(config.terminal_loss_penalty)
        builder.add_terminal_reward(terminal_reward)

    terminal_reason = str(state.forced_game_over_reason or "")
    if not terminal_reason and state.endgame_trigger_player is not None:
        terminal_reason = "score_threshold_endgame"
    if not terminal_reason:
        terminal_reason = "unknown_terminal_reason"

    rollout_sequences = [
        sequence_builders[key].finalize()
        for key in sorted(sequence_builders)
        if sequence_builders[key].step_count > 0
    ]
    elapsed_seconds = float(time.perf_counter() - episode_started_at)
    trace_path = trace_writer.finalize(
        {
            "kind": "terminal",
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
            "update_index": int(update_index),
            "action_count": int(action_count),
            "elapsed_seconds": float(elapsed_seconds),
            "completed_rounds": int(main._completed_rounds(state)),
            "terminal_reason": terminal_reason,
            "scores": {
                str(player_state.name): int(main._final_score_points(state, player_state))
                for player_state in state.players
            },
        }
    )

    return EpisodeRolloutResult(
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        rollout_sequences=rollout_sequences,
        step_count=int(action_count),
        completed_rounds=int(main._completed_rounds(state)),
        terminal_abs_diffs=terminal_abs_diffs,
        terminal_reason=terminal_reason,
        elapsed_seconds=elapsed_seconds,
        trace_path=trace_path,
    )


def _finalize_rollout_advantages(
    *,
    rollout_sequences: Sequence[RolloutSequence],
    config: PPOTrainConfig,
) -> None:
    gamma = float(config.gamma)
    gae_lambda = float(config.gae_lambda)
    for rollout_sequence in rollout_sequences:
        step_count = int(rollout_sequence.step_count)
        if step_count <= 0:
            continue
        advantages = np.zeros((step_count,), dtype=np.float32)
        gae = 0.0
        for step_idx in range(step_count - 1, -1, -1):
            if step_idx == step_count - 1:
                next_value = 0.0
                nonterminal = 0.0
            else:
                next_value = float(rollout_sequence.old_value[step_idx + 1])
                nonterminal = 1.0
            delta = (
                float(rollout_sequence.reward[step_idx])
                + gamma * next_value * nonterminal
                - float(rollout_sequence.old_value[step_idx])
            )
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[step_idx] = float(gae)
        rollout_sequence.advantage = advantages
        rollout_sequence.return_ = rollout_sequence.old_value.astype(np.float32, copy=False) + advantages


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

    config = _restore_config(payload.get("config"))
    config.device = "cpu"
    device = torch.device("cpu")
    model, obs_encoder, action_encoder = _build_model_and_encoders(config=config, device=device)
    model.load_state_dict(dict(payload.get("model_state_dict") or {}))
    model.eval()
    update_index = int(payload.get("update_index", 0) or 0)
    trace_dir_raw = str(payload.get("trace_dir") or "").strip()
    trace_dir = Path(trace_dir_raw) if trace_dir_raw else None

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
                update_index=update_index,
                trace_dir=trace_dir,
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
    trace_dir: Optional[Path] = None,
    rollout_executor: Optional[Executor] = None,
) -> Tuple[List[RolloutSequence], Dict[str, Any]]:
    rollout_sequences: List[RolloutSequence] = []
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
                    update_index=int(update_index),
                    trace_dir=trace_dir,
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
                    "update_index": int(update_index),
                    "trace_dir": str(trace_dir) if trace_dir is not None else "",
                },
            )
            for chunk in chunked_specs
        ]
        for future in futures:
            episode_results.extend(list(future.result()))

    episode_results.sort(key=lambda item: int(item.episode_index))
    for episode_result in episode_results:
        rollout_sequences.extend(episode_result.rollout_sequences)
        episode_rounds.append(int(episode_result.completed_rounds))
        episode_elapsed_seconds.append(float(episode_result.elapsed_seconds))
        terminal_abs_diffs.extend(float(value) for value in episode_result.terminal_abs_diffs)
        terminal_reason_counter[str(episode_result.terminal_reason)] += 1

    _finalize_rollout_advantages(
        rollout_sequences=rollout_sequences,
        config=config,
    )
    step_count = sum(int(sequence.step_count) for sequence in rollout_sequences)

    rollout_stats = {
        "episode_count": len(episode_results),
        "step_count": step_count,
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
        "episode_summaries": [
            {
                "episode_index": int(result.episode_index),
                "episode_seed": int(result.episode_seed),
                "step_count": int(result.step_count),
                "completed_rounds": int(result.completed_rounds),
                "terminal_reason": str(result.terminal_reason),
                "elapsed_seconds": float(result.elapsed_seconds),
                "trace_path": str(result.trace_path or ""),
            }
            for result in episode_results
        ],
    }
    return rollout_sequences, rollout_stats


def _zero_loss_metrics() -> Dict[str, float]:
    return {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "total_loss": 0.0,
    }


def _iter_feedforward_step_refs(
    rollout_sequences: Sequence[RolloutSequence],
    *,
    batch_size: int,
) -> Iterator[List[Tuple[int, int]]]:
    current_batch: List[Tuple[int, int]] = []
    for sequence_idx, rollout_sequence in enumerate(rollout_sequences):
        for step_idx in range(int(rollout_sequence.step_count)):
            current_batch.append((sequence_idx, step_idx))
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []
    if current_batch:
        yield current_batch


def _build_feedforward_training_batch_from_refs(
    rollout_sequences: Sequence[RolloutSequence],
    normalized_advantages: Sequence[np.ndarray],
    step_refs: Sequence[Tuple[int, int]],
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if not step_refs:
        raise ValueError("step_refs must be non-empty.")

    batch_size = len(step_refs)
    first_sequence = rollout_sequences[step_refs[0][0]]
    state_dim = int(first_sequence.state_vec.shape[1])
    action_dim = int(first_sequence.action_features.shape[2])
    max_action_count = max(
        int(rollout_sequences[sequence_idx].action_count[step_idx])
        for sequence_idx, step_idx in step_refs
    )

    state_array = np.zeros((batch_size, state_dim), dtype=np.float32)
    action_array = np.zeros((batch_size, max_action_count, action_dim), dtype=np.float32)
    mask_array = np.zeros((batch_size, max_action_count), dtype=np.bool_)
    action_indices = np.zeros((batch_size,), dtype=np.int64)
    old_logprobs = np.zeros((batch_size,), dtype=np.float32)
    returns = np.zeros((batch_size,), dtype=np.float32)
    advantages = np.zeros((batch_size,), dtype=np.float32)

    for batch_idx, (sequence_idx, step_idx) in enumerate(step_refs):
        rollout_sequence = rollout_sequences[sequence_idx]
        action_count = int(rollout_sequence.action_count[step_idx])
        state_array[batch_idx] = rollout_sequence.state_vec[step_idx]
        action_array[batch_idx, :action_count] = rollout_sequence.action_features[step_idx, :action_count]
        mask_array[batch_idx, :action_count] = rollout_sequence.action_mask[step_idx, :action_count]
        action_indices[batch_idx] = int(rollout_sequence.action_index[step_idx])
        old_logprobs[batch_idx] = float(rollout_sequence.old_logprob[step_idx])
        returns[batch_idx] = float(rollout_sequence.return_[step_idx])
        advantages[batch_idx] = float(normalized_advantages[sequence_idx][step_idx])

    return {
        "state_vec": _to_tensor(state_array, device=device),
        "action_features": _to_tensor(action_array, device=device),
        "action_mask": _to_tensor(mask_array, device=device, dtype=torch.bool),
        "action_index": torch.as_tensor(action_indices, dtype=torch.long, device=device),
        "old_logprob": _to_tensor(old_logprobs, device=device),
        "target_return": _to_tensor(returns, device=device),
        "advantage": _to_tensor(advantages, device=device),
    }


def _tensorize_rollout_sequence(
    rollout_sequence: RolloutSequence,
    normalized_advantage: np.ndarray,
    *,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        "state_vec": _to_tensor(rollout_sequence.state_vec, device=device),
        "action_features": _to_tensor(rollout_sequence.action_features, device=device),
        "action_mask": _to_tensor(rollout_sequence.action_mask, device=device, dtype=torch.bool),
        "action_index": torch.as_tensor(rollout_sequence.action_index, dtype=torch.long, device=device),
        "old_logprob": _to_tensor(rollout_sequence.old_logprob, device=device),
        "target_return": _to_tensor(rollout_sequence.return_, device=device),
        "advantage": _to_tensor(normalized_advantage, device=device),
    }


def _update_model(
    *,
    model: MaskedActorCritic,
    optimizer: torch.optim.Optimizer,
    rollout_sequences: Sequence[RolloutSequence],
    config: PPOTrainConfig,
    device: torch.device,
) -> Dict[str, float]:
    if not rollout_sequences:
        return _zero_loss_metrics()

    advantages = np.concatenate(
        [
            np.asarray(sequence.advantage, dtype=np.float32)
            for sequence in rollout_sequences
            if int(sequence.step_count) > 0
        ],
        axis=0,
    )
    if advantages.size <= 0:
        return _zero_loss_metrics()
    adv_mean = float(np.mean(advantages))
    adv_std = float(np.std(advantages))
    if adv_std < 1e-8:
        adv_std = 1.0
    normalized_advantages = [
        (
            np.asarray(sequence.advantage, dtype=np.float32) - adv_mean
        ) / adv_std
        for sequence in rollout_sequences
    ]
    total_step_count = sum(int(sequence.step_count) for sequence in rollout_sequences)
    if total_step_count <= 0:
        return _zero_loss_metrics()

    epoch_policy_losses: List[float] = []
    epoch_value_losses: List[float] = []
    epoch_entropies: List[float] = []
    epoch_total_losses: List[float] = []

    for _ in range(config.update_epochs):
        optimizer.zero_grad(set_to_none=True)
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_entropy = 0.0
        epoch_total_loss = 0.0
        processed_weight = 0.0

        if not model.use_lstm:
            for step_refs in _iter_feedforward_step_refs(
                rollout_sequences,
                batch_size=_FEEDFORWARD_STEP_BATCH_SIZE,
            ):
                batch = _build_feedforward_training_batch_from_refs(
                    rollout_sequences,
                    normalized_advantages,
                    step_refs,
                    device=device,
                )
                logits, values, _ = model.forward_step(
                    state_vec=batch["state_vec"],
                    action_features=batch["action_features"],
                    action_mask=batch["action_mask"],
                    hidden=None,
                )
                step_log_probs = torch.log_softmax(logits, dim=-1)
                step_probs = torch.softmax(logits, dim=-1)
                chosen_indices = batch["action_index"].unsqueeze(1)
                new_logprob = step_log_probs.gather(1, chosen_indices).squeeze(1)
                entropy = -(step_probs * step_log_probs).sum(dim=-1)
                old_logprob = batch["old_logprob"]
                target_return = batch["target_return"]
                advantage = batch["advantage"]
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
                total_loss = (
                    policy_loss
                    + float(config.value_coef) * value_loss
                    - float(config.entropy_coef) * entropy_mean
                )
                batch_weight = float(len(step_refs)) / float(total_step_count)
                (total_loss * batch_weight).backward()
                processed_weight += batch_weight
                epoch_policy_loss += float(policy_loss.detach().cpu().item()) * batch_weight
                epoch_value_loss += float(value_loss.detach().cpu().item()) * batch_weight
                epoch_entropy += float(entropy_mean.detach().cpu().item()) * batch_weight
                epoch_total_loss += float(total_loss.detach().cpu().item()) * batch_weight
        else:
            for sequence_idx, rollout_sequence in enumerate(rollout_sequences):
                step_count = int(rollout_sequence.step_count)
                if step_count <= 0:
                    continue
                sequence_batch = _tensorize_rollout_sequence(
                    rollout_sequence,
                    normalized_advantages[sequence_idx],
                    device=device,
                )
                hidden = model.init_hidden(1, device=device)
                policy_losses_t: List[torch.Tensor] = []
                value_losses_t: List[torch.Tensor] = []
                entropies_t: List[torch.Tensor] = []

                for step_idx in range(step_count):
                    state_t = sequence_batch["state_vec"][step_idx : step_idx + 1]
                    action_t = sequence_batch["action_features"][step_idx : step_idx + 1]
                    mask_t = sequence_batch["action_mask"][step_idx : step_idx + 1]
                    logits, values, hidden = model.forward_step(
                        state_vec=state_t,
                        action_features=action_t,
                        action_mask=mask_t,
                        hidden=hidden,
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
                    continue

                policy_loss = torch.stack(policy_losses_t).mean()
                value_loss = torch.stack(value_losses_t).mean()
                entropy_mean = torch.stack(entropies_t).mean()
                total_loss = (
                    policy_loss
                    + float(config.value_coef) * value_loss
                    - float(config.entropy_coef) * entropy_mean
                )
                sequence_weight = float(step_count) / float(total_step_count)
                (total_loss * sequence_weight).backward()
                processed_weight += sequence_weight
                epoch_policy_loss += float(policy_loss.detach().cpu().item()) * sequence_weight
                epoch_value_loss += float(value_loss.detach().cpu().item()) * sequence_weight
                epoch_entropy += float(entropy_mean.detach().cpu().item()) * sequence_weight
                epoch_total_loss += float(total_loss.detach().cpu().item()) * sequence_weight

        if processed_weight <= 0.0:
            optimizer.zero_grad(set_to_none=True)
            break

        nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
        optimizer.step()

        epoch_policy_losses.append(epoch_policy_loss)
        epoch_value_losses.append(epoch_value_loss)
        epoch_entropies.append(epoch_entropy)
        epoch_total_losses.append(epoch_total_loss)

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
    slow_trace_dir = output_path / "slow_episode_traces"

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
    if (
        float(getattr(config, "slow_episode_trace_start_seconds", 0.0) or 0.0) > 0.0
    ):
        print(
            "slow episode trace: "
            f"dir={slow_trace_dir} "
            f"start={float(getattr(config, 'slow_episode_trace_start_seconds', 0.0) or 0.0):.2f}s "
            f"stop={float(getattr(config, 'slow_episode_trace_stop_seconds', 0.0) or 0.0):.2f}s"
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
            rollout_sequences, rollout_stats = _collect_rollout(
                model=rollout_model,
                obs_encoder=rollout_obs_encoder,
                action_encoder=rollout_action_encoder,
                config=config,
                rng=rng,
                device=rollout_device,
                update_index=update_idx,
                trace_dir=slow_trace_dir,
                rollout_executor=rollout_executor,
            )
            update_time_sec = float(time.perf_counter() - rollout_started_at)
            model_update_started_at = time.perf_counter()
            update_stats = _update_model(
                model=model,
                optimizer=optimizer,
                rollout_sequences=rollout_sequences,
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
                            "episode_summaries": list(rollout_stats.get("episode_summaries") or []),
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
                slowest_episodes = sorted(
                    list(rollout_stats.get("episode_summaries") or []),
                    key=lambda item: float(item.get("elapsed_seconds", 0.0)),
                    reverse=True,
                )[:3]
                slowest_text = ", ".join(
                    (
                        f"seed={int(item.get('episode_seed', 0))}:"
                        f"{float(item.get('elapsed_seconds', 0.0)):.2f}s/"
                        f"steps={int(item.get('step_count', 0))}/"
                        f"reason={str(item.get('terminal_reason', ''))}"
                    )
                    for item in slowest_episodes
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
                    f"reasons={reason_text} slowest={slowest_text}"
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

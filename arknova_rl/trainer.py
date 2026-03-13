"""Self-play training loop for recurrent PPO."""

from __future__ import annotations

from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, Executor, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import multiprocessing as mp
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import main

from .config import PPOTrainConfig, normalize_torch_device_spec
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


def format_log_timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def _stdout_supports_live_updates() -> bool:
    term = str(os.environ.get("TERM") or "").strip().lower()
    if not term or term == "dumb":
        return False
    return bool(getattr(sys.stdout, "isatty", lambda: False)())


def _stdout_supports_ansi_color() -> bool:
    if "NO_COLOR" in os.environ:
        return False
    return _stdout_supports_live_updates()


def _highlight_log_line(message: str, *, style: str) -> str:
    if not _stdout_supports_ansi_color():
        return message
    palette = {
        "update_summary": "\033[1;96m",
        "fixed_eval": "\033[1;95m",
    }
    prefix = palette.get(str(style), "\033[1m")
    return f"{prefix}{message}\033[0m"


def _log_progress_event(message: str) -> None:
    print(f"[{format_log_timestamp()}] {message}", flush=True)


@dataclass
class _RolloutProgressBar:
    update_index: int
    total_episodes: int
    slot_count: int
    started_at: float = field(default_factory=time.perf_counter)
    completed_episodes: int = 0
    episode_elapsed_total: float = 0.0
    active_slots: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    enabled: bool = field(init=False, default=False)
    last_line_length: int = 0
    closed: bool = False

    def __post_init__(self) -> None:
        self.enabled = _stdout_supports_live_updates() and int(self.total_episodes) > 0

    def slot_started(
        self,
        *,
        slot_index: int,
        episode_index: int,
        episode_seed: int,
    ) -> None:
        if self.closed:
            return
        self.active_slots[int(slot_index)] = {
            "episode_index": int(episode_index),
            "episode_seed": int(episode_seed),
        }
        self._render()

    def episode_completed(
        self,
        *,
        slot_index: int,
        episode_index: int,
        episode_seed: int,
        episode_elapsed_seconds: float,
    ) -> None:
        if self.closed:
            return
        self.completed_episodes = min(
            max(0, int(self.total_episodes)),
            int(self.completed_episodes) + 1,
        )
        self.episode_elapsed_total += float(episode_elapsed_seconds)
        self.active_slots.pop(int(slot_index), None)
        self._render()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.enabled or self.last_line_length <= 0:
            return
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        line = self._format_line()
        padded = line
        if len(line) < self.last_line_length:
            padded += " " * (self.last_line_length - len(line))
        if force or padded.strip():
            sys.stdout.write("\r" + padded)
            sys.stdout.flush()
        self.last_line_length = max(self.last_line_length, len(line))

    def _format_line(self) -> str:
        total = max(1, int(self.total_episodes))
        completed = max(0, min(total, int(self.completed_episodes)))
        width = 24
        filled = min(width, int(width * completed / total))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(0.0, float(time.perf_counter() - self.started_at))
        avg_episode_seconds = (
            float(self.episode_elapsed_total) / float(completed)
            if completed > 0
            else 0.0
        )
        active_parts = [
            f"{slot}:{int(info.get('episode_index', 0)) + 1:03d}"
            for slot, info in sorted(self.active_slots.items())
        ]
        active_text = " ".join(active_parts[:4])
        if len(active_parts) > 4:
            active_text += f" +{len(active_parts) - 4}"
        if not active_text:
            active_text = "-"
        return (
            f"[update {int(self.update_index):04d}] rollout "
            f"[{bar}] {completed:02d}/{total:02d} "
            f"active={active_text} "
            f"avg={avg_episode_seconds:.2f}s elapsed={elapsed:.1f}s"
        )


@dataclass
class _ModelUpdateProgressBar:
    update_index: int
    total_epochs: int
    units_per_epoch: int
    unit_label: str
    started_at: float = field(default_factory=time.perf_counter)
    completed_units: int = 0
    current_epoch: int = 0
    current_epoch_unit: int = 0
    enabled: bool = field(init=False, default=False)
    last_line_length: int = 0
    last_rendered_at: float = 0.0
    min_render_interval_seconds: float = 0.1
    closed: bool = False

    def __post_init__(self) -> None:
        self.enabled = (
            _stdout_supports_live_updates()
            and int(self.total_epochs) > 0
            and int(self.units_per_epoch) > 0
        )

    def start_epoch(self, *, epoch_index: int) -> None:
        if self.closed:
            return
        self.current_epoch = max(0, int(epoch_index))
        self.current_epoch_unit = 0
        self._render(force=True)

    def advance(self) -> None:
        if self.closed:
            return
        self.completed_units += 1
        self.current_epoch_unit += 1
        self._render()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.enabled or self.last_line_length <= 0:
            return
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        total_units = max(1, int(self.total_epochs) * int(self.units_per_epoch))
        if (
            not force
            and self.last_rendered_at > 0.0
            and self.completed_units < total_units
            and now - self.last_rendered_at < float(self.min_render_interval_seconds)
        ):
            return
        line = self._format_line(now=now)
        padded = line
        if len(line) < self.last_line_length:
            padded += " " * (self.last_line_length - len(line))
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()
        self.last_line_length = max(self.last_line_length, len(line))
        self.last_rendered_at = float(now)

    def _format_line(self, *, now: float) -> str:
        total_units = max(1, int(self.total_epochs) * int(self.units_per_epoch))
        completed_units = max(0, min(total_units, int(self.completed_units)))
        width = 24
        filled = min(width, int(width * completed_units / total_units))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(0.0, float(now - self.started_at))
        current_epoch = max(1, min(max(1, int(self.total_epochs)), int(self.current_epoch or 1)))
        current_unit = max(0, min(max(1, int(self.units_per_epoch)), int(self.current_epoch_unit)))
        return (
            f"[update {int(self.update_index):04d}] optimize "
            f"[{bar}] epoch={current_epoch}/{max(1, int(self.total_epochs))} "
            f"{self.unit_label}={current_unit}/{max(1, int(self.units_per_epoch))} "
            f"elapsed={elapsed:.1f}s"
        )


@dataclass
class _FixedEvalProgressBar:
    update_index: int
    total_episodes: int
    opponent_name: str
    started_at: float = field(default_factory=time.perf_counter)
    completed_episodes: int = 0
    enabled: bool = field(init=False, default=False)
    last_line_length: int = 0
    closed: bool = False
    last_rendered_at: float = 0.0
    min_render_interval_seconds: float = 0.1

    def __post_init__(self) -> None:
        self.enabled = _stdout_supports_live_updates() and int(self.total_episodes) > 0
        self._render(force=True)

    def episode_completed(self, *, completed_episodes: int, episode_seed: int) -> None:
        del episode_seed
        if self.closed:
            return
        self.completed_episodes = max(
            0,
            min(max(1, int(self.total_episodes)), int(completed_episodes)),
        )
        self._render()

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if not self.enabled or self.last_line_length <= 0:
            return
        self._render(force=True)
        sys.stdout.write("\n")
        sys.stdout.flush()

    def _render(self, *, force: bool = False) -> None:
        if not self.enabled:
            return
        now = time.perf_counter()
        if (
            not force
            and self.last_rendered_at > 0.0
            and self.completed_episodes < max(1, int(self.total_episodes))
            and now - self.last_rendered_at < float(self.min_render_interval_seconds)
        ):
            return
        line = self._format_line(now=now)
        padded = line
        if len(line) < self.last_line_length:
            padded += " " * (self.last_line_length - len(line))
        sys.stdout.write("\r" + padded)
        sys.stdout.flush()
        self.last_line_length = max(self.last_line_length, len(line))
        self.last_rendered_at = float(now)

    def _format_line(self, *, now: float) -> str:
        total = max(1, int(self.total_episodes))
        completed = max(0, min(total, int(self.completed_episodes)))
        width = 24
        filled = min(width, int(width * completed / total))
        bar = "#" * filled + "-" * (width - filled)
        elapsed = max(0.0, float(now - self.started_at))
        return (
            f"[fixed_eval {int(self.update_index):04d}] "
            f"[{bar}] {completed:02d}/{total:02d} "
            f"vs={self.opponent_name} "
            f"elapsed={elapsed:.1f}s"
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


def _episode_artifact_stem(
    *,
    update_index: int,
    episode_index: int,
    episode_seed: int,
) -> str:
    return (
        f"update_{int(update_index):04d}_episode_{int(episode_index):03d}_seed_{int(episode_seed)}"
    )


def _episode_trace_path(
    *,
    trace_dir: Optional[Path],
    update_index: int,
    episode_index: int,
    episode_seed: int,
) -> Optional[Path]:
    if trace_dir is None:
        return None
    return trace_dir / (
        _episode_artifact_stem(
            update_index=update_index,
            episode_index=episode_index,
            episode_seed=episode_seed,
        )
        + ".jsonl"
    )


def _episode_progress_path(
    *,
    trace_dir: Optional[Path],
    update_index: int,
    episode_index: int,
    episode_seed: int,
) -> Optional[Path]:
    if trace_dir is None:
        return None
    return trace_dir / (
        _episode_artifact_stem(
            update_index=update_index,
            episode_index=episode_index,
            episode_seed=episode_seed,
        )
        + ".progress.json"
    )


def _write_json_snapshot(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8") as handle:
        handle.write(
            json.dumps(_json_safe_trace_value(payload), ensure_ascii=True, sort_keys=True)
        )
        handle.write("\n")
    temp_path.replace(path)


def _read_json_snapshot(path: Optional[Path]) -> Dict[str, Any]:
    if path is None or not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _action_progress_snapshot(action: Any) -> Dict[str, Any]:
    details = getattr(action, "details", None)
    rendered = str(action)
    card_name = getattr(action, "card_name", None)
    action_type = getattr(action, "type", None)
    value = getattr(action, "value", None)
    return {
        "rendered": rendered,
        "type": str(action_type) if action_type is not None else "",
        "card_name": str(card_name or ""),
        "value": (None if value is None else int(value)),
        "details": _json_safe_trace_value(dict(details or {})) if isinstance(details, dict) else {},
    }


def _format_watchdog_progress_summary(progress_snapshot: Dict[str, Any]) -> str:
    if not progress_snapshot:
        return "stage=unknown"
    parts: List[str] = []
    stage = str(progress_snapshot.get("stage") or "").strip()
    if stage:
        parts.append(f"stage={stage}")
    actor_name = str(progress_snapshot.get("actor_name") or "").strip()
    if actor_name:
        parts.append(f"actor={actor_name}")
    current_action = progress_snapshot.get("current_action")
    if isinstance(current_action, dict):
        rendered = str(current_action.get("rendered") or "").strip()
        card_name = str(current_action.get("card_name") or "").strip()
        if rendered:
            parts.append(f"action={rendered!r}")
        if card_name:
            parts.append(f"card={card_name!r}")
    else:
        last_action = progress_snapshot.get("last_completed_action")
        if isinstance(last_action, dict):
            rendered = str(last_action.get("rendered") or "").strip()
            if rendered:
                parts.append(f"last_action={rendered!r}")
    pending = progress_snapshot.get("pending")
    if pending is not None:
        parts.append(f"pending={str(pending)!r}")
    completed_rounds = progress_snapshot.get("completed_rounds")
    if isinstance(completed_rounds, (int, float)):
        parts.append(f"completed_rounds={int(completed_rounds)}")
    legal_action_count = progress_snapshot.get("legal_action_count")
    if isinstance(legal_action_count, (int, float)):
        parts.append(f"legal_actions={int(legal_action_count)}")
    legal_actions_profile = progress_snapshot.get("legal_actions_profile")
    if isinstance(legal_actions_profile, dict):
        phase = str(legal_actions_profile.get("phase") or "").strip()
        current_branch = str(legal_actions_profile.get("current_branch") or "").strip()
        last_subfunction = str(legal_actions_profile.get("last_subfunction") or "").strip()
        if phase:
            parts.append(f"legal_phase={phase}")
        if current_branch:
            parts.append(f"branch={current_branch}")
        if last_subfunction:
            parts.append(f"subfunction={last_subfunction}")
        animals_profile = legal_actions_profile.get("animals_profile")
        if isinstance(animals_profile, dict):
            animals_mode = str(animals_profile.get("mode") or "").strip()
            if animals_mode:
                parts.append(f"animals_mode={animals_mode}")
            current_option = animals_profile.get("current_option")
            if isinstance(current_option, dict):
                option_label = str(current_option.get("rendered") or "").strip()
                if option_label:
                    parts.append(f"animals_option={option_label!r}")
                option_index = current_option.get("index")
                if isinstance(option_index, (int, float)):
                    parts.append(f"animals_option_index={int(option_index)}")
            first_play = animals_profile.get("first_play")
            if isinstance(first_play, dict):
                first_name = str(first_play.get("card_name") or "").strip()
                first_enclosure_index = first_play.get("enclosure_index")
                if first_name:
                    if isinstance(first_enclosure_index, (int, float)):
                        parts.append(f"animals_first={first_name}@E{int(first_enclosure_index)}")
                    else:
                        parts.append(f"animals_first={first_name}")
            second_card = animals_profile.get("second_card")
            if isinstance(second_card, dict):
                second_name = str(second_card.get("card_name") or "").strip()
                if second_name:
                    parts.append(f"animals_second={second_name}")
            current_card_name = str(animals_profile.get("current_card_name") or "").strip()
            if current_card_name:
                parts.append(f"animals_card={current_card_name}")
            current_option_label = str(animals_profile.get("current_option_label") or "").strip()
            if current_option_label:
                parts.append(f"animals_label={current_option_label!r}")
        build_profile = legal_actions_profile.get("build_profile")
        if current_branch == "build" and isinstance(build_profile, dict):
            build_mode = str(build_profile.get("mode") or "").strip()
            if build_mode:
                parts.append(f"build_mode={build_mode}")
            current_option = build_profile.get("current_option")
            if isinstance(current_option, dict):
                option_label = str(current_option.get("rendered") or "").strip()
                if option_label:
                    parts.append(f"build_option={option_label!r}")
            current_option_label = str(build_profile.get("current_option_label") or "").strip()
            if current_option_label:
                parts.append(f"build_label={current_option_label!r}")
            placement_bonuses = [
                str(item).strip()
                for item in list(build_profile.get("placement_bonuses") or [])
                if str(item).strip()
            ]
            if placement_bonuses:
                parts.append(f"build_bonuses={','.join(placement_bonuses)!r}")
            current_bonus = str(build_profile.get("current_bonus") or "").strip()
            if current_bonus:
                parts.append(f"build_bonus={current_bonus}")
            bonus_index = build_profile.get("bonus_index")
            bonus_count = build_profile.get("bonus_count")
            if isinstance(bonus_index, (int, float)) and isinstance(bonus_count, (int, float)):
                parts.append(f"build_bonus_step={int(bonus_index)}/{int(bonus_count)}")
            variant_count_before = build_profile.get("variant_count_before")
            if isinstance(variant_count_before, (int, float)):
                parts.append(f"build_variants_before={int(variant_count_before)}")
            variant_count_after = build_profile.get("variant_count_after")
            if isinstance(variant_count_after, (int, float)):
                parts.append(f"build_variants_after={int(variant_count_after)}")
            deduped_variant_count = build_profile.get("deduped_variant_count")
            if isinstance(deduped_variant_count, (int, float)) and int(deduped_variant_count) > 0:
                parts.append(f"build_variants_deduped={int(deduped_variant_count)}")
            current_sequence_length = build_profile.get("current_sequence_length")
            if isinstance(current_sequence_length, (int, float)) and int(current_sequence_length) > 0:
                parts.append(f"build_sequence_length={int(current_sequence_length)}")
            current_sequence_label = str(build_profile.get("current_sequence_label") or "").strip()
            if current_sequence_label:
                parts.append(f"build_sequence={current_sequence_label!r}")
            resolved_variant_count = build_profile.get("resolved_variant_count")
            if isinstance(resolved_variant_count, (int, float)) and int(resolved_variant_count) > 0:
                parts.append(f"build_resolved_variants={int(resolved_variant_count)}")
    return " ".join(parts) if parts else "stage=unknown"


@dataclass
class _EpisodeProgressTracker:
    trace_dir: Optional[Path]
    update_index: int
    episode_index: int
    episode_seed: int
    enabled: bool = False
    inside_legal_actions_min_write_interval_seconds: float = 0.25
    path: Optional[Path] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    last_written_at: float = 0.0
    last_written_stage: str = ""

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        self.path = _episode_progress_path(
            trace_dir=self.trace_dir,
            update_index=self.update_index,
            episode_index=self.episode_index,
            episode_seed=self.episode_seed,
        )

    def update(self, **kwargs: Any) -> None:
        if self.path is None:
            return
        if not self.payload:
            self.payload = {
                "update_index": int(self.update_index),
                "episode_index": int(self.episode_index),
                "episode_seed": int(self.episode_seed),
            }
        for key, value in kwargs.items():
            self.payload[str(key)] = value
        stage = str(self.payload.get("stage") or "").strip()
        now = time.perf_counter()
        should_write = self.last_written_at <= 0.0 or stage != "inside_legal_actions"
        if not should_write and stage != self.last_written_stage:
            should_write = True
        if (
            not should_write
            and now - self.last_written_at >= float(self.inside_legal_actions_min_write_interval_seconds)
        ):
            should_write = True
        if not should_write:
            return
        _write_json_snapshot(self.path, self.payload)
        self.last_written_at = float(now)
        self.last_written_stage = stage

    def clear(self) -> None:
        if self.path is None:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            return
        finally:
            self.payload = {}
            self.last_written_at = 0.0
            self.last_written_stage = ""


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
            self.path = _episode_trace_path(
                trace_dir=self.trace_dir,
                update_index=self.update_index,
                episode_index=self.episode_index,
                episode_seed=self.episode_seed,
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
            self.path = _episode_trace_path(
                trace_dir=self.trace_dir,
                update_index=self.update_index,
                episode_index=self.episode_index,
                episode_seed=self.episode_seed,
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
            self.path = _episode_trace_path(
                trace_dir=self.trace_dir,
                update_index=self.update_index,
                episode_index=self.episode_index,
                episode_seed=self.episode_seed,
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
    trace_enabled = bool(getattr(config, "slow_episode_trace_enabled", False))
    trace_start_seconds = (
        float(getattr(config, "slow_episode_trace_start_seconds", 0.0) or 0.0)
        if trace_enabled
        else 0.0
    )
    trace_stop_seconds = (
        float(getattr(config, "slow_episode_trace_stop_seconds", 0.0) or 0.0)
        if trace_enabled
        else 0.0
    )
    hidden_by_actor: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {
        player_id: model.init_hidden(1, device=device)
        for player_id in range(len(state.players))
    }

    sequence_builders: Dict[Tuple[int, int], _RolloutSequenceBuilder] = {}
    terminal_abs_diffs: List[float] = []
    action_count = 0
    trace_writer = _SlowEpisodeTraceWriter(
        trace_dir=trace_dir,
        update_index=int(update_index),
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        start_after_seconds=trace_start_seconds,
        stop_after_seconds=trace_stop_seconds,
    )
    progress_tracker = _EpisodeProgressTracker(
        trace_dir=trace_dir,
        update_index=int(update_index),
        episode_index=int(episode_index),
        episode_seed=int(episode_seed),
        enabled=trace_enabled,
    )
    progress_tracker.update(
        stage="episode_started",
        elapsed_seconds=0.0,
        action_count=0,
        current_action=None,
        last_completed_action=None,
        legal_actions_profile=None,
        pending=str(state.pending_decision_kind or ""),
        completed_rounds=int(main._completed_rounds(state)),
        turn_index=int(state.turn_index),
    )

    try:
        while str(state.pending_decision_kind or "").strip() or not state.game_over():
            step_started_at = time.perf_counter()
            actor_id = _current_actor_id(state)
            actor = state.players[actor_id]
            progress_tracker.update(
                stage="before_legal_actions",
                elapsed_seconds=float(step_started_at - episode_started_at),
                action_count=int(action_count),
                actor_id=int(actor_id),
                actor_name=str(actor.name),
                pending=str(state.pending_decision_kind or ""),
                completed_rounds=int(main._completed_rounds(state)),
                turn_index=int(state.turn_index),
                legal_action_count=None,
                legal_actions_profile=None,
                current_action=None,
            )
            with main.legal_actions_profiling(
                lambda profile_snapshot: progress_tracker.update(
                    stage="inside_legal_actions",
                    legal_actions_profile=profile_snapshot,
                ),
                snapshot_copy=False,
            ):
                legal = main.legal_actions(actor, state=state, player_id=actor_id)
            after_legal_at = time.perf_counter()
            if not legal:
                raise RuntimeError(
                    f"No legal actions for actor={actor_id} pending={state.pending_decision_kind!r}."
                )
            progress_tracker.update(
                stage="after_legal_actions",
                elapsed_seconds=float(after_legal_at - episode_started_at),
                action_count=int(action_count),
                actor_id=int(actor_id),
                actor_name=str(actor.name),
                pending=str(state.pending_decision_kind or ""),
                completed_rounds=int(main._completed_rounds(state)),
                turn_index=int(state.turn_index),
                legal_action_count=int(len(legal)),
                legal_actions_profile=None,
            )

            state_vec = obs_encoder.encode_from_state(state, actor_id)
            action_features = action_encoder.encode_many(legal)
            action_mask = np.ones((len(legal),), dtype=np.bool_)
            after_encode_at = time.perf_counter()

            state_t = _to_tensor(state_vec, device=device).unsqueeze(0)
            action_t = _to_tensor(action_features, device=device).unsqueeze(0)
            mask_t = _to_tensor(action_mask, device=device, dtype=torch.bool).unsqueeze(0)

            hidden = hidden_by_actor[actor_id]
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
            hidden_by_actor[actor_id] = (
                next_hidden[0].detach(),
                next_hidden[1].detach(),
            )

            before_diff = _progress_score_diff(state, actor_id)
            endgame_was_triggered = state.endgame_trigger_player is not None
            chosen_action = legal[sampled_index]
            chosen_action_snapshot = _action_progress_snapshot(chosen_action)
            effect_log_cursor = len(state.effect_log)
            pending_before = str(state.pending_decision_kind or "")
            turn_index_before = int(state.turn_index)
            progress_tracker.update(
                stage="before_apply_action",
                elapsed_seconds=float(after_forward_at - episode_started_at),
                action_count=int(action_count),
                actor_id=int(actor_id),
                actor_name=str(actor.name),
                pending=pending_before,
                completed_rounds=int(main._completed_rounds(state)),
                turn_index=turn_index_before,
                legal_action_count=int(len(legal)),
                legal_actions_profile=None,
                current_action=chosen_action_snapshot,
            )
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
            progress_tracker.update(
                stage="after_apply_action",
                elapsed_seconds=float(elapsed_seconds),
                action_count=int(action_count),
                actor_id=int(actor_id),
                actor_name=str(actor.name),
                pending=str(state.pending_decision_kind or ""),
                completed_rounds=int(main._completed_rounds(state)),
                turn_index=int(state.turn_index),
                legal_action_count=int(len(legal)),
                legal_actions_profile=None,
                current_action=None,
                last_completed_action=chosen_action_snapshot,
            )
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
    except Exception as exc:
        progress_tracker.update(
            stage="episode_exception",
            error_type=exc.__class__.__name__,
            error_message=str(exc),
        )
        raise

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
    progress_tracker.update(
        stage="episode_terminal",
        elapsed_seconds=float(elapsed_seconds),
        action_count=int(action_count),
        pending=str(state.pending_decision_kind or ""),
        completed_rounds=int(main._completed_rounds(state)),
        turn_index=int(state.turn_index),
        terminal_reason=terminal_reason,
        current_action=None,
    )
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
    progress_tracker.clear()

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
    episode_specs = list(payload.get("episode_specs") or [])

    episode_results: List[EpisodeRolloutResult] = []
    for episode_index, episode_seed in episode_specs:
        episode_result = _collect_episode_rollout(
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
        episode_results.append(episode_result)
    return episode_results


def _shutdown_rollout_executor(
    rollout_executor: Optional[Executor],
    *,
    force_terminate: bool = False,
) -> None:
    if rollout_executor is None:
        return
    processes = list(getattr(rollout_executor, "_processes", {}).values()) if force_terminate else []
    shutdown = getattr(rollout_executor, "shutdown", None)
    if callable(shutdown):
        try:
            shutdown(wait=not force_terminate, cancel_futures=True)
        except TypeError:
            shutdown()
    if not force_terminate:
        return
    for process in processes:
        if process is None:
            continue
        try:
            if process.is_alive():
                process.terminate()
        except Exception:
            continue
    deadline = time.perf_counter() + 2.0
    for process in processes:
        if process is None:
            continue
        try:
            process.join(timeout=max(0.0, deadline - time.perf_counter()))
        except Exception:
            continue
    for process in processes:
        if process is None:
            continue
        try:
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
        except Exception:
            continue
    final_deadline = time.perf_counter() + 1.0
    for process in processes:
        if process is None:
            continue
        try:
            process.join(timeout=max(0.0, final_deadline - time.perf_counter()))
        except Exception:
            continue


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
    total_episodes = len(episode_specs)
    worker_count = max(1, int(getattr(config, "rollout_workers", 1) or 1))
    trace_enabled = bool(getattr(config, "slow_episode_trace_enabled", False))
    watchdog_timeout_seconds = (
        float(getattr(config, "slow_episode_trace_stop_seconds", 0.0) or 0.0)
        if trace_enabled
        else 0.0
    )
    progress_bar = _RolloutProgressBar(
        update_index=int(update_index),
        total_episodes=total_episodes,
        slot_count=min(max(1, worker_count), max(1, total_episodes)),
    )

    episode_results: List[EpisodeRolloutResult] = []
    try:
        if rollout_executor is None or worker_count <= 1 or len(episode_specs) <= 1:
            for episode_idx, episode_seed in episode_specs:
                progress_bar.slot_started(
                    slot_index=1,
                    episode_index=int(episode_idx),
                    episode_seed=int(episode_seed),
                )
                episode_result = _collect_episode_rollout(
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
                episode_results.append(episode_result)
                progress_bar.episode_completed(
                    slot_index=1,
                    episode_index=int(episode_idx),
                    episode_seed=int(episode_seed),
                    episode_elapsed_seconds=float(episode_result.elapsed_seconds),
                )
        else:
            cpu_model_state = _state_dict_to_cpu(model.state_dict())
            slot_count = min(worker_count, len(episode_specs))
            pending_episode_specs = deque(episode_specs)
            active_futures: Dict[Any, Dict[str, Any]] = {}

            def _submit_episode_to_slot(*, slot_index: int, episode_spec: Tuple[int, int]) -> None:
                episode_idx, episode_seed = int(episode_spec[0]), int(episode_spec[1])
                slot_started_at = time.perf_counter()
                task_seed = int(rng.randint(0, 10_000_000))
                progress_bar.slot_started(
                    slot_index=int(slot_index),
                    episode_index=episode_idx,
                    episode_seed=episode_seed,
                )
                future = rollout_executor.submit(
                    _collect_rollout_chunk_worker,
                    {
                        "config": asdict(config),
                        "model_state_dict": cpu_model_state,
                        "episode_specs": [(episode_idx, episode_seed)],
                        "task_seed": int(task_seed),
                        "update_index": int(update_index),
                        "trace_dir": str(trace_dir) if trace_dir is not None else "",
                    },
                )
                active_futures[future] = {
                    "slot_index": int(slot_index),
                    "episode_index": int(episode_idx),
                    "episode_seed": int(episode_seed),
                    "task_seed": int(task_seed),
                    "started_at": float(slot_started_at),
                    "progress_path": _episode_progress_path(
                        trace_dir=trace_dir,
                        update_index=int(update_index),
                        episode_index=int(episode_idx),
                        episode_seed=int(episode_seed),
                    ),
                }

            for slot_index in range(1, slot_count + 1):
                if not pending_episode_specs:
                    break
                _submit_episode_to_slot(
                    slot_index=slot_index,
                    episode_spec=pending_episode_specs.popleft(),
                )

            while active_futures:
                wait_timeout: Optional[float] = None
                watchdog_now: Optional[float] = None
                if watchdog_timeout_seconds > 0.0:
                    watchdog_now = time.perf_counter()
                    remaining_seconds = [
                        max(
                            0.0,
                            watchdog_timeout_seconds - (
                                float(watchdog_now) - float(info["started_at"])
                            ),
                        )
                        for info in active_futures.values()
                    ]
                    wait_timeout = min(remaining_seconds) if remaining_seconds else 0.0
                done, _ = wait(
                    tuple(active_futures.keys()),
                    return_when=FIRST_COMPLETED,
                    timeout=wait_timeout,
                )
                if not done and watchdog_timeout_seconds > 0.0 and watchdog_now is not None:
                    timed_out_entries: List[Tuple[Any, Dict[str, Any], float]] = []
                    for future, info in active_futures.items():
                        slot_elapsed_seconds = float(watchdog_now - float(info["started_at"]))
                        if slot_elapsed_seconds > watchdog_timeout_seconds:
                            timed_out_entries.append((future, dict(info), slot_elapsed_seconds))
                    if timed_out_entries:
                        timed_out_entries.sort(key=lambda item: float(item[1].get("started_at", 0.0)))
                        timed_out_future, timed_out_info, slot_elapsed_seconds = timed_out_entries[0]
                        del timed_out_future
                        progress_path_value = timed_out_info.get("progress_path")
                        progress_path = (
                            progress_path_value
                            if isinstance(progress_path_value, Path)
                            else Path(progress_path_value)
                            if progress_path_value
                            else None
                        )
                        progress_snapshot = _read_json_snapshot(progress_path)
                        progress_summary = _format_watchdog_progress_summary(progress_snapshot)
                        for active_future in tuple(active_futures.keys()):
                            cancel = getattr(active_future, "cancel", None)
                            if callable(cancel):
                                try:
                                    cancel()
                                except Exception:
                                    pass
                        message = (
                            "Rollout watchdog exceeded slow_episode_trace_stop_seconds "
                            f"({watchdog_timeout_seconds:.2f}s): update={int(update_index):04d} "
                            f"slot={int(timed_out_info['slot_index'])}/{slot_count} "
                            f"episode {int(timed_out_info['episode_index']) + 1:03d}/{max(1, total_episodes):03d} "
                            f"seed={int(timed_out_info['episode_seed'])} "
                            f"task_seed={int(timed_out_info['task_seed'])} "
                            f"slot_elapsed={slot_elapsed_seconds:.2f}s {progress_summary}"
                        )
                        if progress_path is not None:
                            message += f" progress_path={progress_path}"
                        progress_bar.close()
                        _log_progress_event(message)
                        raise TimeoutError(message)
                    continue
                for future in done:
                    info = dict(active_futures.pop(future))
                    slot_elapsed_seconds = float(time.perf_counter() - float(info["started_at"]))
                    try:
                        future_results = list(future.result())
                    except Exception:
                        progress_bar.close()
                        _log_progress_event(
                            f"[update {int(update_index):04d}] slot {int(info['slot_index'])}/{slot_count} failed "
                            f"episode {int(info['episode_index']) + 1:03d}/{max(1, total_episodes):03d} "
                            f"elapsed={slot_elapsed_seconds:.2f}s"
                        )
                        raise
                    episode_results.extend(future_results)
                    if future_results:
                        episode_result = future_results[0]
                        episode_elapsed_sec = float(episode_result.elapsed_seconds)
                    else:
                        episode_elapsed_sec = 0.0
                    progress_bar.episode_completed(
                        slot_index=int(info["slot_index"]),
                        episode_index=int(info["episode_index"]),
                        episode_seed=int(info["episode_seed"]),
                        episode_elapsed_seconds=episode_elapsed_sec,
                    )
                    if pending_episode_specs:
                        _submit_episode_to_slot(
                            slot_index=int(info["slot_index"]),
                            episode_spec=pending_episode_specs.popleft(),
                        )
    finally:
        progress_bar.close()

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
    update_index: int = 0,
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
    nonempty_sequence_count = sum(
        1 for sequence in rollout_sequences if int(sequence.step_count) > 0
    )
    total_step_count = sum(int(sequence.step_count) for sequence in rollout_sequences)
    if total_step_count <= 0:
        return _zero_loss_metrics()

    units_per_epoch = max(1, int(nonempty_sequence_count))
    unit_label = "seq"
    progress_bar = _ModelUpdateProgressBar(
        update_index=int(update_index),
        total_epochs=max(0, int(config.update_epochs)),
        units_per_epoch=units_per_epoch,
        unit_label=unit_label,
    )

    epoch_policy_losses: List[float] = []
    epoch_value_losses: List[float] = []
    epoch_entropies: List[float] = []
    epoch_total_losses: List[float] = []

    try:
        for epoch_idx in range(int(config.update_epochs)):
            progress_bar.start_epoch(epoch_index=epoch_idx + 1)
            optimizer.zero_grad(set_to_none=True)
            epoch_policy_loss = 0.0
            epoch_value_loss = 0.0
            epoch_entropy = 0.0
            epoch_total_loss = 0.0
            processed_weight = 0.0

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
                logits, values, hidden = model.forward_sequence(
                    state_vec=sequence_batch["state_vec"],
                    action_features=sequence_batch["action_features"],
                    action_mask=sequence_batch["action_mask"],
                    hidden=hidden,
                )
                del hidden
                step_log_probs = torch.log_softmax(logits, dim=-1)
                step_probs = torch.softmax(logits, dim=-1)
                chosen_indices = sequence_batch["action_index"].unsqueeze(1)
                new_logprob = step_log_probs.gather(1, chosen_indices).squeeze(1)
                entropy = -(step_probs * step_log_probs).sum(dim=-1)

                old_logprob = sequence_batch["old_logprob"]
                target_return = sequence_batch["target_return"]
                advantage = sequence_batch["advantage"]

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
                sequence_weight = float(step_count) / float(total_step_count)
                (total_loss * sequence_weight).backward()
                processed_weight += sequence_weight
                epoch_policy_loss += float(policy_loss.detach().cpu().item()) * sequence_weight
                epoch_value_loss += float(value_loss.detach().cpu().item()) * sequence_weight
                epoch_entropy += float(entropy_mean.detach().cpu().item()) * sequence_weight
                epoch_total_loss += float(total_loss.detach().cpu().item()) * sequence_weight
                progress_bar.advance()

            if processed_weight <= 0.0:
                optimizer.zero_grad(set_to_none=True)
                break

            nn.utils.clip_grad_norm_(model.parameters(), float(config.max_grad_norm))
            optimizer.step()

            epoch_policy_losses.append(epoch_policy_loss)
            epoch_value_losses.append(epoch_value_loss)
            epoch_entropies.append(epoch_entropy)
            epoch_total_losses.append(epoch_total_loss)
    finally:
        progress_bar.close()

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
    requested = normalize_torch_device_spec(requested_device, allow_auto=True)
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
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
    if requested_device_lower in {"", "auto"}:
        print(
            "device policy: "
            f"requested={requested_device_lower or 'cpu'}; selected {device.type} for training and {inference_device.type} for inference."
        )
    print(
        "resolved devices: "
        f"requested={requested_device} training={device.type} inference={inference_device.type}"
    )
    if bool(getattr(config, "slow_episode_trace_enabled", False)):
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
    force_terminate_rollout_executor = False
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
            _log_progress_event(f"starting update {update_idx:04d}")
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
            try:
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
            except TimeoutError:
                force_terminate_rollout_executor = True
                raise
            update_time_sec = float(time.perf_counter() - rollout_started_at)
            model_update_started_at = time.perf_counter()
            update_stats = _update_model(
                model=model,
                optimizer=optimizer,
                rollout_sequences=rollout_sequences,
                config=config,
                device=device,
                update_index=update_idx,
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
                print(
                    _highlight_log_line(
                        (
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
                        ),
                        style="update_summary",
                    )
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
                fixed_eval_progress_bar = _FixedEvalProgressBar(
                    update_index=int(update_idx),
                    total_episodes=int(fixed_eval_episodes),
                    opponent_name=str(fixed_eval_bundle.name),
                )
                try:
                    fixed_eval_metrics = evaluate_policy_matchup(
                        policy_a=current_bundle,
                        policy_b=fixed_eval_bundle,
                        episodes=fixed_eval_episodes,
                        seed=int(config.seed) + int(update_idx) * 10_000,
                        device=inference_device,
                        deterministic=bool(getattr(config, "fixed_eval_deterministic", True)),
                        progress_callback=(
                            lambda completed_episodes, total_episodes, episode_seed:
                            fixed_eval_progress_bar.episode_completed(
                                completed_episodes=int(completed_episodes),
                                episode_seed=int(episode_seed),
                            )
                        ),
                    )
                finally:
                    fixed_eval_progress_bar.close()
                fixed_reason_text = ", ".join(
                    f"{key}:{value}"
                    for key, value in sorted(fixed_eval_metrics.terminal_reason_counts.items())
                ) or "-"
                print(
                    _highlight_log_line(
                        (
                            f"[fixed_eval {update_idx:04d}] "
                            f"vs={fixed_eval_bundle.name} episodes={fixed_eval_metrics.episodes} "
                            f"win_rate={fixed_eval_metrics.win_rate_a:.3f} "
                            f"wins={fixed_eval_metrics.wins_a}/{fixed_eval_metrics.wins_b} draws={fixed_eval_metrics.draws} "
                            f"score_diff={fixed_eval_metrics.avg_score_diff_a_minus_b:.2f} "
                            f"rounds_avg={fixed_eval_metrics.avg_completed_rounds:.2f} "
                            f"reasons={fixed_reason_text}"
                        ),
                        style="fixed_eval",
                    )
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
        _shutdown_rollout_executor(
            rollout_executor,
            force_terminate=force_terminate_rollout_executor,
        )

    return all_metrics

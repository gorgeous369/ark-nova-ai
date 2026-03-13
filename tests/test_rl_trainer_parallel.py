import copy
from contextlib import contextmanager
import json
import random
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arknova_rl.config import PPOTrainConfig
from arknova_rl.trainer import (
    EpisodeRolloutResult,
    RolloutSequence,
    _EpisodeProgressTracker,
    _SlowEpisodeTraceWriter,
    _collect_episode_rollout,
    _collect_rollout,
    _update_model,
    _build_model_and_encoders,
    _collect_rollout_chunk_worker,
    _episode_progress_path,
    _format_watchdog_progress_summary,
    _state_dict_to_cpu,
)


def test_collect_rollout_chunk_worker_returns_episode_results(monkeypatch):
    config = PPOTrainConfig(seed=91, episodes_per_update=2, rollout_workers=2)
    config.resolve_algo_flags()
    observed_specs = []
    loaded_state_dicts = []

    class _FakeModel:
        def load_state_dict(self, state_dict):
            loaded_state_dicts.append(dict(state_dict))

        def eval(self):
            return self

    def _fake_collect_episode_rollout(**kwargs):
        observed_specs.append((int(kwargs["episode_index"]), int(kwargs["episode_seed"])))
        return EpisodeRolloutResult(
            episode_index=int(kwargs["episode_index"]),
            episode_seed=int(kwargs["episode_seed"]),
            rollout_sequences=[
                RolloutSequence(
                    sequence_key=(int(kwargs["episode_index"]), 0),
                    actor_id=0,
                    state_vec=torch.zeros((1, 1), dtype=torch.float32).numpy(),
                    action_features=torch.zeros((1, 1, 1), dtype=torch.float32).numpy(),
                    action_mask=np.ones((1, 1), dtype=np.bool_),
                    action_count=np.ones((1,), dtype=np.int32),
                    action_index=np.zeros((1,), dtype=np.int64),
                    old_logprob=np.zeros((1,), dtype=np.float32),
                    old_value=np.zeros((1,), dtype=np.float32),
                    reward=np.zeros((1,), dtype=np.float32),
                )
            ],
            step_count=1,
            completed_rounds=1,
            terminal_abs_diffs=[0.0],
            terminal_reason="fake_terminal_reason",
            elapsed_seconds=0.0,
        )

    monkeypatch.setattr(
        "arknova_rl.trainer._collect_episode_rollout",
        _fake_collect_episode_rollout,
    )
    monkeypatch.setattr(
        "arknova_rl.trainer._build_model_and_encoders",
        lambda **kwargs: (_FakeModel(), SimpleNamespace(), SimpleNamespace()),
    )

    results = _collect_rollout_chunk_worker(
        {
            "config": asdict(config),
            "model_state_dict": {"fake_weight": torch.tensor([1.0])},
            "episode_specs": [(0, 111), (1, 222)],
            "task_seed": 333,
        }
    )

    assert len(results) == 2
    assert {result.episode_index for result in results} == {0, 1}
    assert observed_specs == [(0, 111), (1, 222)]
    assert all(result.rollout_sequences for result in results)
    assert all(result.completed_rounds > 0 for result in results)
    assert all(result.elapsed_seconds >= 0.0 for result in results)
    assert len(loaded_state_dicts) == 1
    assert float(loaded_state_dicts[0]["fake_weight"].item()) == 1.0


def test_slow_episode_trace_writer_records_only_within_elapsed_window(tmp_path):
    writer = _SlowEpisodeTraceWriter(
        trace_dir=tmp_path,
        update_index=8,
        episode_index=3,
        episode_seed=77,
        start_after_seconds=300.0,
        stop_after_seconds=480.0,
    )

    writer.append(
        {"kind": "action", "name": "too_early"},
        elapsed_seconds=299.0,
        action_count=1,
    )
    writer.append(
        {"kind": "action", "name": "inside_window_a"},
        elapsed_seconds=300.0,
        action_count=2,
    )
    writer.append(
        {"kind": "action", "name": "inside_window_b"},
        elapsed_seconds=479.5,
        action_count=3,
    )
    writer.append(
        {"kind": "action", "name": "too_late"},
        elapsed_seconds=480.5,
        action_count=4,
    )

    trace_path = writer.finalize({"kind": "terminal"})
    assert trace_path

    with open(trace_path, "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]

    assert [record["kind"] for record in records] == [
        "trace_started",
        "action",
        "action",
        "trace_stopped",
    ]
    assert [record.get("name") for record in records if record["kind"] == "action"] == [
        "inside_window_a",
        "inside_window_b",
    ]
    assert records[0]["trace_start_seconds"] == 300.0
    assert records[0]["trace_stop_seconds"] == 480.0
    assert records[-1]["reason"] == "elapsed_limit"


def test_collect_episode_rollout_aborts_training_when_elapsed_exceeds_stop_seconds(
    monkeypatch,
    tmp_path,
):
    config = PPOTrainConfig(
        slow_episode_trace_enabled=True,
        slow_episode_trace_start_seconds=300.0,
        slow_episode_trace_stop_seconds=480.0,
    )
    config.resolve_algo_flags()

    class _FakeModel:
        def init_hidden(self, batch_size, *, device):
            return (
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
            )

        def forward_step(self, **kwargs):
            logits = torch.zeros((1, 1), dtype=torch.float32)
            value = torch.zeros((1, 1), dtype=torch.float32)
            hidden = kwargs.get("hidden")
            return logits, value, hidden

    class _FakeObsEncoder:
        def encode_from_state(self, state, actor_id):
            return np.zeros((1,), dtype=np.float32)

    class _FakeActionEncoder:
        def encode_many(self, legal):
            return np.zeros((len(legal), 1), dtype=np.float32)

    class _FakePlayer:
        def __init__(self):
            self.name = "P1"
            self.money = 0
            self.appeal = 0
            self.conservation = 0
            self.reputation = 0
            self.x_tokens = 0

    class _FakeState:
        def __init__(self):
            self.players = [_FakePlayer()]
            self.pending_decision_kind = ""
            self.pending_decision_player_id = None
            self.current_player = 0
            self.endgame_trigger_player = None
            self.effect_log = []
            self.turn_index = 0
            self.forced_game_over_reason = ""

        def game_over(self):
            return False

    class _FakeAction:
        def __init__(self):
            self.details = {}

        def __str__(self):
            return "fake_action"

    fake_state = _FakeState()
    fake_action = _FakeAction()

    monkeypatch.setattr("arknova_rl.trainer.main.setup_game", lambda **kwargs: fake_state)
    monkeypatch.setattr("arknova_rl.trainer._current_actor_id", lambda state: 0)
    monkeypatch.setattr("arknova_rl.trainer.main.legal_actions", lambda *args, **kwargs: [fake_action])
    monkeypatch.setattr(
        "arknova_rl.trainer.main.apply_action",
        lambda state, action: setattr(state, "turn_index", int(state.turn_index) + 1),
    )
    monkeypatch.setattr("arknova_rl.trainer.main._completed_rounds", lambda state: 7)
    monkeypatch.setattr("arknova_rl.trainer.main._progress_score", lambda player: 0.0)
    monkeypatch.setattr(
        "arknova_rl.trainer.time.perf_counter",
        iter([0.0, 0.0, 10.0, 10.0, 11.0, 11.0, 12.0, 13.0, 13.0, 481.0, 481.0, 481.0, 481.0]).__next__,
    )

    with pytest.raises(TimeoutError, match="slow_episode_trace_stop_seconds.*action='fake_action'"):
        _collect_episode_rollout(
            model=_FakeModel(),
            obs_encoder=_FakeObsEncoder(),
            action_encoder=_FakeActionEncoder(),
            config=config,
            device=torch.device("cpu"),
            episode_index=2,
            episode_seed=12345,
            update_index=8,
            trace_dir=tmp_path,
        )

    trace_files = sorted(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    with open(trace_files[0], "r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    assert [record["kind"] for record in records] == [
        "trace_started",
        "action",
        "trace_stopped",
    ]
    assert records[1]["action"] == "fake_action"
    assert records[1]["timing_seconds"]["legal_actions"] == 1.0
    assert records[1]["timing_seconds"]["encode"] == 1.0
    assert records[1]["timing_seconds"]["forward"] == 1.0
    assert records[1]["timing_seconds"]["apply_action"] == 468.0
    assert records[-1]["reason"] == "episode_timeout"


def test_collect_rollout_parallel_refills_next_episode_on_first_completed_slot(monkeypatch):
    config = PPOTrainConfig(seed=91, episodes_per_update=4, rollout_workers=2)
    config.resolve_algo_flags()

    started = []
    completed_order = [1, 0, 2, 3]
    futures_by_episode = {}

    class _FakeModel:
        def state_dict(self):
            return {"fake_weight": torch.tensor([1.0])}

    class _FakeFuture:
        def __init__(self, episode_index: int, result: EpisodeRolloutResult):
            self.episode_index = int(episode_index)
            self._result = result

        def result(self):
            return [self._result]

        def __hash__(self):
            return hash(id(self))

    class _FakeExecutor:
        def submit(self, fn, payload):
            episode_index, episode_seed = payload["episode_specs"][0]
            result = EpisodeRolloutResult(
                episode_index=int(episode_index),
                episode_seed=int(episode_seed),
                rollout_sequences=[
                    RolloutSequence(
                        sequence_key=(int(episode_index), 0),
                        actor_id=0,
                        state_vec=torch.zeros((1, 1), dtype=torch.float32).numpy(),
                        action_features=torch.zeros((1, 1, 1), dtype=torch.float32).numpy(),
                        action_mask=np.ones((1, 1), dtype=np.bool_),
                        action_count=np.ones((1,), dtype=np.int32),
                        action_index=np.zeros((1,), dtype=np.int64),
                        old_logprob=np.zeros((1,), dtype=np.float32),
                        old_value=np.zeros((1,), dtype=np.float32),
                        reward=np.zeros((1,), dtype=np.float32),
                    )
                ],
                step_count=1,
                completed_rounds=1,
                terminal_abs_diffs=[0.0],
                terminal_reason="fake_terminal_reason",
                elapsed_seconds=float(int(episode_index) + 1),
            )
            future = _FakeFuture(int(episode_index), result)
            futures_by_episode[int(episode_index)] = future
            return future

    def _fake_wait(futures, return_when, timeout=None):
        del return_when
        del timeout
        for episode_index in list(completed_order):
            future = futures_by_episode.get(int(episode_index))
            if future is not None and future in futures:
                completed_order.remove(int(episode_index))
                return {future}, {item for item in futures if item is not future}
        raise AssertionError("No matching future available for fake wait ordering.")

    monkeypatch.setattr("arknova_rl.trainer.wait", _fake_wait)
    monkeypatch.setattr("arknova_rl.trainer._log_progress_event", lambda message: None)
    monkeypatch.setattr(
        "arknova_rl.trainer._RolloutProgressBar.episode_completed",
        lambda self, **kwargs: None,
    )
    monkeypatch.setattr(
        "arknova_rl.trainer._RolloutProgressBar.slot_started",
        lambda self, **kwargs: started.append((int(kwargs["slot_index"]), int(kwargs["episode_index"]))),
    )

    rollout_sequences, rollout_stats = _collect_rollout(
        model=_FakeModel(),
        obs_encoder=SimpleNamespace(),
        action_encoder=SimpleNamespace(),
        config=config,
        rng=random.Random(7),
        device=torch.device("cpu"),
        update_index=12,
        trace_dir=None,
        rollout_executor=_FakeExecutor(),
    )

    assert started == [
        (1, 0),
        (2, 1),
        (2, 2),
        (1, 3),
    ]
    assert len(rollout_sequences) == 4
    assert rollout_stats["episode_count"] == 4
    assert rollout_stats["step_count"] == 4


def test_collect_episode_rollout_persists_current_action_in_progress_snapshot(monkeypatch, tmp_path):
    config = PPOTrainConfig(
        slow_episode_trace_enabled=True,
        slow_episode_trace_start_seconds=300.0,
        slow_episode_trace_stop_seconds=480.0,
    )
    config.resolve_algo_flags()

    class _FakeModel:
        def init_hidden(self, batch_size, *, device):
            return (
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
            )

        def forward_step(self, **kwargs):
            logits = torch.zeros((1, 1), dtype=torch.float32)
            value = torch.zeros((1, 1), dtype=torch.float32)
            hidden = kwargs.get("hidden")
            return logits, value, hidden

    class _FakeObsEncoder:
        def encode_from_state(self, state, actor_id):
            return np.zeros((1,), dtype=np.float32)

    class _FakeActionEncoder:
        def encode_many(self, legal):
            return np.zeros((len(legal), 1), dtype=np.float32)

    class _FakePlayer:
        def __init__(self):
            self.name = "P1"
            self.money = 0
            self.appeal = 0
            self.conservation = 0
            self.reputation = 0
            self.x_tokens = 0

    class _FakeState:
        def __init__(self):
            self.players = [_FakePlayer()]
            self.pending_decision_kind = ""
            self.pending_decision_player_id = None
            self.current_player = 0
            self.endgame_trigger_player = None
            self.effect_log = []
            self.turn_index = 0
            self.forced_game_over_reason = ""

        def game_over(self):
            return False

    class _FakeAction:
        type = "main_action"
        card_name = "animals"
        value = 0

        def __init__(self):
            self.details = {"action_label": "play Snow Leopard"}

        def __str__(self):
            return "animals | play Snow Leopard"

    fake_state = _FakeState()
    fake_action = _FakeAction()

    monkeypatch.setattr("arknova_rl.trainer.main.setup_game", lambda **kwargs: fake_state)
    monkeypatch.setattr("arknova_rl.trainer._current_actor_id", lambda state: 0)
    monkeypatch.setattr("arknova_rl.trainer.main.legal_actions", lambda *args, **kwargs: [fake_action])
    monkeypatch.setattr(
        "arknova_rl.trainer.main.apply_action",
        lambda state, action: (_ for _ in ()).throw(RuntimeError("apply_action hung")),
    )
    monkeypatch.setattr("arknova_rl.trainer.main._completed_rounds", lambda state: 7)
    monkeypatch.setattr("arknova_rl.trainer.main._progress_score", lambda player: 0.0)

    with pytest.raises(RuntimeError, match="apply_action hung"):
        _collect_episode_rollout(
            model=_FakeModel(),
            obs_encoder=_FakeObsEncoder(),
            action_encoder=_FakeActionEncoder(),
            config=config,
            device=torch.device("cpu"),
            episode_index=2,
            episode_seed=12345,
            update_index=8,
            trace_dir=tmp_path,
        )

    progress_path = _episode_progress_path(
        trace_dir=tmp_path,
        update_index=8,
        episode_index=2,
        episode_seed=12345,
    )
    assert progress_path is not None and progress_path.exists()
    with open(progress_path, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["stage"] == "episode_exception"
    assert snapshot["actor_name"] == "P1"
    assert snapshot["current_action"]["card_name"] == "animals"
    assert snapshot["current_action"]["rendered"] == "animals | play Snow Leopard"
    assert snapshot["error_type"] == "RuntimeError"


def test_collect_episode_rollout_persists_legal_actions_profile_snapshot(monkeypatch, tmp_path):
    config = PPOTrainConfig(
        slow_episode_trace_enabled=True,
        slow_episode_trace_start_seconds=300.0,
        slow_episode_trace_stop_seconds=480.0,
    )
    config.resolve_algo_flags()

    class _FakeModel:
        def init_hidden(self, batch_size, *, device):
            return (
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
            )

        def forward_step(self, **kwargs):
            logits = torch.zeros((1, 1), dtype=torch.float32)
            value = torch.zeros((1, 1), dtype=torch.float32)
            hidden = kwargs.get("hidden")
            return logits, value, hidden

    class _FakeObsEncoder:
        def encode_from_state(self, state, actor_id):
            return np.zeros((1,), dtype=np.float32)

    class _FakeActionEncoder:
        def encode_many(self, legal):
            return np.zeros((len(legal), 1), dtype=np.float32)

    class _FakePlayer:
        def __init__(self):
            self.name = "P2"
            self.money = 0
            self.appeal = 0
            self.conservation = 0
            self.reputation = 0
            self.x_tokens = 0

    class _FakeState:
        def __init__(self):
            self.players = [_FakePlayer(), _FakePlayer()]
            self.pending_decision_kind = ""
            self.pending_decision_player_id = None
            self.current_player = 1
            self.endgame_trigger_player = None
            self.effect_log = []
            self.turn_index = 159
            self.forced_game_over_reason = ""

        def game_over(self):
            return False

    fake_state = _FakeState()
    callback_holder = {"callback": None, "snapshot_copy": None}

    @contextmanager
    def _fake_legal_actions_profiling(callback, *, snapshot_copy=True):
        callback_holder["callback"] = callback
        callback_holder["snapshot_copy"] = bool(snapshot_copy)
        try:
            yield
        finally:
            callback_holder["callback"] = None

    def _fake_legal_actions(*args, **kwargs):
        assert callback_holder["callback"] is not None
        callback_holder["callback"](
            {
                "phase": "list_legal_build_options",
                "current_branch": "build",
                "last_subfunction": "list_legal_build_options",
                "branch_metrics": {
                    "build": {
                        "elapsed_seconds": 12.5,
                        "template_action_count": 1,
                        "concrete_action_count": 17,
                    }
                },
                "abstract_action_count": 31,
                "annotated_action_count": 6,
                "pruned_action_count": 5,
                "concrete_action_count": 17,
            }
        )
        raise RuntimeError("legal_actions hung")

    monkeypatch.setattr("arknova_rl.trainer.main.setup_game", lambda **kwargs: fake_state)
    monkeypatch.setattr("arknova_rl.trainer._current_actor_id", lambda state: 1)
    monkeypatch.setattr("arknova_rl.trainer.main.legal_actions_profiling", _fake_legal_actions_profiling)
    monkeypatch.setattr("arknova_rl.trainer.main.legal_actions", _fake_legal_actions)
    monkeypatch.setattr("arknova_rl.trainer.main._completed_rounds", lambda state: 79)
    monkeypatch.setattr("arknova_rl.trainer.main._progress_score", lambda player: 0.0)

    with pytest.raises(RuntimeError, match="legal_actions hung"):
        _collect_episode_rollout(
            model=_FakeModel(),
            obs_encoder=_FakeObsEncoder(),
            action_encoder=_FakeActionEncoder(),
            config=config,
            device=torch.device("cpu"),
            episode_index=15,
            episode_seed=1347467,
            update_index=42,
            trace_dir=tmp_path,
        )
    assert callback_holder["snapshot_copy"] is False

    progress_path = _episode_progress_path(
        trace_dir=tmp_path,
        update_index=42,
        episode_index=15,
        episode_seed=1347467,
    )
    assert progress_path is not None and progress_path.exists()
    with open(progress_path, "r", encoding="utf-8") as handle:
        snapshot = json.load(handle)
    assert snapshot["stage"] == "episode_exception"
    assert snapshot["actor_name"] == "P2"
    assert snapshot["legal_actions_profile"]["current_branch"] == "build"
    assert snapshot["legal_actions_profile"]["last_subfunction"] == "list_legal_build_options"
    assert snapshot["legal_actions_profile"]["branch_metrics"]["build"]["concrete_action_count"] == 17
    assert snapshot["error_type"] == "RuntimeError"


def test_collect_rollout_parallel_watchdog_reports_progress_snapshot(monkeypatch, tmp_path):
    config = PPOTrainConfig(
        seed=91,
        episodes_per_update=2,
        rollout_workers=2,
        slow_episode_trace_enabled=True,
        slow_episode_trace_start_seconds=30.0,
        slow_episode_trace_stop_seconds=60.0,
    )
    config.resolve_algo_flags()

    class _FakeModel:
        def state_dict(self):
            return {"fake_weight": torch.tensor([1.0])}

    class _FakeFuture:
        def __hash__(self):
            return hash(id(self))

        def cancel(self):
            return True

    class _FakeExecutor:
        def submit(self, fn, payload):
            del fn
            episode_index, episode_seed = payload["episode_specs"][0]
            progress_path = _episode_progress_path(
                trace_dir=tmp_path,
                update_index=21,
                episode_index=int(episode_index),
                episode_seed=int(episode_seed),
            )
            assert progress_path is not None
            with open(progress_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "stage": "before_apply_action",
                        "actor_name": "P1",
                        "current_action": {
                            "rendered": "animals | play Tiger",
                            "card_name": "animals",
                        },
                        "completed_rounds": 12,
                        "pending": "",
                    },
                    handle,
                )
            return _FakeFuture()

    perf_counter_values = iter([0.0, 0.5, 61.0])
    monkeypatch.setattr("arknova_rl.trainer.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(
        "arknova_rl.trainer.wait",
        lambda futures, return_when, timeout=None: (set(), set(futures)),
    )
    monkeypatch.setattr("arknova_rl.trainer._log_progress_event", lambda message: None)

    with pytest.raises(TimeoutError, match="play Tiger.*progress_path="):
        _collect_rollout(
            model=_FakeModel(),
            obs_encoder=SimpleNamespace(),
            action_encoder=SimpleNamespace(),
            config=config,
            rng=random.Random(7),
            device=torch.device("cpu"),
            update_index=21,
            trace_dir=tmp_path,
            rollout_executor=_FakeExecutor(),
        )


def test_format_watchdog_progress_summary_includes_build_profile_details():
    summary = _format_watchdog_progress_summary(
        {
            "stage": "inside_legal_actions",
            "actor_name": "P1",
            "pending": "",
            "completed_rounds": 53,
            "legal_actions_profile": {
                "phase": "enumerate_build_bonus_choice_variants",
                "current_branch": "build",
                "last_subfunction": "_enumerate_build_bonus_choice_variants",
                "build_profile": {
                    "mode": "expand_bonus",
                    "current_option": {
                        "rendered": "enclosure_3 cells=[(0,0),(1,0),(2,0)]",
                    },
                    "current_option_label": "enclosure_3 cells=[(0,0),(1,0),(2,0)]",
                    "placement_bonuses": ["action_to_slot_1", "card_in_reputation_range"],
                    "current_bonus": "card_in_reputation_range",
                    "bonus_index": 2,
                    "bonus_count": 2,
                    "variant_count_before": 5,
                    "variant_count_after": 10,
                    "deduped_variant_count": 10,
                    "current_sequence_length": 1,
                    "current_sequence_label": "enclosure_3 cells=[(0,0),(1,0),(2,0)]",
                },
            },
        }
    )

    assert "build_mode=expand_bonus" in summary
    assert "build_option='enclosure_3 cells=[(0,0),(1,0),(2,0)]'" in summary
    assert "build_bonus=card_in_reputation_range" in summary
    assert "build_bonus_step=2/2" in summary
    assert "build_variants_before=5" in summary
    assert "build_variants_after=10" in summary


def test_episode_progress_tracker_throttles_inside_legal_actions_writes(monkeypatch, tmp_path):
    writes = []
    perf_counter_values = iter([1.0, 1.1, 1.2, 1.7])

    monkeypatch.setattr("arknova_rl.trainer.time.perf_counter", lambda: next(perf_counter_values))
    monkeypatch.setattr(
        "arknova_rl.trainer._write_json_snapshot",
        lambda path, payload: writes.append((path, copy.deepcopy(payload))),
    )

    tracker = _EpisodeProgressTracker(
        trace_dir=tmp_path,
        update_index=4,
        episode_index=2,
        episode_seed=12345,
        enabled=True,
        inside_legal_actions_min_write_interval_seconds=0.5,
    )

    tracker.update(stage="before_legal_actions")
    tracker.update(stage="inside_legal_actions", legal_actions_profile={"phase": "scan_option"})
    tracker.update(stage="inside_legal_actions", legal_actions_profile={"phase": "expand_bonus"})
    tracker.update(stage="inside_legal_actions", legal_actions_profile={"phase": "completed_bonus_variants"})

    assert len(writes) == 3
    assert writes[1][1]["legal_actions_profile"]["phase"] == "scan_option"
    assert writes[2][1]["legal_actions_profile"]["phase"] == "completed_bonus_variants"


def test_update_model_drives_progress_bar(monkeypatch):
    events = []

    class _FakeProgressBar:
        def __init__(self, *, update_index, total_epochs, units_per_epoch, unit_label):
            events.append(("init", int(update_index), int(total_epochs), int(units_per_epoch), str(unit_label)))

        def start_epoch(self, *, epoch_index):
            events.append(("start", int(epoch_index)))

        def advance(self):
            events.append(("advance",))

        def close(self):
            events.append(("close",))

    class _TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state_head = torch.nn.Linear(1, 1)
            self.action_head = torch.nn.Linear(1, 1)

        def init_hidden(self, batch_size, *, device):
            return (
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
                torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
            )

        def forward_step(self, *, state_vec, action_features, action_mask, hidden=None):
            del action_mask
            logits = self.action_head(action_features).squeeze(-1)
            values = self.state_head(state_vec).squeeze(-1)
            return logits, values, hidden

        def forward_sequence(self, *, state_vec, action_features, action_mask, hidden=None):
            return self.forward_step(
                state_vec=state_vec,
                action_features=action_features,
                action_mask=action_mask,
                hidden=hidden,
            )

    monkeypatch.setattr("arknova_rl.trainer._ModelUpdateProgressBar", _FakeProgressBar)

    model = _TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sequence = RolloutSequence(
        sequence_key=(0, 0),
        actor_id=0,
        state_vec=np.asarray([[0.0], [1.0]], dtype=np.float32),
        action_features=np.asarray([[[0.0]], [[1.0]]], dtype=np.float32),
        action_mask=np.asarray([[True], [True]], dtype=np.bool_),
        action_count=np.asarray([1, 1], dtype=np.int32),
        action_index=np.asarray([0, 0], dtype=np.int64),
        old_logprob=np.asarray([0.0, 0.0], dtype=np.float32),
        old_value=np.asarray([0.0, 0.0], dtype=np.float32),
        reward=np.asarray([1.0, 1.0], dtype=np.float32),
        advantage=np.asarray([1.0, 0.5], dtype=np.float32),
        return_=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    config = PPOTrainConfig(update_epochs=2)
    config.resolve_algo_flags()

    metrics = _update_model(
        model=model,
        optimizer=optimizer,
        rollout_sequences=[sequence],
        config=config,
        device=torch.device("cpu"),
        update_index=47,
    )

    assert metrics["total_loss"] == pytest.approx(metrics["total_loss"])
    assert events[0] == ("init", 47, 2, 1, "seq")
    assert ("start", 1) in events
    assert ("start", 2) in events
    assert events.count(("advance",)) == 2
    assert events[-1] == ("close",)


def test_update_model_recurrent_path_handles_large_action_indices():
    class _TinyRecurrentModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state_head = torch.nn.Linear(1, 1)
            self.logit_bias = torch.nn.Parameter(torch.zeros((512,), dtype=torch.float32))

        def init_hidden(self, batch_size, *, device):
            del batch_size
            return (
                torch.zeros((1, 1, 1), dtype=torch.float32, device=device),
                torch.zeros((1, 1, 1), dtype=torch.float32, device=device),
            )

        def forward_step(self, *, state_vec, action_features, action_mask, hidden=None):
            del action_features
            logits = self.logit_bias.unsqueeze(0).expand(state_vec.shape[0], -1)
            logits = logits.masked_fill(~action_mask, -1e9)
            values = self.state_head(state_vec).squeeze(-1)
            return logits, values, hidden

        def forward_sequence(self, *, state_vec, action_features, action_mask, hidden=None):
            return self.forward_step(
                state_vec=state_vec,
                action_features=action_features,
                action_mask=action_mask,
                hidden=hidden,
            )

    model = _TinyRecurrentModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    sequence = RolloutSequence(
        sequence_key=(0, 0),
        actor_id=0,
        state_vec=np.asarray([[0.0], [1.0]], dtype=np.float32),
        action_features=np.zeros((2, 512, 1), dtype=np.float32),
        action_mask=np.ones((2, 512), dtype=np.bool_),
        action_count=np.asarray([512, 512], dtype=np.int32),
        action_index=np.asarray([511, 510], dtype=np.int64),
        old_logprob=np.asarray([0.0, 0.0], dtype=np.float32),
        old_value=np.asarray([0.0, 0.0], dtype=np.float32),
        reward=np.asarray([1.0, 1.0], dtype=np.float32),
        advantage=np.asarray([1.0, 0.5], dtype=np.float32),
        return_=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    config = PPOTrainConfig(update_epochs=1)
    config.resolve_algo_flags()

    metrics = _update_model(
        model=model,
        optimizer=optimizer,
        rollout_sequences=[sequence],
        config=config,
        device=torch.device("cpu"),
        update_index=55,
    )

    assert metrics["total_loss"] == pytest.approx(metrics["total_loss"])

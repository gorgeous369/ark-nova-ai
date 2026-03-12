import json
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from arknova_rl.config import PPOTrainConfig
from arknova_rl.trainer import (
    EpisodeRolloutResult,
    RolloutSequence,
    _SlowEpisodeTraceWriter,
    _collect_episode_rollout,
    _build_model_and_encoders,
    _collect_rollout_chunk_worker,
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
        slow_episode_trace_start_seconds=300.0,
        slow_episode_trace_stop_seconds=480.0,
    )
    config.resolve_algo_flags()

    class _FakeModel:
        use_lstm = False

        def forward_step(self, **kwargs):
            logits = torch.zeros((1, 1), dtype=torch.float32)
            value = torch.zeros((1, 1), dtype=torch.float32)
            return logits, value, None

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
        iter([0.0, 10.0, 11.0, 12.0, 13.0, 481.0]).__next__,
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

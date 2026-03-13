import importlib.util
from pathlib import Path

from arknova_rl.config import PPOTrainConfig


def _load_train_self_play_module():
    module_path = Path("tools/rl/train_self_play.py")
    spec = importlib.util.spec_from_file_location("train_self_play_cli_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_train_self_play_cli_defaults_follow_config_defaults():
    module = _load_train_self_play_module()
    args = module.parse_args([])
    defaults = PPOTrainConfig()

    assert args.seed == defaults.seed
    assert args.device == defaults.device
    assert args.rollout_workers == defaults.rollout_workers
    assert args.updates == defaults.total_updates
    assert args.episodes_per_update == defaults.episodes_per_update
    assert args.lr == defaults.learning_rate
    assert args.hidden_size == defaults.hidden_size
    assert args.lstm_size == defaults.lstm_size
    assert args.action_hidden_size == defaults.action_hidden_size
    assert args.step_reward_scale == defaults.step_reward_scale
    assert args.terminal_reward_scale == defaults.terminal_reward_scale
    assert args.endgame_trigger_reward == defaults.endgame_trigger_reward
    assert args.endgame_speed_bonus == defaults.endgame_speed_bonus
    assert args.terminal_win_bonus == defaults.terminal_win_bonus
    assert args.terminal_loss_penalty == defaults.terminal_loss_penalty
    assert args.checkpoint_interval == defaults.checkpoint_interval
    assert args.slow_episode_trace is False
    assert args.slow_episode_trace_start_seconds is None
    assert args.slow_episode_trace_stop_seconds is None
    assert args.fixed_eval_interval == defaults.fixed_eval_interval
    assert args.fixed_eval_episodes == defaults.fixed_eval_episodes
    assert args.fixed_eval_opponent == defaults.fixed_eval_opponent

    enabled, start_seconds, stop_seconds = module._resolve_slow_episode_trace_settings(args, defaults=defaults)
    assert enabled is False
    assert start_seconds == defaults.slow_episode_trace_start_seconds
    assert stop_seconds == defaults.slow_episode_trace_stop_seconds


def test_train_self_play_cli_slow_episode_trace_flag_uses_default_window():
    module = _load_train_self_play_module()
    args = module.parse_args(["--slow-episode-trace"])

    defaults = PPOTrainConfig()
    enabled, start_seconds, stop_seconds = module._resolve_slow_episode_trace_settings(args, defaults=defaults)

    assert enabled is True
    assert start_seconds == defaults.slow_episode_trace_start_seconds
    assert stop_seconds == defaults.slow_episode_trace_stop_seconds


def test_train_self_play_cli_explicit_trace_seconds_enable_trace_without_flag():
    module = _load_train_self_play_module()
    args = module.parse_args(["--slow-episode-trace-start-seconds", "120"])

    defaults = PPOTrainConfig()
    enabled, start_seconds, stop_seconds = module._resolve_slow_episode_trace_settings(args, defaults=defaults)

    assert enabled is True
    assert start_seconds == 120.0
    assert stop_seconds == defaults.slow_episode_trace_stop_seconds

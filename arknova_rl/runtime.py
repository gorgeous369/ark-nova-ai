"""Shared runtime helpers for Ark Nova RL training and evaluation."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

import main

from .config import PPOTrainConfig
from .encoding import ActionFeatureEncoder, ObservationEncoder
from .model import MaskedActorCritic


def current_actor_id(state: main.GameState) -> int:
    if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None:
        return int(state.pending_decision_player_id)
    return int(state.current_player)


def restore_config(raw_config: Any) -> PPOTrainConfig:
    config = PPOTrainConfig()
    if isinstance(raw_config, dict):
        known_fields = {field.name for field in fields(PPOTrainConfig)}
        if (
            "slow_episode_trace_start_seconds" not in raw_config
            and "slow_episode_trace_seconds" in raw_config
        ):
            config.slow_episode_trace_start_seconds = float(
                raw_config.get("slow_episode_trace_seconds") or 0.0
            )
        for key, value in raw_config.items():
            key_text = str(key)
            if key_text in known_fields:
                setattr(config, key_text, value)
    config.resolve_algo_flags()
    return config


def load_torch_checkpoint(path: Path, *, device: torch.device) -> Dict[str, Any]:
    load_kwargs: Dict[str, Any] = {
        "map_location": device,
    }
    try:
        checkpoint = torch.load(path, weights_only=False, **load_kwargs)
    except TypeError:
        checkpoint = torch.load(path, **load_kwargs)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint payload must be a dict.")
    return checkpoint


def build_model_and_encoders(
    *,
    config: PPOTrainConfig,
    device: torch.device,
) -> Tuple[MaskedActorCritic, ObservationEncoder, ActionFeatureEncoder]:
    obs_encoder = ObservationEncoder()
    action_encoder = ActionFeatureEncoder()

    probe_state = main.setup_game(seed=config.seed, player_names=["P1", "P2"])
    probe_actor_id = current_actor_id(probe_state)
    probe_actor = probe_state.players[probe_actor_id]
    legal = main.legal_actions(
        probe_actor,
        state=probe_state,
        player_id=probe_actor_id,
    )
    if not legal:
        raise RuntimeError("Probe state has no legal actions; cannot infer model dims.")

    state_vec = obs_encoder.encode_from_state(probe_state, probe_actor_id)
    action_features = action_encoder.encode_many(legal)
    model = MaskedActorCritic(
        state_dim=int(state_vec.shape[0]),
        action_dim=int(action_features.shape[1]),
        hidden_size=int(config.hidden_size),
        lstm_size=int(config.lstm_size),
        action_hidden_size=int(config.action_hidden_size),
        use_lstm=bool(config.use_lstm),
    ).to(device)
    return model, obs_encoder, action_encoder

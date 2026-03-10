"""Checkpoint-based fixed evaluation helpers for self-play policies."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

import main

from .config import PPOTrainConfig
from .encoding import ActionFeatureEncoder, ObservationEncoder
from .model import MaskedActorCritic


@dataclass
class PolicyBundle:
    name: str
    model: MaskedActorCritic
    obs_encoder: ObservationEncoder
    action_encoder: ActionFeatureEncoder


@dataclass
class EvaluationMetrics:
    episodes: int
    wins_a: int
    wins_b: int
    draws: int
    avg_score_a: float
    avg_score_b: float
    avg_score_diff_a_minus_b: float
    avg_completed_rounds: float
    terminal_reason_counts: Dict[str, int]

    @property
    def win_rate_a(self) -> float:
        if self.episodes <= 0:
            return 0.0
        return (float(self.wins_a) + 0.5 * float(self.draws)) / float(self.episodes)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["win_rate_a"] = float(self.win_rate_a)
        return payload


def _current_actor_id(state: main.GameState) -> int:
    if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None:
        return int(state.pending_decision_player_id)
    return int(state.current_player)


def _restore_config(raw_config: Any) -> PPOTrainConfig:
    config = PPOTrainConfig()
    if isinstance(raw_config, dict):
        known_fields = {field.name for field in fields(PPOTrainConfig)}
        for key, value in raw_config.items():
            key_text = str(key)
            if key_text in known_fields:
                setattr(config, key_text, value)
    config.resolve_algo_flags()
    return config


def _load_torch_checkpoint(path: Path, *, device: torch.device) -> Dict[str, Any]:
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


def _build_model_and_encoders(
    *,
    config: PPOTrainConfig,
    device: torch.device,
) -> Tuple[MaskedActorCritic, ObservationEncoder, ActionFeatureEncoder]:
    obs_encoder = ObservationEncoder()
    action_encoder = ActionFeatureEncoder()

    probe_state = main.setup_game(seed=config.seed, player_names=["P1", "P2"])
    probe_actor_id = _current_actor_id(probe_state)
    probe_actor = probe_state.players[probe_actor_id]
    legal = main.legal_actions(probe_actor, state=probe_state, player_id=probe_actor_id)
    if not legal:
        raise RuntimeError("Probe state has no legal actions; cannot infer model dims.")

    state_vec, global_vec = obs_encoder.encode_from_state(probe_state, probe_actor_id)
    action_features = action_encoder.encode_many(legal)
    model = MaskedActorCritic(
        state_dim=int(state_vec.shape[0]),
        action_dim=int(action_features.shape[1]),
        global_state_dim=int(global_vec.shape[0]),
        hidden_size=int(config.hidden_size),
        lstm_size=int(config.lstm_size),
        action_hidden_size=int(config.action_hidden_size),
        use_lstm=bool(config.use_lstm),
        use_centralized_value=bool(config.use_centralized_value),
    ).to(device)
    return model, obs_encoder, action_encoder


def load_policy_bundle_from_checkpoint(
    checkpoint_path: Path,
    *,
    device: torch.device,
    name: Optional[str] = None,
) -> PolicyBundle:
    path = Path(checkpoint_path)
    checkpoint = _load_torch_checkpoint(path, device=device)
    config = _restore_config(checkpoint.get("config"))
    model, obs_encoder, action_encoder = _build_model_and_encoders(config=config, device=device)
    model_state = checkpoint.get("model_state_dict")
    if not isinstance(model_state, dict):
        raise ValueError(f"Checkpoint is missing model_state_dict: {path}")
    model.load_state_dict(model_state)
    model.eval()
    return PolicyBundle(
        name=str(name or path.stem),
        model=model,
        obs_encoder=obs_encoder,
        action_encoder=action_encoder,
    )


def evaluate_policy_matchup(
    *,
    policy_a: PolicyBundle,
    policy_b: PolicyBundle,
    episodes: int,
    seed: int,
    device: torch.device,
    deterministic: bool = True,
) -> EvaluationMetrics:
    wins_a = 0
    wins_b = 0
    draws = 0
    total_score_a = 0.0
    total_score_b = 0.0
    total_rounds = 0.0
    terminal_reason_counts: Counter[str] = Counter()

    prev_training_a = bool(policy_a.model.training)
    prev_training_b = bool(policy_b.model.training)
    policy_a.model.eval()
    policy_b.model.eval()

    eval_rng = np.random.default_rng(int(seed))

    try:
        for episode_idx in range(max(0, int(episodes))):
            episode_seed = int(eval_rng.integers(0, 2**31 - 1))
            state = main.setup_game(seed=episode_seed, player_names=["P1", "P2"])
            if episode_idx % 2 == 0:
                seat_policies = {0: policy_a, 1: policy_b}
                owner_by_seat = {0: "A", 1: "B"}
            else:
                seat_policies = {0: policy_b, 1: policy_a}
                owner_by_seat = {0: "B", 1: "A"}

            hidden_by_seat: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
            for seat, bundle in seat_policies.items():
                if bundle.model.use_lstm:
                    hidden_by_seat[seat] = bundle.model.init_hidden(1, device=device)
                else:
                    hidden_by_seat[seat] = None

            while str(state.pending_decision_kind or "").strip() or not state.game_over():
                actor_id = _current_actor_id(state)
                actor = state.players[actor_id]
                bundle = seat_policies[actor_id]
                legal = main.legal_actions(actor, state=state, player_id=actor_id)
                if not legal:
                    raise RuntimeError(
                        f"No legal actions during evaluation for actor={actor_id} pending={state.pending_decision_kind!r}."
                    )

                state_vec, global_vec = bundle.obs_encoder.encode_from_state(state, actor_id)
                action_features = bundle.action_encoder.encode_many(legal)
                action_mask = np.ones((len(legal),), dtype=np.float32)

                state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=device).unsqueeze(0)
                global_t = torch.as_tensor(global_vec, dtype=torch.float32, device=device).unsqueeze(0)
                action_t = torch.as_tensor(action_features, dtype=torch.float32, device=device).unsqueeze(0)
                mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=device).unsqueeze(0)

                hidden = hidden_by_seat.get(actor_id)
                with torch.no_grad():
                    logits, _, next_hidden = bundle.model.forward_step(
                        state_vec=state_t,
                        action_features=action_t,
                        action_mask=mask_t,
                        hidden=hidden,
                        global_state_vec=global_t if bundle.model.use_centralized_value else None,
                    )
                if bundle.model.use_lstm:
                    hidden_by_seat[actor_id] = (
                        next_hidden[0].detach(),
                        next_hidden[1].detach(),
                    ) if next_hidden is not None else None

                if deterministic:
                    chosen_index = int(torch.argmax(logits[0]).item())
                else:
                    distribution = torch.distributions.Categorical(logits=logits[0])
                    chosen_index = int(distribution.sample().item())
                main.apply_action(state, legal[chosen_index])

            score_by_owner: Dict[str, float] = {"A": 0.0, "B": 0.0}
            for seat in range(len(state.players)):
                owner = owner_by_seat[seat]
                score_by_owner[owner] = float(main._final_score_points(state, state.players[seat]))

            score_a = float(score_by_owner["A"])
            score_b = float(score_by_owner["B"])
            total_score_a += score_a
            total_score_b += score_b
            total_rounds += float(main._completed_rounds(state))

            if score_a > score_b:
                wins_a += 1
            elif score_b > score_a:
                wins_b += 1
            else:
                draws += 1

            terminal_reason = str(state.forced_game_over_reason or "")
            if not terminal_reason and state.endgame_trigger_player is not None:
                terminal_reason = "score_threshold_endgame"
            if not terminal_reason:
                terminal_reason = "unknown_terminal_reason"
            terminal_reason_counts[terminal_reason] += 1
    finally:
        policy_a.model.train(prev_training_a)
        policy_b.model.train(prev_training_b)

    episode_count = max(0, int(episodes))
    if episode_count <= 0:
        return EvaluationMetrics(
            episodes=0,
            wins_a=0,
            wins_b=0,
            draws=0,
            avg_score_a=0.0,
            avg_score_b=0.0,
            avg_score_diff_a_minus_b=0.0,
            avg_completed_rounds=0.0,
            terminal_reason_counts={},
        )

    avg_score_a = total_score_a / float(episode_count)
    avg_score_b = total_score_b / float(episode_count)
    return EvaluationMetrics(
        episodes=episode_count,
        wins_a=int(wins_a),
        wins_b=int(wins_b),
        draws=int(draws),
        avg_score_a=float(avg_score_a),
        avg_score_b=float(avg_score_b),
        avg_score_diff_a_minus_b=float(avg_score_a - avg_score_b),
        avg_completed_rounds=float(total_rounds / float(episode_count)),
        terminal_reason_counts=dict(terminal_reason_counts),
    )


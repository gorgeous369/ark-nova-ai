"""Self-play training loop for Masked PPO / Recurrent PPO / MAPPO."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

import main

from .config import PPOTrainConfig
from .encoding import ActionFeatureEncoder, ObservationEncoder
from .model import MaskedActorCritic


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
    terminal_reason_counts: Dict[str, int]
    policy_loss: float
    value_loss: float
    entropy: float
    total_loss: float


def _current_actor_id(state: main.GameState) -> int:
    if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None:
        return int(state.pending_decision_player_id)
    return int(state.current_player)


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


def _to_tensor(array: np.ndarray, *, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(array, dtype=torch.float32, device=device)


def _collect_rollout(
    *,
    model: MaskedActorCritic,
    obs_encoder: ObservationEncoder,
    action_encoder: ActionFeatureEncoder,
    config: PPOTrainConfig,
    rng: random.Random,
    device: torch.device,
    update_index: int,
) -> Tuple[List[RolloutStep], Dict[str, Any]]:
    rollout_steps: List[RolloutStep] = []
    sequence_indices: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
    episode_rounds: List[int] = []
    terminal_abs_diffs: List[float] = []
    terminal_reason_counter: Counter[str] = Counter()

    for episode_idx in range(config.episodes_per_update):
        episode_seed = rng.randint(0, 10_000_000)
        state = main.setup_game(seed=episode_seed, player_names=["P1", "P2"])
        hidden_by_actor: Dict[int, Optional[Tuple[torch.Tensor, torch.Tensor]]] = {}
        if model.use_lstm:
            for player_id in range(len(state.players)):
                hidden_by_actor[player_id] = model.init_hidden(1, device=device)

        while str(state.pending_decision_kind or "").strip() or not state.game_over():
            actor_id = _current_actor_id(state)
            actor = state.players[actor_id]
            legal = main.legal_actions(actor, state=state, player_id=actor_id)
            if not legal:
                raise RuntimeError(
                    f"No legal actions for actor={actor_id} pending={state.pending_decision_kind!r}."
                )

            state_vec, global_vec = obs_encoder.encode_from_state(state, actor_id)
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
            chosen_action = legal[sampled_index]
            main.apply_action(state, chosen_action)
            after_diff = _progress_score_diff(state, actor_id)
            reward = (after_diff - before_diff) * float(config.step_reward_scale)

            step = RolloutStep(
                sequence_key=(episode_idx, actor_id),
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
            seq_key = (episode_idx, player_id)
            if seq_key not in sequence_indices or not sequence_indices[seq_key]:
                continue
            terminal_diff = _terminal_score_diff(state, player_id)
            terminal_abs_diffs.append(abs(float(terminal_diff)))
            last_step_idx = sequence_indices[seq_key][-1]
            rollout_steps[last_step_idx].reward += float(config.terminal_reward_scale) * float(terminal_diff)

        episode_rounds.append(int(main._completed_rounds(state)))
        terminal_reason = str(state.forced_game_over_reason or "")
        if not terminal_reason and state.endgame_trigger_player is not None:
            terminal_reason = "score_threshold_endgame"
        if not terminal_reason:
            terminal_reason = "unknown_terminal_reason"
        terminal_reason_counter[terminal_reason] += 1

    for sequence, indices in sequence_indices.items():
        del sequence
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

    rollout_stats = {
        "episode_count": config.episodes_per_update,
        "step_count": len(rollout_steps),
        "avg_completed_rounds": float(np.mean(episode_rounds)) if episode_rounds else 0.0,
        "avg_terminal_score_diff_abs": (
            float(np.mean(terminal_abs_diffs))
            if terminal_abs_diffs
            else 0.0
        ),
        "terminal_reason_counts": dict(terminal_reason_counter),
        "sequence_indices": sequence_indices,
    }
    return rollout_steps, rollout_stats


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
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }

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
    for _ in range(config.update_epochs):
        policy_losses_t: List[torch.Tensor] = []
        value_losses_t: List[torch.Tensor] = []
        entropies_t: List[torch.Tensor] = []

        for _, indices in ordered_sequences:
            hidden = model.init_hidden(1, device=device) if model.use_lstm else None
            for step_idx in indices:
                step = rollout_steps[step_idx]
                state_t = _to_tensor(step.state_vec, device=device).unsqueeze(0)
                global_t = _to_tensor(step.global_state_vec, device=device).unsqueeze(0)
                action_t = _to_tensor(step.action_features, device=device).unsqueeze(0)
                mask_t = _to_tensor(step.action_mask, device=device).unsqueeze(0)
                logits, values, hidden = model.forward_step(
                    state_vec=state_t,
                    action_features=action_t,
                    action_mask=mask_t,
                    hidden=hidden,
                    global_state_vec=global_t if model.use_centralized_value else None,
                )
                dist = Categorical(logits=logits[0])
                action_index_t = torch.tensor(step.action_index, dtype=torch.long, device=device)
                new_logprob = dist.log_prob(action_index_t)
                entropy = dist.entropy()
                new_value = values.squeeze(0)

                old_logprob = torch.tensor(float(step.old_logprob), dtype=torch.float32, device=device)
                target_return = torch.tensor(float(step.return_), dtype=torch.float32, device=device)
                advantage = torch.tensor(float(normalized_advantages[step_idx]), dtype=torch.float32, device=device)

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
        return {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "total_loss": 0.0,
        }
    return {
        "policy_loss": float(np.mean(epoch_policy_losses)),
        "value_loss": float(np.mean(epoch_value_losses)),
        "entropy": float(np.mean(epoch_entropies)),
        "total_loss": float(np.mean(epoch_total_losses)),
    }


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


def train_self_play(
    *,
    config: PPOTrainConfig,
    output_dir: Path,
) -> List[TrainUpdateMetrics]:
    config.resolve_algo_flags()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    rng = random.Random(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    model, obs_encoder, action_encoder = _build_model_and_encoders(config=config, device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config.learning_rate))

    all_metrics: List[TrainUpdateMetrics] = []
    for update_idx in range(1, config.total_updates + 1):
        rollout_steps, rollout_stats = _collect_rollout(
            model=model,
            obs_encoder=obs_encoder,
            action_encoder=action_encoder,
            config=config,
            rng=rng,
            device=device,
            update_index=update_idx,
        )
        update_stats = _update_model(
            model=model,
            optimizer=optimizer,
            rollout_steps=rollout_steps,
            sequence_indices=rollout_stats["sequence_indices"],
            config=config,
            device=device,
        )
        metrics = TrainUpdateMetrics(
            step_count=int(rollout_stats["step_count"]),
            episode_count=int(rollout_stats["episode_count"]),
            avg_completed_rounds=float(rollout_stats["avg_completed_rounds"]),
            avg_terminal_score_diff_abs=float(rollout_stats["avg_terminal_score_diff_abs"]),
            terminal_reason_counts=dict(rollout_stats["terminal_reason_counts"]),
            policy_loss=float(update_stats["policy_loss"]),
            value_loss=float(update_stats["value_loss"]),
            entropy=float(update_stats["entropy"]),
            total_loss=float(update_stats["total_loss"]),
        )
        all_metrics.append(metrics)

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
                f"policy={metrics.policy_loss:.4f} value={metrics.value_loss:.4f} "
                f"entropy={metrics.entropy:.4f} total={metrics.total_loss:.4f} "
                f"reasons={reason_text}"
            )

        if update_idx % int(config.checkpoint_interval) == 0 or update_idx == config.total_updates:
            checkpoint_path = output_path / f"checkpoint_{update_idx:04d}.pt"
            torch.save(
                {
                    "update": int(update_idx),
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": asdict(config),
                    "metrics": [asdict(item) for item in all_metrics],
                },
                checkpoint_path,
            )
            print(f"saved checkpoint: {checkpoint_path}")

    return all_metrics

"""Observation/action encoders for Ark Nova RL training."""

from __future__ import annotations

import hashlib
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import main


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _norm(value: Any, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return _safe_float(value, 0.0) / float(scale)


def _stable_bucket(token: str, bucket_count: int) -> int:
    if bucket_count <= 0:
        return 0
    digest = hashlib.sha256(token.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int(value % bucket_count)


def _hash_tokens(tokens: Sequence[str], dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if dim <= 0:
        return vec
    for token in tokens:
        normalized = str(token).strip().lower()
        if not normalized:
            continue
        bucket = _stable_bucket(normalized, dim)
        vec[bucket] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0.0:
        vec = vec / norm
    return vec


def _tokenize_text(text: str) -> List[str]:
    return [token for token in re.split(r"[^a-zA-Z0-9_]+", str(text).strip().lower()) if token]


class ActionFeatureEncoder:
    """Encodes concrete `Action` candidates into fixed vectors."""

    def __init__(self, text_hash_dim: int = 64) -> None:
        self.main_actions: Tuple[str, ...] = tuple(main.MAIN_ACTION_CARDS)
        self.pending_kinds: Tuple[str, ...] = (
            "cards_discard",
            "break_discard",
            "opening_draft_keep",
            "conservation_reward",
            "",
        )
        self.text_hash_dim = int(text_hash_dim)
        self.feature_dim = (
            3  # action type one-hot
            + (len(self.main_actions) + 1)  # action card one-hot (+none)
            + len(self.pending_kinds)  # pending kind one-hot
            + 10  # scalar/detail summary
            + self.text_hash_dim  # hashed action text tokens
        )

    def _action_type_one_hot(self, action: main.Action) -> List[float]:
        kind = action.type
        return [
            1.0 if kind == main.ActionType.MAIN_ACTION else 0.0,
            1.0 if kind == main.ActionType.PENDING_DECISION else 0.0,
            1.0 if kind == main.ActionType.X_TOKEN else 0.0,
        ]

    def _main_action_one_hot(self, action: main.Action) -> List[float]:
        card_name = str(action.card_name or "").strip().lower()
        values: List[float] = []
        for item in self.main_actions:
            values.append(1.0 if item == card_name else 0.0)
        values.append(1.0 if not card_name else 0.0)
        return values

    def _pending_kind_one_hot(self, details: Dict[str, Any]) -> List[float]:
        kind = str(details.get("pending_kind") or "").strip().lower()
        return [1.0 if kind == item else 0.0 for item in self.pending_kinds]

    def _detail_summary(self, action: main.Action) -> List[float]:
        details = dict(action.details or {})
        from_display_indices = list(details.get("from_display_indices") or [])
        discard_ids = list(details.get("discard_card_instance_ids") or [])
        discard_indices = list(details.get("discard_hand_indices") or [])
        selections = list(details.get("selections") or [])
        association_sequence = list(details.get("association_task_sequence") or [])
        sponsor_selections = list(details.get("sponsor_selections") or [])

        return [
            _norm(action.value or 0, 5.0),
            _norm(details.get("effective_strength", 0), 5.0),
            _norm(len(from_display_indices), 6.0),
            _norm(details.get("from_deck_count", 0), 6.0),
            _norm(max(len(discard_ids), len(discard_indices)), 8.0),
            _norm(len(selections), 4.0),
            _norm(len(association_sequence), 4.0),
            _norm(len(sponsor_selections), 4.0),
            1.0 if bool(details.get("concrete")) else 0.0,
            1.0 if bool(details.get("use_break_ability")) else 0.0,
        ]

    def _text_features(self, action: main.Action) -> np.ndarray:
        details = dict(action.details or {})
        tokens = _tokenize_text(str(action))
        for key in sorted(details):
            tokens.extend(_tokenize_text(key))
            value = details.get(key)
            if isinstance(value, (str, int, float, bool)):
                tokens.extend(_tokenize_text(str(value)))
        return _hash_tokens(tokens, self.text_hash_dim)

    def encode(self, action: main.Action) -> np.ndarray:
        details = dict(action.details or {})
        vec: List[float] = []
        vec.extend(self._action_type_one_hot(action))
        vec.extend(self._main_action_one_hot(action))
        vec.extend(self._pending_kind_one_hot(details))
        vec.extend(self._detail_summary(action))
        base = np.asarray(vec, dtype=np.float32)
        hashed = self._text_features(action)
        return np.concatenate([base, hashed], axis=0).astype(np.float32)

    def encode_many(self, actions: Sequence[main.Action]) -> np.ndarray:
        if not actions:
            return np.zeros((0, self.feature_dim), dtype=np.float32)
        return np.stack([self.encode(action) for action in actions], axis=0).astype(np.float32)


class ObservationEncoder:
    """Encodes state observations into vectors for actor/critic."""

    def __init__(
        self,
        *,
        max_players: int = 4,
        hand_hash_dim: int = 64,
        final_hash_dim: int = 32,
        draft_hash_dim: int = 32,
    ) -> None:
        self.max_players = int(max_players)
        self.main_actions: Tuple[str, ...] = tuple(main.MAIN_ACTION_CARDS)
        self.association_tasks: Tuple[str, ...] = tuple(main.ASSOCIATION_TASK_KINDS)
        self.pending_kinds: Tuple[str, ...] = (
            "",
            "cards_discard",
            "break_discard",
            "opening_draft_keep",
            "conservation_reward",
        )
        self.hand_hash_dim = int(hand_hash_dim)
        self.final_hash_dim = int(final_hash_dim)
        self.draft_hash_dim = int(draft_hash_dim)

    def _encode_current_player_one_hot(self, player_index: int) -> List[float]:
        return [
            1.0 if int(player_index) == idx else 0.0
            for idx in range(self.max_players)
        ]

    def _encode_pending_kind_one_hot(self, pending_kind: str) -> List[float]:
        normalized = str(pending_kind or "").strip().lower()
        return [1.0 if normalized == item else 0.0 for item in self.pending_kinds]

    def _encode_pending_player_one_hot(self, player_id: Optional[int]) -> List[float]:
        values = [0.0 for _ in range(self.max_players + 1)]
        if player_id is None:
            values[-1] = 1.0
            return values
        idx = int(player_id)
        if 0 <= idx < self.max_players:
            values[idx] = 1.0
        else:
            values[-1] = 1.0
        return values

    def _encode_public_player(self, player: Optional[Dict[str, Any]]) -> np.ndarray:
        info = dict(player or {})
        action_upgraded = dict(info.get("action_upgraded") or {})
        action_order = list(info.get("action_order") or [])
        association_workers = dict(info.get("association_workers_by_task") or {})
        vec: List[float] = [
            1.0 if player is not None else 0.0,
            _norm(info.get("money", 0), 250.0),
            _norm(info.get("appeal", 0), 200.0),
            _norm(info.get("conservation", 0), 20.0),
            _norm(info.get("reputation", 0), 15.0),
            _norm(info.get("workers", 0), 5.0),
            _norm(info.get("x_tokens", 0), 5.0),
            _norm(info.get("hand_limit", 0), 10.0),
            _norm(info.get("hand_count", 0), 30.0),
            _norm(info.get("final_scoring_count", 0), 5.0),
            _norm(info.get("workers_on_association_board", 0), 5.0),
            _norm(info.get("supported_conservation_project_actions", 0), 12.0),
            _norm(info.get("map_left_track_unlocked_count", 0), 10.0),
            _norm(info.get("zoo_cards_count", 0), 80.0),
            _norm(info.get("pouched_cards_count", 0), 40.0),
            _norm(len(info.get("partner_zoos") or []), 5.0),
            _norm(len(info.get("universities") or []), 5.0),
            _norm(len(info.get("supported_conservation_projects") or []), 20.0),
            _norm(len(info.get("claimed_conservation_reward_spaces") or []), 3.0),
            _norm(len(info.get("claimed_reputation_milestones") or []), 8.0),
            1.0 if bool(info.get("map_completion_reward_claimed", False)) else 0.0,
        ]
        for action_name in self.main_actions:
            vec.append(1.0 if bool(action_upgraded.get(action_name, False)) else 0.0)
        for action_name in self.main_actions:
            if action_name in action_order:
                vec.append(_norm(action_order.index(action_name) + 1, float(len(self.main_actions))))
            else:
                vec.append(0.0)
        for task_kind in self.association_tasks:
            vec.append(_norm(association_workers.get(task_kind, 0), 4.0))
        return np.asarray(vec, dtype=np.float32)

    def encode_public_observation(self, public_obs: Dict[str, Any]) -> np.ndarray:
        info = dict(public_obs or {})
        pending_kind = str(info.get("pending_decision_kind") or "")
        pending_player = info.get("pending_decision_player_id")
        conservation_slots = dict(info.get("conservation_project_slots") or {})
        total_slot_count = 0
        filled_slot_count = 0
        for levels in conservation_slots.values():
            level_map = dict(levels or {})
            for owner in level_map.values():
                total_slot_count += 1
                if owner is not None:
                    filled_slot_count += 1

        vec: List[float] = [
            _norm(info.get("player_count", 0), float(self.max_players)),
            _norm(info.get("turn_index", 0), 200.0),
            _norm(info.get("break_progress", 0), 20.0),
            _norm(info.get("break_max", 0), 20.0),
            _norm(info.get("zoo_deck_count", 0), 300.0),
            _norm(info.get("zoo_discard_count", 0), 300.0),
            _norm(len(info.get("zoo_display") or []), 6.0),
            _norm(info.get("donation_progress", 0), 20.0),
            _norm(info.get("final_scoring_deck_count", 0), 20.0),
            _norm(info.get("final_scoring_discard_count", 0), 20.0),
            _norm(info.get("unused_base_conservation_projects_count", 0), 20.0),
            _norm(len(info.get("available_partner_zoos") or []), 5.0),
            _norm(len(info.get("available_universities") or []), 5.0),
            _norm(total_slot_count, 60.0),
            _norm(filled_slot_count, 60.0),
            1.0 if bool(info.get("forced_game_over", False)) else 0.0,
            1.0 if info.get("endgame_trigger_player") is not None else 0.0,
            1.0 if info.get("break_trigger_player") is not None else 0.0,
        ]
        vec.extend(self._encode_current_player_one_hot(_safe_int(info.get("current_player"), 0)))
        vec.extend(self._encode_pending_kind_one_hot(pending_kind))
        vec.extend(self._encode_pending_player_one_hot(pending_player))

        players = list(info.get("players") or [])
        for idx in range(self.max_players):
            player_info = players[idx] if idx < len(players) else None
            vec.extend(self._encode_public_player(player_info))

        return np.asarray(vec, dtype=np.float32)

    def _encode_hand_summary(self, cards: Sequence[Dict[str, Any]]) -> np.ndarray:
        animal_count = 0
        sponsor_count = 0
        project_count = 0
        other_count = 0
        total_cost = 0
        total_size = 0
        total_appeal = 0
        total_conservation = 0
        total_rep_gain = 0
        tokens: List[str] = []

        for card in cards:
            card_type = str(card.get("card_type") or "").strip().lower()
            if card_type == "animal":
                animal_count += 1
            elif card_type == "sponsor":
                sponsor_count += 1
            elif card_type == "conservation_project":
                project_count += 1
            else:
                other_count += 1
            total_cost += _safe_int(card.get("cost", 0), 0)
            total_size += _safe_int(card.get("size", 0), 0)
            total_appeal += _safe_int(card.get("appeal", 0), 0)
            total_conservation += _safe_int(card.get("conservation", 0), 0)
            total_rep_gain += _safe_int(card.get("reputation_gain", 0), 0)
            tokens.append(f"num:{_safe_int(card.get('number', -1), -1)}")
            tokens.extend(_tokenize_text(str(card.get("name") or "")))
            tokens.append(f"type:{card_type}")

        count = max(1, len(cards))
        vec = np.asarray(
            [
                _norm(len(cards), 30.0),
                _norm(animal_count, 30.0),
                _norm(sponsor_count, 30.0),
                _norm(project_count, 30.0),
                _norm(other_count, 30.0),
                _norm(total_cost, 300.0),
                _norm(total_size, 120.0),
                _norm(total_appeal, 120.0),
                _norm(total_conservation, 40.0),
                _norm(total_rep_gain, 40.0),
                _norm(total_cost / count, 30.0),
                _norm(total_size / count, 10.0),
                _norm(total_appeal / count, 10.0),
                _norm(total_conservation / count, 5.0),
            ],
            dtype=np.float32,
        )
        hashed = _hash_tokens(tokens, self.hand_hash_dim)
        return np.concatenate([vec, hashed], axis=0).astype(np.float32)

    def _encode_final_scoring_summary(self, cards: Sequence[Dict[str, Any]]) -> np.ndarray:
        tokens: List[str] = []
        for card in cards:
            tokens.extend(_tokenize_text(str(card.get("data_id") or "")))
            tokens.extend(_tokenize_text(str(card.get("title") or "")))
        vec = np.asarray([_norm(len(cards), 8.0)], dtype=np.float32)
        hashed = _hash_tokens(tokens, self.final_hash_dim)
        return np.concatenate([vec, hashed], axis=0).astype(np.float32)

    def _encode_opening_draft_summary(self, cards: Sequence[Dict[str, Any]]) -> np.ndarray:
        tokens: List[str] = []
        for card in cards:
            tokens.append(f"num:{_safe_int(card.get('number', -1), -1)}")
            tokens.extend(_tokenize_text(str(card.get("name") or "")))
        vec = np.asarray([_norm(len(cards), 8.0)], dtype=np.float32)
        hashed = _hash_tokens(tokens, self.draft_hash_dim)
        return np.concatenate([vec, hashed], axis=0).astype(np.float32)

    def encode_private_observation(self, private_obs: Dict[str, Any]) -> np.ndarray:
        info = dict(private_obs or {})
        player = dict(info.get("player") or {})
        hand_cards = list(player.get("hand") or [])
        final_cards = list(player.get("final_scoring_cards") or [])
        opening_draft_cards = list(player.get("opening_draft_drawn") or [])
        pouched_cards = list(player.get("pouched_cards") or [])
        pouched_by_host = dict(player.get("pouched_cards_by_host") or {})

        vec: List[float] = [
            _norm(player.get("hand_count", 0), 30.0),
            _norm(player.get("final_scoring_count", 0), 8.0),
            _norm(player.get("opening_draft_drawn_count", 0), 8.0),
            _norm(len(pouched_cards), 30.0),
            _norm(len(pouched_by_host), 30.0),
            _norm(player.get("legacy_private_deck_count", 0), 80.0),
            _norm(player.get("legacy_private_discard_count", 0), 80.0),
        ]
        hand_summary = self._encode_hand_summary(hand_cards)
        final_summary = self._encode_final_scoring_summary(final_cards)
        draft_summary = self._encode_opening_draft_summary(opening_draft_cards)
        base = np.asarray(vec, dtype=np.float32)
        return np.concatenate([base, hand_summary, final_summary, draft_summary], axis=0).astype(np.float32)

    def encode_from_state(self, state: main.GameState, viewer_player_id: int) -> Tuple[np.ndarray, np.ndarray]:
        public_obs = main.build_public_observation(state, viewer_player_id=viewer_player_id)
        private_obs = main.build_private_observation(state, viewer_player_id=viewer_player_id)
        public_vec = self.encode_public_observation(public_obs)
        private_vec = self.encode_private_observation(private_obs)
        local_vec = np.concatenate([public_vec, private_vec], axis=0).astype(np.float32)
        global_vec = public_vec.astype(np.float32)
        return local_vec, global_vec


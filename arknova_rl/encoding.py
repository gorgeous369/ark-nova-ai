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


def _normalized_action_cache_value(value: Any) -> Any:
    if isinstance(value, dict):
        normalized_items: List[Tuple[str, Any]] = []
        for raw_key in sorted(value):
            key = str(raw_key).strip().lower()
            if (
                not key
                or key == "action_label"
                or "instance_id" in key
                or key.endswith("_ids")
            ):
                continue
            normalized_items.append((key, _normalized_action_cache_value(value.get(raw_key))))
        return tuple(normalized_items)
    if isinstance(value, (list, tuple)):
        return tuple(_normalized_action_cache_value(item) for item in value)
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (bool, int, float)):
        return value
    if value is None:
        return None
    return repr(value)


class ActionFeatureEncoder:
    """Encodes concrete `Action` candidates into fixed vectors."""

    def __init__(self, text_hash_dim: int = 32, cache_size: int = 8192) -> None:
        self.main_actions: Tuple[str, ...] = tuple(main.MAIN_ACTION_CARDS)
        self.pending_kinds: Tuple[str, ...] = (
            "cards_discard",
            "break_discard",
            "break_card_draw_choice",
            "opening_draft_keep",
            "conservation_reward",
            "revealed_cards_keep",
            "revealed_final_scoring_keep",
            "final_scoring_discard",
            "digging_choice",
            "",
        )
        self.text_hash_dim = int(text_hash_dim)
        self.cache_size = max(0, int(cache_size))
        self.feature_dim = (
            3  # action type one-hot
            + (len(self.main_actions) + 1)  # action card one-hot (+none)
            + len(self.pending_kinds)  # pending kind one-hot
            + 10  # scalar/detail summary
            + self.text_hash_dim  # hashed action text tokens
        )
        self._encode_cache: Dict[Tuple[Any, ...], np.ndarray] = {}

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

    def _append_text_feature_tokens(
        self,
        tokens: List[str],
        *,
        prefix: str,
        value: Any,
    ) -> None:
        if isinstance(value, bool):
            tokens.append(f"{prefix}:{1 if value else 0}")
            return
        if isinstance(value, (int, float)):
            tokens.append(f"{prefix}:{int(value)}")
            return
        if isinstance(value, str):
            value_text = value.strip().lower()
            if value_text:
                tokens.append(f"{prefix}:{value_text}")
            return
        if isinstance(value, dict):
            for child_key in sorted(value):
                child_key_text = str(child_key).strip().lower()
                if (
                    not child_key_text
                    or child_key_text == "action_label"
                    or "instance_id" in child_key_text
                    or child_key_text.endswith("_ids")
                ):
                    continue
                child_prefix = f"{prefix}.{child_key_text}"
                tokens.append(f"key:{child_prefix}")
                self._append_text_feature_tokens(
                    tokens,
                    prefix=child_prefix,
                    value=value.get(child_key),
                )
            return
        if isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                self._append_text_feature_tokens(
                    tokens,
                    prefix=f"{prefix}[{idx}]",
                    value=item,
                )

    def _text_features(self, action: main.Action) -> np.ndarray:
        details = dict(action.details or {})
        tokens: List[str] = [f"type:{action.type.value}"]
        card_name = str(action.card_name or "").strip().lower()
        if card_name:
            tokens.append(f"card:{card_name}")
        for key in sorted(details):
            key_text = str(key).strip().lower()
            if (
                not key_text
                or key_text == "action_label"
                or "instance_id" in key_text
                or key_text.endswith("_ids")
            ):
                continue
            tokens.append(f"key:{key_text}")
            self._append_text_feature_tokens(
                tokens,
                prefix=key_text,
                value=details.get(key),
            )
        return _hash_tokens(tokens, self.text_hash_dim)

    def _cache_key(self, action: main.Action) -> Tuple[Any, ...]:
        return (
            action.type.value,
            int(action.value or 0) if action.value is not None else None,
            str(action.card_name or "").strip().lower(),
            _normalized_action_cache_value(action.details or {}),
        )

    def encode(self, action: main.Action) -> np.ndarray:
        cache_key = self._cache_key(action)
        cached = self._encode_cache.get(cache_key)
        if cached is not None:
            return cached
        details = dict(action.details or {})
        vec: List[float] = []
        vec.extend(self._action_type_one_hot(action))
        vec.extend(self._main_action_one_hot(action))
        vec.extend(self._pending_kind_one_hot(details))
        vec.extend(self._detail_summary(action))
        base = np.asarray(vec, dtype=np.float32)
        hashed = self._text_features(action)
        encoded = np.concatenate([base, hashed], axis=0).astype(np.float32)
        if self.cache_size > 0:
            if len(self._encode_cache) >= self.cache_size:
                self._encode_cache.clear()
            self._encode_cache[cache_key] = encoded
        return encoded

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
        card_text_hash_dim: int = 12,
        setup_card_text_hash_dim: int = 8,
        set_hash_dim: int = 24,
        pending_payload_hash_dim: int = 16,
        map_name_hash_dim: int = 16,
        max_map_grid_cells: int = 64,
        max_map_buildings: int = 48,
        max_sponsor_buildings: int = 24,
        max_sponsor_building_cells: int = 8,
        max_display_cards: int = 6,
        max_conservation_slots: int = 64,
        max_zoo_cards: int = 80,
        max_enclosures: int = 40,
        max_enclosure_objects: int = 40,
        max_private_hand_cards: int = 30,
        max_private_final_cards: int = 8,
        max_private_draft_cards: int = 8,
        max_private_pouched_cards: int = 40,
    ) -> None:
        self.max_players = int(max_players)
        self.main_actions: Tuple[str, ...] = tuple(main.MAIN_ACTION_CARDS)
        self.association_tasks: Tuple[str, ...] = tuple(main.ASSOCIATION_TASK_KINDS)
        self.pending_kinds: Tuple[str, ...] = (
            "",
            "cards_discard",
            "break_discard",
            "break_card_draw_choice",
            "opening_draft_keep",
            "conservation_reward",
            "revealed_cards_keep",
            "revealed_final_scoring_keep",
            "final_scoring_discard",
            "digging_choice",
        )
        self.hand_hash_dim = int(hand_hash_dim)
        self.final_hash_dim = int(final_hash_dim)
        self.draft_hash_dim = int(draft_hash_dim)
        self.card_text_hash_dim = int(card_text_hash_dim)
        self.setup_card_text_hash_dim = int(setup_card_text_hash_dim)
        self.set_hash_dim = int(set_hash_dim)
        self.pending_payload_hash_dim = int(pending_payload_hash_dim)
        self.map_name_hash_dim = int(map_name_hash_dim)
        self.max_map_grid_cells = int(max_map_grid_cells)
        self.max_map_buildings = int(max_map_buildings)
        self.max_sponsor_buildings = int(max_sponsor_buildings)
        self.max_sponsor_building_cells = int(max_sponsor_building_cells)
        self.max_display_cards = int(max_display_cards)
        self.max_conservation_slots = int(max_conservation_slots)
        self.max_zoo_cards = int(max_zoo_cards)
        self.max_enclosures = int(max_enclosures)
        self.max_enclosure_objects = int(max_enclosure_objects)
        self.max_private_hand_cards = int(max_private_hand_cards)
        self.max_private_final_cards = int(max_private_final_cards)
        self.max_private_draft_cards = int(max_private_draft_cards)
        self.max_private_pouched_cards = int(max_private_pouched_cards)
        self._empty_global_vec = np.zeros((0,), dtype=np.float32)
        self.map_coord_min = -20
        self.map_coord_max = 20
        self.map_building_types: Tuple[str, ...] = tuple(
            str(item.name).strip().lower()
            for item in main.BuildingType
        )
        self.map_building_type_to_index = {
            name: idx for idx, name in enumerate(self.map_building_types)
        }
        self.map_building_rotations: Tuple[str, ...] = tuple(
            str(item.name).strip().lower()
            for item in main.Rotation
        )
        self.map_building_rotation_to_index = {
            name: idx for idx, name in enumerate(self.map_building_rotations)
        }
        self.rotation_types: Tuple[str, ...] = tuple(
            str(item.name).strip().lower()
            for item in main.Rotation
        )
        self.rotation_type_to_index = {
            name: idx for idx, name in enumerate(self.rotation_types)
        }
        self.card_types: Tuple[str, ...] = (
            "animal",
            "sponsor",
            "conservation_project",
        )
        self.card_type_to_index = {
            name: idx for idx, name in enumerate(self.card_types)
        }
        self.pending_resume_kinds: Tuple[str, ...] = (
            "",
            "turn_finalize",
            "break_remaining",
            "other",
        )
        self.pending_resume_kind_set = {
            kind for kind in self.pending_resume_kinds if kind != "other"
        }
        self._empty_private_vec: Optional[np.ndarray] = None

    def _coord_pair(self, raw_coord: Any) -> Optional[Tuple[int, int]]:
        if not isinstance(raw_coord, (list, tuple)) or len(raw_coord) != 2:
            return None
        return (_safe_int(raw_coord[0], 0), _safe_int(raw_coord[1], 0))

    def _norm_coord(self, value: Any) -> float:
        low = int(self.map_coord_min)
        high = int(self.map_coord_max)
        if high <= low:
            return 0.0
        raw = _safe_int(value, 0)
        clipped = max(low, min(high, raw))
        return float(clipped - low) / float(high - low)

    def _encode_map_name_hash(self, map_name: str) -> np.ndarray:
        tokens = _tokenize_text(str(map_name or ""))
        if not tokens:
            tokens = ["map:unknown"]
        return _hash_tokens(tokens, self.map_name_hash_dim)

    def _encode_map_grid_slots(self, raw_cells: Sequence[Any]) -> List[float]:
        cells: List[Tuple[int, int]] = []
        for raw_cell in raw_cells:
            parsed = self._coord_pair(raw_cell)
            if parsed is not None:
                cells.append(parsed)
        cells = sorted(set(cells))
        values: List[float] = []
        for idx in range(self.max_map_grid_cells):
            if idx >= len(cells):
                values.extend([0.0, 0.0, 0.0])
                continue
            x_coord, y_coord = cells[idx]
            values.extend([1.0, self._norm_coord(x_coord), self._norm_coord(y_coord)])
        return values

    def _encode_map_building_slots(self, raw_buildings: Sequence[Any]) -> List[float]:
        entries: List[Tuple[str, Tuple[int, int], str, int]] = []
        for raw in raw_buildings:
            if not isinstance(raw, dict):
                continue
            origin = self._coord_pair(raw.get("origin"))
            if origin is None:
                continue
            building_type = str(raw.get("building_type") or "").strip().lower()
            rotation = str(raw.get("rotation") or "").strip().lower()
            empty_spaces = _safe_int(raw.get("empty_spaces", 0), 0)
            entries.append((building_type, origin, rotation, empty_spaces))
        entries.sort(
            key=lambda item: (
                item[0],
                int(item[1][0]),
                int(item[1][1]),
                item[2],
                int(item[3]),
            )
        )

        type_slot_dim = len(self.map_building_types) + 1
        rotation_slot_dim = len(self.map_building_rotations) + 1
        values: List[float] = []
        for idx in range(self.max_map_buildings):
            if idx >= len(entries):
                values.extend([0.0] * (1 + type_slot_dim + rotation_slot_dim + 2 + 1))
                continue
            building_type, (origin_x, origin_y), rotation, empty_spaces = entries[idx]
            values.append(1.0)
            type_one_hot = [0.0] * type_slot_dim
            type_index = self.map_building_type_to_index.get(
                building_type,
                len(self.map_building_types),
            )
            type_one_hot[type_index] = 1.0
            values.extend(type_one_hot)
            rotation_one_hot = [0.0] * rotation_slot_dim
            rotation_index = self.map_building_rotation_to_index.get(
                rotation,
                len(self.map_building_rotations),
            )
            rotation_one_hot[rotation_index] = 1.0
            values.extend(rotation_one_hot)
            values.extend([self._norm_coord(origin_x), self._norm_coord(origin_y)])
            values.append(_norm(empty_spaces, 5.0))
        return values

    def _encode_sponsor_building_slots(self, raw_buildings: Sequence[Any]) -> List[float]:
        entries: List[Tuple[int, float, List[Tuple[int, int]]]] = []
        for raw in raw_buildings:
            if not isinstance(raw, dict):
                continue
            sponsor_number = _safe_int(raw.get("sponsor_number", 0), 0)
            label = str(raw.get("label") or "").strip().lower()
            label_norm = _norm(_stable_bucket(label, 2048), 2047.0)
            cells: List[Tuple[int, int]] = []
            for raw_cell in list(raw.get("cells") or []):
                parsed = self._coord_pair(raw_cell)
                if parsed is not None:
                    cells.append(parsed)
            cells = sorted(set(cells))
            entries.append((sponsor_number, label_norm, cells))
        entries.sort(key=lambda item: (int(item[0]), float(item[1]), tuple(item[2])))

        values: List[float] = []
        for idx in range(self.max_sponsor_buildings):
            if idx >= len(entries):
                values.extend([0.0] * (4 + self.max_sponsor_building_cells * 2))
                continue
            sponsor_number, label_norm, cells = entries[idx]
            values.append(1.0)
            values.append(_norm(sponsor_number, 600.0))
            values.append(_norm(len(cells), float(max(1, self.max_sponsor_building_cells))))
            values.append(label_norm)
            for cell_idx in range(self.max_sponsor_building_cells):
                if cell_idx >= len(cells):
                    values.extend([0.0, 0.0])
                    continue
                x_coord, y_coord = cells[cell_idx]
                values.extend([self._norm_coord(x_coord), self._norm_coord(y_coord)])
        return values

    def _encode_token_hash(self, values: Sequence[Any], *, dim: int, dedupe: bool = True) -> List[float]:
        tokens = [str(value).strip().lower() for value in values if str(value).strip()]
        if dedupe:
            tokens = sorted(set(tokens))
        return _hash_tokens(tokens, dim).tolist()

    def _encode_scalar_token_map_hash(self, payload: Any) -> List[float]:
        if not isinstance(payload, dict):
            return [0.0] * max(0, self.set_hash_dim)
        tokens: List[str] = []
        for key in sorted(payload):
            value = payload.get(key)
            key_text = str(key).strip().lower()
            if not key_text:
                continue
            if isinstance(value, bool):
                value_text = "1" if value else "0"
            elif isinstance(value, (int, float)):
                value_text = str(int(value))
            elif isinstance(value, str):
                value_text = value.strip().lower()
            elif value is None:
                value_text = "none"
            else:
                continue
            tokens.append(f"{key_text}:{value_text}")
        return _hash_tokens(tokens, self.set_hash_dim).tolist()

    def _encode_main_action_count_map(self, payload: Any, *, scale: float = 6.0) -> List[float]:
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        values: List[float] = []
        total = 0
        for action_name in self.main_actions:
            count = max(0, _safe_int(data.get(action_name, 0), 0))
            total += count
            values.append(_norm(count, scale))
        values.append(_norm(total, scale * float(max(1, len(self.main_actions)))))
        return values

    def _encode_strength_count_entries(self, payload: Any) -> List[float]:
        counts: Dict[int, int] = {}
        if isinstance(payload, dict):
            for raw_strength, raw_count in payload.items():
                strength = _safe_int(raw_strength, 0)
                count = max(0, _safe_int(raw_count, 0))
                if count > 0:
                    counts[strength] = counts.get(strength, 0) + count
        elif isinstance(payload, list):
            for raw_entry in payload:
                if not isinstance(raw_entry, dict):
                    continue
                strength = _safe_int(raw_entry.get("strength", 0), 0)
                count = max(0, _safe_int(raw_entry.get("count", 0), 0))
                if count > 0:
                    counts[strength] = counts.get(strength, 0) + count

        values = [_norm(counts.get(strength, 0), 6.0) for strength in range(1, 6)]
        other_total = sum(count for strength, count in counts.items() if strength < 1 or strength > 5)
        distinct = sum(1 for count in counts.values() if count > 0)
        values.extend(
            [
                _norm(other_total, 10.0),
                _norm(distinct, 8.0),
                _norm(sum(counts.values()), 20.0),
            ]
        )
        return values

    def _encode_number_count_entries(
        self,
        payload: Any,
        *,
        number_key: str,
        count_key: str = "count",
        number_scale: float = 600.0,
        count_scale: float = 10.0,
    ) -> List[float]:
        counts: Dict[int, int] = {}
        if isinstance(payload, dict):
            for raw_number, raw_count in payload.items():
                number = _safe_int(raw_number, -1)
                count = max(0, _safe_int(raw_count, 0))
                if count > 0:
                    counts[number] = counts.get(number, 0) + count
        elif isinstance(payload, list):
            for raw_entry in payload:
                if not isinstance(raw_entry, dict):
                    continue
                number = _safe_int(raw_entry.get(number_key, -1), -1)
                count = max(0, _safe_int(raw_entry.get(count_key, 0), 0))
                if count > 0:
                    counts[number] = counts.get(number, 0) + count

        total = sum(counts.values())
        distinct = len(counts)
        max_number = max(counts.keys()) if counts else 0
        max_count = max(counts.values()) if counts else 0
        tokens = [f"{int(number)}:{int(count)}" for number, count in sorted(counts.items())]

        values: List[float] = [
            _norm(total, count_scale * 4.0),
            _norm(distinct, 16.0),
            _norm(max_number, number_scale),
            _norm(max_count, count_scale),
        ]
        values.extend(_hash_tokens(tokens, self.set_hash_dim).tolist())
        return values

    def _encode_dense_count_slots(
        self,
        raw_counts: Sequence[Any],
        *,
        max_slots: int,
        scale: float,
    ) -> List[float]:
        counts = list(raw_counts or [])
        values: List[float] = []
        for idx in range(max(0, int(max_slots))):
            count = _safe_int(counts[idx], 0) if idx < len(counts) else 0
            values.append(_norm(max(0, count), scale))
        return values

    def _rotation_one_hot(self, raw_rotation: Any) -> List[float]:
        slot_dim = len(self.rotation_types) + 1
        values = [0.0] * slot_dim
        normalized = str(raw_rotation or "").strip().lower()
        index = self.rotation_type_to_index.get(normalized, len(self.rotation_types))
        values[index] = 1.0
        return values

    def _card_type_one_hot(self, raw_type: Any) -> List[float]:
        slot_dim = len(self.card_types) + 1
        values = [0.0] * slot_dim
        normalized = str(raw_type or "").strip().lower()
        index = self.card_type_to_index.get(normalized, len(self.card_types))
        values[index] = 1.0
        return values

    def _encode_card_slot(self, raw_card: Optional[Dict[str, Any]], *, include_private_tokens: bool) -> List[float]:
        slot_dim = 1 + (len(self.card_types) + 1) + 15 + self.card_text_hash_dim
        if not isinstance(raw_card, dict):
            return [0.0] * slot_dim

        required_icons = list(raw_card.get("required_icons") or [])
        icon_total = 0
        icon_kinds = 0
        icon_tokens: List[str] = []
        for raw_icon in required_icons:
            if not isinstance(raw_icon, dict):
                continue
            icon_name = str(raw_icon.get("icon") or "").strip().lower()
            icon_count = _safe_int(raw_icon.get("count", 0), 0)
            if icon_count <= 0 and not icon_name:
                continue
            icon_total += max(0, icon_count)
            if icon_name:
                icon_kinds += 1
                icon_tokens.append(f"icon:{icon_name}:{icon_count}")

        badges = [str(item).strip().lower() for item in list(raw_card.get("badges") or []) if str(item).strip()]
        effects = list(raw_card.get("effects") or [])
        effect_tokens: List[str] = []
        for effect in effects:
            if not isinstance(effect, dict):
                continue
            kind = str(effect.get("kind") or "").strip().lower()
            value = str(effect.get("value") or "").strip().lower()
            if kind or value:
                effect_tokens.append(f"effect:{kind}:{value}")

        tokens: List[str] = []
        tokens.extend(_tokenize_text(str(raw_card.get("name") or "")))
        tokens.extend(_tokenize_text(str(raw_card.get("ability_title") or "")))
        tokens.extend(_tokenize_text(str(raw_card.get("ability_text") or "")))
        tokens.extend([f"badge:{item}" for item in badges])
        tokens.extend(icon_tokens)
        tokens.extend(effect_tokens)
        tokens.append(f"num:{_safe_int(raw_card.get('number', -1), -1)}")
        tokens.append(f"type:{str(raw_card.get('card_type') or '').strip().lower()}")

        values: List[float] = [1.0]
        values.extend(self._card_type_one_hot(raw_card.get("card_type")))
        values.extend(
            [
                _norm(raw_card.get("number", 0), 600.0),
                _norm(raw_card.get("cost", 0), 50.0),
                _norm(raw_card.get("size", 0), 15.0),
                _norm(raw_card.get("appeal", 0), 20.0),
                _norm(raw_card.get("conservation", 0), 8.0),
                _norm(raw_card.get("reputation_gain", 0), 8.0),
                _norm(raw_card.get("required_water_adjacency", 0), 3.0),
                _norm(raw_card.get("required_rock_adjacency", 0), 3.0),
                _norm(icon_total, 8.0),
                _norm(icon_kinds, 8.0),
                _norm(len(badges), 8.0),
                _norm(len(effects), 8.0),
                _norm(raw_card.get("reptile_house_size", 0) or 0, 5.0),
                _norm(raw_card.get("large_bird_aviary_size", 0) or 0, 5.0),
                0.0,
            ]
        )
        values.extend(_hash_tokens(tokens, self.card_text_hash_dim).tolist())
        return values

    def _encode_card_slots(
        self,
        raw_cards: Sequence[Any],
        *,
        max_cards: int,
        include_private_tokens: bool,
    ) -> List[float]:
        cards = [dict(card) for card in raw_cards if isinstance(card, dict)]
        slot_values: List[float] = []
        empty_slot = self._encode_card_slot(None, include_private_tokens=include_private_tokens)
        for idx in range(max(0, int(max_cards))):
            if idx >= len(cards):
                slot_values.extend(empty_slot)
                continue
            slot_values.extend(
                self._encode_card_slot(cards[idx], include_private_tokens=include_private_tokens)
            )
        return slot_values

    def _encode_setup_card_slots(self, raw_cards: Sequence[Any], *, max_cards: int) -> List[float]:
        slot_dim = 1 + 2 + self.setup_card_text_hash_dim
        cards = [dict(card) for card in raw_cards if isinstance(card, dict)]
        values: List[float] = []
        for idx in range(max(0, int(max_cards))):
            if idx >= len(cards):
                values.extend([0.0] * slot_dim)
                continue
            card = cards[idx]
            data_id = str(card.get("data_id") or "").strip().lower()
            title = str(card.get("title") or "").strip().lower()
            tokens = _tokenize_text(data_id) + _tokenize_text(title)
            values.extend(
                [
                    1.0,
                    _norm(_stable_bucket(data_id, 4096), 4095.0),
                    _norm(_stable_bucket(title, 4096), 4095.0),
                ]
            )
            values.extend(_hash_tokens(tokens, self.setup_card_text_hash_dim).tolist())
        return values

    def _encode_enclosure_slots(self, raw_enclosures: Sequence[Any]) -> List[float]:
        slot_dim = 1 + 8 + (len(self.rotation_types) + 1) + self.set_hash_dim
        enclosures = [dict(item) for item in raw_enclosures if isinstance(item, dict)]
        enclosures.sort(
            key=lambda item: (
                _safe_int((item.get("origin") or [0, 0])[0] if isinstance(item.get("origin"), (list, tuple)) else 0, 0),
                _safe_int((item.get("origin") or [0, 0])[1] if isinstance(item.get("origin"), (list, tuple)) else 0, 0),
                _safe_int(item.get("size", 0), 0),
                str(item.get("rotation") or ""),
                str(item.get("enclosure_type") or ""),
            )
        )
        values: List[float] = []
        for idx in range(self.max_enclosures):
            if idx >= len(enclosures):
                values.extend([0.0] * slot_dim)
                continue
            enclosure = enclosures[idx]
            origin = self._coord_pair(enclosure.get("origin"))
            origin_x, origin_y = origin if origin is not None else (0, 0)
            type_token = str(enclosure.get("enclosure_type") or "").strip().lower()
            values.append(1.0)
            values.extend(
                [
                    _norm(enclosure.get("size", 0), 10.0),
                    1.0 if bool(enclosure.get("occupied", False)) else 0.0,
                    self._norm_coord(origin_x),
                    self._norm_coord(origin_y),
                    _norm(enclosure.get("used_capacity", 0), 10.0),
                    _norm(enclosure.get("animal_capacity", 0), 10.0),
                    _norm(
                        _safe_int(enclosure.get("used_capacity", 0), 0)
                        / float(max(1, _safe_int(enclosure.get("animal_capacity", 0), 0))),
                        1.0,
                    ),
                    _norm(_stable_bucket(type_token, 2048), 2047.0),
                ]
            )
            values.extend(self._rotation_one_hot(enclosure.get("rotation")))
            values.extend(_hash_tokens([type_token], self.set_hash_dim).tolist())
        return values

    def _encode_enclosure_object_slots(self, raw_objects: Sequence[Any]) -> List[float]:
        slot_dim = 1 + 8 + (len(self.rotation_types) + 1) + self.set_hash_dim
        objects = [dict(item) for item in raw_objects if isinstance(item, dict)]
        objects.sort(
            key=lambda item: (
                _safe_int((item.get("origin") or [0, 0])[0] if isinstance(item.get("origin"), (list, tuple)) else 0, 0),
                _safe_int((item.get("origin") or [0, 0])[1] if isinstance(item.get("origin"), (list, tuple)) else 0, 0),
                _safe_int(item.get("size", 0), 0),
                str(item.get("rotation") or ""),
                str(item.get("enclosure_type") or ""),
            )
        )
        values: List[float] = []
        for idx in range(self.max_enclosure_objects):
            if idx >= len(objects):
                values.extend([0.0] * slot_dim)
                continue
            enclosure = objects[idx]
            origin = self._coord_pair(enclosure.get("origin"))
            origin_x, origin_y = origin if origin is not None else (0, 0)
            type_token = str(enclosure.get("enclosure_type") or "").strip().lower()
            values.append(1.0)
            values.extend(
                [
                    _norm(enclosure.get("size", 0), 10.0),
                    self._norm_coord(origin_x),
                    self._norm_coord(origin_y),
                    _norm(enclosure.get("adjacent_rock", 0), 6.0),
                    _norm(enclosure.get("adjacent_water", 0), 6.0),
                    _norm(enclosure.get("animals_inside", 0), 10.0),
                    _norm(_stable_bucket(type_token, 2048), 2047.0),
                    _norm(
                        _safe_int(enclosure.get("adjacent_rock", 0), 0)
                        + _safe_int(enclosure.get("adjacent_water", 0), 0),
                        12.0,
                    ),
                ]
            )
            values.extend(self._rotation_one_hot(enclosure.get("rotation")))
            values.extend(_hash_tokens([type_token], self.set_hash_dim).tolist())
        return values

    def _encode_conservation_slot_details(self, conservation_slots: Dict[str, Any]) -> List[float]:
        entries: List[Tuple[str, str, Optional[int]]] = []
        for project_id, levels in sorted(conservation_slots.items()):
            if not isinstance(levels, dict):
                continue
            for level_name, owner in sorted(levels.items()):
                owner_value: Optional[int]
                if owner is None:
                    owner_value = None
                else:
                    owner_value = _safe_int(owner, -1)
                entries.append((str(project_id), str(level_name), owner_value))

        values: List[float] = []
        owner_slot_dim = self.max_players + 1
        for idx in range(self.max_conservation_slots):
            if idx >= len(entries):
                values.extend([0.0] * (1 + 2 + owner_slot_dim))
                continue
            project_id, level_name, owner = entries[idx]
            values.append(1.0)
            values.append(_norm(_stable_bucket(project_id.lower(), 4096), 4095.0))
            values.append(_norm(_stable_bucket(level_name.lower(), 1024), 1023.0))
            owner_one_hot = [0.0] * owner_slot_dim
            if owner is None or owner < 0 or owner >= self.max_players:
                owner_one_hot[-1] = 1.0
            else:
                owner_one_hot[int(owner)] = 1.0
            values.extend(owner_one_hot)
        return values

    def _encode_pending_payload(self, payload: Any) -> List[float]:
        data = dict(payload or {}) if isinstance(payload, dict) else {}
        resume_player = data.get("resume_turn_player_id")
        resume_player_vec = [0.0] * (self.max_players + 1)
        if resume_player is None:
            resume_player_vec[-1] = 1.0
        else:
            idx = _safe_int(resume_player, -1)
            if 0 <= idx < self.max_players:
                resume_player_vec[idx] = 1.0
            else:
                resume_player_vec[-1] = 1.0

        resume_kind = str(data.get("resume_kind") or "").strip().lower()
        resume_kind_one_hot: List[float] = []
        for item in self.pending_resume_kinds:
            if item == "other":
                is_other = bool(resume_kind) and resume_kind not in self.pending_resume_kind_set
                resume_kind_one_hot.append(1.0 if is_other else 0.0)
                continue
            resume_kind_one_hot.append(1.0 if resume_kind == item else 0.0)

        tokens = self._encode_scalar_token_map_hash(data)
        key_hash = self._encode_token_hash(sorted(data.keys()), dim=self.pending_payload_hash_dim, dedupe=True)
        values: List[float] = [
            _norm(data.get("discard_target", 0), 12.0),
            _norm(data.get("keep_target", 0), 8.0),
            _norm(data.get("draft_card_count", 0), 8.0),
            _norm(data.get("threshold", 0), 12.0),
            _norm(data.get("break_hand_limit_index", 0), 6.0),
            _norm(data.get("break_income_index", 0), 6.0),
            1.0 if bool(data.get("resume_turn_consumed_venom", False)) else 0.0,
            1.0 if bool(data.get("break_triggered", False)) else 0.0,
            1.0 if bool(data.get("consumed_venom", False)) else 0.0,
            _norm(len(data), 12.0),
        ]
        values.extend(resume_player_vec)
        values.extend(resume_kind_one_hot)
        values.extend(tokens)
        values.extend(key_hash)
        return values

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
        action_order_positions = {
            str(action_name): idx + 1 for idx, action_name in enumerate(action_order)
        }
        association_workers = dict(info.get("association_workers_by_task") or {})
        zoo_map_grid = list(info.get("zoo_map_grid") or [])
        zoo_map_buildings = list(info.get("zoo_map_buildings") or [])
        sponsor_buildings = list(info.get("sponsor_buildings") or [])
        partner_zoos = list(info.get("partner_zoos") or [])
        universities = list(info.get("universities") or [])
        supported_projects = list(info.get("supported_conservation_projects") or [])
        claimed_reward_spaces = list(info.get("claimed_conservation_reward_spaces") or [])
        claimed_reputation = list(info.get("claimed_reputation_milestones") or [])
        map_claimed_indices = list(info.get("map_left_track_claimed_indices") or [])
        map_unlocked_effects = list(info.get("map_left_track_unlocked_effects") or [])
        claimed_partner_thresholds = list(info.get("claimed_partner_zoo_thresholds") or [])
        claimed_university_thresholds = list(info.get("claimed_university_thresholds") or [])
        zoo_cards = list(info.get("zoo_cards") or [])
        enclosures = list(info.get("enclosures") or [])
        enclosure_objects = list(info.get("enclosure_objects") or [])
        multiplier_tokens = dict(info.get("multiplier_tokens_on_actions") or {})
        venom_tokens = dict(info.get("venom_tokens_on_actions") or {})
        constriction_tokens = dict(info.get("constriction_tokens_on_actions") or {})
        extra_actions_granted = dict(info.get("extra_actions_granted") or {})
        extra_strength_actions = list(info.get("extra_strength_actions") or [])
        sponsor_tokens = list(info.get("sponsor_tokens_by_number") or [])
        sponsor_mode = str(info.get("sponsor_waza_assignment_mode") or "").strip().lower()
        vec: List[float] = [
            1.0 if player is not None else 0.0,
        ]
        vec.extend(self._encode_pending_player_one_hot(info.get("player_id")))
        vec.extend(
            [
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
            _norm(len(partner_zoos), 5.0),
            _norm(len(universities), 5.0),
            _norm(len(supported_projects), 20.0),
            _norm(len(claimed_reward_spaces), 3.0),
            _norm(len(claimed_reputation), 8.0),
            1.0 if bool(info.get("map_completion_reward_claimed", False)) else 0.0,
            1.0 if bool(info.get("zoo_map_present", False)) else 0.0,
            _norm(info.get("zoo_map_grid_count", len(zoo_map_grid)), 80.0),
            _norm(info.get("zoo_map_building_count", len(zoo_map_buildings)), 60.0),
            _norm(len(sponsor_buildings), 30.0),
            _norm(len(enclosures), 40.0),
            _norm(len(enclosure_objects), 40.0),
            _norm(len(map_claimed_indices), 16.0),
            _norm(len(map_unlocked_effects), 16.0),
            _norm(len(claimed_partner_thresholds), 6.0),
            _norm(len(claimed_university_thresholds), 6.0),
            ]
        )
        for action_name in self.main_actions:
            vec.append(1.0 if bool(action_upgraded.get(action_name, False)) else 0.0)
        for action_name in self.main_actions:
            action_position = action_order_positions.get(action_name, 0)
            vec.append(_norm(action_position, float(len(self.main_actions))))
        for task_kind in self.association_tasks:
            vec.append(_norm(association_workers.get(task_kind, 0), 4.0))
        vec.extend(self._encode_main_action_count_map(multiplier_tokens, scale=4.0))
        vec.extend(self._encode_main_action_count_map(venom_tokens, scale=2.0))
        vec.extend(self._encode_main_action_count_map(constriction_tokens, scale=2.0))
        vec.extend(self._encode_main_action_count_map(extra_actions_granted, scale=4.0))
        vec.append(_norm(info.get("extra_any_actions", 0), 8.0))
        vec.extend(self._encode_strength_count_entries(extra_strength_actions))
        vec.append(_norm(info.get("camouflage_condition_ignores", 0), 8.0))
        vec.extend(
            self._encode_number_count_entries(
                sponsor_tokens,
                number_key="number",
                count_key="count",
                number_scale=600.0,
                count_scale=10.0,
            )
        )
        vec.extend(
            [
                1.0 if sponsor_mode == "small" else 0.0,
                1.0 if sponsor_mode == "large" else 0.0,
                1.0 if sponsor_mode not in {"", "small", "large"} else 0.0,
                _norm(info.get("sponsor_ignore_large_condition_charges", 0), 8.0),
            ]
        )
        vec.extend(self._encode_token_hash(partner_zoos, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(universities, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(supported_projects, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(claimed_reward_spaces, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(claimed_reputation, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(map_claimed_indices, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(map_unlocked_effects, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(claimed_partner_thresholds, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(claimed_university_thresholds, dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_map_grid_slots(zoo_map_grid))
        vec.extend(self._encode_map_building_slots(zoo_map_buildings))
        vec.extend(
            self._encode_card_slots(
                zoo_cards,
                max_cards=self.max_zoo_cards,
                include_private_tokens=False,
            )
        )
        vec.extend(self._encode_enclosure_slots(enclosures))
        vec.extend(self._encode_enclosure_object_slots(enclosure_objects))
        vec.extend(self._encode_sponsor_building_slots(sponsor_buildings))
        return np.asarray(vec, dtype=np.float32)

    def encode_public_observation(self, public_obs: Dict[str, Any]) -> np.ndarray:
        info = dict(public_obs or {})
        pending_kind = str(info.get("pending_decision_kind") or "")
        pending_player = info.get("pending_decision_player_id")
        pending_payload = dict(info.get("pending_decision_payload_public") or {})
        conservation_slots = dict(info.get("conservation_project_slots") or {})
        opening_setup = dict(info.get("opening_setup") or {})
        total_slot_count = 0
        filled_slot_count = 0
        for levels in conservation_slots.values():
            level_map = dict(levels or {})
            for owner in level_map.values():
                total_slot_count += 1
                if owner is not None:
                    filled_slot_count += 1

        shared_bonus_tiles = dict(info.get("shared_conservation_bonus_tiles") or {})
        claimed_bonus_tiles = dict(info.get("claimed_conservation_bonus_tiles") or {})
        shared_bonus_tokens: List[str] = []
        claimed_bonus_tokens: List[str] = []
        for threshold, tiles in sorted(shared_bonus_tiles.items()):
            for tile in sorted(list(tiles or [])):
                shared_bonus_tokens.append(f"{threshold}:{str(tile).strip().lower()}")
        for threshold, tiles in sorted(claimed_bonus_tiles.items()):
            for tile in sorted(list(tiles or [])):
                claimed_bonus_tokens.append(f"{threshold}:{str(tile).strip().lower()}")

        opening_tokens: List[str] = []
        opening_tokens.extend(
            f"c2:{str(item).strip().lower()}"
            for item in list(opening_setup.get("conservation_space_2_fixed_options") or [])
        )
        opening_tokens.extend(
            f"c5:{str(item).strip().lower()}"
            for item in list(opening_setup.get("conservation_space_5_bonus_tiles") or [])
        )
        opening_tokens.extend(
            f"c8:{str(item).strip().lower()}"
            for item in list(opening_setup.get("conservation_space_8_bonus_tiles") or [])
        )
        opening_tokens.append(
            f"c10:{str(opening_setup.get('conservation_space_10_rule') or '').strip().lower()}"
        )
        for project in list(opening_setup.get("base_conservation_projects") or []):
            if isinstance(project, dict):
                opening_tokens.append(f"base:{str(project.get('data_id') or '').strip().lower()}")
        for blocked in list(opening_setup.get("two_player_blocked_project_levels") or []):
            if not isinstance(blocked, dict):
                continue
            opening_tokens.append(
                "blocked:"
                f"{str(blocked.get('project_data_id') or '').strip().lower()}:"
                f"{str(blocked.get('blocked_level') or '').strip().lower()}"
            )

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
            _norm(info.get("endgame_trigger_turn_index", 0), 200.0),
            1.0 if info.get("endgame_trigger_player") is not None else 0.0,
            1.0 if info.get("break_trigger_player") is not None else 0.0,
        ]
        vec.extend(self._encode_map_name_hash(str(info.get("map_image_name") or "")).tolist())
        vec.extend(self._encode_current_player_one_hot(_safe_int(info.get("current_player"), 0)))
        vec.extend(self._encode_pending_kind_one_hot(pending_kind))
        vec.extend(self._encode_pending_player_one_hot(pending_player))
        vec.extend(self._encode_pending_player_one_hot(info.get("endgame_trigger_player")))
        vec.extend(self._encode_pending_player_one_hot(info.get("break_trigger_player")))
        vec.extend(self._encode_pending_payload(pending_payload))
        vec.extend(self._encode_token_hash(info.get("available_partner_zoos") or [], dim=self.set_hash_dim, dedupe=True))
        vec.extend(self._encode_token_hash(info.get("available_universities") or [], dim=self.set_hash_dim, dedupe=True))
        vec.extend(_hash_tokens(shared_bonus_tokens, self.set_hash_dim).tolist())
        vec.extend(_hash_tokens(claimed_bonus_tokens, self.set_hash_dim).tolist())
        vec.extend(_hash_tokens(opening_tokens, self.set_hash_dim).tolist())
        vec.extend(
            self._encode_card_slots(
                list(info.get("zoo_display") or []),
                max_cards=self.max_display_cards,
                include_private_tokens=False,
            )
        )
        vec.extend(self._encode_conservation_slot_details(conservation_slots))

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
        opening_draft_kept_indices = list(player.get("opening_draft_kept_indices") or [])
        pouched_cards = list(player.get("pouched_cards") or [])
        pouched_by_host = dict(player.get("pouched_cards_by_host") or {})
        pouched_by_host_cards = dict(player.get("pouched_cards_by_host_cards") or {})
        pouched_by_host_public = list(player.get("pouched_cards_by_host_public") or [])
        pouched_host_public_counts = list(player.get("pouched_host_public_counts") or [])
        host_public_card_tokens: List[str] = []
        host_card_total = 0
        host_with_cards = 0
        host_max_stack = 0
        if pouched_by_host_public:
            for raw_entry in pouched_by_host_public:
                if not isinstance(raw_entry, dict):
                    continue
                host_public_index_raw = raw_entry.get("host_public_index")
                host_public_index = (
                    None
                    if host_public_index_raw is None
                    else _safe_int(host_public_index_raw, -1)
                )
                host_number = raw_entry.get("host_number")
                cards = [dict(card) for card in list(raw_entry.get("cards") or []) if isinstance(card, dict)]
                card_count = len(cards)
                host_card_total += card_count
                if card_count > 0:
                    host_with_cards += 1
                    host_max_stack = max(host_max_stack, card_count)
                if host_public_index is not None and host_public_index >= 0:
                    host_token = f"slot:{host_public_index}"
                elif host_number is not None:
                    host_token = f"host_number:{_safe_int(host_number, -1)}"
                else:
                    host_token = "slot:unknown"
                for card in cards:
                    card_number = _safe_int(card.get("number", -1), -1)
                    card_type = str(card.get("card_type") or "").strip().lower()
                    host_public_card_tokens.append(
                        f"{host_token}|num:{card_number}|type:{card_type}"
                    )
        elif pouched_by_host_cards:
            for raw_cards in pouched_by_host_cards.values():
                cards = [dict(card) for card in list(raw_cards or []) if isinstance(card, dict)]
                card_count = len(cards)
                host_card_total += card_count
                if card_count > 0:
                    host_with_cards += 1
                    host_max_stack = max(host_max_stack, card_count)
                for card in cards:
                    card_number = _safe_int(card.get("number", -1), -1)
                    card_type = str(card.get("card_type") or "").strip().lower()
                    host_public_card_tokens.append(f"slot:unknown|num:{card_number}|type:{card_type}")
        else:
            for host, raw_ids in sorted(pouched_by_host.items(), key=lambda item: str(item[0])):
                del host
                ids = [str(item).strip().lower() for item in list(raw_ids or []) if str(item).strip()]
                card_count = len(ids)
                host_card_total += card_count
                if card_count > 0:
                    host_with_cards += 1
                    host_max_stack = max(host_max_stack, card_count)

        vec: List[float] = [
            *self._encode_pending_player_one_hot(info.get("viewer_player_id")),
            _norm(player.get("hand_count", 0), 30.0),
            _norm(player.get("final_scoring_count", 0), 8.0),
            _norm(player.get("opening_draft_drawn_count", 0), 8.0),
            _norm(len(opening_draft_kept_indices), 8.0),
            _norm(len(pouched_cards), 30.0),
            _norm(len(pouched_by_host), 30.0),
            _norm(host_card_total, 40.0),
            _norm(host_with_cards, 40.0),
            _norm(host_max_stack, 10.0),
            _norm(player.get("zoo_cards_public_count", 0), 80.0),
            _norm(player.get("legacy_private_deck_count", 0), 80.0),
            _norm(player.get("legacy_private_discard_count", 0), 80.0),
        ]
        vec.extend(self._encode_token_hash(opening_draft_kept_indices, dim=self.set_hash_dim, dedupe=True))
        vec.extend(
            self._encode_dense_count_slots(
                pouched_host_public_counts,
                max_slots=self.max_zoo_cards,
                scale=4.0,
            )
        )
        vec.extend(_hash_tokens(host_public_card_tokens, self.set_hash_dim).tolist())
        hand_summary = self._encode_hand_summary(hand_cards)
        final_summary = self._encode_final_scoring_summary(final_cards)
        draft_summary = self._encode_opening_draft_summary(opening_draft_cards)
        hand_slots = self._encode_card_slots(
            hand_cards,
            max_cards=self.max_private_hand_cards,
            include_private_tokens=True,
        )
        final_slots = self._encode_setup_card_slots(final_cards, max_cards=self.max_private_final_cards)
        draft_slots = self._encode_card_slots(
            opening_draft_cards,
            max_cards=self.max_private_draft_cards,
            include_private_tokens=True,
        )
        pouched_slots = self._encode_card_slots(
            pouched_cards,
            max_cards=self.max_private_pouched_cards,
            include_private_tokens=True,
        )
        base = np.asarray(vec, dtype=np.float32)
        return np.concatenate(
            [
                base,
                hand_summary,
                final_summary,
                draft_summary,
                np.asarray(hand_slots, dtype=np.float32),
                np.asarray(final_slots, dtype=np.float32),
                np.asarray(draft_slots, dtype=np.float32),
                np.asarray(pouched_slots, dtype=np.float32),
            ],
            axis=0,
        ).astype(np.float32)

    def encode_global_state_from_state(
        self,
        state: main.GameState,
        *,
        public_vec: Optional[np.ndarray] = None,
        private_obs_by_player: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> np.ndarray:
        if public_vec is None:
            viewer_id = 0 if state.players else 0
            public_obs = main.build_public_observation(state, viewer_player_id=viewer_id)
            public_vec = self.encode_public_observation(public_obs)

        private_blocks: List[np.ndarray] = []
        if self._empty_private_vec is None:
            self._empty_private_vec = self.encode_private_observation({})
        for player_id in range(self.max_players):
            if player_id >= len(state.players):
                private_blocks.append(np.zeros_like(self._empty_private_vec))
                continue
            if private_obs_by_player is not None and player_id < len(private_obs_by_player):
                private_obs = private_obs_by_player[player_id]
            else:
                private_obs = main.build_private_observation(state, viewer_player_id=player_id)
            private_blocks.append(self.encode_private_observation(private_obs))

        return np.concatenate(
            [public_vec.astype(np.float32)] + [block.astype(np.float32) for block in private_blocks],
            axis=0,
        ).astype(np.float32)

    def encode_local_state_from_state(
        self,
        state: main.GameState,
        viewer_player_id: int,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        public_obs = main.build_public_observation(state, viewer_player_id=viewer_player_id)
        private_obs = main.build_private_observation(state, viewer_player_id=viewer_player_id)
        public_vec = self.encode_public_observation(public_obs)
        private_vec = self.encode_private_observation(private_obs)
        return (
            np.concatenate([public_vec, private_vec], axis=0).astype(np.float32),
            public_vec,
            private_obs,
        )

    def encode_from_state(
        self,
        state: main.GameState,
        viewer_player_id: int,
        *,
        include_global: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        local_vec, public_vec, viewer_private_obs = self.encode_local_state_from_state(state, viewer_player_id)
        if not include_global:
            return local_vec, self._empty_global_vec
        private_obs_by_player: List[Dict[str, Any]] = []
        for player_id in range(len(state.players)):
            if player_id == int(viewer_player_id):
                private_obs_by_player.append(viewer_private_obs)
                continue
            private_obs_by_player.append(
                main.build_private_observation(state, viewer_player_id=player_id)
            )
        global_vec = self.encode_global_state_from_state(
            state,
            public_vec=public_vec,
            private_obs_by_player=private_obs_by_player,
        )
        return local_vec, global_vec

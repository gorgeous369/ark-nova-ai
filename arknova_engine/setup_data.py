"""Data loading helpers extracted from main setup flow.

This module keeps card/map dataset parsing and setup-pool extraction outside
the runtime loop so main.py can stay focused on game rules and interaction.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar


TCard = TypeVar("TCard")
TSetupCard = TypeVar("TSetupCard")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _safe_card_number(raw_value: object) -> Optional[int]:
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str) and raw_value.isdigit():
        return int(raw_value)
    return None


def _safe_int(raw_value: object, default: int = 0) -> int:
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, str):
        value = raw_value.strip()
        if value.startswith("-"):
            payload = value[1:]
            if payload.isdigit():
                return int(value)
        if value.isdigit():
            return int(value)
    return default


def resolve_cards_dataset_path(repo_root: Optional[Path] = None) -> Path:
    root = repo_root or _repo_root()
    preferred = root / "data" / "cards" / "cards.base.json"
    if preferred.exists():
        return preferred
    return root / "data" / "cards" / "cards.json"


@lru_cache(maxsize=8)
def _load_cards_payload(dataset_path: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Cards payload must be object: {path}")
    return payload


def load_animal_cards_from_dataset(
    *,
    card_factory: Callable[..., TCard],
    allowed_card_types: Sequence[str],
    canonical_icon_key: Callable[[str], str],
    dataset_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[TCard, ...]:
    path = dataset_path or resolve_cards_dataset_path(repo_root=repo_root)
    if not path.exists():
        return tuple()

    payload = _load_cards_payload(str(path))
    cards = payload.get("cards", [])
    if not isinstance(cards, list):
        return tuple()

    zoo_cards: List[TCard] = []
    seen_instance: Dict[str, int] = {}
    allowed_types = {str(card_type).strip() for card_type in allowed_card_types}
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_type = str(card.get("type") or "").strip()
        if card_type not in allowed_types:
            continue
        number = _safe_card_number(card.get("number"))
        if number is None:
            continue

        title = str(card.get("title") or card.get("data_id") or f"CARD_{number}")
        cost = _safe_int(card.get("cost"), default=_safe_int(card.get("level"), default=0))
        size = _safe_int(card.get("enclosure_size"), default=0)
        appeal = _safe_int(card.get("appeal"), default=0)
        conservation = _safe_int(card.get("conservation"), default=0)
        reputation_gain = _safe_int(
            card.get("reputation_gain"),
            default=_safe_int(card.get("reputation"), default=0),
        )
        raw_badges = card.get("badges") if isinstance(card.get("badges"), list) else []
        badges = tuple(str(badge) for badge in raw_badges if isinstance(badge, str) and badge.strip())
        raw_required_icons = card.get("required_icons") if isinstance(card.get("required_icons"), dict) else {}
        required_icons: List[Tuple[str, int]] = []
        for raw_icon, raw_need in raw_required_icons.items():
            if not isinstance(raw_icon, str):
                continue
            need = _safe_int(raw_need, default=0)
            if need <= 0:
                continue
            required_icons.append((canonical_icon_key(raw_icon), need))
        required_icons_tuple = tuple(sorted(required_icons))
        required_water_adjacency = _safe_int(card.get("required_water_adjacency"), default=0)
        required_rock_adjacency = _safe_int(card.get("required_rock_adjacency"), default=0)
        ability_title = str(card.get("ability_title") or "").strip()
        ability_text = str(card.get("ability_text") or "").strip()
        raw_effects = card.get("effects") if isinstance(card.get("effects"), list) else []
        parsed_effects: List[Tuple[str, str]] = []
        for raw_effect in raw_effects:
            if not isinstance(raw_effect, dict):
                continue
            kind = str(raw_effect.get("kind") or "").strip()
            effect_text = str(raw_effect.get("text") or "").strip()
            if not kind and not effect_text:
                continue
            parsed_effects.append((kind, effect_text))
        effects_tuple = tuple(parsed_effects)
        reptile_house_size = (
            _safe_int(card.get("reptile_house_size"), default=0)
            if "reptile_house_size" in card
            else None
        )
        large_bird_aviary_size = (
            _safe_int(card.get("large_bird_aviary_size"), default=0)
            if "large_bird_aviary_size" in card
            else None
        )
        base_instance = str(number)
        counter = seen_instance.get(base_instance, 0) + 1
        seen_instance[base_instance] = counter
        instance_id = base_instance if counter == 1 else f"{base_instance}-{counter}"

        zoo_cards.append(
            card_factory(
                name=title,
                cost=cost,
                size=size,
                appeal=appeal,
                conservation=conservation,
                reputation_gain=reputation_gain,
                card_type=card_type,
                badges=badges,
                required_water_adjacency=required_water_adjacency,
                required_rock_adjacency=required_rock_adjacency,
                required_icons=required_icons_tuple,
                ability_title=ability_title,
                ability_text=ability_text,
                effects=effects_tuple,
                reptile_house_size=reptile_house_size,
                large_bird_aviary_size=large_bird_aviary_size,
                number=number,
                instance_id=instance_id,
            )
        )

    zoo_cards.sort(key=lambda c: (getattr(c, "number", -1), getattr(c, "instance_id", "")))
    return tuple(zoo_cards)


@lru_cache(maxsize=64)
def _load_map_tiles_payload(tiles_path: str) -> Dict[str, Any]:
    path = Path(tiles_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Map tiles payload must be object: {path}")
    return payload


def build_map_tile_bonus_map(
    *,
    map_image_name: str,
    repo_root: Optional[Path] = None,
) -> Dict[Tuple[int, int], str]:
    root = repo_root or _repo_root()
    path = root / "data" / "maps" / "tiles" / f"{map_image_name}.tiles.json"
    if not path.exists():
        raise FileNotFoundError(f"Map tiles file not found: {path}")
    payload = _load_map_tiles_payload(str(path))
    tiles = payload.get("tiles", [])
    if not isinstance(tiles, list):
        return {}
    bonus_map: Dict[Tuple[int, int], str] = {}
    for raw_tile in tiles:
        if not isinstance(raw_tile, dict):
            continue
        x = raw_tile.get("x")
        y = raw_tile.get("y")
        bonus = raw_tile.get("placement_bonus")
        if isinstance(x, int) and isinstance(y, int) and isinstance(bonus, str) and bonus:
            bonus_map[(x, y)] = bonus
    return bonus_map


def load_base_setup_card_pools(
    *,
    setup_card_factory: Callable[..., TSetupCard],
    dataset_path: Optional[Path] = None,
    repo_root: Optional[Path] = None,
) -> Tuple[List[TSetupCard], List[TSetupCard]]:
    path = dataset_path or resolve_cards_dataset_path(repo_root=repo_root)
    if not path.exists():
        fallback_final = [
            setup_card_factory(data_id=f"F{i:03d}_Fallback", title=f"Final Scoring {i}")
            for i in range(1, 12)
        ]
        fallback_projects = [
            setup_card_factory(data_id=f"P{100 + i:03d}_Fallback", title=f"Base Project {i}")
            for i in range(1, 13)
        ]
        return fallback_final, fallback_projects

    payload = _load_cards_payload(str(path))
    cards = payload.get("cards", [])
    if not isinstance(cards, list):
        return [], []

    final_cards: List[TSetupCard] = []
    base_projects: List[TSetupCard] = []
    for card in cards:
        if not isinstance(card, dict):
            continue
        card_type = card.get("type")
        card_id = str(card.get("data_id", "")).strip()
        if not card_id:
            continue
        title = str(card.get("title") or card.get("subtitle") or card_id)
        number = _safe_card_number(card.get("number"))
        if card_type == "final_scoring" and number is not None and 1 <= number <= 11:
            final_cards.append(setup_card_factory(data_id=card_id, title=title))
        elif card_type == "conservation_project" and number is not None and 101 <= number <= 112:
            base_projects.append(setup_card_factory(data_id=card_id, title=title))

    final_cards.sort(key=lambda c: str(getattr(c, "data_id", "")))
    base_projects.sort(key=lambda c: str(getattr(c, "data_id", "")))

    if len(final_cards) < 11 or len(base_projects) < 12:
        raise ValueError(
            "Card dataset is missing base final scoring cards or base conservation project cards."
        )
    return final_cards, base_projects

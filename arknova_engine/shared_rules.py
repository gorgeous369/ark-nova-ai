from __future__ import annotations

from typing import Callable, Dict, Iterable, List, MutableMapping, Optional, Sequence, Tuple, TypeVar


TKey = TypeVar("TKey")


MAX_X_TOKENS = 5
DEFAULT_NUM_PLAYERS = 2
DEFAULT_MAP_IMAGE_NAME = "plan1a"
STARTING_MONEY = 25
STARTING_HAND_DRAW_COUNT = 8
STARTING_HAND_KEEP_COUNT = 4
FINAL_SCORING_DECK_SIZE = 11
DONATION_COST_TRACK: Tuple[int, ...] = (2, 5, 7, 10, 12)

MAIN_ACTION_CARDS: Tuple[str, ...] = (
    "animals",
    "cards",
    "build",
    "association",
    "sponsors",
)
ALL_PARTNER_ZOOS: Tuple[str, ...] = ("africa", "europe", "asia", "america", "australia")
BREAK_TRACK_BY_PLAYERS: Dict[int, int] = {2: 9, 3: 12, 4: 15}

CARDS_DRAW_TABLE_BASE: Tuple[int, ...] = (1, 1, 2, 2, 3)
CARDS_DISCARD_TABLE_BASE: Tuple[int, ...] = (1, 0, 1, 0, 1)
ANIMALS_PLAY_LIMIT_BASE: Tuple[int, ...] = (0, 1, 1, 1, 2)

CARDS_DRAW_TABLE_UPGRADED: Tuple[int, ...] = (1, 2, 2, 3, 4)
CARDS_DISCARD_TABLE_UPGRADED: Tuple[int, ...] = (0, 1, 0, 1, 1)
ANIMALS_PLAY_LIMIT_UPGRADED: Tuple[int, ...] = (1, 1, 2, 2, 2)
CARDS_SNAP_ALLOWED_BASE: Tuple[bool, ...] = (False, False, False, False, True)
CARDS_SNAP_ALLOWED_UPGRADED: Tuple[bool, ...] = (False, False, True, True, True)


def ordered_action_card_names() -> List[str]:
    return list(MAIN_ACTION_CARDS)


def list_factory(items: Iterable[TKey]) -> Callable[[], List[TKey]]:
    cached_items = tuple(items)

    def factory() -> List[TKey]:
        return list(cached_items)

    return factory


def action_order_factory(action_factory: Callable[[str], TKey]) -> Callable[[], List[TKey]]:
    def factory() -> List[TKey]:
        return [action_factory(action_name) for action_name in MAIN_ACTION_CARDS]

    return factory


def zero_map(keys: Iterable[TKey]) -> Dict[TKey, int]:
    return {key: 0 for key in keys}


def false_map(keys: Iterable[TKey]) -> Dict[TKey, bool]:
    return {key: False for key in keys}


def zero_map_factory(keys: Iterable[TKey]) -> Callable[[], Dict[TKey, int]]:
    cached_keys = tuple(keys)

    def factory() -> Dict[TKey, int]:
        return zero_map(cached_keys)

    return factory


def false_map_factory(keys: Iterable[TKey]) -> Callable[[], Dict[TKey, bool]]:
    cached_keys = tuple(keys)

    def factory() -> Dict[TKey, bool]:
        return false_map(cached_keys)

    return factory


def reset_int_map(mapping: MutableMapping[TKey, int], keys: Iterable[TKey]) -> None:
    for key in keys:
        mapping[key] = 0


def clear_action_token_maps(action_keys: Iterable[TKey], *token_maps: MutableMapping[TKey, int]) -> None:
    cached_keys = tuple(action_keys)
    for token_map in token_maps:
        reset_int_map(token_map, cached_keys)


def action_base_strength(action_order: Sequence[TKey], action: TKey) -> int:
    try:
        if hasattr(action_order, "index"):
            return action_order.index(action) + 1
        return list(action_order).index(action) + 1
    except ValueError as exc:
        raise ValueError(f"Action {action!r} not found in action order.") from exc


def move_action_to_slot(action_order: Sequence[TKey], action: TKey, slot_number: int) -> List[TKey]:
    order = list(action_order)
    if action not in order:
        return order
    idx = order.index(action)
    card = order.pop(idx)
    target_index = max(0, min(len(order), int(slot_number) - 1))
    order.insert(target_index, card)
    return order


def break_income_order(player_count: int, trigger_player: Optional[int]) -> List[int]:
    total = max(0, int(player_count))
    if trigger_player is None:
        return list(range(total))
    trigger = int(trigger_player)
    if trigger < 0 or trigger >= total:
        return list(range(total))
    return [(trigger + offset) % total for offset in range(total)]


def recall_workers(
    available_workers: int,
    workers_on_board: int,
    *,
    max_workers: Optional[int] = None,
) -> Tuple[int, int]:
    total = max(0, int(available_workers)) + max(0, int(workers_on_board))
    if max_workers is not None:
        total = min(total, int(max_workers))
    return total, 0


def cards_table_values(strength: int, upgraded: bool) -> Tuple[int, int, bool]:
    if strength <= 0:
        return 0, 0, False
    idx = min(5, max(1, int(strength))) - 1
    if upgraded:
        return (
            CARDS_DRAW_TABLE_UPGRADED[idx],
            CARDS_DISCARD_TABLE_UPGRADED[idx],
            CARDS_SNAP_ALLOWED_UPGRADED[idx],
        )
    return (
        CARDS_DRAW_TABLE_BASE[idx],
        CARDS_DISCARD_TABLE_BASE[idx],
        CARDS_SNAP_ALLOWED_BASE[idx],
    )


def animals_play_limit(strength: int, upgraded: bool) -> int:
    if strength <= 0:
        return 0
    idx = min(5, max(1, int(strength))) - 1
    if upgraded:
        return ANIMALS_PLAY_LIMIT_UPGRADED[idx]
    return ANIMALS_PLAY_LIMIT_BASE[idx]


def current_donation_cost(donation_progress: int) -> int:
    idx = min(max(0, int(donation_progress)), len(DONATION_COST_TRACK) - 1)
    return DONATION_COST_TRACK[idx]

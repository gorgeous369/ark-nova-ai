"""Scoring and track progression helpers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Sequence


def bga_conservation_points(conservation: int) -> int:
    cp = max(0, conservation)
    first_ten = min(cp, 10)
    above_ten = max(0, cp - 10)
    return first_ten * 2 + above_ten * 3


def progress_score(player: Any) -> int:
    return int(player.appeal) + bga_conservation_points(int(player.conservation))


def break_income_from_appeal(appeal: int) -> int:
    value = max(0, appeal)
    if value <= 4:
        return 5 + value
    if value <= 16:
        return 10 + (value - 5) // 2
    if value <= 31:
        return 16 + (value - 17) // 3
    if value <= 55:
        return 21 + (value - 32) // 4
    if value <= 95:
        return 27 + (value - 56) // 5
    return 35 + (value - 96) // 6


def discard_down_to_limit(player: Any) -> List[Any]:
    if len(player.hand) <= int(player.hand_limit):
        return []
    overflow = len(player.hand) - int(player.hand_limit)
    discarded = list(player.hand[-overflow:])
    del player.hand[-overflow:]
    return discarded


def upgrade_one_action_card(player: Any, priority: Sequence[str]) -> bool:
    for action in priority:
        if not bool(player.action_upgraded.get(action, False)):
            player.action_upgraded[action] = True
            return True
    return False


def take_one_reputation_bonus_card(
    *,
    state: Any,
    player: Any,
    reputation_display_limit_fn: Callable[[int], int],
    replenish_display_fn: Callable[[Any], None],
) -> None:
    if state.zoo_deck:
        player.hand.append(state.zoo_deck.pop(0))
        return
    limit = int(reputation_display_limit_fn(int(player.reputation)))
    if limit <= 0 or not state.zoo_display:
        return
    idx = 0
    if idx < limit:
        player.hand.append(state.zoo_display.pop(idx))
        replenish_display_fn(state)


def apply_reputation_milestone_reward(
    *,
    state: Any,
    player: Any,
    milestone: int,
    upgrade_one_action_card_fn: Callable[[Any], bool],
    take_one_reputation_bonus_card_fn: Callable[[Any, Any], None],
    max_workers: int,
    max_x_tokens: int,
) -> None:
    if milestone == 5:
        upgrade_one_action_card_fn(player)
        return
    if milestone == 8:
        player.workers = min(max_workers, int(player.workers) + 1)
        return
    if milestone in {10, 13}:
        take_one_reputation_bonus_card_fn(state, player)
        return
    if milestone in {11, 14}:
        player.conservation = int(player.conservation) + 1
        return
    if milestone in {12, 15}:
        player.x_tokens = min(max_x_tokens, int(player.x_tokens) + 1)


def increase_reputation(
    *,
    state: Any,
    player: Any,
    amount: int,
    association_action_key: str,
    reputation_cap_without_upgrade: int,
    reputation_cap_with_upgrade: int,
    milestone_values: Iterable[int],
    apply_milestone_reward_fn: Callable[[Any, Any, int], None],
) -> None:
    if amount <= 0:
        return
    has_association_upgrade = bool(player.action_upgraded.get(association_action_key, False))
    reputation_cap = reputation_cap_with_upgrade if has_association_upgrade else reputation_cap_without_upgrade
    old_rep = int(player.reputation)
    target_rep = old_rep + amount
    if reputation_cap == reputation_cap_with_upgrade and target_rep > reputation_cap_with_upgrade:
        overflow = target_rep - reputation_cap_with_upgrade
        player.appeal = int(player.appeal) + overflow
    player.reputation = min(reputation_cap, target_rep)
    for milestone in milestone_values:
        if old_rep < milestone <= int(player.reputation) and milestone not in player.claimed_reputation_milestones:
            player.claimed_reputation_milestones.add(milestone)
            apply_milestone_reward_fn(state, player, milestone)


def advance_break_track(
    *,
    state: Any,
    steps: int,
    trigger_player: int,
    max_x_tokens: int,
) -> bool:
    if steps <= 0:
        return False
    if int(state.break_progress) >= int(state.break_max):
        return False
    state.break_progress = min(int(state.break_max), int(state.break_progress) + steps)
    if int(state.break_progress) >= int(state.break_max):
        player = state.players[trigger_player]
        player.x_tokens = min(max_x_tokens, int(player.x_tokens) + 1)
        return True
    return False


def resolve_break(
    *,
    state: Any,
    association_task_kinds: Sequence[str],
    max_workers: int,
    discard_down_to_limit_fn: Callable[[Any], List[Any]],
    break_income_from_appeal_fn: Callable[[int], int],
    refresh_association_market_fn: Callable[[Any], None],
) -> None:
    for player in state.players:
        state.zoo_discard.extend(discard_down_to_limit_fn(player))
    for player in state.players:
        player.money = int(player.money) + int(break_income_from_appeal_fn(int(player.appeal)))
        if int(player.workers_on_association_board) > 0:
            player.workers = min(max_workers, int(player.workers) + int(player.workers_on_association_board))
            player.workers_on_association_board = 0
            for task in association_task_kinds:
                player.association_workers_by_task[task] = 0
    refresh_association_market_fn(state)
    state.break_progress = 0

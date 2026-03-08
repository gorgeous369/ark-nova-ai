"""Console interaction helpers."""

from __future__ import annotations

from itertools import combinations
import random
from typing import Any, Callable, List, Optional, Sequence, Set, Tuple


def enumerate_index_combinations(
    *,
    total_items: int,
    min_choose: int,
    max_choose: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    total = max(0, int(total_items))
    lower = max(0, int(min_choose))
    upper = total if max_choose is None else min(total, max(0, int(max_choose)))
    if lower > upper:
        return []
    return [
        choice
        for choose_count in range(lower, upper + 1)
        for choice in combinations(range(total), choose_count)
    ]


def prompt_card_combination_indices(
    *,
    title: str,
    cards: Sequence[Any],
    format_card_line: Callable[[Any], str],
    min_choose: int,
    max_choose: Optional[int],
    action_label: str,
    combination_header: Optional[str] = None,
) -> List[int]:
    choices = enumerate_index_combinations(
        total_items=len(cards),
        min_choose=min_choose,
        max_choose=max_choose,
    )
    if not choices:
        return []

    print(title)
    for idx, card in enumerate(cards, start=1):
        print(f"{idx}. {format_card_line(card)}")

    header = str(combination_header or f"{action_label.title()} combinations:").strip()
    if header:
        print(header)
    for idx, choice in enumerate(choices, start=1):
        label = ",".join(str(value + 1) for value in choice)
        print(f"{idx}. {action_label} [{label}]")

    while True:
        raw = input(f"Select combination [1-{len(choices)}]: ").strip()
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue
        picked = int(raw)
        if not (1 <= picked <= len(choices)):
            print("Index out of range, try again.")
            continue
        return list(choices[picked - 1])


def prompt_opening_draft_indices(
    *,
    player_name: str,
    drafted_cards: List[Any],
    format_card_line: Callable[[Any], str],
) -> List[int]:
    return prompt_card_combination_indices(
        title=f"\n{player_name} opening draft: choose 4 cards to keep from 8:",
        cards=drafted_cards,
        format_card_line=format_card_line,
        min_choose=4,
        max_choose=4,
        action_label="keep",
        combination_header="Opening draft combinations:",
    )


def resolve_manual_opening_drafts(
    *,
    state: Any,
    manual_player_names: Set[str],
    draw_opening_draft_cards_fn: Callable[[Any, Any], List[Any]],
    apply_opening_draft_selection_fn: Callable[..., None],
    prompt_opening_draft_indices_fn: Callable[[str, List[Any]], List[int]],
) -> None:
    for player in state.players:
        if player.name not in manual_player_names:
            continue
        if player.opening_draft_kept_indices:
            continue
        if not player.opening_draft_drawn:
            draft = draw_opening_draft_cards_fn(player, state)
        else:
            draft = list(player.opening_draft_drawn)
        kept = prompt_opening_draft_indices_fn(player.name, draft)
        apply_opening_draft_selection_fn(
            player=player,
            drafted_cards=draft,
            rng=random.Random(0),
            kept_indices=kept,
            discard_sink=state.zoo_discard,
        )

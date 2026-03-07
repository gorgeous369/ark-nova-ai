"""Console interaction helpers."""

from __future__ import annotations

import random
from typing import Any, Callable, List, Set


def prompt_opening_draft_indices(
    *,
    player_name: str,
    drafted_cards: List[Any],
    format_card_line: Callable[[Any], str],
) -> List[int]:
    print(f"\n{player_name} opening draft: choose 4 cards to keep from 8:")
    for idx, card in enumerate(drafted_cards, start=1):
        print(f"{idx}. {format_card_line(card)}")
    while True:
        raw = input("Enter 4 numbers separated by space (e.g. 1 3 6 8): ").strip()
        parts = raw.split()
        if len(parts) != 4:
            print("Please enter exactly 4 numbers.")
            continue
        if not all(part.isdigit() for part in parts):
            print("Only numbers are allowed.")
            continue
        chosen = [int(part) for part in parts]
        if any(not (1 <= value <= len(drafted_cards)) for value in chosen):
            print("Index out of range, try again.")
            continue
        if len(set(chosen)) != 4:
            print("Please choose 4 different cards.")
            continue
        return sorted(value - 1 for value in chosen)


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

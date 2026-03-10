from __future__ import annotations

from typing import Callable, Iterable, Optional

import main


DEFAULT_PLAYER_NAMES = ["P1", "P2"]


def make_state(seed: int, *, include_marine_world: bool = False) -> main.GameState:
    return main.setup_game(
        seed=seed,
        player_names=list(DEFAULT_PLAYER_NAMES),
        include_marine_world=include_marine_world,
    )


def set_action_strength(player: main.PlayerState, action_name: str, strength: int) -> None:
    slot_index = max(1, min(len(main.MAIN_ACTION_CARDS), int(strength))) - 1
    action_order = [name for name in main.MAIN_ACTION_CARDS if name != action_name]
    action_order.insert(slot_index, action_name)
    player.action_order = action_order


def take_card_by_number(state: main.GameState, number: int) -> main.AnimalCard:
    for zone in (state.zoo_deck, state.zoo_display, state.zoo_discard):
        for idx, card in enumerate(zone):
            if card.number == number:
                return zone.pop(idx)
    for player in state.players:
        for idx, card in enumerate(player.hand):
            if card.number == number:
                return player.hand.pop(idx)
    raise AssertionError(f"Card #{number} not found in known zones")


def find_action(
    actions: Iterable[main.Action],
    *,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> main.Action:
    for action in actions:
        details = dict(action.details or {})
        if any(details.get(key) != value for key, value in expected_details.items()):
            continue
        if predicate is not None and not predicate(action):
            continue
        return action
    raise AssertionError(f"Action not found for details={expected_details}")


def has_action(
    actions: Iterable[main.Action],
    *,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> bool:
    try:
        find_action(actions, predicate=predicate, **expected_details)
    except AssertionError:
        return False
    return True


def materialize_first_action(state: main.GameState, player_id: int) -> main.Action:
    player = state.players[player_id]
    legal_actions = main.legal_actions(
        player,
        state=state,
        player_id=player_id,
    )
    if not legal_actions:
        raise AssertionError("No legal concrete actions available.")
    for action in legal_actions:
        if action.type == main.ActionType.PENDING_DECISION:
            return action
    for action in legal_actions:
        if action.type == main.ActionType.MAIN_ACTION and action.card_name == "build":
            return action
    for action in legal_actions:
        if (
            action.type == main.ActionType.MAIN_ACTION
            and action.card_name == "sponsors"
            and bool((action.details or {}).get("use_break_ability"))
        ):
            return action
    for action in legal_actions:
        if action.type == main.ActionType.X_TOKEN:
            return action
    return legal_actions[0]

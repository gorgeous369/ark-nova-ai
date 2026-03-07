"""Turn/action dispatch helpers."""

from __future__ import annotations

from typing import Any, Callable, List


def legal_actions(
    *,
    player: Any,
    action_factory: Callable[..., Any],
    main_action_type: Any,
    x_token_action_type: Any,
    max_x_tokens: int,
) -> List[Any]:
    actions: List[Any] = []
    max_x_spend = max(0, min(max_x_tokens, int(player.x_tokens)))
    for action_name in player.action_order:
        for x_spent in range(0, max_x_spend + 1):
            actions.append(action_factory(main_action_type, value=x_spent, card_name=action_name))
    if int(player.x_tokens) < max_x_tokens:
        actions.append(action_factory(x_token_action_type))
    return actions


def apply_action(
    *,
    state: Any,
    action: Any,
    validate_card_zones_fn: Callable[[Any], None],
    rotate_action_card_to_slot_1_fn: Callable[[Any, str], int],
    perform_main_action_fn: Callable[[Any, Any, int, str, int, Any], bool],
    resolve_break_fn: Callable[[Any], None],
    main_action_type: Any,
    x_token_action_type: Any,
    max_x_tokens: int,
) -> None:
    validate_card_zones_fn(state)
    player_id = int(state.current_player)
    player = state.players[player_id]
    break_triggered = False

    if action.type == main_action_type:
        if not action.card_name:
            raise ValueError("Main action requires card_name.")
        chosen = str(action.card_name)
        x_spent = int(action.value or 0)
        if x_spent < 0:
            raise ValueError("x_spent cannot be negative.")
        if x_spent > int(player.x_tokens):
            raise ValueError("Not enough X-tokens for selected action.")
        base_strength = rotate_action_card_to_slot_1_fn(player, chosen)
        player.x_tokens = int(player.x_tokens) - x_spent
        strength = base_strength + x_spent
        break_triggered = bool(
            perform_main_action_fn(state, player, player_id, chosen, strength, action.details)
        )
    elif action.type == x_token_action_type:
        if int(player.x_tokens) >= max_x_tokens:
            raise ValueError("Cannot take X action at max X-token limit.")
        chosen = action.card_name or player.action_order[-1]
        rotate_action_card_to_slot_1_fn(player, chosen)
        player.x_tokens = min(max_x_tokens, int(player.x_tokens) + 1)
    else:
        raise ValueError("Unsupported action type in this runner.")

    if break_triggered:
        resolve_break_fn(state)

    validate_card_zones_fn(state)
    state.turn_index = int(state.turn_index) + 1
    state.current_player = (int(state.current_player) + 1) % len(state.players)

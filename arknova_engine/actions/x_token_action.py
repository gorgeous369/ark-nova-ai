from __future__ import annotations

from arknova_engine.base_game import GameState, MAX_X_TOKENS, MainAction


def run_x_token_action(gs: GameState, chosen_action: MainAction) -> None:
    player_id = gs.current_player
    player = gs.current()

    if chosen_action not in player.action_order:
        raise ValueError(f"Action {chosen_action} is not available in current action order.")
    if player.x_tokens >= MAX_X_TOKENS:
        raise ValueError("Cannot perform X-token action at maximum X-token limit.")

    player.x_tokens += 1
    gs.action_log.append(f"P{player_id} XTOKEN action={chosen_action.value}")
    gs._complete_main_action(player_id, chosen_action)

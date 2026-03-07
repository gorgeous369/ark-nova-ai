from __future__ import annotations

from typing import List, Optional

from arknova_engine.base_game import GameState, MainAction


def run_cards_action(
    gs: GameState,
    from_display_indices: Optional[List[int]] = None,
    from_deck_count: Optional[int] = None,
    snap_display_index: Optional[int] = None,
    discard_hand_indices: Optional[List[int]] = None,
    x_spent: int = 0,
) -> None:
    player_id = gs.current_player
    player = gs.current()
    base_strength = gs._slot_strength(player, MainAction.CARDS)
    gs._consume_x_tokens_for_action(player, x_spent)
    strength = base_strength + x_spent
    upgraded = player.action_upgraded[MainAction.CARDS]
    draw_target, discard_target, snap_allowed = gs._cards_table_values(strength, upgraded)

    gs.advance_break_token(2, player_id=player_id)

    if from_display_indices is None:
        from_display_indices = []
    if snap_display_index is not None:
        if from_display_indices or from_deck_count:
            raise ValueError("Snap cannot be combined with normal draw selections.")
        if not snap_allowed:
            raise ValueError("Snap is not available at this action strength.")
        gs._validate_display_index(snap_display_index)
        player.hand.append(gs.display.pop(snap_display_index))
        gs._replenish_display_end_of_turn()
        gs.action_log.append(
            f"P{player_id} CARDS snap strength={strength} upgraded={upgraded} idx={snap_display_index}"
        )
        gs._complete_main_action(player_id, MainAction.CARDS)
        return

    if not upgraded and from_display_indices:
        raise ValueError("Cards side I cannot draw from display (except snap).")

    for idx in from_display_indices:
        gs._validate_display_index(idx)
    if upgraded:
        max_accessible = player.reputation + 1
        for idx in from_display_indices:
            if idx + 1 > max_accessible:
                raise ValueError("Selected display card is outside reputation range.")

    if from_deck_count is None:
        from_deck_count = draw_target - len(from_display_indices)
    if from_deck_count < 0:
        raise ValueError("from_deck_count cannot be negative.")

    total_draw = len(from_display_indices) + from_deck_count
    if total_draw > draw_target:
        raise ValueError(f"Cards action can draw at most {draw_target} card(s) at this strength.")

    if from_deck_count > len(gs.deck):
        raise ValueError("Not enough cards in deck for requested draw.")

    taken_display = gs._take_display_cards(from_display_indices)
    player.hand.extend(taken_display)
    for _ in range(from_deck_count):
        player.hand.append(gs.deck.pop(0))

    gs._resolve_cards_discard(player, discard_target, discard_hand_indices)
    if from_display_indices:
        gs._replenish_display_end_of_turn()

    gs.action_log.append(
        "P{} CARDS draw={} display={} deck={} discard={} strength={} upgraded={}".format(
            player_id,
            total_draw,
            len(from_display_indices),
            from_deck_count,
            discard_target,
            strength,
            upgraded,
        )
    )
    gs._complete_main_action(player_id, MainAction.CARDS)

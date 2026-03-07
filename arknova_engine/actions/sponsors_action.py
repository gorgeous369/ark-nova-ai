from __future__ import annotations

from typing import List, Optional

from arknova_engine.base_game import CardSource, GameState, MainAction, SponsorCard, SponsorPlaySelection


def run_sponsors_action(
    gs: GameState,
    selections: Optional[List[SponsorPlaySelection]] = None,
    use_break_ability: bool = False,
    x_spent: int = 0,
) -> None:
    player_id = gs.current_player
    player = gs.current()
    base_strength = gs._slot_strength(player, MainAction.SPONSORS)
    gs._consume_x_tokens_for_action(player, x_spent)
    strength = base_strength + x_spent
    is_sponsors_upgraded = player.action_upgraded[MainAction.SPONSORS]

    if selections is None:
        selections = []
    if use_break_ability and selections:
        raise ValueError("Sponsors break alternative cannot be combined with sponsor plays.")

    if use_break_ability:
        gs.advance_break_token(strength, player_id=player_id)
        money_gain = strength * (2 if is_sponsors_upgraded else 1)
        player.money += money_gain
        gs.action_log.append(
            f"P{player_id} SPONSORS break strength={strength} upgraded={is_sponsors_upgraded} money+={money_gain}"
        )
        gs._complete_main_action(player_id, MainAction.SPONSORS)
        return

    if not selections:
        raise ValueError("Sponsors action must either play sponsor card(s) or use break alternative.")

    if not is_sponsors_upgraded and len(selections) != 1:
        raise ValueError("Sponsors side I requires exactly one sponsor card.")

    total_level = 0
    for selection in selections:
        sponsor, _ = gs._sponsor_from_source(
            player=player,
            selection=selection,
            is_sponsors_upgraded=is_sponsors_upgraded,
        )
        total_level += sponsor.level

    level_cap = strength + 1 if is_sponsors_upgraded else strength
    if total_level > level_cap:
        raise ValueError(
            f"Sponsor levels total {total_level} exceeds allowed maximum {level_cap}."
        )

    took_from_display = False
    pending_after_finish_effects: List[str] = []
    for selection in selections:
        sponsor, display_extra_cost = gs._sponsor_from_source(
            player=player,
            selection=selection,
            is_sponsors_upgraded=is_sponsors_upgraded,
        )
        gs._validate_sponsor_conditions(
            player=player,
            sponsor=sponsor,
            is_sponsors_upgraded=is_sponsors_upgraded,
        )

        if player.money < display_extra_cost:
            raise ValueError("Insufficient money for sponsor display additional cost.")

        if selection.source == CardSource.HAND:
            popped = player.hand.pop(selection.source_index)
        else:
            popped = gs.display.pop(selection.source_index)
            took_from_display = True
        if not isinstance(popped, SponsorCard):
            raise ValueError("Selected card is not a SponsorCard.")

        player.money -= display_extra_cost
        player.appeal += popped.appeal_gain
        player.reputation += popped.reputation_gain
        player.conservation += popped.conservation_gain
        player.recurring_break_income += popped.recurring_break_income_gain
        for icon, amount in popped.granted_icons.items():
            player.zoo_icons[icon] = player.zoo_icons.get(icon, 0) + amount
        player.played_sponsors.append(popped)

        if popped.after_finishing_effect_label:
            pending_after_finish_effects.append(popped.after_finishing_effect_label)

    if took_from_display:
        gs._replenish_display_end_of_turn()

    if pending_after_finish_effects:
        gs.action_log.append(
            f"P{player_id} SPONSORS after_finish_effects={pending_after_finish_effects}"
        )
    gs.action_log.append(
        "P{} SPONSORS play={} total_level={} strength={} upgraded={}".format(
            player_id,
            len(selections),
            total_level,
            strength,
            is_sponsors_upgraded,
        )
    )
    gs._complete_main_action(player_id, MainAction.SPONSORS)

from __future__ import annotations

from typing import List

from arknova_engine.base_game import (
    AnimalCard,
    AnimalPlacement,
    AnimalPlaySelection,
    CardSource,
    GameState,
    MainAction,
)
from arknova_engine.map_model import BuildingSubType


def run_animals_action(
    gs: GameState,
    selections: List[AnimalPlaySelection],
    x_spent: int = 0,
    take_strength5_reputation_bonus: bool = False,
) -> None:
    player_id = gs.current_player
    player = gs.current()
    base_strength = gs._slot_strength(player, MainAction.ANIMALS)
    gs._consume_x_tokens_for_action(player, x_spent)
    strength = base_strength + x_spent
    is_animals_upgraded = player.action_upgraded[MainAction.ANIMALS]
    play_limit = gs._animals_play_limit(strength, is_animals_upgraded)

    if len(selections) > play_limit:
        raise ValueError(f"Animals action can play at most {play_limit} animal card(s) at this strength.")

    if take_strength5_reputation_bonus:
        if not is_animals_upgraded or strength < 5:
            raise ValueError("Strength-5 reputation bonus is only available on upgraded Animals at strength 5+.")
        player.reputation += 1

    took_from_display = False
    pending_after_finish_effects: List[str] = []

    for selection in selections:
        animal, display_extra_cost = gs._animal_from_source(
            player=player,
            selection=selection,
            is_animals_upgraded=is_animals_upgraded,
        )

        gs._validate_animal_conditions(
            player=player,
            animal=animal,
            is_animals_upgraded=is_animals_upgraded,
        )

        enclosure = player.zoo_map.buildings.get(selection.enclosure_origin)
        if enclosure is None:
            raise ValueError("Target enclosure does not exist.")
        spaces_used = gs._validate_animal_enclosure(player, animal, enclosure)

        discounted_cost = max(0, animal.cost - gs._partner_zoo_discount(player, animal))
        total_cost = discounted_cost + display_extra_cost
        if player.money < total_cost:
            raise ValueError("Insufficient money to play selected animal.")
        if selection.source == CardSource.HAND:
            popped = player.hand.pop(selection.source_index)
        else:
            popped = gs.display.pop(selection.source_index)
            took_from_display = True
        if not isinstance(popped, AnimalCard):
            raise ValueError("Selected card is not an AnimalCard.")
        player.money -= total_cost

        if enclosure.type.subtype == BuildingSubType.ENCLOSURE_BASIC:
            enclosure.empty_spaces = 0
        else:
            enclosure.empty_spaces -= spaces_used

        player.appeal += animal.appeal_gain
        player.reputation += animal.reputation_gain
        player.conservation += animal.conservation_gain

        for icon_group in (animal.continent_icons, animal.category_icons):
            for icon, amount in icon_group.items():
                player.zoo_icons[icon] = player.zoo_icons.get(icon, 0) + amount
        if animal.required_water_adjacency:
            player.zoo_icons["water"] = (
                player.zoo_icons.get("water", 0) + animal.required_water_adjacency
            )
        if animal.required_rock_adjacency:
            player.zoo_icons["rock"] = (
                player.zoo_icons.get("rock", 0) + animal.required_rock_adjacency
            )

        player.played_animals.append(popped)
        player.animal_placements[popped.card_id] = AnimalPlacement(
            enclosure_origin=selection.enclosure_origin,
            enclosure_type=enclosure.type,
            spaces_used=spaces_used,
        )
        if popped.after_finishing_effect_label:
            pending_after_finish_effects.append(popped.after_finishing_effect_label)

    if took_from_display:
        gs._replenish_display_end_of_turn()

    if pending_after_finish_effects:
        gs.action_log.append(
            f"P{player_id} ANIMALS after_finish_effects={pending_after_finish_effects}"
        )

    gs.action_log.append(
        "P{} ANIMALS play={} strength={} upgraded={} rep_bonus={}".format(
            player_id,
            len(selections),
            strength,
            is_animals_upgraded,
            take_strength5_reputation_bonus,
        )
    )
    gs._complete_main_action(player_id, MainAction.ANIMALS)

from __future__ import annotations

from typing import List, Set

from arknova_engine.base_game import BuildSelection, BuildingType, MainAction, GameState


def run_build_action(
    gs: GameState,
    selections: List[BuildSelection],
    x_spent: int = 0,
    has_diversity_researcher: bool = False,
) -> None:
    player_id = gs.current_player
    player = gs.current()

    base_strength = gs._slot_strength(player, MainAction.BUILD)
    gs._consume_x_tokens_for_action(player, x_spent)
    strength = base_strength + x_spent

    is_build_upgraded = player.action_upgraded[MainAction.BUILD]
    if not is_build_upgraded and len(selections) != 1:
        raise ValueError("Build side I requires exactly one building.")
    if is_build_upgraded and len(selections) < 1:
        raise ValueError("Build side II requires at least one building.")

    remaining_size = strength
    built_this_action: Set[BuildingType] = set()

    for selection in selections:
        legal = player.zoo_map.legal_building_placements(
            is_build_upgraded=is_build_upgraded,
            has_diversity_researcher=has_diversity_researcher,
            max_building_size=remaining_size,
            already_built_buildings=built_this_action,
        )
        picked = gs._find_matching_building(legal, selection)
        if picked is None:
            raise ValueError(f"Illegal build selection: {selection}")

        size = len(picked.layout)
        cost = size * 2
        if player.money < cost:
            raise ValueError("Insufficient money to build selected building.")

        player.money -= cost
        player.zoo_map.add_building(picked)
        gs._apply_building_placement_effects(player, picked)
        built_this_action.add(picked.type)
        remaining_size -= size

    if gs._map_cover_bonus_reached(player):
        player.map_cover_bonus_claimed = True
        player.appeal += 7

    gs.action_log.append(
        f"P{player_id} BUILD strength={strength} upgraded={is_build_upgraded} buildings={len(selections)}"
    )
    gs._complete_main_action(player_id, MainAction.BUILD)

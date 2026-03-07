"""Setup flow helpers extracted from main game loop."""

from __future__ import annotations

import random
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


def draw_opening_draft_cards(
    *,
    player: Any,
    state: Any,
    draw_from_zoo_deck: Callable[[Any, int], List[Any]],
) -> List[Any]:
    drafted_cards = list(draw_from_zoo_deck(state, 8))
    player.opening_draft_drawn = list(drafted_cards)
    return drafted_cards


def apply_opening_draft_selection(
    *,
    player: Any,
    drafted_cards: List[Any],
    rng: random.Random,
    kept_indices: Optional[Sequence[int]] = None,
    discard_sink: Optional[List[Any]] = None,
) -> None:
    if len(drafted_cards) <= 4:
        player.hand = drafted_cards
        player.opening_draft_kept_indices = list(range(len(drafted_cards)))
        return

    if kept_indices is None:
        selected = sorted(rng.sample(range(len(drafted_cards)), 4))
    else:
        if len(kept_indices) != 4:
            raise ValueError("Opening draft must keep exactly 4 cards.")
        if len(set(kept_indices)) != 4:
            raise ValueError("Opening draft kept indices must be unique.")
        selected = sorted(int(idx) for idx in kept_indices)
        if selected[0] < 0 or selected[-1] >= len(drafted_cards):
            raise ValueError("Opening draft kept index out of range.")

    selected_set = set(selected)
    player.hand = [card for idx, card in enumerate(drafted_cards) if idx in selected_set]
    discarded_cards = [card for idx, card in enumerate(drafted_cards) if idx not in selected_set]
    if discard_sink is not None:
        discard_sink.extend(discarded_cards)
    player.opening_draft_kept_indices = selected


def draw_final_scoring_cards_for_players(
    *,
    players: List[Any],
    rng: random.Random,
    final_pool: Sequence[Any],
) -> List[Any]:
    deck = list(final_pool)
    rng.shuffle(deck)
    needed = len(players) * 2
    if len(deck) < needed:
        raise ValueError("Not enough final scoring cards for setup.")
    for player in players:
        player.final_scoring_cards = [deck.pop(0), deck.pop(0)]
    return deck


def build_opening_setup_info(
    *,
    rng: random.Random,
    base_projects_pool: Sequence[Any],
    conservation_bonus_tile_pool: Sequence[str],
    conservation_space_2_fixed_options: Sequence[str],
    conservation_space_10_rule: str,
    two_player_blocked_levels: Sequence[str],
    opening_setup_info_factory: Callable[..., Any],
    blocked_project_level_factory: Callable[..., Any],
) -> Any:
    bonus_pool = list(conservation_bonus_tile_pool)
    rng.shuffle(bonus_pool)
    cp5_tiles = bonus_pool[:2]
    cp8_tiles = bonus_pool[2:4]

    project_pool = list(base_projects_pool)
    rng.shuffle(project_pool)
    selected_projects = project_pool[:3]

    blocked_levels = [
        blocked_project_level_factory(
            project_data_id=project.data_id,
            project_title=project.title,
            blocked_level=two_player_blocked_levels[idx],
        )
        for idx, project in enumerate(selected_projects)
    ]

    return opening_setup_info_factory(
        conservation_space_2_fixed_options=list(conservation_space_2_fixed_options),
        conservation_space_5_bonus_tiles=cp5_tiles,
        conservation_space_8_bonus_tiles=cp8_tiles,
        conservation_space_10_rule=conservation_space_10_rule,
        base_conservation_projects=selected_projects,
        two_player_blocked_project_levels=blocked_levels,
    )


def setup_game_state(
    *,
    seed: int,
    player_names: Optional[List[str]],
    manual_opening_draft_player_names: Optional[Set[str]],
    build_deck: Callable[[], List[Any]],
    map_image_name: str,
    load_map_data: Callable[[str], Any],
    build_map_tile_bonus_map: Callable[[str], Dict[Tuple[int, int], str]],
    player_state_factory: Callable[..., Any],
    game_state_factory: Callable[..., Any],
    ark_map_factory: Callable[..., Any],
    main_action_names: Sequence[str],
    draw_final_scoring_cards_for_players_fn: Callable[[List[Any], random.Random], List[Any]],
    build_opening_setup_info_fn: Callable[[random.Random], Any],
    load_base_setup_card_pools_fn: Callable[[], Tuple[List[Any], List[Any]]],
    refresh_association_market: Callable[[Any], None],
    draw_opening_draft_cards_fn: Callable[[Any, Any], List[Any]],
    apply_opening_draft_selection_fn: Callable[..., None],
    replenish_zoo_display: Callable[[Any], None],
    validate_card_zones: Callable[[Any], None],
    break_track_by_players: Dict[int, int],
    max_turns_per_player: int = 16,
) -> Any:
    rng = random.Random(seed)
    if player_names is None:
        player_names = ["HeuristicAI", "RandomAI"]
    if len(player_names) != 2:
        raise ValueError("This prototype currently supports exactly 2 players.")
    manual_players = manual_opening_draft_player_names or set()

    zoo_deck = build_deck()
    rng.shuffle(zoo_deck)
    map_data = load_map_data(map_image_name)
    map_bonus_map = build_map_tile_bonus_map(map_image_name)

    action_pool = [action for action in main_action_names if action != "animals"]
    p1_actions = ["animals"] + rng.sample(action_pool, len(action_pool))
    p2_actions = ["animals"] + rng.sample(action_pool, len(action_pool))

    p1 = player_state_factory(name=player_names[0], action_order=p1_actions, zoo_map=ark_map_factory(map_data=map_data))
    p2 = player_state_factory(name=player_names[1], action_order=p2_actions, zoo_map=ark_map_factory(map_data=map_data))
    players = [p1, p2]

    final_scoring_deck = draw_final_scoring_cards_for_players_fn(players, rng)
    opening_setup = build_opening_setup_info_fn(rng)
    _, base_project_pool = load_base_setup_card_pools_fn()
    selected_project_ids = {project.data_id for project in opening_setup.base_conservation_projects}
    unused_base_projects = [project for project in base_project_pool if project.data_id not in selected_project_ids]
    shared_tiles = {
        5: list(opening_setup.conservation_space_5_bonus_tiles),
        8: list(opening_setup.conservation_space_8_bonus_tiles),
    }
    claimed_tiles = {5: [], 8: []}

    state = game_state_factory(
        players=players,
        map_image_name=map_image_name,
        map_tile_bonuses=map_bonus_map,
        opening_setup=opening_setup,
        shared_conservation_bonus_tiles=shared_tiles,
        claimed_conservation_bonus_tiles=claimed_tiles,
        zoo_deck=zoo_deck,
        zoo_discard=[],
        zoo_display=[],
        unused_base_conservation_projects=unused_base_projects,
        final_scoring_deck=final_scoring_deck,
        final_scoring_discard=[],
        break_progress=0,
        break_max=break_track_by_players[len(players)],
        max_turns_per_player=max_turns_per_player,
    )
    refresh_association_market(state)
    state.conservation_project_slots = {
        project.data_id: {
            "left_level": None,
            "middle_level": None,
            "right_level": None,
        }
        for project in opening_setup.base_conservation_projects
    }

    for player in players:
        draft = draw_opening_draft_cards_fn(player, state)
        if player.name in manual_players:
            continue
        apply_opening_draft_selection_fn(
            player=player,
            drafted_cards=draft,
            rng=rng,
            kept_indices=None,
            discard_sink=state.zoo_discard,
        )

    replenish_zoo_display(state)
    validate_card_zones(state)
    return state

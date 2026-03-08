import pytest

from main import (
    CONSERVATION_BONUS_TILE_POOL,
    CONSERVATION_FIXED_MONEY_OPTION,
    CONSERVATION_SPACE_10_RULE,
    CONSERVATION_SPACE_2_FIXED_OPTIONS,
    TWO_PLAYER_BLOCKED_LEVELS,
    setup_game,
)


def test_setup_game_contains_requested_opening_information():
    state = setup_game(seed=406, player_names=["P1", "P2"])
    opening = state.opening_setup

    assert opening.conservation_space_2_fixed_options == list(CONSERVATION_SPACE_2_FIXED_OPTIONS)

    cp5 = opening.conservation_space_5_bonus_tiles
    cp8 = opening.conservation_space_8_bonus_tiles
    assert len(cp5) == 2
    assert len(cp8) == 2
    assert len(set(cp5 + cp8)) == 4
    for tile in cp5 + cp8:
        assert tile in CONSERVATION_BONUS_TILE_POOL
    assert opening.conservation_space_10_rule == CONSERVATION_SPACE_10_RULE
    assert state.shared_conservation_bonus_tiles[5] == cp5
    assert state.shared_conservation_bonus_tiles[8] == cp8
    assert state.claimed_conservation_bonus_tiles[5] == []
    assert state.claimed_conservation_bonus_tiles[8] == []

    projects = opening.base_conservation_projects
    blocked = opening.two_player_blocked_project_levels
    assert len(projects) == 3
    assert len(blocked) == 3
    assert [item.blocked_level for item in blocked] == list(TWO_PLAYER_BLOCKED_LEVELS)
    assert [item.project_data_id for item in blocked] == [project.data_id for project in projects]

    all_final_ids = []
    for player in state.players:
        assert len(player.final_scoring_cards) == 2
        for card in player.final_scoring_cards:
            all_final_ids.append(card.data_id)
            number = int(card.data_id[1:4])
            assert 1 <= number <= 11
    assert len(set(all_final_ids)) == 4
    assert len(state.final_scoring_deck) == 7


def test_setup_game_opening_information_is_seed_deterministic():
    state_1 = setup_game(seed=999, player_names=["P1", "P2"])
    state_2 = setup_game(seed=999, player_names=["P1", "P2"])

    assert state_1.opening_setup == state_2.opening_setup
    assert state_1.players[0].final_scoring_cards == state_2.players[0].final_scoring_cards
    assert state_1.players[1].final_scoring_cards == state_2.players[1].final_scoring_cards


def test_selected_base_conservation_projects_are_removed_from_public_zoo_card_zones():
    state = setup_game(seed=42, player_names=["P1", "P2"])
    selected_numbers = {int(project.data_id[1:4]) for project in state.opening_setup.base_conservation_projects}

    public_zones = [state.zoo_deck, state.zoo_display, state.zoo_discard]
    player_zones = [player.hand for player in state.players]
    all_zones = public_zones + player_zones

    leaked = [
        (card.number, card.name)
        for zone in all_zones
        for card in zone
        if str(getattr(card, "card_type", "") or "") == "conservation_project" and int(card.number) in selected_numbers
    ]

    assert leaked == []


def test_cp_5_and_8_random_tiles_are_shared_while_5coins_is_repeatable():
    state = setup_game(seed=1234, player_names=["P1", "P2"])
    p0 = state.players[0]
    p1 = state.players[1]

    p0.conservation = 5
    p1.conservation = 5
    state.shared_conservation_bonus_tiles[5] = ["10_money", "partner_zoo"]
    random_tile = "10_money"

    state.claim_conservation_reward(player_id=0, threshold=5, reward=random_tile)
    assert random_tile not in state.shared_conservation_bonus_tiles[5]
    assert random_tile in state.claimed_conservation_bonus_tiles[5]

    with pytest.raises(ValueError, match="not available"):
        state.claim_conservation_reward(player_id=1, threshold=5, reward=random_tile)

    money_before = p1.money
    state.claim_conservation_reward(
        player_id=1,
        threshold=5,
        reward=CONSERVATION_FIXED_MONEY_OPTION,
    )
    assert p1.money == money_before + 5

    p0.conservation = 8
    p1.conservation = 8
    p0_money_before = p0.money
    state.claim_conservation_reward(
        player_id=0,
        threshold=8,
        reward=CONSERVATION_FIXED_MONEY_OPTION,
    )
    assert p0.money == p0_money_before + 5
    assert CONSERVATION_FIXED_MONEY_OPTION in state.available_conservation_reward_choices(
        player_id=1,
        threshold=8,
    )

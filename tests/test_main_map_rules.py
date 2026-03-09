from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation
from main import (
    AnimalCard,
    Enclosure,
    EnclosureObject,
    _apply_map_break_recurring_effects_for_player,
    _apply_map_partner_threshold_rewards,
    _apply_map_university_threshold_rewards,
    _ensure_player_map_initialized,
    _on_map_conservation_project_supported,
    _perform_animals_action_effect,
    _resolve_break,
    list_legal_association_options,
    setup_game,
)


def test_plan1a_map_rules_loaded_with_tower_tag():
    state = setup_game(seed=701, player_names=["P1", "P2"])

    assert "left_track_unlocks" in state.map_rules
    assert len(state.map_rules["left_track_unlocks"]) == 7
    assert state.map_rules["map_completion_reward"]["effect"] == "gain_appeal_7"
    assert "passive_effects" in state.map_rules
    assert "tower" in [str(tag).lower() for tag in state.map_tile_tags.get((6, 1), [])]


def test_map_left_track_unlock_is_immediate_and_recurs_each_break_step5():
    state = setup_game(seed=702, player_names=["P1", "P2"])
    p0 = state.players[0]
    p1 = state.players[1]

    hand_before = len(p0.hand)
    _on_map_conservation_project_supported(state=state, player=p0, player_id=0)

    assert p0.map_left_track_unlocked_count == 1
    assert p0.map_left_track_unlocked_effects == ["draw_1_card_deck_or_reputation_range"]
    assert len(p0.hand) == hand_before + 1

    # Avoid interactive hand-limit discard prompts during break.
    p0.hand_limit = 99
    p1.hand_limit = 99
    _resolve_break(state)

    assert len(p0.hand) == hand_before + 2
    assert any("map_break_step5:P1:draw_1_card_deck_or_reputation_range" in line for line in state.effect_log)


def test_map_left_track_unlock_can_choose_any_remaining_reward_by_index():
    state = setup_game(seed=702, player_names=["P1", "P2"])
    p0 = state.players[0]

    money_before = p0.money
    _on_map_conservation_project_supported(
        state=state,
        player=p0,
        player_id=0,
        effect_details={"unlock_index": 5},
    )

    assert p0.money == money_before + 12
    assert p0.map_left_track_unlocked_count == 1
    assert 5 in p0.map_left_track_claimed_indices
    assert p0.map_left_track_unlocked_effects == []
    assert any("map_left_track_unlock_6:P1:gain_12_coins" in line for line in state.effect_log)


def test_map_partner_and_university_threshold_rewards_trigger_once(monkeypatch):
    state = setup_game(seed=703, player_names=["P1", "P2"])
    p0 = state.players[0]

    monkeypatch.setattr("builtins.input", lambda _: "1")

    p0.partner_zoos.update({"asia", "africa"})
    upgraded_before = sum(1 for key in p0.action_upgraded if p0.action_upgraded[key])
    _apply_map_partner_threshold_rewards(state=state, player=p0, player_id=0)
    assert 2 in p0.claimed_partner_zoo_thresholds
    assert sum(1 for key in p0.action_upgraded if p0.action_upgraded[key]) == upgraded_before + 1

    # Triggering again at the same count must do nothing.
    _apply_map_partner_threshold_rewards(state=state, player=p0, player_id=0)
    assert sum(1 for key in p0.action_upgraded if p0.action_upgraded[key]) == upgraded_before + 1

    workers_before = p0.workers
    p0.partner_zoos.add("america")
    _apply_map_partner_threshold_rewards(state=state, player=p0, player_id=0)
    assert 3 in p0.claimed_partner_zoo_thresholds
    assert p0.workers == workers_before + 1

    cons_before = p0.conservation
    p0.partner_zoos.add("europe")
    _apply_map_partner_threshold_rewards(state=state, player=p0, player_id=0)
    assert 4 in p0.claimed_partner_zoo_thresholds
    assert p0.conservation == cons_before + 2

    universities_before = sum(1 for key in p0.action_upgraded if p0.action_upgraded[key])
    p0.universities.update({"science_2", "science_1_reputation_2"})
    _apply_map_university_threshold_rewards(state=state, player=p0, player_id=0)
    assert 2 in p0.claimed_university_thresholds
    assert sum(1 for key in p0.action_upgraded if p0.action_upgraded[key]) == universities_before + 1

    cons_before = p0.conservation
    p0.universities.add("reputation_1_hand_limit_5")
    _apply_map_university_threshold_rewards(state=state, player=p0, player_id=0)
    assert 3 in p0.claimed_university_thresholds
    assert p0.conservation == cons_before + 1


def test_observation_tower_passive_grants_2_appeal_when_adjacent_standard_enclosure_is_occupied():
    state = setup_game(seed=704, player_names=["P1", "P2"])
    p0 = state.players[0]

    _ensure_player_map_initialized(state, p0)
    assert p0.zoo_map is not None

    # (6,0) is adjacent to the tower tile at (6,1) on plan1a.
    p0.zoo_map.add_building(Building(BuildingType.SIZE_1, HexTile(6, 0), Rotation.ROT_0))
    p0.enclosures = [Enclosure(size=1, occupied=False, origin=(6, 0), rotation="ROT_0")]
    p0.enclosure_objects = [
        EnclosureObject(
            size=1,
            enclosure_type="enclosure_1",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(6, 0),
            rotation="ROT_0",
        )
    ]

    p0.money = 10
    p0.appeal = 0
    p0.hand = [
        AnimalCard(
            name="Test Animal",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9901,
            instance_id="test-tower-animal",
        )
    ]

    _perform_animals_action_effect(
        state=state,
        player=p0,
        strength=3,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert p0.appeal == 3
    assert any("map_passive:tower_adjacent_standard_enclosure_occupied:+2_appeal" in line for line in state.effect_log)


def test_association_side_i_hides_partner_options_after_two_partners():
    state = setup_game(seed=705, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.partner_zoos = {"asia", "africa"}
    p0.action_upgraded["association"] = False

    state.available_partner_zoos = {"america", "australia", "europe"}
    opts = list_legal_association_options(state=state, player_id=0, strength=4)

    assert all(opt.get("task_kind") != "partner_zoo" for opt in opts)

    p0.action_upgraded["association"] = True
    opts_upgraded = list_legal_association_options(state=state, player_id=0, strength=4)
    assert any(opt.get("task_kind") == "partner_zoo" for opt in opts_upgraded)


def test_association_hides_partner_options_after_four_partners_even_side_ii():
    state = setup_game(seed=706, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.partner_zoos = {"asia", "africa", "america", "europe"}
    p0.action_upgraded["association"] = True

    state.available_partner_zoos = {"australia"}
    opts = list_legal_association_options(state=state, player_id=0, strength=4)

    assert all(opt.get("task_kind") != "partner_zoo" for opt in opts)


def test_map_break_recurring_play_sponsor_by_paying_cost_auto_fills_waza_mode():
    state = setup_game(seed=707, player_names=["P1", "P2"])
    p0 = state.players[0]

    sponsor_227 = next(card for card in state.zoo_deck if card.number == 227)
    state.zoo_deck.remove(sponsor_227)
    p0.hand = [sponsor_227]
    p0.money = 30
    p0.reputation = 6
    p0.map_left_track_unlocked_effects = ["play_1_sponsor_by_paying_cost"]

    _apply_map_break_recurring_effects_for_player(
        state=state,
        player=p0,
        player_id=0,
    )

    assert all(card.instance_id != sponsor_227.instance_id for card in p0.hand)
    assert any(card.instance_id == sponsor_227.instance_id for card in p0.zoo_cards)
    assert p0.sponsor_waza_assignment_mode in {"small", "large"}
    assert any("map_break_step5:P1:play_1_sponsor_by_paying_cost:played=227" in line for line in state.effect_log)

import copy

import pytest

from tests.helpers import make_state, set_action_strength, take_card_by_number

from main import (
    _apply_build_placement_bonus,
    Action,
    ActionType,
    AnimalCard,
    Enclosure,
    EnclosureObject,
    SponsorBuilding,
    Building,
    BuildingType,
    HexTile,
    Terrain,
    _resolve_break,
    _resolve_break_remaining_stages,
    _final_score_points,
    _apply_sponsor_passive_triggers_on_card_play,
    _player_icon_snapshot,
    _player_border_coords,
    _sponsor_endgame_bonus,
    _sponsor_requirements_met,
    _sponsor_unique_base_footprint,
    _list_legal_sponsor_unique_building_cells,
    apply_action,
    legal_actions,
    list_legal_animals_options,
    list_legal_build_options,
)


def test_sponsors_play_from_hand_is_free_and_applies_immediate_effect():
    state = make_state(701)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 4)

    sponsor_220 = take_card_by_number(state, 220)
    player.hand = [sponsor_220]
    money_before = player.money

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_220.instance_id,
                    }
                ],
            },
        ),
    )

    # Sponsor cards from hand are free; card #220 immediate effect gives +3 money.
    assert player.money == money_before + 3
    assert any(card.number == 220 and card.card_type == "sponsor" for card in player.zoo_cards)
    assert state.break_progress == 0


@pytest.mark.parametrize(
    ("universities", "expected_gain"),
    [
        (set(), 0),
        ({"science_2"}, 2),
        ({"science_2", "science_1_reputation_2"}, 5),
        ({"science_2", "science_1_reputation_2", "reputation_1_hand_limit_5"}, 10),
    ],
)
def test_sponsor_203_immediate_money_scales_with_university_count(universities, expected_gain):
    state = make_state(70101 + expected_gain)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 4)
    sponsor_203 = take_card_by_number(state, 203)
    player.hand = [sponsor_203]
    player.universities = set(universities)
    money_before = player.money

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_203.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.money == money_before + expected_gain


def test_sponsor_205_printed_reputation_and_conservation_apply_only_once():
    state = make_state(70106)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 4)
    reputation_before = player.reputation
    conservation_before = player.conservation

    player.zoo_cards.extend([take_card_by_number(state, 201), take_card_by_number(state, 223)])
    sponsor_205 = take_card_by_number(state, 205)
    player.hand = [sponsor_205]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_205.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.reputation == reputation_before + 2
    assert player.conservation == conservation_before + 1


@pytest.mark.parametrize(("sponsor_number", "seed"), [(255, 701061), (256, 701062)])
def test_sponsor_255_and_256_only_gain_4_appeal_once(sponsor_number, seed):
    state = make_state(seed)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 4)
    player.money = 20
    appeal_before = player.appeal
    sponsor_card = take_card_by_number(state, sponsor_number)
    legal = _list_legal_sponsor_unique_building_cells(state=state, player=player, sponsor_number=sponsor_number)
    assert legal
    player.hand = [sponsor_card]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_card.instance_id,
                    }
                ],
                "sponsor_unique_building_selections": [
                    {
                        "card_instance_id": sponsor_card.instance_id,
                        "cells": [list(cell) for cell in legal[0]],
                    }
                ],
            },
        ),
    )

    assert player.appeal == appeal_before + 4
    assert any(card.number == sponsor_number for card in player.zoo_cards)


def test_sponsor_206_immediate_appeal_scales_with_supported_projects():
    state = make_state(70107)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 5)
    appeal_before = player.appeal

    player.supported_conservation_project_actions = 3
    player.zoo_cards = [
        AnimalCard(
            name=f"Science{i}",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Science",),
            number=9050 + i,
            instance_id=f"sci-206-{i}",
        )
        for i in range(4)
    ]
    sponsor_206 = take_card_by_number(state, 206)
    player.hand = [sponsor_206]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_206.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.appeal == appeal_before + 6


def test_sponsor_207_uses_data_max_appeal_requirement():
    state = make_state(70108)
    player = state.players[0]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 4)
    player.appeal = 26
    sponsor_207 = take_card_by_number(state, 207)
    player.hand = [sponsor_207]

    with pytest.raises(ValueError, match="appeal_must_be_at_most_25"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="sponsors",
                details={
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": 0,
                            "card_instance_id": sponsor_207.instance_id,
                        }
                    ],
                },
            ),
        )


def test_sponsor_264_uses_data_max_appeal_requirement():
    state = make_state(701081)
    player = state.players[0]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 5)
    player.appeal = 26
    sponsor_264 = take_card_by_number(state, 264)
    player.hand = [sponsor_264]

    with pytest.raises(ValueError, match="appeal_must_be_at_most_25"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="sponsors",
                details={
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": 0,
                            "card_instance_id": sponsor_264.instance_id,
                        }
                    ],
                },
            ),
        )


def test_sponsor_207_counts_bear_as_distinct_animal_category():
    state = make_state(70109)
    player = state.players[0]
    other = state.players[1]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 4)
    conservation_before = player.conservation
    other_money_before = other.money

    player.zoo_cards = [
        AnimalCard(
            name="Bear Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Bear",),
            number=9070,
            instance_id="bear-207",
        ),
        AnimalCard(
            name="Africa Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Africa",),
            number=9071,
            instance_id="africa-207",
        ),
    ]
    sponsor_207 = take_card_by_number(state, 207)
    player.hand = [sponsor_207]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_207.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.conservation == conservation_before + 1
    assert other.money == other_money_before + 2


def test_sponsor_226_reputation_from_data_applies_only_once():
    state = make_state(701095)
    player = state.players[0]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 5)
    sponsor_226 = take_card_by_number(state, 226)
    player.hand = [sponsor_226]
    reputation_before = player.reputation

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_226.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.reputation == reputation_before + 2
    assert any(card.number == 226 for card in player.zoo_cards)


def test_sponsor_254_reputation_and_conservation_from_data_apply_only_once():
    state = make_state(701096)
    player = state.players[0]
    state.current_player = 0
    player.money = 20
    set_action_strength(player, "sponsors", 5)
    sponsor_254 = take_card_by_number(state, 254)
    legal = _list_legal_sponsor_unique_building_cells(state=state, player=player, sponsor_number=254)
    assert legal
    player.hand = [sponsor_254]
    reputation_before = player.reputation
    conservation_before = player.conservation

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_254.instance_id,
                    }
                ],
                "sponsor_unique_building_selections": [
                    {
                        "card_instance_id": sponsor_254.instance_id,
                        "cells": [list(cell) for cell in legal[0]],
                    }
                ],
                "sponsor_draw_card_choices": [
                    {
                        "draw_source": "deck",
                    }
                ],
            },
        ),
    )

    assert player.reputation == reputation_before + 1
    assert player.conservation == conservation_before + 1
    assert any(card.number == 254 for card in player.zoo_cards)


def test_sponsor_261_appeal_and_conservation_from_data_apply_only_once():
    state = make_state(701097)
    player = state.players[0]
    state.current_player = 0
    player.money = 20
    set_action_strength(player, "sponsors", 3)
    sponsor_261 = take_card_by_number(state, 261)
    player.hand = [sponsor_261]
    appeal_before = player.appeal
    conservation_before = player.conservation

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_261.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.appeal == appeal_before + 1
    assert player.conservation == conservation_before + 1
    assert any(card.number == 261 for card in player.zoo_cards)


def test_sponsor_262_immediate_counts_bear_as_distinct_animal_category():
    state = make_state(70110)
    player = state.players[0]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 5)
    money_before = player.money

    player.zoo_cards = [
        AnimalCard(
            name="Bear Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Bear",),
            number=9072,
            instance_id="bear-262",
        ),
        AnimalCard(
            name="Africa Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Africa",),
            number=9073,
            instance_id="africa-262",
        ),
    ]
    sponsor_262 = take_card_by_number(state, 262)
    player.hand = [sponsor_262]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_262.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.money == money_before + 4


def test_sponsor_262_passive_counts_new_bear_icon_as_distinct_category():
    state = make_state(70111)
    owner = state.players[0]
    owner.zoo_cards = [
        AnimalCard(
            name="Explorer",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=262,
            instance_id="s262-passive",
        ),
        AnimalCard(
            name="Africa Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Africa",),
            number=9074,
            instance_id="africa-existing-262",
        ),
    ]
    appeal_before = owner.appeal
    money_before = owner.money

    played_card = AnimalCard(
        name="Bear Animal",
        cost=0,
        size=1,
        appeal=0,
        conservation=0,
        card_type="animal",
        badges=("Bear",),
        number=9075,
        instance_id="bear-played-262",
    )
    owner.zoo_cards.append(played_card)

    _apply_sponsor_passive_triggers_on_card_play(
        state=state,
        played_by_player_id=0,
        played_card=played_card,
    )

    assert owner.appeal == appeal_before + 1
    assert owner.money == money_before + 2


def test_sponsorship_cards_231_to_235_use_level_3_but_cost_0_and_do_not_count_their_own_badge():
    state = make_state(7011)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 4)

    sponsor_233 = take_card_by_number(state, 233)
    player.hand = [sponsor_233]
    player.zoo_cards = [
        AnimalCard(
            name="Bird Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            badges=("Bird",),
            instance_id="bird-animal",
        )
    ]
    money_before = player.money

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_233.instance_id,
                    }
                ],
            },
        ),
    )

    snapshot = _player_icon_snapshot(player)
    assert player.money == money_before
    assert player.appeal == 1
    assert snapshot["categories"]["Bird"] == 1


def test_field_research_type_d_orcas_requires_two_science_icons_and_one_sea_animal():
    state = make_state(70115)
    player = state.players[0]

    orcas = AnimalCard(
        name="FIELD RESEARCH TYPE D ORCAS",
        cost=6,
        size=0,
        appeal=0,
        conservation=0,
        reputation_gain=3,
        card_type="sponsor",
        required_icons=(("science", 2), ("seaanimal", 1)),
        badges=("Science",),
        number=277,
        instance_id="s-277",
    )

    player.zoo_cards.append(take_card_by_number(state, 223))
    ok, reason = _sponsor_requirements_met(player=player, card=orcas, sponsors_upgraded=True)
    assert ok is False
    assert reason == "icon_seaanimal_1"

    player.zoo_cards.append(
        AnimalCard(
            name="Sea Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("SeaAnimal",),
            number=9701,
            instance_id="sea-animal",
        )
    )
    ok, reason = _sponsor_requirements_met(player=player, card=orcas, sponsors_upgraded=True)
    assert ok is True
    assert reason == "ok"


def test_sponsor_play_applies_printed_reputation_and_badges_for_field_research_type_d_orcas():
    state = make_state(70116)
    player = state.players[0]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 5)
    reputation_before = player.reputation

    player.zoo_cards.extend(
        [
            take_card_by_number(state, 223),
            AnimalCard(
                name="Sea Animal",
                cost=0,
                size=1,
                appeal=0,
                conservation=0,
                card_type="animal",
                badges=("SeaAnimal",),
                number=9702,
                instance_id="sea-animal-2",
            ),
        ]
    )
    orcas = AnimalCard(
        name="FIELD RESEARCH TYPE D ORCAS",
        cost=6,
        size=0,
        appeal=0,
        conservation=0,
        reputation_gain=3,
        card_type="sponsor",
        required_icons=(("science", 2), ("seaanimal", 1)),
        badges=("Science",),
        number=277,
        instance_id="s-277-play",
    )
    player.hand = [orcas]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": orcas.instance_id,
                    }
                ],
            },
        ),
    )

    icons = _player_icon_snapshot(player)
    assert player.reputation == reputation_before + 3
    assert icons["science"] == 3
    assert any(card.instance_id == orcas.instance_id for card in player.zoo_cards)


def test_sponsor_238_self_triggers_bird_money_when_played_from_hand():
    state = make_state(7012)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 4)

    sponsor_238 = take_card_by_number(state, 238)
    player.hand = [sponsor_238]
    money_before = player.money

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_238.instance_id,
                    }
                ],
            },
        ),
    )

    assert player.money == money_before + 3
    assert any(card.number == 238 for card in player.zoo_cards)


def test_sponsors_break_alternative_upgraded_gives_double_money():
    state = make_state(702)
    player = state.players[0]
    state.current_player = 0
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 4)
    money_before = player.money

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={"use_break_ability": True, "sponsor_selections": []},
        ),
    )

    assert player.money == money_before + 8
    assert state.break_progress == 4


def test_sponsor_206_requires_4_science_icons_to_play():
    state = make_state(703)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "sponsors", 5)

    sponsor_206 = take_card_by_number(state, 206)
    player.hand = [sponsor_206]

    with pytest.raises(ValueError, match="icon_science_4"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="sponsors",
                details={
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": 0,
                            "card_instance_id": sponsor_206.instance_id,
                        }
                    ],
                },
            ),
        )

    player.zoo_cards = [
        AnimalCard(
            name=f"Science{i}",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            card_type="animal",
            badges=("Science",),
            number=9000 + i,
            instance_id=f"sci-{i}",
        )
        for i in range(4)
    ]
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]
    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_206.instance_id,
                    }
                ],
            },
        ),
    )
    assert any(card.number == 206 for card in player.zoo_cards)


def test_break_income_applies_sponsor_income_effects(monkeypatch):
    state = make_state(704)
    player = state.players[0]

    sponsor_206 = take_card_by_number(state, 206)
    sponsor_209 = take_card_by_number(state, 209)
    sponsor_220 = take_card_by_number(state, 220)
    sponsor_231 = take_card_by_number(state, 231)
    player.zoo_cards.extend([sponsor_206, sponsor_209, sponsor_220, sponsor_231])
    player.zoo_cards.extend(
        [
            AnimalCard(
                name=f"P{i}",
                cost=0,
                size=1,
                appeal=0,
                conservation=0,
                card_type="animal",
                badges=("Primate",),
                number=9100 + i,
                instance_id=f"prim-{i}",
            )
            for i in range(2)
        ]
    )
    player.money = 0
    player.appeal = 0
    player.conservation = 0
    player.x_tokens = 0

    monkeypatch.setattr("builtins.input", lambda _: "1")
    _resolve_break(state)

    # appeal income 5 + sponsor220 income 3 + sponsor231 (2 primate icons, sponsor card itself does not count) income 3
    assert player.money == 11
    assert player.conservation == 1
    assert player.x_tokens == 1


def test_sponsor_263_allows_large_animal_to_ignore_one_condition():
    state = make_state(705)
    player = state.players[0]
    player.hand = [
        AnimalCard(
            name="LargeNeedsScience",
            cost=10,
            size=4,
            appeal=7,
            conservation=1,
            card_type="animal",
            required_icons=(("science", 1),),
            number=9901,
            instance_id="large-needs-science",
        )
    ]
    player.money = 20
    player.enclosures = [Enclosure(size=5, occupied=False)]
    options_without = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert options_without == []

    sponsor_263 = take_card_by_number(state, 263)
    player.zoo_cards.append(sponsor_263)
    options_with = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert len(options_with) == 1


def test_sponsor_263_places_free_size_5_enclosure_without_money():
    state = make_state(7051)
    player = state.players[0]
    state.current_player = 0
    player.money = 0
    player.reputation = 6
    player.action_upgraded["sponsors"] = True
    set_action_strength(player, "sponsors", 5)
    sponsor_263 = take_card_by_number(state, 263)
    player.hand = [sponsor_263]

    sponsor_actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "sponsors"
        and not bool((action.details or {}).get("use_break_ability"))
    ]
    assert sponsor_actions

    apply_action(state, sponsor_actions[0])

    assert player.money == 0
    assert any(building.type == BuildingType.SIZE_5 for building in player.zoo_map.buildings.values())
    assert any(enclosure.size == 5 for enclosure in player.enclosures)
    assert any(card.number == 263 for card in player.zoo_cards)


def test_sponsor_210_places_free_kiosk_when_playing_america_icon():
    state = make_state(706)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]  # animals strength=2
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="Expert210",
            cost=4,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=210,
            instance_id="s210",
        )
    )
    player.hand = [
        AnimalCard(
            name="AmericaAnimal",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            badges=("America",),
            number=9801,
            instance_id="a-9801",
        )
    ]
    player.enclosures = [Enclosure(size=1, occupied=False)]
    kiosks_before = sum(1 for b in player.zoo_map.buildings.values() if b.type == BuildingType.KIOSK)

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )

    kiosks_after = sum(1 for b in player.zoo_map.buildings.values() if b.type == BuildingType.KIOSK)
    assert kiosks_after == kiosks_before + 1


def test_sponsor_228_allows_extra_small_animal_and_take_small_from_display():
    state = make_state(707)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]  # animals strength=2
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="SmallProgram228",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=228,
            instance_id="s228",
        )
    )
    first_small = AnimalCard(
        name="SmallA",
        cost=1,
        size=1,
        appeal=2,
        conservation=0,
        card_type="animal",
        number=9802,
        instance_id="a-9802",
    )
    second_small = AnimalCard(
        name="SmallB",
        cost=1,
        size=1,
        appeal=2,
        conservation=0,
        card_type="animal",
        number=9803,
        instance_id="a-9803",
    )
    display_small = AnimalCard(
        name="DisplaySmall",
        cost=2,
        size=1,
        appeal=1,
        conservation=0,
        card_type="animal",
        number=9804,
        instance_id="a-9804",
    )
    player.hand = [first_small, second_small]
    player.enclosures = [Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False)]
    state.zoo_display = [display_small]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )

    played_numbers = {card.number for card in player.zoo_cards}
    assert 9802 in played_numbers
    assert 9803 in played_numbers
    assert any(card.number == 9804 for card in player.hand)


def test_sponsor_228_respects_selected_extra_small_animal_choice():
    state = make_state(70701)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="SmallProgram228",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=228,
            instance_id="s228-select",
        )
    )
    trigger_small = AnimalCard("TriggerSmall", 1, 1, 1, 0, card_type="animal", number=98228, instance_id="trigger-small")
    preferred_small = AnimalCard("PreferredSmall", 1, 1, 1, 0, card_type="animal", number=98229, instance_id="preferred-small")
    auto_small = AnimalCard("AutoSmall", 1, 1, 4, 0, card_type="animal", number=98230, instance_id="auto-small")
    player.hand = [trigger_small, preferred_small, auto_small]
    player.enclosures = [Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False)]
    state.zoo_display = []

    sequence_index = next(
        int(option["index"]) - 1
        for option in list_legal_animals_options(state=state, player_id=0, strength=2)
        if [str(play.get("card_instance_id") or "") for play in list(option.get("plays") or [])] == [trigger_small.instance_id]
    )

    chosen_action = next(
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and int((action.details or {}).get("animals_sequence_index", -1)) == sequence_index
        and list((action.details or {}).get("sponsor_228_extra_play_choices") or [])
        == [{"card_instance_id": preferred_small.instance_id, "enclosure_index": 1}]
    )

    apply_action(state, chosen_action)

    played_numbers = {card.number for card in player.zoo_cards}
    hand_numbers = {card.number for card in player.hand}
    assert trigger_small.number in played_numbers
    assert preferred_small.number in played_numbers
    assert auto_small.number in hand_numbers


def test_break_income_sponsor_201_uses_pending_draw_choice():
    state = make_state(7071)
    player = state.players[0]
    for other in state.players:
        other.hand = []
    state.break_trigger_player = 0
    player.zoo_cards.append(
        AnimalCard(
            name="ScienceLab201",
            cost=3,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=201,
            instance_id="s201",
        )
    )
    display_card = AnimalCard(
        name="DisplayChoice",
        cost=1,
        size=1,
        appeal=0,
        conservation=0,
        card_type="animal",
        number=9811,
        instance_id="display-choice",
    )
    deck_card = AnimalCard(
        name="DeckChoice",
        cost=1,
        size=1,
        appeal=0,
        conservation=0,
        card_type="animal",
        number=9812,
        instance_id="deck-choice",
    )
    player.reputation = 3
    state.zoo_display = [
        display_card,
        AnimalCard("BreakDisplayB", 1, 1, 0, 0, card_type="animal", number=9819, instance_id="break-display-b"),
        AnimalCard("BreakDisplayC", 1, 1, 0, 0, card_type="animal", number=9820, instance_id="break-display-c"),
    ]
    state.zoo_deck = [
        deck_card,
        AnimalCard("BreakDeckB", 1, 1, 0, 0, card_type="animal", number=9821, instance_id="break-deck-b"),
        AnimalCard("BreakDeckC", 1, 1, 0, 0, card_type="animal", number=9822, instance_id="break-deck-c"),
        AnimalCard("BreakDeckD", 1, 1, 0, 0, card_type="animal", number=9823, instance_id="break-deck-d"),
    ]

    _resolve_break_remaining_stages(state, preprocessed=True)

    assert state.pending_decision_kind == "break_card_draw_choice"

    actions = legal_actions(player, state=state, player_id=0)
    display_action = next(
        action
        for action in actions
        if (action.details or {}).get("draw_source") == "display"
    )
    apply_action(state, display_action)

    assert state.pending_decision_kind == ""
    assert any(card.instance_id == "display-choice" for card in player.hand)
    assert state.break_progress == 0


def test_animals_legal_actions_expand_sponsor_214_action_choices():
    state = make_state(7072)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="AfricaExpert214",
            cost=4,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=214,
            instance_id="s214",
        )
    )
    player.hand = [
        AnimalCard(
            name="AfricaAnimal",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            badges=("Africa",),
            number=9813,
            instance_id="a-9813",
        )
    ]
    player.enclosures = [Enclosure(size=1, occupied=False)]

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"
    ]

    chosen_actions = {
        str(((action.details or {}).get("sponsor_action_to_slot_1_choices") or [{}])[0].get("action_name") or "")
        for action in actions
        if (action.details or {}).get("sponsor_action_to_slot_1_choices")
    }

    assert chosen_actions == set(player.action_order)


def test_animals_legal_actions_expand_sponsor_210_free_build_choices():
    state = make_state(7073)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="AmericaExpert210",
            cost=4,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=210,
            instance_id="s210-choices",
        )
    )
    player.hand = [
        AnimalCard(
            name="AmericaAnimal",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            badges=("America",),
            number=9814,
            instance_id="a-9814",
        )
    ]
    player.enclosures = [Enclosure(size=1, occupied=False)]

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"
    ]

    variants = [
        list((action.details or {}).get("sponsor_free_building_placement_choices") or [])
        for action in actions
    ]
    assert any(choice_list and bool(choice_list[0].get("skip")) for choice_list in variants)
    assert len(
        {
            tuple(
                tuple(cell)
                for cell in list(((choice_list[0].get("selection") or {}).get("cells") or []))
            )
            for choice_list in variants
            if choice_list and not bool(choice_list[0].get("skip"))
        }
    ) > 1


def test_animals_legal_actions_expand_sponsor_228_display_take_choices():
    state = make_state(7074)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="SmallProgram228",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=228,
            instance_id="s228-choices",
        )
    )
    player.hand = [
        AnimalCard("Small1", 1, 1, 1, 0, card_type="animal", number=9815, instance_id="small-1"),
        AnimalCard("Small2", 1, 1, 1, 0, card_type="animal", number=9816, instance_id="small-2"),
    ]
    player.enclosures = [Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False)]
    state.zoo_display = [
        AnimalCard("DispSmall1", 1, 1, 1, 0, card_type="animal", number=9817, instance_id="disp-small-1"),
        AnimalCard("DispSmall2", 1, 1, 1, 0, card_type="animal", number=9818, instance_id="disp-small-2"),
    ]

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"
    ]

    variants = [list((action.details or {}).get("sponsor_display_take_choices") or []) for action in actions]
    chosen_indices = {
        int(choice_list[0].get("display_index", -1))
        for choice_list in variants
        if choice_list and not bool(choice_list[0].get("skip"))
    }
    assert chosen_indices == {0, 1}


def test_animals_legal_actions_expand_sponsor_228_extra_play_choices():
    state = make_state(70741)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="SmallProgram228",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=228,
            instance_id="s228-extra-choices",
        )
    )
    trigger_small = AnimalCard("TriggerSmall", 1, 1, 1, 0, card_type="animal", number=98231, instance_id="trigger-small-choices")
    option_a = AnimalCard("OptionA", 1, 1, 1, 0, card_type="animal", number=98232, instance_id="option-a")
    option_b = AnimalCard("OptionB", 1, 1, 4, 0, card_type="animal", number=98233, instance_id="option-b")
    player.hand = [trigger_small, option_a, option_b]
    player.enclosures = [Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False)]
    state.zoo_display = []

    sequence_index = next(
        int(option["index"]) - 1
        for option in list_legal_animals_options(state=state, player_id=0, strength=2)
        if [str(play.get("card_instance_id") or "") for play in list(option.get("plays") or [])] == [trigger_small.instance_id]
    )

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "animals"
        and int((action.details or {}).get("animals_sequence_index", -1)) == sequence_index
    ]

    variants = [list((action.details or {}).get("sponsor_228_extra_play_choices") or []) for action in actions]
    assert any(choice_list and bool(choice_list[0].get("skip")) for choice_list in variants)

    card_to_enclosures = {}
    for choice_list in variants:
        if not choice_list or bool(choice_list[0].get("skip")):
            continue
        entry = choice_list[0]
        card_id = str(entry.get("card_instance_id") or "")
        enclosure_index = int(entry.get("enclosure_index", -1))
        card_to_enclosures.setdefault(card_id, set()).add(enclosure_index)

    assert card_to_enclosures == {
        option_a.instance_id: {1, 2},
        option_b.instance_id: {1, 2},
    }


def test_animals_sponsor_249_uses_pending_revealed_keep_choice():
    state = make_state(7075)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="BirdHouse249",
            cost=4,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=249,
            instance_id="s249",
        )
    )
    player.hand = [
        AnimalCard(
            name="BirdAnimal",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            badges=("Bird",),
            number=9819,
            instance_id="bird-play",
        )
    ]
    player.enclosures = [Enclosure(size=1, occupied=False)]
    keep_first = AnimalCard("KeepFirst", 1, 1, 0, 0, card_type="animal", number=9820, instance_id="keep-first")
    keep_second = AnimalCard("KeepSecond", 1, 1, 0, 0, card_type="sponsor", number=9821, instance_id="keep-second")
    state.zoo_deck = [keep_first, keep_second]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )

    assert state.pending_decision_kind == "revealed_cards_keep"

    actions = legal_actions(player, state=state, player_id=0)
    chosen = next(
        action
        for action in actions
        if list((action.details or {}).get("keep_card_instance_ids") or []) == ["keep-second"]
    )
    apply_action(state, chosen)

    assert state.pending_decision_kind == ""
    assert any(card.instance_id == "keep-second" for card in player.hand)
    assert all(card.instance_id != "keep-first" for card in player.hand)


def test_animals_pending_keep_actions_expand_followup_display_choices():
    state = make_state(70751)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "animals", 5)
    player.money = 20
    player.hand = [
        AnimalCard(
            name="Perceiver",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            ability_title="Perception 2",
            card_type="animal",
            number=98211,
            instance_id="perceiver",
        ),
        AnimalCard(
            name="Snapper",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            ability_title="Snapping 1",
            card_type="animal",
            number=98216,
            instance_id="snapper",
        ),
    ]
    player.enclosures = [Enclosure(size=1, occupied=False), Enclosure(size=1, occupied=False)]
    state.zoo_deck = [
        AnimalCard("KeepFirst", 1, 1, 0, 0, card_type="animal", number=98212, instance_id="keep-first"),
        AnimalCard("KeepSecond", 1, 1, 0, 0, card_type="animal", number=98213, instance_id="keep-second"),
    ]
    state.zoo_display = [
        AnimalCard("DispFirst", 1, 1, 0, 0, card_type="animal", number=98214, instance_id="disp-first"),
        AnimalCard("DispSecond", 1, 1, 0, 0, card_type="animal", number=98215, instance_id="disp-second"),
    ]

    sequence_index = next(
        int(option["index"]) - 1
        for option in list_legal_animals_options(state=state, player_id=0, strength=5)
        if [str(play.get("card_instance_id") or "") for play in list(option.get("plays") or [])]
        == ["perceiver", "snapper"]
    )

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": sequence_index},
        ),
    )

    assert state.pending_decision_kind == "revealed_cards_keep"

    actions = legal_actions(player, state=state, player_id=0)
    keep_second_actions = [
        action
        for action in actions
        if list((action.details or {}).get("keep_card_instance_ids") or []) == ["keep-second"]
    ]
    chosen_indices = {
        int(choice_list[0].get("display_index", -1))
        for choice_list in [
            list((action.details or {}).get("display_take_choices") or [])
            for action in keep_second_actions
        ]
        if choice_list
    }
    assert chosen_indices == {0, 1}

    chosen = next(
        action
        for action in keep_second_actions
        if int(
            (((action.details or {}).get("display_take_choices") or [{}])[0]).get("display_index", -1)
        ) == 1
    )
    apply_action(state, chosen)

    assert state.pending_decision_kind == ""
    assert any(card.instance_id == "keep-second" for card in player.hand)
    assert any(card.instance_id == "disp-second" for card in player.hand)
    assert all(card.instance_id != "disp-first" for card in player.hand)


def test_animals_sponsor_252_uses_pending_revealed_keep_choice():
    state = make_state(7076)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.extend(
        [
            AnimalCard(
                name="PredatorLab252",
                cost=4,
                size=0,
                appeal=0,
                conservation=0,
                card_type="sponsor",
                number=252,
                instance_id="s252",
            ),
            AnimalCard(
                name="PredatorResident",
                cost=0,
                size=1,
                appeal=0,
                conservation=0,
                card_type="animal",
                badges=("Predator",),
                number=9822,
                instance_id="predator-resident",
            ),
        ]
    )
    player.hand = [
        AnimalCard(
            name="PredatorPlay",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            badges=("Predator",),
            number=9823,
            instance_id="predator-play",
        )
    ]
    player.enclosures = [Enclosure(size=1, occupied=False)]
    state.zoo_deck = [
        AnimalCard("AnimalChoiceA", 1, 1, 0, 0, card_type="animal", number=9824, instance_id="animal-a"),
        AnimalCard("AnimalChoiceB", 1, 1, 0, 0, card_type="animal", number=9825, instance_id="animal-b"),
        AnimalCard("AnimalChoiceC", 1, 1, 0, 0, card_type="animal", number=9826, instance_id="animal-c"),
    ]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )

    assert state.pending_decision_kind == "revealed_cards_keep"

    actions = legal_actions(player, state=state, player_id=0)
    chosen = next(
        action
        for action in actions
        if list((action.details or {}).get("keep_card_instance_ids") or []) == ["animal-b"]
    )
    apply_action(state, chosen)

    assert state.pending_decision_kind == ""
    assert any(card.instance_id == "animal-b" for card in player.hand)
    assert all(card.instance_id != "animal-a" for card in player.hand)


def test_animals_sponsor_252_discards_revealed_cards_when_no_animals_match():
    state = make_state(70761)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]
    player.money = 20
    player.zoo_cards.extend(
        [
            AnimalCard(
                name="PredatorLab252",
                cost=4,
                size=0,
                appeal=0,
                conservation=0,
                card_type="sponsor",
                number=252,
                instance_id="s252-empty",
            ),
            AnimalCard(
                name="PredatorResident",
                cost=0,
                size=1,
                appeal=0,
                conservation=0,
                card_type="animal",
                badges=("Predator",),
                number=9827,
                instance_id="predator-resident-empty",
            ),
        ]
    )
    player.hand = [
        AnimalCard(
            name="PredatorPlay",
            cost=0,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            badges=("Predator",),
            number=9828,
            instance_id="predator-play-empty",
        )
    ]
    player.enclosures = [Enclosure(size=1, occupied=False)]
    state.zoo_deck = [
        AnimalCard("SponsorReveal", 1, 1, 0, 0, card_type="sponsor", number=9829, instance_id="sponsor-reveal"),
        AnimalCard(
            "ProjectReveal",
            1,
            0,
            0,
            0,
            card_type="conservation_project",
            number=9830,
            instance_id="project-reveal",
        ),
        AnimalCard("SponsorRevealB", 1, 1, 0, 0, card_type="sponsor", number=9831, instance_id="sponsor-reveal-b"),
    ]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )

    assert state.pending_decision_kind == ""
    assert any(card.instance_id == "sponsor-reveal" for card in state.zoo_discard)
    assert any(card.instance_id == "project-reveal" for card in state.zoo_discard)
    assert any(card.instance_id == "sponsor-reveal-b" for card in state.zoo_discard)


def test_sponsor_253_plays_sponsor_from_hand_on_herbivore_trigger():
    state = make_state(708)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]  # animals strength=2
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="OkapiStable253",
            cost=6,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            badges=("Herbivore",),
            number=253,
            instance_id="s253",
        )
    )
    player.sponsor_tokens_by_number[253] = 1
    sponsor_220 = take_card_by_number(state, 220)
    herbivore_animal = AnimalCard(
        name="HerbivoreAnimal",
        cost=0,
        size=1,
        appeal=1,
        conservation=0,
        card_type="animal",
        badges=("Herbivore",),
        number=9805,
        instance_id="a-9805",
    )
    player.hand = [herbivore_animal, sponsor_220]
    player.enclosures = [Enclosure(size=1, occupied=False)]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )

    assert any(card.number == 220 for card in player.zoo_cards)
    assert player.sponsor_tokens_by_number.get(253, 0) == 0
    assert player.money == 19  # 20 - level(4) + immediate(3)


def test_animals_legal_actions_expand_sponsor_253_followup_choices():
    state = make_state(7081)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]  # animals strength=2
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="OkapiStable253",
            cost=6,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            badges=("Herbivore",),
            number=253,
            instance_id="s253",
        )
    )
    player.sponsor_tokens_by_number[253] = 1
    sponsor_220 = take_card_by_number(state, 220)
    sponsor_238 = take_card_by_number(state, 238)
    herbivore_animal = AnimalCard(
        name="HerbivoreAnimal",
        cost=0,
        size=1,
        appeal=1,
        conservation=0,
        card_type="animal",
        badges=("Herbivore",),
        number=9806,
        instance_id="a-9806",
    )
    player.hand = [herbivore_animal, sponsor_220, sponsor_238]
    player.enclosures = [Enclosure(size=1, occupied=False)]

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION and action.card_name == "animals"
    ]

    variants = {
        tuple(
            "skip"
            if bool(entry.get("skip"))
            else str(entry.get("card_instance_id") or "")
            for entry in list((action.details or {}).get("sponsor_253_plays") or [])
        )
        for action in actions
    }

    assert variants == {
        ("skip",),
        (str(sponsor_220.instance_id),),
        (str(sponsor_238.instance_id),),
    }

    chosen = next(
        action
        for action in actions
        if [str(entry.get("card_instance_id") or "") for entry in list((action.details or {}).get("sponsor_253_plays") or [])]
        == [str(sponsor_220.instance_id)]
    )
    apply_action(state, chosen)

    assert any(card.number == 220 for card in player.zoo_cards)
    assert player.sponsor_tokens_by_number.get(253, 0) == 0


def test_sponsors_legal_actions_expand_sponsor_253_followup_choices():
    state = make_state(7082)
    player = state.players[0]
    state.current_player = 0
    player.action_order = ["cards", "animals", "build", "association", "sponsors"]  # sponsors strength=5
    player.money = 20
    player.zoo_cards.append(
        AnimalCard(
            name="OkapiStable253",
            cost=6,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            badges=("Herbivore",),
            number=253,
            instance_id="s253",
        )
    )
    player.sponsor_tokens_by_number[253] = 1
    trigger_sponsor = AnimalCard(
        name="TriggerHerbSponsor",
        cost=4,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        badges=("Herbivore",),
        number=9810,
        instance_id="s-9810",
    )
    sponsor_220 = take_card_by_number(state, 220)
    sponsor_238 = take_card_by_number(state, 238)
    player.hand = [trigger_sponsor, sponsor_220, sponsor_238]

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "sponsors"
        and not bool((action.details or {}).get("use_break_ability"))
        and [str(selection.get("card_instance_id") or "") for selection in list((action.details or {}).get("sponsor_selections") or [])]
        == ["s-9810"]
    ]

    variants = {
        tuple(
            "skip"
            if bool(entry.get("skip"))
            else str(entry.get("card_instance_id") or "")
            for entry in list((action.details or {}).get("sponsor_253_plays") or [])
        )
        for action in actions
    }

    assert variants == {
        ("skip",),
        (str(sponsor_220.instance_id),),
        (str(sponsor_238.instance_id),),
    }


def test_sponsors_actions_filter_display_sequences_that_exceed_total_money():
    state = make_state(7083)
    player = state.players[0]
    state.current_player = 0
    player.money = 3
    player.reputation = 6
    player.action_upgraded["sponsors"] = True
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]  # sponsors strength=5

    display_filler = AnimalCard(
        name="DisplaySponsorFiller",
        cost=1,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=9801,
        instance_id="display-filler",
    )
    sponsor_a = AnimalCard(
        name="DisplaySponsorA",
        cost=1,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=9802,
        instance_id="display-a",
    )
    sponsor_b = AnimalCard(
        name="DisplaySponsorB",
        cost=1,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=9803,
        instance_id="display-b",
    )
    state.zoo_display = [display_filler, sponsor_a, sponsor_b]

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "sponsors"
        and not bool((action.details or {}).get("use_break_ability"))
    ]

    selection_lists = [
        [
            str(selection.get("card_instance_id") or "")
            for selection in list((action.details or {}).get("sponsor_selections") or [])
        ]
        for action in actions
    ]

    assert [str(sponsor_a.instance_id)] in selection_lists
    assert [str(sponsor_b.instance_id)] in selection_lists
    assert [str(sponsor_a.instance_id), str(sponsor_b.instance_id)] not in selection_lists
    assert [str(sponsor_b.instance_id), str(sponsor_a.instance_id)] not in selection_lists


def test_sponsor_257_break_income_counts_adjacent_non_empty_buildings_only(monkeypatch):
    state = make_state(709)
    player = state.players[0]
    player.money = 0
    player.appeal = 0
    player.zoo_cards.append(
        AnimalCard(
            name="SideEntrance257",
            cost=3,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=257,
            instance_id="s257",
        )
    )
    player.sponsor_buildings.append(
        SponsorBuilding(sponsor_number=257, cells=((0, 0), (1, 0)), label="SIDE ENTRANCE")
    )
    player.zoo_map.add_building(Building(BuildingType.KIOSK, HexTile(1, 1)))
    player.zoo_map.add_building(Building(BuildingType.SIZE_1, HexTile(0, 1)))
    player.enclosures.append(Enclosure(size=1, occupied=False, origin=(0, 1), rotation="ROT_0"))
    player.enclosure_objects.append(
        EnclosureObject(
            size=1,
            enclosure_type="enclosure_1",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(0, 1),
            rotation="ROT_0",
        )
    )

    monkeypatch.setattr("builtins.input", lambda _: "1")
    _resolve_break(state)

    # base income 5 + kiosk income 1 (adjacent unique building)
    # + side entrance income 2 (kiosk only; empty standard enclosure excluded)
    assert player.money == 8


def test_playing_sponsor_243_places_unique_building():
    state = make_state(710)
    player = state.players[0]
    state.current_player = 0
    player.reputation = 3
    player.money = 20
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]  # sponsors strength=5
    sponsor_243 = take_card_by_number(state, 243)
    legal = _list_legal_sponsor_unique_building_cells(state=state, player=player, sponsor_number=243)
    assert legal
    player.hand = [sponsor_243]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_243.instance_id,
                    }
                ],
                "sponsor_unique_building_selections": [
                    {
                        "card_instance_id": sponsor_243.instance_id,
                        "cells": [list(cell) for cell in legal[0]],
                    }
                ],
            },
        ),
    )

    assert any(item.sponsor_number == 243 for item in player.sponsor_buildings)


def test_sponsor_243_requires_explicit_unique_building_selection():
    state = make_state(715)
    player = state.players[0]
    state.current_player = 0
    player.money = 20
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]
    legal = _list_legal_sponsor_unique_building_cells(state=state, player=player, sponsor_number=243)
    assert len(legal) > 1
    sponsor_243 = AnimalCard(
        name="Unique Build Sponsor",
        cost=4,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=243,
        instance_id="s243-test",
    )
    player.hand = [sponsor_243]

    with pytest.raises(
        ValueError,
        match="Sponsor #243 requires explicit sponsor_unique_building_selections when multiple placements exist.",
    ):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="sponsors",
                details={
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": 0,
                            "card_instance_id": sponsor_243.instance_id,
                        }
                    ],
                },
            ),
        )

    assert player.hand == [sponsor_243]
    assert player.zoo_cards == []


@pytest.mark.parametrize(
    ("seed", "sponsor_number", "reputation", "sponsors_upgraded"),
    [
        (716, 244, 3, False),
        (717, 247, 0, True),
    ],
)
def test_playing_sponsor_244_and_247_places_unique_building(
    seed,
    sponsor_number,
    reputation,
    sponsors_upgraded,
):
    state = make_state(seed)
    player = state.players[0]
    state.current_player = 0
    player.reputation = reputation
    player.money = 20
    player.action_upgraded["sponsors"] = sponsors_upgraded
    set_action_strength(player, "sponsors", 5)
    sponsor_card = take_card_by_number(state, sponsor_number)
    legal = _list_legal_sponsor_unique_building_cells(
        state=state,
        player=player,
        sponsor_number=sponsor_number,
    )
    assert legal
    player.hand = [sponsor_card]

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_card.instance_id,
                    }
                ],
                "sponsor_unique_building_selections": [
                    {
                        "card_instance_id": sponsor_card.instance_id,
                        "cells": [list(cell) for cell in legal[0]],
                    }
                ],
            },
        ),
    )

    assert any(
        item.sponsor_number == sponsor_number and item.cells == legal[0]
        for item in player.sponsor_buildings
    )


@pytest.mark.parametrize(
    ("seed", "sponsor_number", "reputation", "sponsors_upgraded"),
    [
        (718, 244, 3, False),
        (719, 247, 0, True),
    ],
)
def test_sponsor_244_and_247_require_explicit_unique_building_selection(
    seed,
    sponsor_number,
    reputation,
    sponsors_upgraded,
):
    state = make_state(seed)
    player = state.players[0]
    state.current_player = 0
    player.reputation = reputation
    player.money = 20
    player.action_upgraded["sponsors"] = sponsors_upgraded
    set_action_strength(player, "sponsors", 5)
    legal = _list_legal_sponsor_unique_building_cells(
        state=state,
        player=player,
        sponsor_number=sponsor_number,
    )
    assert len(legal) > 1
    sponsor_card = take_card_by_number(state, sponsor_number)
    player.hand = [sponsor_card]

    with pytest.raises(
        ValueError,
        match=(
            f"Sponsor #{sponsor_number} requires explicit sponsor_unique_building_selections "
            "when multiple placements exist."
        ),
    ):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="sponsors",
                details={
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": 0,
                            "card_instance_id": sponsor_card.instance_id,
                        }
                    ],
                },
            ),
        )

    assert player.hand == [sponsor_card]
    assert player.zoo_cards == []


def test_sponsor_227_requires_explicit_mode():
    state = make_state(716)
    player = state.players[0]
    state.current_player = 0
    player.money = 20
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]
    sponsor_227 = AnimalCard(
        name="WAZA",
        cost=4,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=227,
        instance_id="s227-test",
    )
    player.hand = [sponsor_227]

    with pytest.raises(
        ValueError,
        match="Sponsor #227 requires explicit sponsor_227_mode",
    ):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="sponsors",
                details={
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": 0,
                            "card_instance_id": sponsor_227.instance_id,
                        }
                    ],
                },
            ),
        )

    assert player.hand == [sponsor_227]
    assert player.zoo_cards == []


def test_sponsor_227_puts_non_kept_revealed_cards_to_deck_bottom():
    state = make_state(7161)
    player = state.players[0]
    state.current_player = 0
    player.money = 20
    player.reputation = 6
    player.action_order = ["animals", "cards", "build", "association", "sponsors"]
    sponsor_227 = AnimalCard(
        name="WAZA",
        cost=4,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=227,
        instance_id="s227-bottom",
    )
    reveal_1 = AnimalCard(
        name="Reveal Sponsor",
        cost=1,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=99001,
        instance_id="reveal-1",
    )
    reveal_2 = AnimalCard(
        name="Reveal Large",
        cost=1,
        size=4,
        appeal=0,
        conservation=0,
        card_type="animal",
        number=99002,
        instance_id="reveal-2",
    )
    keep_small = AnimalCard(
        name="Keep Small",
        cost=1,
        size=2,
        appeal=0,
        conservation=0,
        card_type="animal",
        number=99003,
        instance_id="keep-small",
    )
    tail_card = AnimalCard(
        name="Tail",
        cost=1,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        number=99004,
        instance_id="tail-card",
    )
    player.hand = [sponsor_227]
    state.zoo_deck = [reveal_1, reveal_2, keep_small, tail_card]
    state.zoo_discard = []

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": False,
                "sponsor_227_mode": "small",
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": 0,
                        "card_instance_id": sponsor_227.instance_id,
                    }
                ],
            },
        ),
    )

    assert any(card.instance_id == sponsor_227.instance_id for card in player.zoo_cards)
    assert any(card.instance_id == keep_small.instance_id for card in player.hand)
    assert [card.instance_id for card in state.zoo_deck] == [
        tail_card.instance_id,
        reveal_1.instance_id,
        reveal_2.instance_id,
    ]
    assert all(
        card.instance_id not in {reveal_1.instance_id, reveal_2.instance_id, keep_small.instance_id}
        for card in state.zoo_discard
    )


def test_final_score_includes_sponsor_endgame_bonus():
    state = make_state(711)
    player = state.players[0]
    player.appeal = 10
    player.conservation = 0
    player.final_scoring_cards = []
    player.zoo_cards = [
        AnimalCard(
            name="QuarantineLab",
            cost=3,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=225,
            instance_id="s225",
        ),
        AnimalCard("AF", 0, 0, 0, 0, badges=("Africa",), number=9901, instance_id="c1"),
        AnimalCard("EU", 0, 0, 0, 0, badges=("Europe",), number=9902, instance_id="c2"),
        AnimalCard("AS", 0, 0, 0, 0, badges=("Asia",), number=9903, instance_id="c3"),
        AnimalCard("AM", 0, 0, 0, 0, badges=("America",), number=9904, instance_id="c4"),
        AnimalCard("AU", 0, 0, 0, 0, badges=("Australia",), number=9905, instance_id="c5"),
    ]

    assert _final_score_points(state, player) == 12  # 10 appeal + 1 CP => +2 points (BGA CP scoring)


def test_sponsor_241_endgame_requires_connected_water_spaces():
    state = make_state(7111)
    player = state.players[0]
    player.zoo_map = copy.deepcopy(player.zoo_map)
    player.zoo_cards = [
        AnimalCard(
            name="Hydrologist",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=241,
            instance_id="s241",
        )
    ]
    player.zoo_map.map_data.terrain.clear()
    player.zoo_map.map_data.terrain.update(
        {
            HexTile(0, 0): Terrain.WATER,
            HexTile(1, 0): Terrain.WATER,
            HexTile(5, 1): Terrain.WATER,
        }
    )

    assert _sponsor_endgame_bonus(state, player) == (0, 0)

    player.zoo_map.map_data.terrain.clear()
    player.zoo_map.map_data.terrain.update(
        {
            HexTile(0, 0): Terrain.WATER,
            HexTile(1, 0): Terrain.WATER,
            HexTile(0, 1): Terrain.WATER,
        }
    )

    assert _sponsor_endgame_bonus(state, player) == (0, 1)


def test_sponsor_242_endgame_requires_connected_rock_spaces():
    state = make_state(7112)
    player = state.players[0]
    player.zoo_map = copy.deepcopy(player.zoo_map)
    player.zoo_cards = [
        AnimalCard(
            name="Geologist",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=242,
            instance_id="s242",
        )
    ]
    player.zoo_map.map_data.terrain.clear()
    player.zoo_map.map_data.terrain.update(
        {
            HexTile(0, 0): Terrain.ROCK,
            HexTile(1, 0): Terrain.ROCK,
            HexTile(5, 1): Terrain.ROCK,
        }
    )

    assert _sponsor_endgame_bonus(state, player) == (0, 0)

    player.zoo_map.map_data.terrain.clear()
    player.zoo_map.map_data.terrain.update(
        {
            HexTile(0, 0): Terrain.ROCK,
            HexTile(1, 0): Terrain.ROCK,
            HexTile(0, 1): Terrain.ROCK,
        }
    )

    assert _sponsor_endgame_bonus(state, player) == (0, 1)


@pytest.mark.parametrize(("sponsor_number", "terrain", "seed"), [(258, Terrain.WATER, 7113), (259, Terrain.ROCK, 7114)])
def test_sponsor_258_and_259_endgame_counts_only_unconnected_terrain_spaces(sponsor_number, terrain, seed):
    state = make_state(seed)
    player = state.players[0]
    player.zoo_map = copy.deepcopy(player.zoo_map)
    player.zoo_cards = [
        AnimalCard(
            name=f"Sponsor{sponsor_number}",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=sponsor_number,
            instance_id=f"s{sponsor_number}",
        )
    ]
    player.zoo_map.map_data.terrain.clear()
    player.zoo_map.map_data.terrain.update(
        {
            HexTile(0, 0): terrain,
            HexTile(1, 0): terrain,
            HexTile(5, 1): terrain,
        }
    )

    assert _sponsor_endgame_bonus(state, player) == (0, 0)

    player.zoo_map.map_data.terrain[HexTile(8, 0)] = terrain

    assert _sponsor_endgame_bonus(state, player) == (0, 1)


def test_sponsor_260_endgame_counts_only_connected_groups_of_6_fillable_spaces():
    state = make_state(7115)
    player = state.players[0]
    player.zoo_map = copy.deepcopy(player.zoo_map)
    player.zoo_cards = [
        AnimalCard(
            name="NativeFarmAnimals",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=260,
            instance_id="s260",
        )
    ]
    player.zoo_map.grid = [
        HexTile(0, 0),
        HexTile(1, 0),
        HexTile(0, 1),
        HexTile(1, 1),
        HexTile(10, 0),
        HexTile(11, 0),
        HexTile(10, 1),
        HexTile(11, 1),
    ]
    player.zoo_map.map_data.terrain.clear()

    assert _sponsor_endgame_bonus(state, player) == (0, 0)

    player.zoo_map.grid.extend([HexTile(2, 0), HexTile(2, 1)])

    assert _sponsor_endgame_bonus(state, player) == (0, 1)


def test_sponsor_264_endgame_counts_only_isolated_placement_bonus_spaces():
    state = make_state(7116)
    player = state.players[0]
    player.zoo_cards = [
        AnimalCard(
            name="FreeRangeNewWorldMonkeys",
            cost=5,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=264,
            instance_id="s264",
        )
    ]
    state.map_tile_bonuses = {
        (0, 0): "card",
        (1, 0): "card",
        (5, 1): "card",
    }

    assert _sponsor_endgame_bonus(state, player) == (0, 0)

    state.map_tile_bonuses[(8, 0)] = "card"

    assert _sponsor_endgame_bonus(state, player) == (0, 1)


def _find_engineer_build_and_extra_options(state):
    for option in list_legal_build_options(state=state, player_id=0, strength=1):
        if list(option.get("placement_bonuses") or []):
            continue
        scratch = copy.deepcopy(state)
        scratch_player = scratch.players[0]
        apply_action(
            scratch,
            Action(
                ActionType.MAIN_ACTION,
                card_name="build",
                details={
                    "selections": [
                        {
                            "building_type": str(option.get("building_type") or ""),
                            "cells": [list(cell) for cell in option.get("cells") or []],
                        }
                    ],
                    "_skip_post_build_effects": True,
                },
            ),
        )
        extra_options = [
            extra
            for extra in list_legal_build_options(state=scratch, player_id=0, strength=int(option.get("size", 0)))
            if extra.get("building_type") == option.get("building_type")
            and not list(extra.get("placement_bonuses") or [])
            and int(extra.get("cost", 0)) <= int(scratch_player.money)
        ]
        if extra_options:
            return option, extra_options[0]
    raise AssertionError("Expected at least one Engineer extra-build candidate.")


def _find_archeologist_border_build_option(state):
    player = state.players[0]
    border_coords = sorted(_player_border_coords(player))
    for option in list_legal_build_options(state=state, player_id=0, strength=1):
        cells = [tuple(cell) for cell in option.get("cells") or []]
        if len(cells) != 1 or cells[0] not in border_coords:
            continue
        extra_coords = [coord for coord in border_coords if coord != cells[0]]
        if len(extra_coords) >= 2:
            return option, cells[0], extra_coords[0], extra_coords[1]
    raise AssertionError("Expected a border build option with at least two other border spaces.")


def test_archeologist_uses_selected_extra_bonus_space():
    state = make_state(71703)
    player = state.players[0]
    player.zoo_cards = [take_card_by_number(state, 221)]

    border_coords = sorted(_player_border_coords(player))
    primary_coord, first_extra_coord, selected_extra_coord = border_coords[:3]
    state.map_tile_bonuses = {
        primary_coord: "5coins",
        first_extra_coord: "5coins",
        selected_extra_coord: "x_token",
    }

    money_before = player.money
    x_tokens_before = player.x_tokens

    _apply_build_placement_bonus(
        state=state,
        player=player,
        bonus="5coins",
        details={"archaeologist_bonus_choices": [{"coord": [selected_extra_coord[0], selected_extra_coord[1]]}]},
        bonus_index=0,
        bonus_coord=primary_coord,
    )

    assert player.money == money_before + 5
    assert player.x_tokens == x_tokens_before + 1


def test_archeologist_build_actions_expand_extra_bonus_choices():
    state = make_state(71704)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "build", 1)
    player.money = 20
    player.zoo_cards = [take_card_by_number(state, 221)]

    option, primary_coord, extra_coord_a, extra_coord_b = _find_archeologist_border_build_option(state)
    state.map_tile_bonuses = {
        primary_coord: "5coins",
        extra_coord_a: "x_token",
        extra_coord_b: "reputation",
    }

    selection = {
        "building_type": str(option.get("building_type") or ""),
        "cells": [list(cell) for cell in option.get("cells") or []],
    }

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "build"
        and list((action.details or {}).get("selections") or []) == [selection]
    ]

    choice_coords = {
        tuple(((action.details or {}).get("archaeologist_bonus_choices") or [{}])[0].get("coord") or ())
        for action in actions
        if (action.details or {}).get("archaeologist_bonus_choices")
    }

    assert choice_coords == {extra_coord_a, extra_coord_b}


def test_engineer_build_actions_expand_skip_and_extra_build_variants():
    state = make_state(71701)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "build", 1)
    player.money = 20
    player.zoo_cards = [take_card_by_number(state, 217)]

    option, extra_option = _find_engineer_build_and_extra_options(state)
    base_selection = {
        "building_type": str(option.get("building_type") or ""),
        "cells": [list(cell) for cell in option.get("cells") or []],
    }
    extra_selection = {
        "building_type": str(extra_option.get("building_type") or ""),
        "cells": [list(cell) for cell in extra_option.get("cells") or []],
    }

    actions = [
        action
        for action in legal_actions(player, state=state, player_id=0)
        if action.type == ActionType.MAIN_ACTION
        and action.card_name == "build"
        and list((action.details or {}).get("selections") or []) == [base_selection]
    ]

    assert any(
        list((action.details or {}).get("engineer_extra_build_choices") or []) == [{"skip": True}]
        for action in actions
    )
    assert any(
        list((action.details or {}).get("engineer_extra_build_choices") or []) == [{"selection": extra_selection}]
        for action in actions
    )


def test_engineer_extra_build_can_be_skipped():
    state = make_state(71702)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "build", 1)
    player.money = 20
    player.zoo_cards = [take_card_by_number(state, 217)]

    option, _extra_option = _find_engineer_build_and_extra_options(state)
    selection = {
        "building_type": str(option.get("building_type") or ""),
        "cells": [list(cell) for cell in option.get("cells") or []],
    }
    money_before = player.money
    building_count_before = len(player.zoo_map.buildings)

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="build",
            details={
                "selections": [selection],
                "engineer_extra_build_choices": [{"skip": True}],
            },
        ),
    )

    assert player.money == money_before - int(option.get("cost", 0))
    assert len(player.zoo_map.buildings) == building_count_before + 1
    assert not any(str(entry).startswith("sponsor_217_extra_build") for entry in state.effect_log)


def test_engineer_extra_build_uses_explicit_selection():
    state = make_state(71703)
    player = state.players[0]
    state.current_player = 0
    set_action_strength(player, "build", 1)
    player.money = 20
    player.zoo_cards = [take_card_by_number(state, 217)]

    option, extra_option = _find_engineer_build_and_extra_options(state)
    selection = {
        "building_type": str(option.get("building_type") or ""),
        "cells": [list(cell) for cell in option.get("cells") or []],
    }
    extra_selection = {
        "building_type": str(extra_option.get("building_type") or ""),
        "cells": [list(cell) for cell in extra_option.get("cells") or []],
    }
    money_before = player.money
    building_count_before = len(player.zoo_map.buildings)

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="build",
            details={
                "selections": [selection],
                "engineer_extra_build_choices": [{"selection": extra_selection}],
            },
        ),
    )

    assert player.money == money_before - int(option.get("cost", 0)) - int(extra_option.get("cost", 0))
    assert len(player.zoo_map.buildings) == building_count_before + 2
    assert any(str(entry).startswith("sponsor_217_extra_build") for entry in state.effect_log)


def test_sponsor_badge_overrides_do_not_double_count_dataset_badges():
    state = make_state(712)
    player = state.players[0]
    science_lab = take_card_by_number(state, 201)
    science_institute = take_card_by_number(state, 223)
    player.zoo_cards.extend([science_lab, science_institute])

    icons = _player_icon_snapshot(player)
    # #201 provides 1 science icon, #223 provides 2 science icons.
    assert icons["science"] == 3


def test_sponsor_passive_science_trigger_uses_icon_count_not_card_count():
    state = make_state(713)
    owner = state.players[0]
    actor_id = 0
    owner.zoo_cards.append(take_card_by_number(state, 202))
    rep_before = owner.reputation

    played_card = AnimalCard(
        name="DOUBLE SCIENCE CARD",
        cost=0,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        badges=("Science", "Science"),
        number=9998,
        instance_id="double-sci",
    )
    _apply_sponsor_passive_triggers_on_card_play(
        state=state,
        played_by_player_id=actor_id,
        played_card=played_card,
    )

    assert owner.reputation == rep_before + 2


def test_sponsor_unique_shapes_use_configured_footprints():
    assert _sponsor_unique_base_footprint(243) == ((0, 0), (0, 1), (1, 1))
    assert _sponsor_unique_base_footprint(249) == ((0, 0), (1, 0), (2, 0))
    assert _sponsor_unique_base_footprint(250) == ((0, 0), (1, 0), (1, 1), (2, -1))
    assert _sponsor_unique_base_footprint(254) == ((0, 0), (1, 0), (1, 1))
    assert _sponsor_unique_base_footprint(257) == ((0, 0), (1, 0))


def test_sponsor_unique_legal_cells_match_shape_sizes():
    state = make_state(714)
    player = state.players[0]
    for sponsor_number, expected_size in [(243, 3), (244, 4), (249, 3), (250, 4), (254, 3), (257, 2)]:
        legal = _list_legal_sponsor_unique_building_cells(
            state=state,
            player=player,
            sponsor_number=sponsor_number,
        )
        assert legal
        assert all(len(cells) == expected_size for cells in legal)

import pytest

from tests.helpers import make_state, set_action_strength, take_card_by_number

from main import (
    Action,
    ActionType,
    AnimalCard,
    Enclosure,
    EnclosureObject,
    SponsorBuilding,
    Building,
    BuildingType,
    HexTile,
    _resolve_break,
    _resolve_break_remaining_stages,
    _final_score_points,
    _apply_sponsor_passive_triggers_on_card_play,
    _player_icon_snapshot,
    _sponsor_unique_base_footprint,
    _list_legal_sponsor_unique_building_cells,
    apply_action,
    legal_actions,
    list_legal_animals_options,
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

from main import (
    AnimalCard,
    Action,
    ActionType,
    Enclosure,
    EnclosureObject,
    _player_icon_snapshot,
    apply_action,
    legal_actions,
    list_legal_animals_options,
    list_legal_build_options,
    setup_game,
)
from arknova_engine.map_model import BuildingType


def test_build_options_initial_follow_border_adjacency_rule():
    state = setup_game(seed=601, player_names=["P1", "P2"])
    p0 = state.players[0]
    assert p0.zoo_map is not None

    options = list_legal_build_options(state=state, player_id=0, strength=3)
    assert options

    border = p0.zoo_map.border_tiles(ignore_terrain=False)
    for opt in options:
        layout_tiles = {(x, y) for x, y in opt["cells"]}
        assert any((tile.x, tile.y) in layout_tiles for tile in border)

    size3_footprints = {
        tuple(sorted((tile[0], tile[1]) for tile in opt["cells"]))
        for opt in options
        if opt["building_type"] == "SIZE_3"
    }
    assert len(size3_footprints) >= 2


def test_build_options_are_deduplicated_by_footprint():
    state = setup_game(seed=604, player_names=["P1", "P2"])
    options = list_legal_build_options(state=state, player_id=0, strength=5)

    keys = {
        (
            opt["building_type"],
            tuple(sorted((tile[0], tile[1]) for tile in opt["cells"])),
        )
        for opt in options
    }
    assert len(keys) == len(options)


def test_size5_options_match_all_unique_legal_footprints():
    state = setup_game(seed=606, player_names=["P1", "P2"])
    p0 = state.players[0]
    assert p0.zoo_map is not None

    options = list_legal_build_options(state=state, player_id=0, strength=5)
    ui_size5 = {
        tuple(sorted((cell[0], cell[1]) for cell in opt["cells"]))
        for opt in options
        if opt["building_type"] == "SIZE_5"
    }

    raw_legal = p0.zoo_map.legal_building_placements(
        is_build_upgraded=p0.action_upgraded["build"],
        has_diversity_researcher=False,
        max_building_size=5,
        already_built_buildings=set(),
    )
    engine_size5 = {
        tuple(sorted((tile.x, tile.y) for tile in b.layout))
        for b in raw_legal
        if b.type == BuildingType.SIZE_5
    }

    assert ui_size5 == engine_size5


def test_build_action_uses_serialized_selection_and_places_building():
    state = setup_game(seed=602, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["build", "animals", "cards", "association", "sponsors"]
    p0.money = 50
    assert p0.zoo_map is not None

    options = list_legal_build_options(state=state, player_id=0, strength=1)
    pick = next(opt for opt in options if opt["building_type"] == "SIZE_1")
    money_before = p0.money

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="build",
            details={
                "selections": [
                    {
                        "building_type": pick["building_type"],
                        "cells": pick["cells"],
                    }
                ]
            },
        ),
    )

    assert p0.money == money_before - pick["cost"]
    assert any(
        building.type.name == pick["building_type"]
        and sorted((tile.x, tile.y) for tile in building.layout)
        == sorted((cell[0], cell[1]) for cell in pick["cells"])
        for building in p0.zoo_map.buildings.values()
    )
    assert p0.enclosure_objects
    obj = p0.enclosure_objects[-1]
    assert obj.size == pick["size"]
    assert obj.enclosure_type == "enclosure_1"
    assert isinstance(obj.adjacent_rock, int)
    assert isinstance(obj.adjacent_water, int)
    assert obj.animals_inside == 0


def test_partner_zoo_discount_applies_per_matching_continent_badge():
    state = setup_game(seed=608, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.partner_zoos.add("asia")
    p0.money = 20
    p0.enclosures = [Enclosure(size=2, occupied=False)]
    p0.enclosure_objects = [
        EnclosureObject(
            size=2,
            enclosure_type="enclosure_2",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(0, 0),
            rotation="ROT_0",
        )
    ]
    p0.hand = [
        AnimalCard(
            name="ASIA TEST ANIMAL",
            cost=14,
            size=2,
            appeal=7,
            conservation=0,
            reputation_gain=0,
            card_type="animal",
            badges=("Asia", "Reptile"),
            required_icons=(("reptile", 2),),
            number=9974,
            instance_id="9974",
        )
    ]
    p0.zoo_cards = [
        AnimalCard(
            name="REPTILE ICON SOURCE",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            reputation_gain=0,
            card_type="animal",
            badges=("Reptile", "Reptile"),
            number=9975,
            instance_id="9975",
        )
    ]

    options = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert len(options) == 1
    play = options[0]["plays"][0]
    assert play["card_base_cost"] == 14
    assert play["card_partner_discount"] == 3
    assert play["card_discount"] == 3
    assert play["card_cost"] == 11
    assert options[0]["total_cost"] == 11


def test_partner_zoo_requirement_must_match_the_animal_continent_badge():
    state = setup_game(seed=609, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.money = 20
    p0.enclosures = [Enclosure(size=2, occupied=False)]
    p0.enclosure_objects = [
        EnclosureObject(
            size=2,
            enclosure_type="enclosure_2",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(0, 0),
            rotation="ROT_0",
        )
    ]
    p0.hand = [
        AnimalCard(
            name="ASIA PARTNER REQUIRED",
            cost=10,
            size=2,
            appeal=5,
            conservation=0,
            reputation_gain=0,
            card_type="animal",
            badges=("Asia", "Reptile"),
            required_icons=(("partner_zoo", 1),),
            number=9976,
            instance_id="9976",
        )
    ]
    p0.zoo_cards = []

    p0.partner_zoos = {"africa"}
    assert list_legal_animals_options(state=state, player_id=0, strength=2) == []

    p0.partner_zoos = {"asia"}
    options = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert len(options) == 1
    assert options[0]["plays"][0]["card_cost"] == 7


def test_build_placement_bonus_x_token_is_applied():
    state = setup_game(seed=603, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["build", "animals", "cards", "association", "sponsors"]
    p0.money = 50
    p0.x_tokens = 0

    options = list_legal_build_options(state=state, player_id=0, strength=1)
    pick = next(opt for opt in options if "x_token" in opt["placement_bonuses"])

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="build",
            details={
                "selections": [
                    {
                        "building_type": pick["building_type"],
                        "cells": pick["cells"],
                    }
                ]
            },
        ),
    )

    assert p0.x_tokens == 1


def test_animals_action_updates_enclosure_object_animals_inside():
    state = setup_game(seed=605, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.money = 50
    state.current_player = 0
    p0.action_order = ["build", "animals", "cards", "association", "sponsors"]

    options = list_legal_build_options(state=state, player_id=0, strength=1)
    pick = next(opt for opt in options if opt["building_type"] == "SIZE_1")
    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="build",
            details={
                "selections": [
                    {
                        "building_type": pick["building_type"],
                        "cells": pick["cells"],
                    }
                ]
            },
        ),
    )

    p0.hand = [
        AnimalCard(
            name="TEST_ANIMAL",
            cost=1,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=999001,
            instance_id="999001",
        )
    ]
    state.current_player = 0
    p0.action_order = ["build", "animals", "cards", "association", "sponsors"]
    apply_action(state, Action(ActionType.MAIN_ACTION, card_name="animals"))

    assert p0.enclosure_objects
    assert p0.enclosure_objects[0].animals_inside == 1


def test_upgraded_animals_options_include_two_play_permutations():
    state = setup_game(seed=607, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.money = 50
    p0.action_upgraded["animals"] = True
    p0.action_order = ["cards", "build", "animals", "association", "sponsors"]  # animals strength=3
    p0.enclosures = [
        Enclosure(size=2, occupied=False, origin=(0, 0), rotation="ROT_0"),
        Enclosure(size=2, occupied=False, origin=(1, -1), rotation="ROT_0"),
    ]
    p0.enclosure_objects = [
        EnclosureObject(2, "enclosure_2", 0, 0, 0, (0, 0), "ROT_0"),
        EnclosureObject(2, "enclosure_2", 0, 0, 0, (1, -1), "ROT_0"),
    ]
    p0.hand = [
        AnimalCard(
            name="A1",
            cost=3,
            size=2,
            appeal=2,
            conservation=0,
            card_type="animal",
            number=910001,
            instance_id="910001",
        ),
        AnimalCard(
            name="A2",
            cost=4,
            size=2,
            appeal=3,
            conservation=1,
            card_type="animal",
            number=910002,
            instance_id="910002",
        ),
    ]

    options = list_legal_animals_options(state=state, player_id=0, strength=3)

    assert options
    assert any(len(opt["plays"]) == 2 for opt in options)
    two_play_option = next(opt for opt in options if len(opt["plays"]) == 2)

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": int(two_play_option["index"]) - 1},
        ),
    )

    assert len(p0.hand) == 0
    assert sum(1 for e in p0.enclosures if e.occupied) == 2


def test_upgraded_animals_strength5_grants_one_reputation():
    state = setup_game(seed=608, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.reputation = 0
    p0.action_upgraded["animals"] = True
    p0.action_upgraded["association"] = True
    p0.action_order = ["cards", "build", "association", "sponsors", "animals"]  # animals strength=5

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"skip_animals_action": True},
        ),
    )

    assert p0.reputation == 1


def test_animals_option_totals_include_reputation_and_apply_it():
    state = setup_game(seed=609, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.money = 50
    p0.action_upgraded["association"] = True
    p0.action_order = ["cards", "animals", "build", "association", "sponsors"]  # animals strength=2
    p0.enclosures = [Enclosure(size=2, occupied=False, origin=(0, 0), rotation="ROT_0")]
    p0.enclosure_objects = [EnclosureObject(2, "enclosure_2", 0, 0, 0, (0, 0), "ROT_0")]
    p0.hand = [
        AnimalCard(
            name="REP_ANIMAL",
            cost=3,
            size=2,
            appeal=2,
            reputation_gain=2,
            conservation=1,
            card_type="animal",
            number=920001,
            instance_id="920001",
        )
    ]

    options = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert options
    assert options[0]["total_reputation"] == 2

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": 0},
        ),
    )
    assert p0.reputation == 3


def test_americas_badge_maps_to_continent_america():
    state = setup_game(seed=610, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.zoo_cards = [
        AnimalCard(
            name="COATI",
            cost=11,
            size=2,
            appeal=4,
            conservation=0,
            reputation_gain=0,
            card_type="animal",
            badges=("Americas", "Predator"),
            number=414,
            instance_id="414",
        )
    ]

    icons = _player_icon_snapshot(p0)
    assert icons["continents"]["America"] == 1
    assert icons["continents"]["Africa"] == 0
    assert icons["categories"].get("Americas", 0) == 0
    assert icons["categories"]["Predator"] == 1


def test_animals_required_icons_gate_playability():
    state = setup_game(seed=611, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.money = 50
    p0.action_order = ["cards", "animals", "build", "association", "sponsors"]  # animals strength=2
    p0.enclosures = [Enclosure(size=2, occupied=False, origin=(0, 0), rotation="ROT_0")]
    p0.enclosure_objects = [EnclosureObject(2, "enclosure_2", 0, 0, 0, (0, 0), "ROT_0")]
    gated = AnimalCard(
        name="SCIENCE_GATED",
        cost=3,
        size=2,
        appeal=2,
        conservation=1,
        reputation_gain=0,
        card_type="animal",
        badges=("Primate",),
        required_icons=(("science", 2),),
        number=930001,
        instance_id="930001",
    )
    p0.hand = [gated]

    options_without_science = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert options_without_science == []

    p0.zoo_cards = [
        AnimalCard(
            name="SCI_A",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            reputation_gain=0,
            card_type="animal",
            badges=("Science",),
            number=930101,
            instance_id="930101",
        ),
        AnimalCard(
            name="SCI_B",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            reputation_gain=0,
            card_type="animal",
            badges=("Science",),
            number=930102,
            instance_id="930102",
        ),
    ]

    options_with_science = list_legal_animals_options(state=state, player_id=0, strength=2)
    assert len(options_with_science) == 1


def test_upgraded_build_can_place_reptile_house_and_register_host_enclosure():
    state = setup_game(seed=612, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.money = 50
    p0.action_upgraded["build"] = True
    p0.action_order = ["animals", "cards", "association", "sponsors", "build"]

    options = list_legal_build_options(state=state, player_id=0, strength=5)
    pick = next(opt for opt in options if opt["building_type"] == "REPTILE_HOUSE")

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="build",
            details={
                "selections": [
                    {
                        "building_type": pick["building_type"],
                        "cells": pick["cells"],
                    }
                ]
            },
        ),
    )

    reptile_enclosures = [e for e in p0.enclosures if e.enclosure_type == "reptile_house"]
    assert len(reptile_enclosures) == 1
    assert reptile_enclosures[0].animal_capacity == 5
    assert reptile_enclosures[0].used_capacity == 0


def test_reptile_house_can_host_multiple_animals_with_special_space_values():
    state = setup_game(seed=613, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.money = 50
    p0.action_upgraded["animals"] = True
    p0.action_order = ["cards", "build", "animals", "association", "sponsors"]
    p0.enclosures = [
        Enclosure(
            size=5,
            occupied=False,
            origin=(0, 0),
            rotation="ROT_0",
            enclosure_type="reptile_house",
            used_capacity=0,
            animal_capacity=5,
        )
    ]
    p0.enclosure_objects = [
        EnclosureObject(5, "reptile_house", 0, 0, 0, (0, 0), "ROT_0")
    ]
    p0.hand = [
        AnimalCard(
            name="R1",
            cost=4,
            size=4,
            appeal=2,
            conservation=0,
            card_type="animal",
            badges=("Reptile",),
            reptile_house_size=0,
            number=940001,
            instance_id="940001",
        ),
        AnimalCard(
            name="R2",
            cost=5,
            size=5,
            appeal=3,
            conservation=1,
            card_type="animal",
            badges=("Reptile",),
            reptile_house_size=2,
            number=940002,
            instance_id="940002",
        ),
    ]

    options = list_legal_animals_options(state=state, player_id=0, strength=3)
    assert any(
        len(opt["plays"]) == 2 and all(int(play["enclosure_index"]) == 0 for play in opt["plays"])
        for opt in options
    )
    two_play = next(
        opt
        for opt in options
        if len(opt["plays"]) == 2 and all(int(play["enclosure_index"]) == 0 for play in opt["plays"])
    )

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="animals",
            details={"animals_sequence_index": int(two_play["index"]) - 1},
        ),
    )

    assert len(p0.hand) == 0
    assert p0.enclosures[0].used_capacity == 2
    assert p0.enclosure_objects[0].animals_inside == 2


def test_standard_enclosure_still_holds_only_one_animal_even_when_animals_is_upgraded():
    state = setup_game(seed=614, player_names=["P1", "P2"])
    p0 = state.players[0]
    p0.money = 50
    p0.action_upgraded["animals"] = True
    p0.action_order = ["cards", "build", "animals", "association", "sponsors"]
    p0.enclosures = [Enclosure(size=5, occupied=False, origin=(0, 0), rotation="ROT_0")]
    p0.enclosure_objects = [EnclosureObject(5, "enclosure_5", 0, 0, 0, (0, 0), "ROT_0")]
    p0.hand = [
        AnimalCard(
            name="A1",
            cost=2,
            size=2,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=950001,
            instance_id="950001",
        ),
        AnimalCard(
            name="A2",
            cost=3,
            size=3,
            appeal=2,
            conservation=0,
            card_type="animal",
            number=950002,
            instance_id="950002",
        ),
    ]

    options = list_legal_animals_options(state=state, player_id=0, strength=3)
    assert options
    assert not any(len(opt["plays"]) == 2 for opt in options)


def test_legal_actions_filters_two_play_animals_sequence_invalidated_by_trade_effect():
    state = setup_game(seed=615, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.money = 50
    p0.action_upgraded["animals"] = True
    p0.action_order = ["cards", "build", "animals", "association", "sponsors"]  # animals strength=3
    p0.reputation = 10
    p0.enclosures = [
        Enclosure(size=2, occupied=False, origin=(0, 0), rotation="ROT_0"),
        Enclosure(size=2, occupied=False, origin=(1, -1), rotation="ROT_0"),
    ]
    p0.enclosure_objects = [
        EnclosureObject(2, "enclosure_2", 0, 0, 0, (0, 0), "ROT_0"),
        EnclosureObject(2, "enclosure_2", 0, 0, 0, (1, -1), "ROT_0"),
    ]
    p0.hand = [
        AnimalCard(
            name="Target",
            cost=3,
            size=2,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=960001,
            instance_id="960001",
        ),
        AnimalCard(
            name="Trader",
            cost=3,
            size=2,
            appeal=1,
            conservation=0,
            card_type="animal",
            ability_title="Trade",
            number=960002,
            instance_id="960002",
        ),
    ]
    state.zoo_display = [
        AnimalCard(
            name="DisplayCard",
            cost=2,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=960100,
            instance_id="960100",
        )
    ]

    options = list_legal_animals_options(state=state, player_id=0, strength=3)
    invalid_idx = next(
        int(option["index"]) - 1
        for option in options
        if [str(play.get("card_instance_id") or "") for play in option.get("plays") or []] == ["960002", "960001"]
    )

    actions = legal_actions(p0, state=state, player_id=0)
    animal_indices = {
        int((action.details or {}).get("animals_sequence_index", -1))
        for action in actions
        if action.type == ActionType.MAIN_ACTION and str(action.card_name or "") == "animals"
    }

    assert invalid_idx not in animal_indices

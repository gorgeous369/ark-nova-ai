from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation
from main import (
    AnimalCard,
    Enclosure,
    EnclosureObject,
    SetupCardRef,
    _ensure_player_map_initialized,
    _perform_animals_action_effect,
    setup_game,
)


def _prepare_basic_animals_play_state(seed: int = 610):
    state = setup_game(seed=seed, player_names=["P1", "P2"])
    player = state.players[0]
    player.money = 25
    player.reputation = 6
    player.hand = []
    player.zoo_cards = []
    player.enclosures = [
        Enclosure(size=2, occupied=False, origin=(0, 0), rotation="ROT_0"),
    ]
    player.enclosure_objects = [
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
    return state, player


def test_mark_effect_marks_first_display_animal_in_range():
    state, player = _prepare_basic_animals_play_state(seed=611)
    player.hand = [
        AnimalCard(
            name="Marker",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Mark",
            card_type="animal",
            number=9001,
            instance_id="test-mark",
        )
    ]
    state.zoo_display = [
        AnimalCard(
            name="Display Animal",
            cost=3,
            size=1,
            appeal=2,
            conservation=0,
            card_type="animal",
            number=9101,
            instance_id="disp-animal",
        ),
        AnimalCard(
            name="Display Sponsor",
            cost=3,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=9102,
            instance_id="disp-sponsor",
        ),
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert "disp-animal" in state.marked_display_card_ids
    assert any("effect[mark_display_animal] marked=1" in line for line in state.effect_log)


def test_dominance_effect_takes_specific_unused_base_project():
    state, player = _prepare_basic_animals_play_state(seed=612)
    state.unused_base_conservation_projects = [
        SetupCardRef(data_id="P108_Primates", title="PRIMATES"),
        SetupCardRef(data_id="P103_Africa", title="AFRICA"),
    ]
    player.hand = [
        AnimalCard(
            name="Dominance Animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Dominance",
            card_type="animal",
            number=9002,
            instance_id="test-dominance",
        )
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert all(project.data_id != "P108_Primates" for project in state.unused_base_conservation_projects)
    assert any(card.card_type == "conservation_project" and "PRIMATES" in card.name for card in player.hand)
    assert any("effect[take_specific_base_project] target=primates taken=1" in line for line in state.effect_log)


def test_cut_down_effect_removes_one_empty_standard_enclosure_and_refunds():
    state, player = _prepare_basic_animals_play_state(seed=613)
    _ensure_player_map_initialized(state, player)
    assert player.zoo_map is not None
    player.zoo_map.add_building(Building(BuildingType.SIZE_2, HexTile(0, 0), Rotation.ROT_0))
    player.zoo_map.add_building(Building(BuildingType.SIZE_1, HexTile(2, 0), Rotation.ROT_0))
    player.enclosures.append(Enclosure(size=1, occupied=False, origin=(2, 0), rotation="ROT_0"))
    player.enclosure_objects.append(
        EnclosureObject(
            size=1,
            enclosure_type="enclosure_1",
            adjacent_rock=0,
            adjacent_water=0,
            animals_inside=0,
            origin=(2, 0),
            rotation="ROT_0",
        )
    )
    player.money = 10
    player.hand = [
        AnimalCard(
            name="Cutter",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Cut Down",
            card_type="animal",
            number=9003,
            instance_id="test-cut-down",
        )
    ]

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert len(player.enclosures) == 1
    assert player.money == 12
    assert any("effect[remove_empty_enclosure_refund] removed=1 refunded=2" in line for line in state.effect_log)


def test_shark_attack_effect_discards_animals_from_display_and_gains_money():
    state, player = _prepare_basic_animals_play_state(seed=614)
    player.reputation = 15
    player.money = 5
    player.hand = [
        AnimalCard(
            name="Shark",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            ability_title="Shark Attack 2",
            card_type="animal",
            number=9004,
            instance_id="test-shark",
        )
    ]
    state.zoo_display = [
        AnimalCard(
            name="A1",
            cost=3,
            size=1,
            appeal=5,
            conservation=0,
            card_type="animal",
            number=9201,
            instance_id="disp-a1",
        ),
        AnimalCard(
            name="A2",
            cost=4,
            size=1,
            appeal=3,
            conservation=0,
            card_type="animal",
            number=9202,
            instance_id="disp-a2",
        ),
        AnimalCard(
            name="S1",
            cost=4,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            number=9203,
            instance_id="disp-s1",
        ),
    ]
    state.zoo_deck = [
        AnimalCard(
            name="Top1",
            cost=3,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9301,
            instance_id="deck-1",
        ),
        AnimalCard(
            name="Top2",
            cost=3,
            size=1,
            appeal=1,
            conservation=0,
            card_type="animal",
            number=9302,
            instance_id="deck-2",
        ),
    ]
    discard_before = len(state.zoo_discard)

    _perform_animals_action_effect(
        state=state,
        player=player,
        strength=2,
        details={"animals_sequence_index": 0},
        player_id=0,
    )

    assert len(state.zoo_discard) == discard_before + 2
    assert player.money == 9
    assert any("effect[shark_attack] discarded=2 money=+4" in line for line in state.effect_log)

import itertools

import main

from main import (
    AnimalCard,
    Building,
    BuildingType,
    HexTile,
    Rotation,
    SetupCardRef,
    _bga_conservation_points,
    _final_score_points,
    _final_scoring_conservation_bonus_from_cards,
    setup_game,
)


_CARD_IDS = itertools.count(9000)


def _next_card_number() -> int:
    return next(_CARD_IDS)


def _animal(
    *,
    size: int = 2,
    badges=(),
    required_icons=(),
    required_rock_adjacency: int = 0,
    required_water_adjacency: int = 0,
) -> AnimalCard:
    return AnimalCard(
        name=f"Animal{_next_card_number()}",
        cost=0,
        size=size,
        appeal=0,
        conservation=0,
        card_type="animal",
        badges=tuple(badges),
        required_icons=tuple(required_icons),
        required_rock_adjacency=required_rock_adjacency,
        required_water_adjacency=required_water_adjacency,
        number=_next_card_number(),
        instance_id=f"animal-{_next_card_number()}",
    )


def _sponsor(
    *,
    badges=(),
    required_icons=(),
    number: int | None = None,
) -> AnimalCard:
    card_number = _next_card_number() if number is None else number
    return AnimalCard(
        name=f"Sponsor{card_number}",
        cost=0,
        size=0,
        appeal=0,
        conservation=0,
        card_type="sponsor",
        badges=tuple(badges),
        required_icons=tuple(required_icons),
        number=card_number,
        instance_id=f"sponsor-{card_number}",
    )


def _fresh_state(seed: int = 901):
    state = setup_game(seed=seed, player_names=["P1", "P2"])
    for player in state.players:
        player.appeal = 0
        player.conservation = 0
        player.reputation = 1
        player.zoo_cards = []
        player.partner_zoos.clear()
        player.universities.clear()
        player.supported_conservation_projects.clear()
        player.supported_conservation_project_actions = 0
        player.sponsor_buildings = []
        if player.zoo_map is not None:
            player.zoo_map.buildings.clear()
    return state, state.players[0], state.players[1]


def _only_final_card(player, data_id: str) -> None:
    player.final_scoring_cards = [SetupCardRef(data_id=data_id, title=data_id)]


def _building_at(building_type: BuildingType, x: int, y: int) -> Building:
    return Building(type=building_type, origin_hex=HexTile(x, y), rotation=Rotation.ROT_0)


def test_setup_game_uses_full_17_card_final_scoring_pool():
    state = setup_game(seed=902, player_names=["P1", "P2"], include_marine_world=True)

    all_cards = list(state.final_scoring_deck)
    for player in state.players:
        all_cards.extend(player.final_scoring_cards)

    numbers = sorted(int(card.data_id[1:4]) for card in all_cards)
    assert len(all_cards) == 17
    assert len({card.data_id for card in all_cards}) == 17
    assert numbers[0] == 1
    assert numbers[-1] == 17
    assert len(state.final_scoring_deck) == 13


def test_final_scoring_f001_large_animal_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F001_LargeAnimalZoo")
    p0.zoo_cards = [_animal(size=4), _animal(size=5), _animal(size=4)]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f002_small_animal_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F002_SmallAnimalZoo")
    p0.zoo_cards = [_animal(size=2) for _ in range(8)]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f003_research_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F003_ResearchZoo")
    p0.zoo_cards = [_animal(badges=("Science",)) for _ in range(5)]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f004_architectural_zoo(monkeypatch):
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F004_ArchitecturalZoo")
    monkeypatch.setattr(main, "_all_terrain_spaces_adjacent_to_buildings", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(main, "_all_border_spaces_adjacent_to_buildings", lambda *_args, **_kwargs: True)
    monkeypatch.setattr(main, "_is_map_completely_covered", lambda *_args, **_kwargs: True)

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 4


def test_final_scoring_f005_conservation_zoo_uses_supported_actions():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F005_ConservationZoo")
    p0.supported_conservation_project_actions = 5

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 4


def test_final_scoring_f006_naturalists_zoo(monkeypatch):
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F006_NaturalistsZoo")
    monkeypatch.setattr(main, "_count_empty_fillable_spaces", lambda *_args, **_kwargs: 18)

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f007_favorite_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F007_FavoriteZoo")
    p0.reputation = 12

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f008_sponsored_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F008_SponsoredZoo")
    p0.zoo_cards = [_sponsor() for _ in range(7)]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f009_diverse_species_zoo():
    state, p0, p1 = _fresh_state()
    _only_final_card(p0, "F009_DiverseSpeciesZoo")
    p0.zoo_cards = [
        _animal(badges=("Bird",)),
        _animal(badges=("Bird",)),
        _animal(badges=("Herbivore",)),
        _animal(badges=("Predator",)),
    ]
    p1.zoo_cards = [
        _animal(badges=("Bird",)),
        _animal(badges=("Predator",)),
        _animal(badges=("Reptile",)),
    ]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 2


def test_final_scoring_f010_climbing_park():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F010_ClimbingPark")
    p0.zoo_cards = [_animal(badges=("Rock",)) for _ in range(5)]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f011_aquatic_park():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F011_AquaticPark")
    p0.zoo_cards = [_animal(badges=("Water",)) for _ in range(6)]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f012_designer_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F012_DesignerZoo")
    p0.zoo_map.buildings = {
        HexTile(0, 0): _building_at(BuildingType.SIZE_1, 0, 0),
        HexTile(1, 0): _building_at(BuildingType.SIZE_2, 1, 0),
        HexTile(2, 0): _building_at(BuildingType.SIZE_3, 2, 0),
        HexTile(3, 0): _building_at(BuildingType.SIZE_4, 3, 0),
        HexTile(4, 0): _building_at(BuildingType.SIZE_5, 4, 0),
        HexTile(5, 0): _building_at(BuildingType.REPTILE_HOUSE, 5, 0),
        HexTile(6, 0): _building_at(BuildingType.LARGE_BIRD_AVIARY, 6, 0),
    }

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f013_specialized_habitat_zoo_skips_supported_base_project():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F013_SpecializedHabitatZoo")
    p0.zoo_cards = [
        _animal(badges=("Europe",)),
        _animal(badges=("Europe",)),
        _animal(badges=("Europe",)),
        _animal(badges=("Europe",)),
        _animal(badges=("Europe",)),
        _animal(badges=("Africa",)),
        _animal(badges=("Africa",)),
        _animal(badges=("Africa",)),
        _animal(badges=("Africa",)),
    ]
    p0.supported_conservation_projects.add("P107_Europe")

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 2


def test_final_scoring_f014_specialized_species_zoo_skips_supported_base_project():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F014_SpecializedSpeciesZoo")
    p0.zoo_cards = [
        _animal(badges=("Bird",)),
        _animal(badges=("Bird",)),
        _animal(badges=("Bird",)),
        _animal(badges=("Bird",)),
        _animal(badges=("Bird",)),
        _animal(badges=("Predator",)),
        _animal(badges=("Predator",)),
        _animal(badges=("Predator",)),
        _animal(badges=("Predator",)),
    ]
    p0.supported_conservation_projects.add("P112_Birds")

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 2


def test_final_scoring_f015_catered_picnic_areas():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F015_CateredPicnicAreas")
    p0.zoo_map.buildings = {
        HexTile(0, 0): _building_at(BuildingType.KIOSK, 0, 0),
        HexTile(1, 0): _building_at(BuildingType.KIOSK, 1, 0),
        HexTile(2, 0): _building_at(BuildingType.KIOSK, 2, 0),
        HexTile(3, 0): _building_at(BuildingType.KIOSK, 3, 0),
        HexTile(4, 0): _building_at(BuildingType.PAVILION, 4, 0),
        HexTile(5, 0): _building_at(BuildingType.PAVILION, 5, 0),
        HexTile(6, 0): _building_at(BuildingType.PAVILION, 6, 0),
    }

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 2


def test_final_scoring_f016_accessible_zoo():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F016_AccessibleZoo")
    p0.zoo_cards = [
        _animal(
            required_icons=(("Science", 2), ("Predator", 1)),
            required_rock_adjacency=1,
            required_water_adjacency=1,
        ),
        _sponsor(required_icons=(("Science", 2), ("Reputation", 5), ("Bird", 1))),
        _sponsor(required_icons=(("Herbivore", 1), ("Predator", 1), ("Reputation", 3))),
    ]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 3


def test_final_scoring_f017_international_zoo_counts_own_partner_zoo_twice():
    state, p0, p1 = _fresh_state()
    _only_final_card(p0, "F017_InternationalZoo")
    p0.zoo_cards = [_animal(badges=("Africa",))]
    p0.partner_zoos.add("africa")
    p1.zoo_cards = [_animal(badges=("Africa",)), _animal(badges=("Africa",))]

    assert _final_scoring_conservation_bonus_from_cards(state, p0) == 1


def test_final_score_points_include_final_scoring_conservation_bonus():
    state, p0, _ = _fresh_state()
    _only_final_card(p0, "F007_FavoriteZoo")
    p0.appeal = 30
    p0.conservation = 8
    p0.reputation = 12

    expected = 30 + _bga_conservation_points(11)
    assert _final_score_points(state, p0) == expected

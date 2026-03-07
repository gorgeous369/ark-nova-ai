import pytest

from arknova_engine.base_game import (
    AssociationTask,
    AssociationTaskSelection,
    AnimalCard,
    AnimalPlacement,
    ConservationProjectCard,
    ConservationProjectSlot,
    ConservationRequirementKind,
    MainAction,
    create_base_game,
)
from arknova_engine.map_model import Building, BuildingType, HexTile, Rotation


def _set_association_strength_slot(state, slot_index: int) -> None:
    order = [MainAction.ANIMALS, MainAction.CARDS, MainAction.BUILD, MainAction.SPONSORS]
    order.insert(slot_index - 1, MainAction.ASSOCIATION)
    state.current().action_order = order


def _project(
    card_id: str,
    requirement_kind: ConservationRequirementKind,
    slots,
    *,
    requirement_icon: str | None = None,
    requirement_group: str | None = None,
) -> ConservationProjectCard:
    return ConservationProjectCard(
        card_id=card_id,
        title=card_id,
        requirement_kind=requirement_kind,
        requirement_icon=requirement_icon,
        requirement_group=requirement_group,
        slots=tuple(slots),
    )


def test_association_side_i_single_reputation_task():
    state = create_base_game(num_players=2, seed=31)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    _set_association_strength_slot(state, 2)

    state.perform_association_action(
        [AssociationTaskSelection(task=AssociationTask.REPUTATION)]
    )

    assert player.reputation == 2
    assert player.active_workers == 0
    assert player.workers_on_association_board == 1
    assert player.association_workers_by_task[AssociationTask.REPUTATION] == 1
    assert player.action_order[0] == MainAction.ASSOCIATION
    assert state.current_player == 1


def test_association_side_i_rejects_multiple_tasks():
    state = create_base_game(num_players=2, seed=32)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.active_workers = 3
    _set_association_strength_slot(state, 5)

    with pytest.raises(ValueError, match="side I requires exactly one"):
        state.perform_association_action(
            [
                AssociationTaskSelection(task=AssociationTask.REPUTATION),
                AssociationTaskSelection(
                    task=AssociationTask.PARTNER_ZOO,
                    partner_zoo="africa",
                ),
            ]
        )


def test_association_side_ii_multiple_tasks_total_value_cap():
    state = create_base_game(num_players=2, seed=33)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = True
    player.active_workers = 3
    _set_association_strength_slot(state, 5)

    state.perform_association_action(
        [
            AssociationTaskSelection(task=AssociationTask.REPUTATION),
            AssociationTaskSelection(
                task=AssociationTask.PARTNER_ZOO,
                partner_zoo="africa",
            ),
        ]
    )

    assert player.reputation == 2
    assert "africa" in player.partner_zoos
    assert player.zoo_icons["africa"] == 1
    assert player.active_workers == 1
    assert player.workers_on_association_board == 2

    state2 = create_base_game(num_players=2, seed=34)
    player2 = state2.current()
    player2.action_upgraded[MainAction.ASSOCIATION] = True
    player2.active_workers = 3
    _set_association_strength_slot(state2, 4)
    with pytest.raises(ValueError, match="exceeds action strength"):
        state2.perform_association_action(
            [
                AssociationTaskSelection(task=AssociationTask.REPUTATION),
                AssociationTaskSelection(
                    task=AssociationTask.PARTNER_ZOO,
                    partner_zoo="asia",
                ),
            ]
        )


def test_association_side_ii_rejects_duplicate_tasks():
    state = create_base_game(num_players=2, seed=35)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = True
    player.active_workers = 3
    _set_association_strength_slot(state, 5)

    with pytest.raises(ValueError, match="only different tasks"):
        state.perform_association_action(
            [
                AssociationTaskSelection(task=AssociationTask.REPUTATION),
                AssociationTaskSelection(task=AssociationTask.REPUTATION),
            ]
        )


def test_association_worker_escalation_on_same_task():
    state = create_base_game(num_players=2, seed=36)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    _set_association_strength_slot(state, 5)
    player.association_workers_by_task[AssociationTask.REPUTATION] = 1
    player.workers_on_association_board = 1

    player.active_workers = 1
    with pytest.raises(ValueError, match="Not enough active association workers"):
        state.perform_association_action(
            [AssociationTaskSelection(task=AssociationTask.REPUTATION)]
        )

    player.active_workers = 2
    state.perform_association_action([AssociationTaskSelection(task=AssociationTask.REPUTATION)])
    assert player.association_workers_by_task[AssociationTask.REPUTATION] == 3
    assert player.workers_on_association_board == 3
    assert player.active_workers == 0


def test_association_donation_rules_and_cost_progression():
    side_i_state = create_base_game(num_players=2, seed=37)
    side_i_player = side_i_state.current()
    side_i_player.action_upgraded[MainAction.ASSOCIATION] = False
    _set_association_strength_slot(side_i_state, 2)
    with pytest.raises(ValueError, match="Donation is only available"):
        side_i_state.perform_association_action(
            [AssociationTaskSelection(task=AssociationTask.REPUTATION)],
            make_donation=True,
        )

    state = create_base_game(num_players=2, seed=38)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = True
    _set_association_strength_slot(state, 2)
    state.perform_association_action(
        [AssociationTaskSelection(task=AssociationTask.REPUTATION)],
        make_donation=True,
    )
    assert player.money == 23
    assert player.conservation == 1
    assert player.reputation == 2
    assert player.player_tokens_in_supply == 17
    assert state.donation_progress == 1

    high_state = create_base_game(num_players=2, seed=39)
    high_player = high_state.current()
    high_player.action_upgraded[MainAction.ASSOCIATION] = True
    _set_association_strength_slot(high_state, 2)
    high_state.donation_progress = 4
    high_state.perform_association_action(
        [AssociationTaskSelection(task=AssociationTask.REPUTATION)],
        make_donation=True,
    )
    assert high_player.money == 13
    assert high_player.player_tokens_in_supply == 18
    assert high_state.donation_progress == 5


def test_association_university_and_conservation_project_support():
    state = create_base_game(num_players=2, seed=40)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.active_workers = 1
    _set_association_strength_slot(state, 4)

    picked_uni = sorted(state.available_universities)[0]
    state.perform_association_action(
        [
            AssociationTaskSelection(
                task=AssociationTask.UNIVERSITY,
                university=picked_uni,
                university_science_icons=2,
                university_reputation_gain=2,
            )
        ]
    )
    assert picked_uni in player.universities
    assert player.reputation == 2
    assert player.zoo_icons["science"] == 2

    state.current_player = 0
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.active_workers = 1
    _set_association_strength_slot(state, 5)
    player.reputation = 2
    player.money = 25
    state.base_conservation_projects = [
        _project(
            "p_reptiles",
            ConservationRequirementKind.ICON_COUNT,
            [ConservationProjectSlot("right", conservation_gain=2, required_icons=2)],
            requirement_icon="reptile",
            requirement_group="category",
        )
    ]
    state.conservation_project_slot_owners = {}
    state._initialize_conservation_project_slots()
    player.zoo_icons["reptile"] = 2

    state.perform_association_action(
        [
            AssociationTaskSelection(
                task=AssociationTask.CONSERVATION_PROJECT,
                project_id="p_reptiles",
                project_slot_id="right",
                map_support_space_index=5,
            )
        ]
    )

    assert player.money == 37
    assert player.conservation == 2
    assert player.reputation == 2
    assert player.claimed_map_support_spaces == {5}
    assert state.conservation_project_slot_owners["p_reptiles"]["right"] == 0


def test_conservation_project_rejects_repeat_support_and_occupied_slot():
    state = create_base_game(num_players=2, seed=41)
    player0 = state.players[0]
    player1 = state.players[1]
    player0.action_upgraded[MainAction.ASSOCIATION] = False
    player1.action_upgraded[MainAction.ASSOCIATION] = False
    player0.active_workers = 1
    player1.active_workers = 1
    _set_association_strength_slot(state, 5)
    player0.zoo_icons["bird"] = 2
    player1.zoo_icons["bird"] = 2
    state.base_conservation_projects = [
        _project(
            "p_birds",
            ConservationRequirementKind.ICON_COUNT,
            [
                ConservationProjectSlot("middle", conservation_gain=4, required_icons=4),
                ConservationProjectSlot("right", conservation_gain=2, required_icons=2),
            ],
            requirement_icon="bird",
            requirement_group="category",
        )
    ]
    state.conservation_project_slot_owners = {}
    state._initialize_conservation_project_slots()

    state.perform_association_action(
        [
            AssociationTaskSelection(
                task=AssociationTask.CONSERVATION_PROJECT,
                project_id="p_birds",
                project_slot_id="right",
                map_support_space_index=6,
            )
        ]
    )

    state.current_player = 0
    player0.active_workers = 1
    _set_association_strength_slot(state, 5)
    with pytest.raises(ValueError, match="already supported by this player"):
        state.perform_association_action(
            [
                AssociationTaskSelection(
                    task=AssociationTask.CONSERVATION_PROJECT,
                    project_id="p_birds",
                    project_slot_id="middle",
                    map_support_space_index=5,
                )
            ]
        )

    state.current_player = 1
    _set_association_strength_slot(state, 5)
    with pytest.raises(ValueError, match="already occupied"):
        state.perform_association_action(
            [
                AssociationTaskSelection(
                    task=AssociationTask.CONSERVATION_PROJECT,
                    project_id="p_birds",
                    project_slot_id="right",
                    map_support_space_index=5,
                )
            ]
        )


def test_new_conservation_project_from_display_shifts_row_and_discards_rightmost():
    state = create_base_game(num_players=2, seed=42)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = True
    player.active_workers = 1
    player.reputation = 2
    player.money = 25
    player.zoo_icons["predator"] = 2
    _set_association_strength_slot(state, 5)

    left_project = _project(
        "p_left",
        ConservationRequirementKind.ICON_COUNT,
        [ConservationProjectSlot("right", conservation_gain=2, required_icons=2)],
        requirement_icon="predator",
        requirement_group="category",
    )
    right_project = _project(
        "p_right",
        ConservationRequirementKind.ICON_COUNT,
        [ConservationProjectSlot("right", conservation_gain=2, required_icons=2)],
        requirement_icon="predator",
        requirement_group="category",
    )
    new_project = _project(
        "p_new",
        ConservationRequirementKind.ICON_COUNT,
        [ConservationProjectSlot("right", conservation_gain=2, required_icons=2)],
        requirement_icon="predator",
        requirement_group="category",
    )
    state.conservation_projects_in_play = [left_project, right_project]
    state.conservation_project_slot_owners = {
        "p_left": {"right": None},
        "p_right": {"right": 1},
    }
    state.display = ["c1", "c2", new_project, "c4", "c5", "c6"]
    state.deck = ["next_cp"]
    player1_supply_before = state.players[1].player_tokens_in_supply

    state.perform_association_action(
        [
            AssociationTaskSelection(
                task=AssociationTask.CONSERVATION_PROJECT,
                project_from_display_index=2,
                project_slot_id="right",
                map_support_space_index=6,
            )
        ]
    )

    assert [project.card_id for project in state.conservation_projects_in_play] == ["p_new", "p_left"]
    assert state.players[1].player_tokens_in_supply == player1_supply_before + 1
    assert any(getattr(card, "card_id", "") == "p_right" for card in state.discard)
    assert state.display == ["c1", "c2", "c4", "c5", "c6", "next_cp"]
    assert player.money == 22
    assert player.x_tokens == 3


def test_release_project_from_special_enclosure_uses_printed_standard_size():
    state = create_base_game(num_players=2, seed=43)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.active_workers = 1
    _set_association_strength_slot(state, 5)

    reptile_origin = HexTile(2, 1)
    reptile_house = Building(BuildingType.REPTILE_HOUSE, reptile_origin, Rotation.ROT_0)
    player.zoo_map.add_building(reptile_house)
    reptile_house.empty_spaces = 3

    tortoise = AnimalCard(
        card_id="african_spurred_tortoise",
        cost=0,
        min_enclosure_size=3,
        appeal_gain=6,
        category_icons={"reptile": 1},
        continent_icons={"africa": 1},
        special_enclosure_spaces={BuildingType.REPTILE_HOUSE: 2},
    )
    player.played_animals = [tortoise]
    player.animal_placements[tortoise.card_id] = AnimalPlacement(
        enclosure_origin=reptile_origin,
        enclosure_type=BuildingType.REPTILE_HOUSE,
        spaces_used=2,
    )
    player.zoo_icons["reptile"] = 1
    player.zoo_icons["africa"] = 1
    player.appeal = 10
    state.base_conservation_projects = [
        _project(
            "p_release_reptile",
            ConservationRequirementKind.RELEASE,
            [
                ConservationProjectSlot("left", conservation_gain=5, release_size_bucket="4+"),
                ConservationProjectSlot("middle", conservation_gain=4, release_size_bucket="3"),
                ConservationProjectSlot("right", conservation_gain=3, release_size_bucket="2-"),
            ],
            requirement_icon="reptile",
            requirement_group="category",
        )
    ]
    state.conservation_project_slot_owners = {}
    state._initialize_conservation_project_slots()

    state.perform_association_action(
        [
            AssociationTaskSelection(
                task=AssociationTask.CONSERVATION_PROJECT,
                project_id="p_release_reptile",
                project_slot_id="middle",
                released_animal_index=0,
                map_support_space_index=5,
            )
        ]
    )

    assert player.conservation == 4
    assert player.money == 37
    assert player.appeal == 4
    assert player.played_animals == []
    assert player.zoo_icons == {}
    assert reptile_house.empty_spaces == 5
    assert state.discard[-1].card_id == "african_spurred_tortoise"


def test_release_project_frees_smallest_occupied_standard_enclosure():
    state = create_base_game(num_players=2, seed=44)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.active_workers = 1
    _set_association_strength_slot(state, 5)

    size4_origin = HexTile(0, 1)
    size3_origin = HexTile(2, 0)
    size4 = Building(BuildingType.SIZE_4, size4_origin, Rotation.ROT_0)
    size3 = Building(BuildingType.SIZE_3, size3_origin, Rotation.ROT_0)
    player.zoo_map.add_building(size4)
    player.zoo_map.add_building(size3)
    size4.empty_spaces = 0
    size3.empty_spaces = 0

    bird = AnimalCard(
        card_id="release_bird",
        cost=0,
        min_enclosure_size=2,
        appeal_gain=3,
        category_icons={"bird": 1},
    )
    player.played_animals = [bird]
    player.animal_placements[bird.card_id] = AnimalPlacement(
        enclosure_origin=size4_origin,
        enclosure_type=BuildingType.SIZE_4,
        spaces_used=1,
    )
    player.zoo_icons["bird"] = 1
    player.appeal = 8
    state.base_conservation_projects = [
        _project(
            "p_release_bird",
            ConservationRequirementKind.RELEASE,
            [
                ConservationProjectSlot("left", conservation_gain=5, release_size_bucket="4+"),
                ConservationProjectSlot("middle", conservation_gain=4, release_size_bucket="3"),
                ConservationProjectSlot("right", conservation_gain=3, release_size_bucket="2-"),
            ],
            requirement_icon="bird",
            requirement_group="category",
        )
    ]
    state.conservation_project_slot_owners = {}
    state._initialize_conservation_project_slots()

    state.perform_association_action(
        [
            AssociationTaskSelection(
                task=AssociationTask.CONSERVATION_PROJECT,
                project_id="p_release_bird",
                project_slot_id="right",
                released_animal_index=0,
                map_support_space_index=6,
            )
        ]
    )

    assert size3.empty_spaces == 1
    assert size4.empty_spaces == 0
    assert player.appeal == 5
    assert player.played_animals == []


def test_release_project_rejects_wrong_size_bucket():
    state = create_base_game(num_players=2, seed=45)
    player = state.current()
    player.action_upgraded[MainAction.ASSOCIATION] = False
    player.active_workers = 1
    _set_association_strength_slot(state, 5)

    reptile_origin = HexTile(2, 1)
    reptile_house = Building(BuildingType.REPTILE_HOUSE, reptile_origin, Rotation.ROT_0)
    player.zoo_map.add_building(reptile_house)
    reptile_house.empty_spaces = 3

    reptile = AnimalCard(
        card_id="wrong_bucket",
        cost=0,
        min_enclosure_size=3,
        appeal_gain=2,
        category_icons={"reptile": 1},
        special_enclosure_spaces={BuildingType.REPTILE_HOUSE: 2},
    )
    player.played_animals = [reptile]
    player.animal_placements[reptile.card_id] = AnimalPlacement(
        enclosure_origin=reptile_origin,
        enclosure_type=BuildingType.REPTILE_HOUSE,
        spaces_used=2,
    )
    state.base_conservation_projects = [
        _project(
            "p_release_wrong_bucket",
            ConservationRequirementKind.RELEASE,
            [
                ConservationProjectSlot("left", conservation_gain=5, release_size_bucket="4+"),
                ConservationProjectSlot("middle", conservation_gain=4, release_size_bucket="3"),
            ],
            requirement_icon="reptile",
            requirement_group="category",
        )
    ]
    state.conservation_project_slot_owners = {}
    state._initialize_conservation_project_slots()

    with pytest.raises(ValueError, match="release size bucket"):
        state.perform_association_action(
            [
                AssociationTaskSelection(
                    task=AssociationTask.CONSERVATION_PROJECT,
                    project_id="p_release_wrong_bucket",
                    project_slot_id="left",
                    released_animal_index=0,
                    map_support_space_index=6,
                )
            ]
        )

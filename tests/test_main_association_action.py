import pytest

from main import (
    Action,
    ActionType,
    AnimalCard,
    BASE_CONSERVATION_PROJECT_LEVELS,
    _blocked_level_by_project,
    _project_requirement_value,
    _resolve_break,
    apply_action,
    setup_game,
)


def _seed_project_icons(player, project_id: str, count: int) -> None:
    number = int(project_id[1:4])
    cards = []
    if number == 101:  # species diversity: unique categories
        for idx in range(count):
            cards.append(
                AnimalCard(
                    name=f"cat-{idx}",
                    cost=0,
                    size=0,
                    appeal=0,
                    conservation=0,
                    badges=(f"Category{idx}",),
                )
            )
    elif number == 102:  # habitat diversity: unique continents
        continents = ("Africa", "Europe", "Asia", "Americas", "Australia")
        for idx in range(min(count, len(continents))):
            cards.append(
                AnimalCard(
                    name=f"continent-{idx}",
                    cost=0,
                    size=0,
                    appeal=0,
                    conservation=0,
                    badges=(continents[idx],),
                )
            )
    elif number == 103:
        badge = "Africa"
    elif number == 104:
        badge = "Americas"
    elif number == 105:
        badge = "Australia"
    elif number == 106:
        badge = "Asia"
    elif number == 107:
        badge = "Europe"
    elif number == 108:
        badge = "Primate"
    elif number == 109:
        badge = "Reptile"
    elif number == 110:
        badge = "Predator"
    elif number == 111:
        badge = "Herbivore"
    elif number == 112:
        badge = "Bird"
    else:
        badge = "Primate"

    if number != 101 and number != 102:
        for idx in range(count):
            cards.append(
                AnimalCard(
                    name=f"icon-{idx}",
                    cost=0,
                    size=0,
                    appeal=0,
                    conservation=0,
                    badges=(badge,),
                )
            )

    player.zoo_cards.extend(cards)


def test_setup_initializes_association_market():
    state = setup_game(seed=601, player_names=["P1", "P2"])

    assert state.available_partner_zoos == {"africa", "europe", "asia", "america", "australia"}
    assert state.available_universities == {
        "reputation_1_hand_limit_5",
        "science_1_reputation_2",
        "science_2",
    }
    assert len(state.conservation_project_slots) == 3
    assert all(
        level_owner is None
        for slots in state.conservation_project_slots.values()
        for level_owner in slots.values()
    )


def test_association_partner_zoo_is_shared_until_break_refresh(monkeypatch):
    state = setup_game(seed=602, player_names=["P1", "P2"])
    p0, p1 = state.players
    state.current_player = 0
    p0.action_order = ["cards", "animals", "association", "build", "sponsors"]  # association strength=3

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "partner_zoo", "partner_zoo": "asia"},
        ),
    )
    assert "asia" in p0.partner_zoos
    assert "asia" not in state.available_partner_zoos

    p1.action_order = ["cards", "animals", "association", "build", "sponsors"]  # strength=3
    with pytest.raises(ValueError, match="not legal|not currently available"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={"task_kind": "partner_zoo", "partner_zoo": "asia"},
            ),
        )

    monkeypatch.setattr("builtins.input", lambda _: "1")
    _resolve_break(state)
    assert "asia" in state.available_partner_zoos


def test_association_university_reputation_type_applies_bonus():
    state = setup_game(seed=603, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "animals", "build", "association", "sponsors"]  # association strength=4
    rep_before = p0.reputation
    hand_limit_before = p0.hand_limit

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "university", "university": "reputation_1_hand_limit_5"},
        ),
    )

    assert "reputation_1_hand_limit_5" in p0.universities
    assert "reputation_1_hand_limit_5" not in state.available_universities
    assert p0.reputation == rep_before + 1
    assert p0.hand_limit == max(hand_limit_before, 5)


def test_association_strength_5_can_support_base_conservation_project():
    state = setup_game(seed=604, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "animals", "build", "sponsors", "association"]  # association strength=5
    project_id = state.opening_setup.base_conservation_projects[0].data_id
    _seed_project_icons(p0, project_id, 5)

    blocked_level = _blocked_level_by_project(state).get(project_id, "")
    level_candidates = [item for item in BASE_CONSERVATION_PROJECT_LEVELS if item[0] != blocked_level]
    expected_gain = max(item[2] for item in level_candidates)

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "conservation_project", "project_id": project_id},
        ),
    )

    assert project_id in p0.supported_conservation_projects
    assert p0.conservation == expected_gain


def test_species_diversity_does_not_count_science_rock_or_water():
    state = setup_game(seed=6041, player_names=["P1", "P2"])
    player = state.players[0]
    player.zoo_cards = [
        AnimalCard(
            name="science-sponsor",
            cost=0,
            size=0,
            appeal=0,
            conservation=0,
            card_type="sponsor",
            badges=("Science",),
            number=201,
        ),
        AnimalCard(
            name="rock-animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            badges=(),
            required_rock_adjacency=1,
        ),
        AnimalCard(
            name="water-animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            badges=(),
            required_water_adjacency=1,
        ),
        AnimalCard(
            name="bird-animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            badges=("Bird",),
        ),
        AnimalCard(
            name="predator-animal",
            cost=0,
            size=1,
            appeal=0,
            conservation=0,
            badges=("Predator",),
        ),
    ]

    assert _project_requirement_value(player, "P101_SpeciesDiversity") == 2


def test_association_conservation_project_slot_is_blocked_if_occupied():
    state = setup_game(seed=607, player_names=["P1", "P2"])
    p0, p1 = state.players
    project_id = state.opening_setup.base_conservation_projects[0].data_id
    _seed_project_icons(p0, project_id, 5)
    _seed_project_icons(p1, project_id, 5)
    blocked_level = _blocked_level_by_project(state).get(project_id, "")
    target_level = next(
        level_name
        for level_name, _, _ in BASE_CONSERVATION_PROJECT_LEVELS
        if level_name != blocked_level
    )

    state.current_player = 0
    p0.action_order = ["cards", "animals", "build", "sponsors", "association"]  # association strength=5
    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={
                "task_kind": "conservation_project",
                "project_id": project_id,
                "project_level": target_level,
            },
        ),
    )
    assert state.conservation_project_slots[project_id][target_level] == 0

    state.current_player = 1
    p1.action_order = ["cards", "animals", "build", "sponsors", "association"]  # strength=5
    with pytest.raises(ValueError, match="not legal|already occupied"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={
                    "task_kind": "conservation_project",
                    "project_id": project_id,
                    "project_level": target_level,
                },
            ),
        )


def test_association_conservation_project_requires_icon_threshold():
    state = setup_game(seed=618, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "animals", "build", "sponsors", "association"]  # association strength=5
    project_id = state.opening_setup.base_conservation_projects[0].data_id

    with pytest.raises(ValueError, match="Requested association task is not legal"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={"task_kind": "conservation_project", "project_id": project_id},
            ),
        )


def test_association_conservation_project_respects_two_player_blocked_level():
    state = setup_game(seed=619, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "animals", "build", "sponsors", "association"]  # association strength=5
    project_id = state.opening_setup.base_conservation_projects[0].data_id
    _seed_project_icons(p0, project_id, 5)
    blocked_level = _blocked_level_by_project(state).get(project_id, "")

    with pytest.raises(ValueError, match="Requested association task is not legal"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={
                    "task_kind": "conservation_project",
                    "project_id": project_id,
                    "project_level": blocked_level,
                },
            ),
        )


def test_association_same_task_second_time_requires_two_workers(monkeypatch):
    state = setup_game(seed=608, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # association strength=2
    p0.workers = 3

    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "reputation"},
        ),
    )
    assert p0.workers == 2
    assert p0.workers_on_association_board == 1
    assert p0.association_workers_by_task["reputation"] == 1

    state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # strength=2
    monkeypatch.setattr("builtins.input", lambda _: "1")
    apply_action(
        state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "reputation"},
        ),
    )
    assert p0.workers == 0
    assert p0.workers_on_association_board == 3
    assert p0.association_workers_by_task["reputation"] == 3

    state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # strength=2
    with pytest.raises(ValueError, match="not legal|cannot be used again"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={"task_kind": "reputation"},
            ),
        )

    monkeypatch.setattr("builtins.input", lambda _: "1")
    _resolve_break(state)
    assert p0.workers == 3
    assert p0.workers_on_association_board == 0
    assert p0.association_workers_by_task["reputation"] == 0


def test_association_donation_requires_upgrade_and_uses_progressive_cost(monkeypatch):
    state = setup_game(seed=605, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # association strength=2

    with pytest.raises(ValueError, match="Donation is only available"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={"task_kind": "reputation", "make_donation": True},
            ),
        )

    upgraded_state = setup_game(seed=606, player_names=["P1", "P2"])
    p0 = upgraded_state.players[0]
    upgraded_state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # strength=2
    p0.action_upgraded["association"] = True
    p0.workers = 3

    apply_action(
        upgraded_state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "reputation", "make_donation": True},
        ),
    )
    assert p0.money == 23
    assert p0.conservation == 1
    assert upgraded_state.donation_progress == 1

    upgraded_state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # strength=2
    monkeypatch.setattr("builtins.input", lambda _: "1")
    apply_action(
        upgraded_state,
        Action(
            ActionType.MAIN_ACTION,
            card_name="association",
            details={"task_kind": "reputation", "make_donation": True},
        ),
    )
    assert p0.money == 18
    assert p0.conservation == 2
    assert upgraded_state.donation_progress == 2


def test_association_donation_cannot_be_done_alone_without_task():
    state = setup_game(seed=607, player_names=["P1", "P2"])
    p0 = state.players[0]
    state.current_player = 0
    p0.action_order = ["cards", "association", "animals", "build", "sponsors"]  # association strength=2
    p0.action_upgraded["association"] = True
    p0.workers = 0

    with pytest.raises(ValueError, match="combined with another association task"):
        apply_action(
            state,
            Action(
                ActionType.MAIN_ACTION,
                card_name="association",
                details={"make_donation": True},
            ),
        )

"""Ark Nova base-game engine (base actions focus).

This module implements a strict rules flow for:
- action card strength and rotation
- Animals action (core)
- Association action (core)
- Cards action
- Build action (side I and side II behavior)
- Break track trigger and Break resolution sequence

It intentionally limits scope to base-action core timing/state transitions before
adding full endgame/scoring rules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

from arknova_engine.map_model import (
    ArkNovaMap,
    BonusResource,
    Building,
    BuildingSubType,
    BuildingType,
    HexTile,
    Rotation,
    Terrain,
    load_map_data_by_image_name,
)


MAX_X_TOKENS = 5
BREAK_MAX_BY_PLAYERS = {2: 9, 3: 12, 4: 15}
PROJECT_ROW_LIMIT_BY_PLAYERS = {2: 2, 3: 3, 4: 4}
ALL_PARTNER_ZOOS = ("africa", "europe", "asia", "america", "australia")
ALL_UNIVERSITIES = tuple(f"university_{i}" for i in range(1, 13))
ALL_CONTINENT_ICONS = ALL_PARTNER_ZOOS
ALL_CATEGORY_ICONS = ("predator", "bird", "herbivore", "reptile", "primate")
BGA_END_GAME_TRIGGER_SCORE = 100

CARDS_DRAW_TABLE_BASE = (1, 1, 2, 2, 3)
CARDS_DISCARD_TABLE_BASE = (1, 0, 1, 0, 1)
CARDS_SNAP_ALLOWED_BASE = (False, False, False, False, True)

CARDS_DRAW_TABLE_UPGRADED = (1, 2, 2, 3, 4)
CARDS_DISCARD_TABLE_UPGRADED = (0, 1, 0, 1, 1)
CARDS_SNAP_ALLOWED_UPGRADED = (False, False, True, True, True)

ANIMALS_PLAY_LIMIT_BASE = (0, 1, 1, 1, 2)
ANIMALS_PLAY_LIMIT_UPGRADED = (1, 1, 2, 2, 2)

ASSOCIATION_REPUTATION_GAIN = 2
ASSOCIATION_TASK_VALUE = {
    "reputation": 2,
    "partner_zoo": 3,
    "university": 4,
    "conservation_project": 5,
}
DONATION_COST_TRACK = (2, 5, 7, 10, 12)

DEFAULT_NUM_PLAYERS = 2
DEFAULT_MAP_IMAGE_NAME = "plan1a"
STARTING_MONEY = 25
STARTING_HAND_DRAW_COUNT = 8
STARTING_HAND_KEEP_COUNT = 4
ZOO_DECK_SIZE = 212
FINAL_SCORING_DECK_SIZE = 11
PLAYER_TOKENS_TOTAL = 25
PLAYER_TOKENS_PLACED_ON_MAP = 7


class MainAction(str, Enum):
    CARDS = "cards"
    BUILD = "build"
    ANIMALS = "animals"
    ASSOCIATION = "association"
    SPONSORS = "sponsors"


class CardSource(str, Enum):
    HAND = "hand"
    DISPLAY = "display"


class AssociationTask(str, Enum):
    REPUTATION = "reputation"
    PARTNER_ZOO = "partner_zoo"
    UNIVERSITY = "university"
    CONSERVATION_PROJECT = "conservation_project"


class ConservationRequirementKind(str, Enum):
    ICON_COUNT = "icon_count"
    DISTINCT_CONTINENT = "distinct_continent"
    DISTINCT_CATEGORY = "distinct_category"
    RELEASE = "release"


def _empty_action_token_map() -> Dict[MainAction, int]:
    return {action: 0 for action in MainAction}


def _empty_association_worker_map() -> Dict[AssociationTask, int]:
    return {task: 0 for task in AssociationTask}


@dataclass(frozen=True)
class BuildSelection:
    building_type: BuildingType
    origin_hex: HexTile
    rotation: Rotation = Rotation.ROT_0


@dataclass(frozen=True)
class AnimalPlaySelection:
    source: CardSource
    source_index: int
    enclosure_origin: HexTile


@dataclass(frozen=True)
class AnimalCard:
    card_id: str
    cost: int
    min_enclosure_size: int
    appeal_gain: int
    required_water_adjacency: int = 0
    required_rock_adjacency: int = 0
    continent_icons: Dict[str, int] = field(default_factory=dict)
    category_icons: Dict[str, int] = field(default_factory=dict)
    required_partner_zoos: Set[str] = field(default_factory=set)
    required_icons: Dict[str, int] = field(default_factory=dict)
    min_reputation: int = 0
    requires_upgraded_animals_action: bool = False
    is_petting_zoo_animal: bool = False
    special_enclosure_spaces: Dict[BuildingType, int] = field(default_factory=dict)
    reputation_gain: int = 0
    conservation_gain: int = 0
    after_finishing_effect_label: Optional[str] = None


@dataclass(frozen=True)
class ConservationProjectSlot:
    slot_id: str
    conservation_gain: int
    reputation_gain: int = 0
    required_icons: int = 0
    release_size_bucket: Optional[str] = None


@dataclass(frozen=True)
class ConservationProjectCard:
    card_id: str
    title: str
    requirement_kind: ConservationRequirementKind
    slots: Tuple[ConservationProjectSlot, ...]
    requirement_icon: Optional[str] = None
    requirement_group: Optional[str] = None
    play_bonus_reputation_gain: int = 0
    play_bonus_conservation_gain: int = 0
    play_bonus_effect_label: Optional[str] = None


@dataclass(frozen=True)
class AnimalPlacement:
    enclosure_origin: HexTile
    enclosure_type: BuildingType
    spaces_used: int


@dataclass(frozen=True)
class MapSupportSpace:
    space_index: int
    effect_code: str
    recurring: bool = False


@dataclass(frozen=True)
class ReleaseIntoWildResolution:
    animal: AnimalCard
    placement: AnimalPlacement
    released_from_standard_origin: Optional[HexTile] = None


@dataclass(frozen=True)
class AssociationTaskSelection:
    task: AssociationTask
    partner_zoo: Optional[str] = None
    university: Optional[str] = None
    university_science_icons: int = 0
    university_reputation_gain: int = 0
    project_id: Optional[str] = None
    project_slot_id: Optional[str] = None
    project_from_hand_index: Optional[int] = None
    project_from_display_index: Optional[int] = None
    released_animal_index: Optional[int] = None
    map_support_space_index: Optional[int] = None


@dataclass(frozen=True)
class SponsorPlaySelection:
    source: CardSource
    source_index: int


@dataclass(frozen=True)
class SponsorCard:
    card_id: str
    level: int
    min_reputation: int = 0
    required_partner_zoos: Set[str] = field(default_factory=set)
    required_icons: Dict[str, int] = field(default_factory=dict)
    requires_upgraded_sponsors_action: bool = False
    granted_icons: Dict[str, int] = field(default_factory=dict)
    appeal_gain: int = 0
    reputation_gain: int = 0
    conservation_gain: int = 0
    recurring_break_income_gain: int = 0
    instant_effect_label: Optional[str] = None
    after_finishing_effect_label: Optional[str] = None
    end_game_effect_label: Optional[str] = None


def _clone_project_slots(
    slot_defs: Sequence[Tuple[str, int, int, int, Optional[str]]]
) -> Tuple[ConservationProjectSlot, ...]:
    return tuple(
        ConservationProjectSlot(
            slot_id=slot_id,
            conservation_gain=conservation_gain,
            reputation_gain=reputation_gain,
            required_icons=required_icons,
            release_size_bucket=release_size_bucket,
        )
        for slot_id, conservation_gain, reputation_gain, required_icons, release_size_bucket in slot_defs
    )


BASE_PROJECT_SLOT_DEFS: Tuple[Tuple[str, int, int, int, Optional[str]], ...] = (
    ("left", 5, 0, 5, None),
    ("middle", 4, 0, 4, None),
    ("right", 2, 0, 2, None),
)
RELEASE_PROJECT_SLOT_DEFS: Tuple[Tuple[str, int, int, int, Optional[str]], ...] = (
    ("left", 5, 0, 0, "4+"),
    ("middle", 4, 0, 0, "3"),
    ("right", 3, 0, 0, "2-"),
)


@lru_cache(maxsize=1)
def _builtin_conservation_project_catalog() -> Dict[str, ConservationProjectCard]:
    catalog: Dict[str, ConservationProjectCard] = {}

    def add_icon_project(
        card_id: str,
        title: str,
        icon: str,
        *,
        requirement_kind: ConservationRequirementKind,
        requirement_group: Optional[str],
    ) -> None:
        catalog[card_id] = ConservationProjectCard(
            card_id=card_id,
            title=title,
            requirement_kind=requirement_kind,
            requirement_icon=icon,
            requirement_group=requirement_group,
            slots=_clone_project_slots(BASE_PROJECT_SLOT_DEFS),
        )

    def add_release_project(card_id: str, title: str, icon: str, group: str) -> None:
        catalog[card_id] = ConservationProjectCard(
            card_id=card_id,
            title=title,
            requirement_kind=ConservationRequirementKind.RELEASE,
            requirement_icon=icon,
            requirement_group=group,
            slots=_clone_project_slots(RELEASE_PROJECT_SLOT_DEFS),
        )

    catalog["P101_SpeciesDiversity"] = ConservationProjectCard(
        card_id="P101_SpeciesDiversity",
        title="SPECIES DIVERSITY",
        requirement_kind=ConservationRequirementKind.DISTINCT_CATEGORY,
        slots=_clone_project_slots(BASE_PROJECT_SLOT_DEFS),
    )
    catalog["P102_HabitatDiversity"] = ConservationProjectCard(
        card_id="P102_HabitatDiversity",
        title="HABITAT DIVERSITY",
        requirement_kind=ConservationRequirementKind.DISTINCT_CONTINENT,
        slots=_clone_project_slots(BASE_PROJECT_SLOT_DEFS),
    )

    add_icon_project(
        "P103_Africa",
        "AFRICA",
        "africa",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="continent",
    )
    add_icon_project(
        "P104_Americas",
        "AMERICAS",
        "america",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="continent",
    )
    add_icon_project(
        "P105_Australia",
        "AUSTRALIA",
        "australia",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="continent",
    )
    add_icon_project(
        "P106_Asia",
        "ASIA",
        "asia",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="continent",
    )
    add_icon_project(
        "P107_Europe",
        "EUROPE",
        "europe",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="continent",
    )
    add_icon_project(
        "P108_Primates",
        "PRIMATES",
        "primate",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="category",
    )
    add_icon_project(
        "P109_Reptiles",
        "REPTILES",
        "reptile",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="category",
    )
    add_icon_project(
        "P110_Predators",
        "PREDATORS",
        "predator",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="category",
    )
    add_icon_project(
        "P111_Herbivores",
        "HERBIVORES",
        "herbivore",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="category",
    )
    add_icon_project(
        "P112_Birds",
        "BIRDS",
        "bird",
        requirement_kind=ConservationRequirementKind.ICON_COUNT,
        requirement_group="category",
    )

    add_release_project(
        "P113_ReleaseBavarianForest",
        "BAVARIAN FOREST NATIONAL PARK",
        "europe",
        "continent",
    )
    add_release_project(
        "P114_ReleaseYosemite",
        "YOSEMITE NATIONAL PARK",
        "america",
        "continent",
    )
    add_release_project(
        "P115_ReleaseAngthong",
        "ANGTHONG NATIONAL PARK",
        "asia",
        "continent",
    )
    add_release_project(
        "P116_ReleaseSerengeti",
        "SERENGETI NATIONAL PARK",
        "africa",
        "continent",
    )
    add_release_project(
        "P117_ReleaseBlueMountains",
        "BLUE MOUNTAINS NATIONAL PARK",
        "australia",
        "continent",
    )
    add_release_project("P118_ReleaseSavanna", "SAVANNA", "predator", "category")
    add_release_project(
        "P119_ReleaseLowMountainRange",
        "LOW MOUNTAIN RANGE",
        "bird",
        "category",
    )
    add_release_project(
        "P120_ReleaseBambooForest",
        "BAMBOO FOREST",
        "herbivore",
        "category",
    )
    add_release_project("P121_ReleaseSeaCave", "SEA CAVE", "reptile", "category")
    add_release_project("P122_ReleaseJungle", "JUNGLE", "primate", "category")
    return catalog


@lru_cache(maxsize=32)
def _load_map_support_spaces_by_image_name(image_name: str) -> Tuple[MapSupportSpace, ...]:
    root = Path(__file__).resolve().parents[1]
    tiles_path = root / "data" / "maps" / "tiles" / f"{image_name}.tiles.json"
    if not tiles_path.exists():
        return ()
    payload = json.loads(tiles_path.read_text(encoding="utf-8"))
    map_rules = payload.get("map_rules", {})
    if not isinstance(map_rules, dict):
        return ()
    raw_spaces = map_rules.get("left_track_unlocks", [])
    if not isinstance(raw_spaces, list):
        return ()

    spaces: List[MapSupportSpace] = []
    for idx, item in enumerate(raw_spaces):
        if not isinstance(item, dict):
            continue
        effect_code = str(item.get("effect") or "").strip().lower()
        if not effect_code:
            continue
        recurring = str(item.get("category") or "").strip().lower() == "purple_recurring_action"
        spaces.append(MapSupportSpace(space_index=idx, effect_code=effect_code, recurring=recurring))
    return tuple(spaces)


@dataclass(frozen=True)
class BGAFinalScoreBreakdown:
    player_id: int
    appeal_points: int
    conservation_points: int
    end_game_bonus_points: int
    total_points: int


@dataclass
class PlayerState:
    zoo_map: ArkNovaMap = field(default_factory=ArkNovaMap)
    money: int = STARTING_MONEY
    appeal: int = 0
    conservation: int = 0
    reputation: int = 0
    x_tokens: int = 0
    hand: List[object] = field(default_factory=list)
    hand_limit: int = 3
    action_order: List[MainAction] = field(
        default_factory=lambda: [
            MainAction.ANIMALS,
            MainAction.CARDS,
            MainAction.BUILD,
            MainAction.ASSOCIATION,
            MainAction.SPONSORS,
        ]
    )
    action_upgraded: Dict[MainAction, bool] = field(
        default_factory=lambda: {action: False for action in MainAction}
    )
    multiplier_tokens_on_actions: Dict[MainAction, int] = field(default_factory=_empty_action_token_map)
    venom_tokens_on_actions: Dict[MainAction, int] = field(default_factory=_empty_action_token_map)
    constriction_tokens_on_actions: Dict[MainAction, int] = field(default_factory=_empty_action_token_map)
    active_workers: int = 1
    workers_on_association_board: int = 0
    association_workers_by_task: Dict[AssociationTask, int] = field(default_factory=_empty_association_worker_map)
    recurring_break_income: int = 0
    partner_zoos: Set[str] = field(default_factory=set)
    universities: Set[str] = field(default_factory=set)
    zoo_icons: Dict[str, int] = field(default_factory=dict)
    played_animals: List[AnimalCard] = field(default_factory=list)
    animal_placements: Dict[str, AnimalPlacement] = field(default_factory=dict)
    played_sponsors: List[SponsorCard] = field(default_factory=list)
    supported_conservation_projects: Set[str] = field(default_factory=set)
    claimed_map_support_spaces: Set[int] = field(default_factory=set)
    recurring_map_effects: List[str] = field(default_factory=list)
    pending_map_effects: List[str] = field(default_factory=list)
    final_scoring_cards: List[str] = field(default_factory=list)
    discarded_final_scoring_cards: List[str] = field(default_factory=list)
    player_tokens_in_supply: int = PLAYER_TOKENS_TOTAL - PLAYER_TOKENS_PLACED_ON_MAP
    map_cover_bonus_claimed: bool = False


@dataclass
class GameState:
    players: List[PlayerState]
    map_image_name: str = DEFAULT_MAP_IMAGE_NAME
    current_player: int = 0
    turn_index: int = 0
    break_progress: int = 0
    break_pending: bool = False
    break_trigger_player: Optional[int] = None
    display: List[object] = field(default_factory=list)
    deck: List[object] = field(default_factory=list)
    discard: List[object] = field(default_factory=list)
    base_conservation_projects: List[ConservationProjectCard] = field(default_factory=list)
    conservation_projects_in_play: List[ConservationProjectCard] = field(default_factory=list)
    conservation_project_slot_owners: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    map_support_spaces: Tuple[MapSupportSpace, ...] = field(default_factory=tuple)
    available_partner_zoos: Set[str] = field(default_factory=set)
    available_universities: Set[str] = field(default_factory=set)
    action_log: List[str] = field(default_factory=list)
    donation_progress: int = 0
    final_scoring_deck: List[str] = field(default_factory=list)
    final_scoring_discard: List[str] = field(default_factory=list)
    end_game_triggered: bool = False
    end_game_trigger_player: Optional[int] = None
    end_game_pending_players: Set[int] = field(default_factory=set)
    game_over: bool = False
    # Optional exact table by appeal index -> income. If missing, fallback heuristic is used.
    appeal_income_table: Optional[List[int]] = None
    end_game_trigger_score: int = BGA_END_GAME_TRIGGER_SCORE

    def __post_init__(self) -> None:
        if len(self.players) not in BREAK_MAX_BY_PLAYERS:
            raise ValueError("Only 2-4 players are supported in base setup.")
        if not self.map_support_spaces:
            self.map_support_spaces = _load_map_support_spaces_by_image_name(self.map_image_name)
        self._initialize_conservation_project_slots()
        self._refill_association_offers()

    @property
    def break_max(self) -> int:
        return BREAK_MAX_BY_PLAYERS[len(self.players)]

    @property
    def conservation_project_row_limit(self) -> int:
        return PROJECT_ROW_LIMIT_BY_PLAYERS[len(self.players)]

    def player(self, player_id: int) -> PlayerState:
        return self.players[player_id]

    def current(self) -> PlayerState:
        return self.player(self.current_player)

    def _initialize_conservation_project_slots(self) -> None:
        for project in self.base_conservation_projects:
            self._ensure_conservation_project_slots(project)
        for project in self.conservation_projects_in_play:
            self._ensure_conservation_project_slots(project)

    def _ensure_conservation_project_slots(self, project: ConservationProjectCard) -> None:
        if project.card_id in self.conservation_project_slot_owners:
            return
        self.conservation_project_slot_owners[project.card_id] = {
            slot.slot_id: None for slot in project.slots
        }

    def _find_conservation_project(self, project_id: str) -> Optional[ConservationProjectCard]:
        for project in self.base_conservation_projects:
            if project.card_id == project_id:
                return project
        for project in self.conservation_projects_in_play:
            if project.card_id == project_id:
                return project
        return None

    def _conservation_project_slot(
        self,
        project: ConservationProjectCard,
        slot_id: str,
    ) -> ConservationProjectSlot:
        for slot in project.slots:
            if slot.slot_id == slot_id:
                return slot
        raise ValueError(f"Conservation project slot '{slot_id}' does not exist.")

    def _slot_strength(self, player: PlayerState, action: MainAction) -> int:
        try:
            return player.action_order.index(action) + 1
        except ValueError as exc:
            raise ValueError(f"Action {action} not found in action order.") from exc

    def _consume_x_tokens_for_action(self, player: PlayerState, x_spent: int) -> None:
        if x_spent < 0:
            raise ValueError("x_spent cannot be negative.")
        if x_spent > player.x_tokens:
            raise ValueError("Not enough X-tokens.")
        player.x_tokens -= x_spent

    def _rotate_action_to_slot_1(self, player: PlayerState, action: MainAction) -> None:
        idx = player.action_order.index(action)
        used = player.action_order.pop(idx)
        player.action_order.insert(0, used)

    def _complete_main_action(self, player_id: int, action: MainAction) -> None:
        player = self.player(player_id)
        reached_trigger_before_break = self._player_reaches_bga_end_trigger(player_id)
        self._rotate_action_to_slot_1(player, action)
        break_resolved = False
        if self.break_pending and self.break_trigger_player == player_id:
            self._resolve_break()
            break_resolved = True
        if not self.end_game_triggered:
            if reached_trigger_before_break:
                self._trigger_end_game(trigger_player=player_id, triggered_during_break=False)
            elif break_resolved and self._player_reaches_bga_end_trigger(player_id):
                self._trigger_end_game(trigger_player=player_id, triggered_during_break=True)
        if self.end_game_triggered and player_id in self.end_game_pending_players:
            self.end_game_pending_players.remove(player_id)
            if not self.end_game_pending_players:
                self.game_over = True
        self.current_player = (player_id + 1) % len(self.players)
        self.turn_index += 1

    def _assert_game_not_over(self) -> None:
        if self.game_over:
            raise ValueError("Game is already over.")

    def _player_reaches_bga_end_trigger(self, player_id: int) -> bool:
        player = self.player(player_id)
        progress_points = player.appeal + self.bga_conservation_points(player.conservation)
        return progress_points >= self.end_game_trigger_score

    def _trigger_end_game(self, trigger_player: int, triggered_during_break: bool) -> None:
        self.end_game_triggered = True
        self.end_game_trigger_player = trigger_player
        self.end_game_pending_players = set(range(len(self.players)))
        if not triggered_during_break:
            self.end_game_pending_players.discard(trigger_player)
        self.action_log.append(
            "END_GAME_TRIGGER player={} during_break={}".format(
                trigger_player,
                triggered_during_break,
            )
        )

    def _find_matching_building(
        self,
        legal: Sequence[Building],
        selection: BuildSelection,
    ) -> Optional[Building]:
        for building in legal:
            if (
                building.type == selection.building_type
                and building.origin_hex == selection.origin_hex
                and building.rotation == selection.rotation
            ):
                return building
        return None

    def _apply_building_placement_effects(self, player: PlayerState, building: Building) -> None:
        if building.type == BuildingType.PAVILION:
            player.appeal += 1
        for tile in building.layout:
            bonus = player.zoo_map.map_data.placement_bonuses.get(tile)
            if bonus is None:
                continue
            if bonus.resource == BonusResource.REPUTATION:
                player.reputation += bonus.amount

    def _map_cover_bonus_reached(self, player: PlayerState) -> bool:
        if player.map_cover_bonus_claimed:
            return False
        covered = player.zoo_map.covered_hexes()
        fillable_tiles = {
            tile
            for tile in player.zoo_map.grid
            if player.zoo_map.map_data.terrain.get(tile) not in {Terrain.WATER, Terrain.ROCK}
        }
        return fillable_tiles.issubset(covered)

    def perform_build_action(
        self,
        selections: List[BuildSelection],
        x_spent: int = 0,
        has_diversity_researcher: bool = False,
    ) -> None:
        self._assert_game_not_over()
        from arknova_engine.actions.build_action import run_build_action

        run_build_action(
            self,
            selections=selections,
            x_spent=x_spent,
            has_diversity_researcher=has_diversity_researcher,
        )

    def _cards_table_values(self, strength: int, upgraded: bool) -> Tuple[int, int, bool]:
        idx = min(5, strength) - 1
        if upgraded:
            return (
                CARDS_DRAW_TABLE_UPGRADED[idx],
                CARDS_DISCARD_TABLE_UPGRADED[idx],
                CARDS_SNAP_ALLOWED_UPGRADED[idx],
            )
        return (
            CARDS_DRAW_TABLE_BASE[idx],
            CARDS_DISCARD_TABLE_BASE[idx],
            CARDS_SNAP_ALLOWED_BASE[idx],
        )

    def _validate_display_index(self, idx: int) -> None:
        if idx < 0 or idx >= len(self.display):
            raise ValueError(f"Display index {idx} out of range.")

    def _animals_play_limit(self, strength: int, upgraded: bool) -> int:
        idx = min(5, strength) - 1
        if upgraded:
            return ANIMALS_PLAY_LIMIT_UPGRADED[idx]
        return ANIMALS_PLAY_LIMIT_BASE[idx]

    def _player_icon_count(self, player: PlayerState, icon: str) -> int:
        count = player.zoo_icons.get(icon, 0)
        if icon in player.partner_zoos:
            count += 1
        if icon == "science":
            count += len(player.universities)
        return count

    def _distinct_project_icon_count(self, player: PlayerState, icons: Sequence[str]) -> int:
        return sum(1 for icon in icons if self._player_icon_count(player, icon) > 0)

    @staticmethod
    def _release_size_bucket(min_enclosure_size: int) -> str:
        if min_enclosure_size >= 4:
            return "4+"
        if min_enclosure_size == 3:
            return "3"
        return "2-"

    def _remove_icons_for_animal(self, player: PlayerState, animal: AnimalCard) -> None:
        for icon_group in (animal.continent_icons, animal.category_icons):
            for icon, amount in icon_group.items():
                remaining = player.zoo_icons.get(icon, 0) - amount
                if remaining > 0:
                    player.zoo_icons[icon] = remaining
                else:
                    player.zoo_icons.pop(icon, None)
        if animal.required_water_adjacency:
            remaining = player.zoo_icons.get("water", 0) - animal.required_water_adjacency
            if remaining > 0:
                player.zoo_icons["water"] = remaining
            else:
                player.zoo_icons.pop("water", None)
        if animal.required_rock_adjacency:
            remaining = player.zoo_icons.get("rock", 0) - animal.required_rock_adjacency
            if remaining > 0:
                player.zoo_icons["rock"] = remaining
            else:
                player.zoo_icons.pop("rock", None)

    def _validate_animal_conditions(
        self,
        player: PlayerState,
        animal: AnimalCard,
        is_animals_upgraded: bool,
    ) -> None:
        if animal.requires_upgraded_animals_action and not is_animals_upgraded:
            raise ValueError("This animal requires upgraded Animals action.")
        if player.reputation < animal.min_reputation:
            raise ValueError("Reputation is too low for this animal.")
        if not animal.required_partner_zoos.issubset(player.partner_zoos):
            raise ValueError("Required partner zoo is missing.")
        for icon, need in animal.required_icons.items():
            if self._player_icon_count(player, icon) < need:
                raise ValueError(f"Not enough icon '{icon}' for this animal.")

    def _adjacent_terrain_count(self, player: PlayerState, building: Building, terrain: Terrain) -> int:
        adjacent: Set[HexTile] = set()
        for tile in building.layout:
            for neighbor in tile.neighbors():
                if neighbor in building.layout:
                    continue
                if player.zoo_map.map_data.terrain.get(neighbor) == terrain:
                    adjacent.add(neighbor)
        return len(adjacent)

    def _validate_animal_enclosure(
        self,
        player: PlayerState,
        animal: AnimalCard,
        enclosure: Building,
    ) -> int:
        if enclosure.type.subtype == BuildingSubType.ENCLOSURE_BASIC:
            if animal.is_petting_zoo_animal:
                raise ValueError("Petting Zoo animals must be placed in a Petting Zoo.")
            if enclosure.empty_spaces <= 0:
                raise ValueError("Target standard enclosure is already occupied.")
            if len(enclosure.layout) < animal.min_enclosure_size:
                raise ValueError("Target enclosure is too small for this animal.")
            spaces_used = 1
        elif enclosure.type.subtype == BuildingSubType.ENCLOSURE_SPECIAL:
            if animal.is_petting_zoo_animal:
                if enclosure.type != BuildingType.PETTING_ZOO:
                    raise ValueError("Petting Zoo animal must be placed in Petting Zoo.")
                spaces_used = 1
            else:
                if enclosure.type not in animal.special_enclosure_spaces:
                    raise ValueError("Animal cannot be placed in this special enclosure.")
                spaces_used = animal.special_enclosure_spaces[enclosure.type]
            if enclosure.empty_spaces < spaces_used:
                raise ValueError("Special enclosure does not have enough empty spaces.")
        else:
            raise ValueError("Animals can only be placed in standard or special enclosures.")

        if self._adjacent_terrain_count(player, enclosure, Terrain.WATER) < animal.required_water_adjacency:
            raise ValueError("Enclosure does not meet water adjacency requirement.")
        if self._adjacent_terrain_count(player, enclosure, Terrain.ROCK) < animal.required_rock_adjacency:
            raise ValueError("Enclosure does not meet rock adjacency requirement.")
        return spaces_used

    def _partner_zoo_discount(self, player: PlayerState, animal: AnimalCard) -> int:
        discount = 0
        for continent, count in animal.continent_icons.items():
            if continent in player.partner_zoos:
                discount += 3 * count
        return discount

    def _animal_from_source(
        self,
        player: PlayerState,
        selection: AnimalPlaySelection,
        is_animals_upgraded: bool,
    ) -> Tuple[AnimalCard, int]:
        if selection.source == CardSource.HAND:
            if selection.source_index < 0 or selection.source_index >= len(player.hand):
                raise ValueError("Hand index is out of range.")
            raw_card = player.hand[selection.source_index]
            display_extra_cost = 0
        elif selection.source == CardSource.DISPLAY:
            if not is_animals_upgraded:
                raise ValueError("Animals side I cannot play from display.")
            self._validate_display_index(selection.source_index)
            if selection.source_index + 1 > player.reputation + 1:
                raise ValueError("Selected display animal is outside reputation range.")
            display_extra_cost = selection.source_index + 1
            raw_card = self.display[selection.source_index]
        else:
            raise ValueError(f"Unsupported animal source: {selection.source}")

        if not isinstance(raw_card, AnimalCard):
            raise ValueError("Selected card is not an AnimalCard.")
        return raw_card, display_extra_cost

    def perform_animals_action(
        self,
        selections: List[AnimalPlaySelection],
        x_spent: int = 0,
        take_strength5_reputation_bonus: bool = False,
    ) -> None:
        self._assert_game_not_over()
        from arknova_engine.actions.animals_action import run_animals_action

        run_animals_action(
            self,
            selections=selections,
            x_spent=x_spent,
            take_strength5_reputation_bonus=take_strength5_reputation_bonus,
        )

    def _association_task_value(self, task: AssociationTask) -> int:
        return ASSOCIATION_TASK_VALUE[task.value]

    def _association_workers_needed(self, player: PlayerState, task: AssociationTask) -> int:
        placed = player.association_workers_by_task[task]
        if placed >= 3:
            raise ValueError("This association task cannot be used again before break.")
        if placed == 0:
            return 1
        return 2

    def _spend_association_workers(self, player: PlayerState, task: AssociationTask, workers: int) -> None:
        if workers <= 0:
            raise ValueError("Association workers spent must be positive.")
        if player.active_workers < workers:
            raise ValueError("Not enough active association workers.")
        player.active_workers -= workers
        player.workers_on_association_board += workers
        player.association_workers_by_task[task] += workers

    def _resolve_map_support_space_effect(
        self,
        player: PlayerState,
        player_id: int,
        effect_code: str,
        *,
        source: str,
    ) -> None:
        if effect_code == "gain_5_coins":
            player.money += 5
            self.action_log.append(f"{source}:P{player_id}:gain_5_coins")
            return
        if effect_code == "gain_12_coins":
            player.money += 12
            self.action_log.append(f"{source}:P{player_id}:gain_12_coins")
            return
        if effect_code == "gain_3_x_tokens":
            player.x_tokens = min(MAX_X_TOKENS, player.x_tokens + 3)
            self.action_log.append(f"{source}:P{player_id}:gain_3_x_tokens")
            return
        if effect_code == "gain_worker_1":
            player.active_workers += 1
            self.action_log.append(f"{source}:P{player_id}:gain_worker_1")
            return
        if effect_code == "draw_1_card_deck_or_reputation_range":
            if self.deck:
                player.hand.append(self.deck.pop(0))
            self.action_log.append(f"{source}:P{player_id}:draw_1_card")
            return
        player.pending_map_effects.append(effect_code)
        self.action_log.append(f"{source}:P{player_id}:pending={effect_code}")

    def _claim_map_support_space(
        self,
        player: PlayerState,
        player_id: int,
        support_space_index: int,
    ) -> None:
        if support_space_index < 0 or support_space_index >= len(self.map_support_spaces):
            raise ValueError("Selected zoo-map support space is out of range.")
        if support_space_index in player.claimed_map_support_spaces:
            raise ValueError("Selected zoo-map support space is already empty.")

        support_space = self.map_support_spaces[support_space_index]
        if support_space.space_index != support_space_index:
            raise ValueError("Selected zoo-map support space is invalid.")

        player.claimed_map_support_spaces.add(support_space_index)
        if support_space.recurring:
            player.recurring_map_effects.append(support_space.effect_code)
        self._resolve_map_support_space_effect(
            player=player,
            player_id=player_id,
            effect_code=support_space.effect_code,
            source=f"map_support_space_{support_space_index}",
        )

    def _next_available_map_support_space(self, player: PlayerState) -> int:
        for support_space in self.map_support_spaces:
            if support_space.space_index not in player.claimed_map_support_spaces:
                return support_space.space_index
        raise ValueError("No player tokens remain on the left side of the zoo map.")

    def _conservation_project_from_selection(
        self,
        player: PlayerState,
        task: AssociationTaskSelection,
        is_association_upgraded: bool,
    ) -> Tuple[ConservationProjectCard, bool]:
        specified_sources = sum(
            value is not None
            for value in (task.project_from_hand_index, task.project_from_display_index)
        )
        if specified_sources > 1:
            raise ValueError("Choose at most one source for a new conservation project.")

        if task.project_from_hand_index is not None:
            if task.project_from_hand_index < 0 or task.project_from_hand_index >= len(player.hand):
                raise ValueError("Conservation project hand index is out of range.")
            raw_project = player.hand[task.project_from_hand_index]
            if not isinstance(raw_project, ConservationProjectCard):
                raise ValueError("Selected hand card is not a ConservationProjectCard.")
            return raw_project, False

        if task.project_from_display_index is not None:
            if not is_association_upgraded:
                raise ValueError("Only upgraded Association can play project cards from display.")
            self._validate_display_index(task.project_from_display_index)
            if task.project_from_display_index + 1 > player.reputation + 1:
                raise ValueError("Selected project card is outside reputation range.")
            raw_project = self.display[task.project_from_display_index]
            if not isinstance(raw_project, ConservationProjectCard):
                raise ValueError("Selected display card is not a ConservationProjectCard.")
            extra_cost = task.project_from_display_index + 1
            if player.money < extra_cost:
                raise ValueError("Insufficient money for display project additional cost.")
            return raw_project, True

        if task.project_id is None:
            raise ValueError("Conservation project task requires a project_id or a new project source.")
        project = self._find_conservation_project(task.project_id)
        if project is None:
            raise ValueError("Selected conservation project is not currently in play.")
        return project, False

    def _consume_conservation_project_source(
        self,
        player: PlayerState,
        task: AssociationTaskSelection,
    ) -> None:
        if task.project_from_hand_index is not None:
            player.hand.pop(task.project_from_hand_index)
            return
        if task.project_from_display_index is not None:
            extra_cost = task.project_from_display_index + 1
            player.money -= extra_cost
            self.display.pop(task.project_from_display_index)

    def _add_conservation_project_to_row(
        self,
        player: PlayerState,
        project: ConservationProjectCard,
    ) -> None:
        if self._find_conservation_project(project.card_id) is not None:
            raise ValueError("Conservation project is already in play.")
        self.conservation_projects_in_play.insert(0, project)
        self._ensure_conservation_project_slots(project)

        if project.play_bonus_conservation_gain:
            player.conservation += project.play_bonus_conservation_gain
        if project.play_bonus_reputation_gain:
            player.reputation += project.play_bonus_reputation_gain
        if project.play_bonus_effect_label:
            player.pending_map_effects.append(project.play_bonus_effect_label)

        while len(self.conservation_projects_in_play) > self.conservation_project_row_limit:
            discarded_project = self.conservation_projects_in_play.pop()
            owners = self.conservation_project_slot_owners.pop(discarded_project.card_id, {})
            for owner in owners.values():
                if owner is None:
                    continue
                self.player(owner).player_tokens_in_supply += 1
            self.discard.append(discarded_project)

    def _prepare_release_into_wild(
        self,
        player: PlayerState,
        project: ConservationProjectCard,
        slot: ConservationProjectSlot,
        task: AssociationTaskSelection,
    ) -> ReleaseIntoWildResolution:
        if task.released_animal_index is None:
            raise ValueError("Release project requires a released_animal_index.")
        if task.released_animal_index < 0 or task.released_animal_index >= len(player.played_animals):
            raise ValueError("released_animal_index is out of range.")
        animal = player.played_animals[task.released_animal_index]
        if self._release_size_bucket(animal.min_enclosure_size) != slot.release_size_bucket:
            raise ValueError("Released animal does not match the selected release size bucket.")
        if project.requirement_icon is None or project.requirement_group is None:
            raise ValueError("Release project is missing its icon requirement.")

        if project.requirement_group == "continent":
            if animal.continent_icons.get(project.requirement_icon, 0) <= 0:
                raise ValueError("Released animal does not match the release project icon.")
        elif project.requirement_group == "category":
            if animal.category_icons.get(project.requirement_icon, 0) <= 0:
                raise ValueError("Released animal does not match the release project icon.")
        else:
            raise ValueError("Unsupported release project requirement group.")

        placement = player.animal_placements.get(animal.card_id)
        if placement is None:
            raise ValueError("Animal placement is unknown for release project resolution.")

        if placement.enclosure_type.subtype == BuildingSubType.ENCLOSURE_BASIC:
            candidate_enclosures = sorted(
                (
                    building
                    for building in player.zoo_map.buildings.values()
                    if building.type.subtype == BuildingSubType.ENCLOSURE_BASIC
                    and building.empty_spaces == 0
                    and len(building.layout) >= animal.min_enclosure_size
                ),
                key=lambda building: (len(building.layout), building.origin_hex.x, building.origin_hex.y),
            )
            if not candidate_enclosures:
                raise ValueError("No occupied standard enclosure can be freed for this release.")
            return ReleaseIntoWildResolution(
                animal=animal,
                placement=placement,
                released_from_standard_origin=candidate_enclosures[0].origin_hex,
            )

        building = player.zoo_map.buildings.get(placement.enclosure_origin)
        if building is None:
            raise ValueError("Animal enclosure is missing for release project resolution.")
        if building.empty_spaces + placement.spaces_used > building.type.max_capacity:
            raise ValueError("Released animal would overfill its special enclosure.")
        return ReleaseIntoWildResolution(animal=animal, placement=placement)

    def _apply_release_into_wild(
        self,
        player: PlayerState,
        resolution: ReleaseIntoWildResolution,
    ) -> None:
        animal = resolution.animal
        player.appeal -= animal.appeal_gain
        self._remove_icons_for_animal(player, animal)

        if resolution.placement.enclosure_type.subtype == BuildingSubType.ENCLOSURE_BASIC:
            if resolution.released_from_standard_origin is None:
                raise ValueError("Missing released standard enclosure origin.")
            building = player.zoo_map.buildings.get(resolution.released_from_standard_origin)
            if building is None:
                raise ValueError("Released standard enclosure does not exist.")
            building.empty_spaces = building.type.max_capacity
        else:
            building = player.zoo_map.buildings.get(resolution.placement.enclosure_origin)
            if building is None:
                raise ValueError("Released special enclosure does not exist.")
            building.empty_spaces = min(
                building.type.max_capacity,
                building.empty_spaces + resolution.placement.spaces_used,
            )

        player.played_animals.remove(animal)
        player.animal_placements.pop(animal.card_id, None)
        self.discard.append(animal)

    def _validate_conservation_project_support(
        self,
        player: PlayerState,
        project: ConservationProjectCard,
        slot: ConservationProjectSlot,
        task: AssociationTaskSelection,
    ) -> Optional[ReleaseIntoWildResolution]:
        if project.requirement_kind == ConservationRequirementKind.ICON_COUNT:
            if project.requirement_icon is None:
                raise ValueError("Icon-count project is missing requirement_icon.")
            if self._player_icon_count(player, project.requirement_icon) < slot.required_icons:
                raise ValueError("Player does not meet this conservation project icon requirement.")
            return None
        if project.requirement_kind == ConservationRequirementKind.DISTINCT_CONTINENT:
            if self._distinct_project_icon_count(player, ALL_CONTINENT_ICONS) < slot.required_icons:
                raise ValueError("Player does not meet this conservation project continent requirement.")
            return None
        if project.requirement_kind == ConservationRequirementKind.DISTINCT_CATEGORY:
            if self._distinct_project_icon_count(player, ALL_CATEGORY_ICONS) < slot.required_icons:
                raise ValueError("Player does not meet this conservation project category requirement.")
            return None
        if project.requirement_kind == ConservationRequirementKind.RELEASE:
            return self._prepare_release_into_wild(player, project, slot, task)
        raise ValueError(f"Unsupported conservation project requirement kind '{project.requirement_kind}'.")

    def _support_conservation_project(
        self,
        player_id: int,
        player: PlayerState,
        project: ConservationProjectCard,
        task: AssociationTaskSelection,
    ) -> None:
        if project.card_id in player.supported_conservation_projects:
            raise ValueError("Conservation project already supported by this player.")
        if task.project_slot_id is None:
            raise ValueError("Conservation project support requires a project_slot_id.")

        slot = self._conservation_project_slot(project, task.project_slot_id)
        owners = self.conservation_project_slot_owners[project.card_id]
        if owners[slot.slot_id] is not None:
            raise ValueError("Selected conservation project slot is already occupied.")

        release_resolution = self._validate_conservation_project_support(
            player=player,
            project=project,
            slot=slot,
            task=task,
        )

        support_space_index = task.map_support_space_index
        if support_space_index is None:
            support_space_index = self._next_available_map_support_space(player)
        if support_space_index < 0 or support_space_index >= len(self.map_support_spaces):
            raise ValueError("Selected zoo-map support space is out of range.")
        if support_space_index in player.claimed_map_support_spaces:
            raise ValueError("Selected zoo-map support space is already empty.")

        owners[slot.slot_id] = player_id
        player.supported_conservation_projects.add(project.card_id)
        if release_resolution is not None:
            self._apply_release_into_wild(player, release_resolution)
        player.conservation += slot.conservation_gain
        player.reputation += slot.reputation_gain
        self._claim_map_support_space(
            player=player,
            player_id=player_id,
            support_space_index=support_space_index,
        )

    def _execute_association_task(
        self,
        player_id: int,
        player: PlayerState,
        task: AssociationTaskSelection,
        is_association_upgraded: bool,
    ) -> bool:
        took_from_display = False

        if task.task == AssociationTask.REPUTATION:
            player.reputation += ASSOCIATION_REPUTATION_GAIN
            return took_from_display

        if task.task == AssociationTask.PARTNER_ZOO:
            if task.partner_zoo is None:
                raise ValueError("Partner zoo task requires a selected partner zoo.")
            continent = task.partner_zoo.lower()
            if continent not in ALL_PARTNER_ZOOS:
                raise ValueError(f"Unsupported partner zoo continent '{task.partner_zoo}'.")
            if continent in player.partner_zoos:
                raise ValueError("Player already has this partner zoo.")
            if continent not in self.available_partner_zoos:
                raise ValueError("Selected partner zoo is not currently available.")
            if len(player.partner_zoos) >= 4:
                raise ValueError("Cannot have more than 4 partner zoos.")
            if len(player.partner_zoos) >= 2 and not is_association_upgraded:
                raise ValueError("Third/fourth partner zoo requires upgraded Association action.")

            player.partner_zoos.add(continent)
            self.available_partner_zoos.discard(continent)
            player.zoo_icons[continent] = player.zoo_icons.get(continent, 0) + 1
            return took_from_display

        if task.task == AssociationTask.UNIVERSITY:
            if task.university is None:
                raise ValueError("University task requires a selected university.")
            if task.university not in ALL_UNIVERSITIES:
                raise ValueError(f"Unsupported university '{task.university}'.")
            if task.university in player.universities:
                raise ValueError("Player already has this university.")
            if task.university not in self.available_universities:
                raise ValueError("Selected university is not currently available.")
            if task.university_science_icons < 0:
                raise ValueError("university_science_icons cannot be negative.")
            if task.university_reputation_gain < 0:
                raise ValueError("university_reputation_gain cannot be negative.")

            player.universities.add(task.university)
            self.available_universities.discard(task.university)
            player.zoo_icons["science"] = player.zoo_icons.get("science", 0) + task.university_science_icons
            player.reputation += task.university_reputation_gain
            return took_from_display

        if task.task == AssociationTask.CONSERVATION_PROJECT:
            project, took_from_display = self._conservation_project_from_selection(
                player=player,
                task=task,
                is_association_upgraded=is_association_upgraded,
            )
            if (
                task.project_from_hand_index is not None
                or task.project_from_display_index is not None
            ):
                if task.project_slot_id is None:
                    raise ValueError("Conservation project support requires a project_slot_id.")
                slot = self._conservation_project_slot(project, task.project_slot_id)
                self._validate_conservation_project_support(
                    player=player,
                    project=project,
                    slot=slot,
                    task=task,
                )
                support_space_index = task.map_support_space_index
                if support_space_index is None:
                    self._next_available_map_support_space(player)
                elif (
                    support_space_index < 0
                    or support_space_index >= len(self.map_support_spaces)
                    or support_space_index in player.claimed_map_support_spaces
                ):
                    raise ValueError("Selected zoo-map support space is invalid.")
                self._consume_conservation_project_source(player=player, task=task)
                self._add_conservation_project_to_row(player=player, project=project)
            self._support_conservation_project(
                player_id=player_id,
                player=player,
                project=project,
                task=task,
            )
            return took_from_display

        raise ValueError(f"Unknown association task '{task.task}'.")

    def _current_donation_cost(self) -> int:
        return DONATION_COST_TRACK[min(self.donation_progress, len(DONATION_COST_TRACK) - 1)]

    def perform_association_action(
        self,
        tasks: List[AssociationTaskSelection],
        make_donation: bool = False,
        x_spent: int = 0,
    ) -> None:
        self._assert_game_not_over()
        from arknova_engine.actions.association_action import run_association_action

        run_association_action(
            self,
            tasks=tasks,
            make_donation=make_donation,
            x_spent=x_spent,
        )

    def _validate_sponsor_conditions(
        self,
        player: PlayerState,
        sponsor: SponsorCard,
        is_sponsors_upgraded: bool,
    ) -> None:
        if sponsor.level <= 0:
            raise ValueError("Sponsor card level must be positive.")
        if sponsor.requires_upgraded_sponsors_action and not is_sponsors_upgraded:
            raise ValueError("This sponsor requires upgraded Sponsors action.")
        if player.reputation < sponsor.min_reputation:
            raise ValueError("Reputation is too low for this sponsor.")
        if not sponsor.required_partner_zoos.issubset(player.partner_zoos):
            raise ValueError("Required partner zoo is missing for this sponsor.")
        for icon, need in sponsor.required_icons.items():
            if self._player_icon_count(player, icon) < need:
                raise ValueError(f"Not enough icon '{icon}' for this sponsor.")

    def _sponsor_from_source(
        self,
        player: PlayerState,
        selection: SponsorPlaySelection,
        is_sponsors_upgraded: bool,
    ) -> Tuple[SponsorCard, int]:
        if selection.source == CardSource.HAND:
            if selection.source_index < 0 or selection.source_index >= len(player.hand):
                raise ValueError("Hand index is out of range.")
            raw_card = player.hand[selection.source_index]
            display_extra_cost = 0
        elif selection.source == CardSource.DISPLAY:
            if not is_sponsors_upgraded:
                raise ValueError("Sponsors side I cannot play from display.")
            self._validate_display_index(selection.source_index)
            if selection.source_index + 1 > player.reputation + 1:
                raise ValueError("Selected sponsor is outside reputation range.")
            raw_card = self.display[selection.source_index]
            display_extra_cost = selection.source_index + 1
        else:
            raise ValueError(f"Unsupported sponsor source: {selection.source}")

        if not isinstance(raw_card, SponsorCard):
            raise ValueError("Selected card is not a SponsorCard.")
        return raw_card, display_extra_cost

    def perform_sponsors_action(
        self,
        selections: Optional[List[SponsorPlaySelection]] = None,
        use_break_ability: bool = False,
        x_spent: int = 0,
    ) -> None:
        self._assert_game_not_over()
        from arknova_engine.actions.sponsors_action import run_sponsors_action

        run_sponsors_action(
            self,
            selections=selections,
            use_break_ability=use_break_ability,
            x_spent=x_spent,
        )

    def perform_x_token_action(self, chosen_action: MainAction) -> None:
        self._assert_game_not_over()
        from arknova_engine.actions.x_token_action import run_x_token_action

        run_x_token_action(self, chosen_action=chosen_action)

    def _take_display_cards(self, indices: List[int]) -> List[object]:
        if len(indices) != len(set(indices)):
            raise ValueError("Display indices must be unique.")
        for idx in indices:
            self._validate_display_index(idx)
        taken: List[object] = []
        for idx in sorted(indices, reverse=True):
            taken.append(self.display.pop(idx))
        taken.reverse()
        return taken

    def _replenish_display_end_of_turn(self) -> None:
        while len(self.display) < 6 and self.deck:
            self.display.append(self.deck.pop(0))

    def _resolve_cards_discard(self, player: PlayerState, discard_count: int, discard_hand_indices: Optional[List[int]]) -> None:
        if discard_count == 0:
            return
        if discard_hand_indices is None:
            if len(player.hand) < discard_count:
                raise ValueError("Not enough cards in hand to discard.")
            discard_hand_indices = list(range(len(player.hand) - discard_count, len(player.hand)))
        if len(discard_hand_indices) != discard_count:
            raise ValueError(f"Exactly {discard_count} hand-card index(es) must be discarded.")
        if len(set(discard_hand_indices)) != len(discard_hand_indices):
            raise ValueError("Discard hand indices must be unique.")
        for idx in discard_hand_indices:
            if idx < 0 or idx >= len(player.hand):
                raise ValueError(f"Discard hand index {idx} out of range.")
        for idx in sorted(discard_hand_indices, reverse=True):
            self.discard.append(player.hand.pop(idx))

    def perform_cards_action(
        self,
        from_display_indices: Optional[List[int]] = None,
        from_deck_count: Optional[int] = None,
        snap_display_index: Optional[int] = None,
        discard_hand_indices: Optional[List[int]] = None,
        x_spent: int = 0,
    ) -> None:
        self._assert_game_not_over()
        from arknova_engine.actions.cards_action import run_cards_action

        run_cards_action(
            self,
            from_display_indices=from_display_indices,
            from_deck_count=from_deck_count,
            snap_display_index=snap_display_index,
            discard_hand_indices=discard_hand_indices,
            x_spent=x_spent,
        )

    def perform_cards_action_stub(self, break_steps: int = 2, x_spent: int = 0) -> None:
        """Minimal Cards action for break-timing tests.

        This intentionally does not implement full Cards draw/snap logic yet.
        """
        self._assert_game_not_over()
        player_id = self.current_player
        player = self.current()
        _ = self._slot_strength(player, MainAction.CARDS)
        self._consume_x_tokens_for_action(player, x_spent)
        self.advance_break_token(break_steps, player_id=player_id)
        self.action_log.append(f"P{player_id} CARDS_STUB break+={break_steps}")
        self._complete_main_action(player_id, MainAction.CARDS)

    def advance_break_token(self, steps: int, player_id: Optional[int] = None) -> None:
        if steps < 0:
            raise ValueError("Break steps cannot be negative.")
        if player_id is None:
            player_id = self.current_player
        if self.break_progress >= self.break_max:
            return

        self.break_progress = min(self.break_progress + steps, self.break_max)
        if self.break_progress == self.break_max and not self.break_pending:
            self.break_pending = True
            self.break_trigger_player = player_id
            player = self.player(player_id)
            player.x_tokens = min(MAX_X_TOKENS, player.x_tokens + 1)

    def _clear_action_tokens(self, player: PlayerState) -> None:
        for action in MainAction:
            player.multiplier_tokens_on_actions[action] = 0
            player.venom_tokens_on_actions[action] = 0
            player.constriction_tokens_on_actions[action] = 0

    def _refill_association_offers(self) -> None:
        claimed_partner_zoos: Set[str] = set()
        for player in self.players:
            claimed_partner_zoos.update(player.partner_zoos)
        self.available_partner_zoos = set(ALL_PARTNER_ZOOS) - claimed_partner_zoos

        claimed_universities: Set[str] = set()
        for player in self.players:
            claimed_universities.update(player.universities)
        # Base board shows 4 university offers at a time.
        self.available_universities = set(sorted(set(ALL_UNIVERSITIES) - claimed_universities)[:4])

    def _replenish_display_after_break(self) -> None:
        for _ in range(min(2, len(self.display))):
            self.discard.append(self.display.pop(0))
        while len(self.display) < 6 and self.deck:
            self.display.append(self.deck.pop(0))

    def _kiosk_income_for_player(self, player: PlayerState) -> int:
        def adjacent(a: Building, b: Building) -> bool:
            for tile_a in a.layout:
                for tile_b in b.layout:
                    if tile_a.distance(tile_b) == 1:
                        return True
            return False

        income = 0
        buildings = list(player.zoo_map.buildings.values())
        kiosks = [b for b in buildings if b.type == BuildingType.KIOSK]
        for kiosk in kiosks:
            for building in buildings:
                if building is kiosk or not adjacent(kiosk, building):
                    continue
                if building.type == BuildingType.PAVILION:
                    income += 1
                elif building.type.subtype == BuildingSubType.ENCLOSURE_SPECIAL:
                    income += 1
                elif building.type.subtype == BuildingSubType.UNIQUE:
                    income += 1
                elif (
                    building.type.subtype == BuildingSubType.ENCLOSURE_BASIC
                    and building.empty_spaces == 0
                ):
                    income += 1
        return income

    def _appeal_income(self, appeal: int) -> int:
        if self.appeal_income_table:
            if appeal < len(self.appeal_income_table):
                return self.appeal_income_table[appeal]
            return self.appeal_income_table[-1]

        # Fallback approximation until full official table is encoded.
        if appeal <= 7:
            return 4 + appeal
        return 11 + (appeal - 7 + 1) // 2

    def _break_income_order(self) -> List[int]:
        if self.break_trigger_player is None:
            return list(range(len(self.players)))
        return [
            (self.break_trigger_player + offset) % len(self.players)
            for offset in range(len(self.players))
        ]

    def _resolve_break(self) -> None:
        # 1) Hand limit discard.
        for player in self.players:
            if len(player.hand) > player.hand_limit:
                over = len(player.hand) - player.hand_limit
                to_discard = player.hand[-over:]
                self.discard.extend(to_discard)
                del player.hand[-over:]

        # 2) Remove temporary tokens on action cards.
        for player in self.players:
            self._clear_action_tokens(player)

        # 3) Recall workers and refill association offers.
        for player in self.players:
            player.active_workers += player.workers_on_association_board
            player.workers_on_association_board = 0
            for task in AssociationTask:
                player.association_workers_by_task[task] = 0
        self._refill_association_offers()

        # 4) Discard display folders 1 and 2, shift and refill.
        self._replenish_display_after_break()

        # 5) Income.
        for player_id in self._break_income_order():
            player = self.player(player_id)
            player.money += self._appeal_income(player.appeal)
            player.money += self._kiosk_income_for_player(player)
            player.money += player.recurring_break_income
            for effect_code in list(player.recurring_map_effects):
                self._resolve_map_support_space_effect(
                    player=player,
                    player_id=player_id,
                    effect_code=effect_code,
                    source="break_recurring_map_effect",
                )

        if not self.end_game_triggered:
            for player_id in self._break_income_order():
                if self._player_reaches_bga_end_trigger(player_id):
                    self._trigger_end_game(trigger_player=player_id, triggered_during_break=True)
                    break

        # 6) Reset break track.
        self.break_progress = 0
        self.break_pending = False
        self.break_trigger_player = None
        self.action_log.append("BREAK_RESOLVED")

    @staticmethod
    def bga_conservation_points(conservation: int) -> int:
        """BGA scoring conversion for conservation points.

        Rule used by this project:
        - first 10 conservation points: 2 VP each
        - conservation points above 10: 3 VP each
        """

        cp = max(0, conservation)
        first_ten = min(cp, 10)
        above_ten = max(0, cp - 10)
        return first_ten * 2 + above_ten * 3

    def bga_final_score(
        self,
        player_id: int,
        final_scoring_conservation_bonus: int = 0,
        end_game_bonus_points: int = 0,
    ) -> BGAFinalScoreBreakdown:
        """Compute one player's final score using the BGA formula."""

        player = self.player(player_id)
        appeal_points = player.appeal
        conservation_total = player.conservation + max(0, final_scoring_conservation_bonus)
        conservation_points = self.bga_conservation_points(conservation_total)
        total_points = appeal_points + conservation_points + end_game_bonus_points
        return BGAFinalScoreBreakdown(
            player_id=player_id,
            appeal_points=appeal_points,
            conservation_points=conservation_points,
            end_game_bonus_points=end_game_bonus_points,
            total_points=total_points,
        )

    def bga_final_scores(
        self,
        final_scoring_conservation_bonus_by_player: Optional[Dict[int, int]] = None,
        end_game_bonus_points_by_player: Optional[Dict[int, int]] = None,
    ) -> List[BGAFinalScoreBreakdown]:
        final_scoring_conservation_bonus_by_player = (
            final_scoring_conservation_bonus_by_player or {}
        )
        end_game_bonus_points_by_player = end_game_bonus_points_by_player or {}
        return [
            self.bga_final_score(
                player_id=player_id,
                final_scoring_conservation_bonus=final_scoring_conservation_bonus_by_player.get(
                    player_id, 0
                ),
                end_game_bonus_points=end_game_bonus_points_by_player.get(player_id, 0),
            )
            for player_id in range(len(self.players))
        ]

    def bga_winner_player_ids(
        self,
        final_scoring_conservation_bonus_by_player: Optional[Dict[int, int]] = None,
        end_game_bonus_points_by_player: Optional[Dict[int, int]] = None,
    ) -> List[int]:
        scores = self.bga_final_scores(
            final_scoring_conservation_bonus_by_player=final_scoring_conservation_bonus_by_player,
            end_game_bonus_points_by_player=end_game_bonus_points_by_player,
        )
        if not scores:
            return []
        top_score = max(score.total_points for score in scores)
        return [score.player_id for score in scores if score.total_points == top_score]


def create_base_game(
    num_players: int = DEFAULT_NUM_PLAYERS,
    seed: int = 7,
    map_image_name: str = DEFAULT_MAP_IMAGE_NAME,
) -> GameState:
    if num_players != DEFAULT_NUM_PLAYERS:
        raise ValueError("Current setup flow is implemented for 2-player games.")

    rng = random.Random(seed)
    map_data = load_map_data_by_image_name(map_image_name)

    players: List[PlayerState] = []
    for player_index in range(num_players):
        actions = [MainAction.CARDS, MainAction.BUILD, MainAction.ASSOCIATION, MainAction.SPONSORS]
        rng.shuffle(actions)
        players.append(
            PlayerState(
                zoo_map=ArkNovaMap(map_data=map_data),
                appeal=player_index,  # 2p setup: start player at 0, second player at 1.
                action_order=[MainAction.ANIMALS] + actions,
            )
        )

    zoo_deck = [f"zoo_card_{i:03d}" for i in range(1, ZOO_DECK_SIZE + 1)]
    rng.shuffle(zoo_deck)

    discard: List[object] = []
    for player in players:
        drafted_cards = [zoo_deck.pop(0) for _ in range(STARTING_HAND_DRAW_COUNT)]
        kept_indices = set(rng.sample(range(STARTING_HAND_DRAW_COUNT), STARTING_HAND_KEEP_COUNT))
        player.hand = [card for idx, card in enumerate(drafted_cards) if idx in kept_indices]
        discard.extend(card for idx, card in enumerate(drafted_cards) if idx not in kept_indices)

    display = [zoo_deck.pop(0) for _ in range(6)]

    final_scoring_deck = [f"final_scoring_{i:02d}" for i in range(1, FINAL_SCORING_DECK_SIZE + 1)]
    rng.shuffle(final_scoring_deck)
    for player in players:
        player.final_scoring_cards = [final_scoring_deck.pop(0), final_scoring_deck.pop(0)]

    builtin_catalog = _builtin_conservation_project_catalog()
    base_project_ids = sorted(
        card_id
        for card_id, project in builtin_catalog.items()
        if project.requirement_kind != ConservationRequirementKind.RELEASE
        and 101 <= int(card_id[1:4]) <= 112
    )
    rng.shuffle(base_project_ids)
    base_conservation_projects = [builtin_catalog[card_id] for card_id in base_project_ids[:3]]

    return GameState(
        players=players,
        map_image_name=map_image_name,
        display=display,
        deck=zoo_deck,
        discard=discard,
        base_conservation_projects=base_conservation_projects,
        final_scoring_deck=final_scoring_deck,
        final_scoring_discard=[],
    )

"""Simplified Ark Nova-like prototype runtime.

Run:
    python main.py

This is a compact baseline you can extend:
- simplified game state
- legal action generation
- human-vs-human turn selection loop
- explicit action details for interface-driven integrations
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from itertools import combinations, product
import json
from pathlib import Path
import random
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from arknova_engine.map_model import (
    ArkNovaMap,
    Building,
    BuildingSubType,
    BuildingType,
    HexTile,
    Rotation,
    Terrain,
    load_map_data_by_image_name,
)
from arknova_engine.card_effects import apply_animal_effect, build_effect_coverage, resolve_card_effect
from arknova_engine.setup_data import (
    build_map_tile_bonus_map as _build_map_tile_bonus_map_impl,
    load_final_scoring_cards_from_dataset as _load_final_scoring_cards_from_dataset_impl,
    load_animal_cards_from_dataset as _load_animal_cards_from_dataset_impl,
    load_base_setup_card_pools as _load_base_setup_card_pools_impl,
    resolve_cards_dataset_path as _resolve_cards_dataset_path_impl,
)
from arknova_engine.setup import (
    apply_opening_draft_selection as _apply_opening_draft_selection_impl,
    build_opening_setup_info as _build_opening_setup_info_impl,
    draw_final_scoring_cards_for_players as _draw_final_scoring_cards_for_players_impl,
    draw_opening_draft_cards as _draw_opening_draft_cards_impl,
    setup_game_state as _setup_game_state_impl,
)
from arknova_engine.console import (
    enumerate_index_combinations as _enumerate_index_combinations_impl,
    prompt_card_combination_indices as _prompt_card_combination_indices_impl,
    prompt_opening_draft_indices as _prompt_opening_draft_indices_impl,
)
from arknova_engine.console_actions import (
    format_animals_play_step_for_human as _format_animals_play_step_for_human_impl,
    prompt_animals_action_details_for_human as _prompt_animals_action_details_for_human_impl,
    prompt_association_action_details_for_human as _prompt_association_action_details_for_human_impl,
    prompt_build_action_details_for_human as _prompt_build_action_details_for_human_impl,
    prompt_cards_action_details_for_human as _prompt_cards_action_details_for_human_impl,
    prompt_sponsors_action_details_for_human as _prompt_sponsors_action_details_for_human_impl,
)
from arknova_engine.scoring import (
    advance_break_track as _advance_break_track_impl,
    bga_conservation_points as _bga_conservation_points_impl,
    break_income_from_appeal as _break_income_from_appeal_impl,
    increase_reputation as _increase_reputation_impl,
    progress_score as _progress_score_impl,
    take_one_reputation_bonus_card as _take_one_reputation_bonus_card_impl,
    upgrade_one_action_card as _upgrade_one_action_card_impl,
)
from arknova_engine.turns import (
    apply_action as _apply_action_impl,
    legal_actions as _legal_actions_impl,
)


class ActionType(str, Enum):
    MAIN_ACTION = "main_action"
    PENDING_DECISION = "pending_decision"
    X_TOKEN = "x_token"
    TAKE_MONEY = "take_money"
    DRAW_CARD = "draw_card"
    BUILD_ENCLOSURE = "build_enclosure"
    PLAY_ANIMAL = "play_animal"
    DONATION = "donation"


@dataclass(frozen=True)
class AnimalCard:
    name: str
    cost: int
    size: int
    appeal: int
    conservation: int
    reputation_gain: int = 0
    card_type: str = "animal"
    badges: Tuple[str, ...] = ()
    required_water_adjacency: int = 0
    required_rock_adjacency: int = 0
    required_icons: Tuple[Tuple[str, int], ...] = ()
    ability_title: str = ""
    ability_text: str = ""
    effects: Tuple[Tuple[str, str], ...] = ()
    reptile_house_size: Optional[int] = None
    large_bird_aviary_size: Optional[int] = None
    number: int = -1
    instance_id: str = ""


@dataclass
class Enclosure:
    size: int
    occupied: bool = False
    origin: Optional[Tuple[int, int]] = None
    rotation: str = "ROT_0"
    enclosure_type: str = "standard"
    used_capacity: int = 0
    animal_capacity: int = 1


@dataclass
class EnclosureObject:
    size: int
    enclosure_type: str
    adjacent_rock: int
    adjacent_water: int
    animals_inside: int
    origin: Tuple[int, int]
    rotation: str


@dataclass
class SponsorBuilding:
    sponsor_number: int
    cells: Tuple[Tuple[int, int], ...]
    label: str = ""


@dataclass(frozen=True)
class Action:
    type: ActionType
    value: Optional[int] = None
    card_name: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:
        if self.type == ActionType.MAIN_ACTION:
            x_spent = int(self.value or 0)
            base = str(self.card_name)
            details = self.details or {}
            if base == "sponsors" and bool(details.get("sponsors_break_only")):
                base = "sponsors_break_only"
            strength_raw = details.get("effective_strength")
            strength = int(strength_raw) if isinstance(strength_raw, int) else None
            if strength is not None:
                if x_spent <= 0:
                    rendered = f"{base}(strength={strength})"
                else:
                    rendered = f"{base}(x={x_spent}, strength={strength})"
            else:
                if x_spent <= 0:
                    rendered = base
                else:
                    rendered = f"{base}(x={x_spent})"
            action_label = str(details.get("action_label") or "").strip()
            if action_label:
                return f"{rendered} | {action_label}"
            return rendered
        if self.type == ActionType.PENDING_DECISION:
            details = self.details or {}
            kind = str(details.get("pending_kind") or "pending").strip() or "pending"
            action_label = str(details.get("action_label") or "").strip()
            if action_label:
                return f"{kind} | {action_label}"
            return kind
        if self.type == ActionType.X_TOKEN:
            if self.card_name:
                return f"X_TOKEN(move={self.card_name})"
            return "X_TOKEN(move any action card to slot 1)"
        if self.type == ActionType.BUILD_ENCLOSURE:
            return f"BUILD(size={self.value})"
        if self.type == ActionType.PLAY_ANIMAL:
            return f"PLAY({self.card_name})"
        return self.type.value.upper()


@dataclass(frozen=True)
class SetupCardRef:
    data_id: str
    title: str


@dataclass(frozen=True)
class TwoPlayerBlockedProjectLevel:
    project_data_id: str
    project_title: str
    blocked_level: str


@dataclass
class OpeningSetupInfo:
    conservation_space_2_fixed_options: List[str] = field(default_factory=list)
    conservation_space_5_bonus_tiles: List[str] = field(default_factory=list)
    conservation_space_8_bonus_tiles: List[str] = field(default_factory=list)
    conservation_space_10_rule: str = ""
    base_conservation_projects: List[SetupCardRef] = field(default_factory=list)
    two_player_blocked_project_levels: List[TwoPlayerBlockedProjectLevel] = field(default_factory=list)


CONSERVATION_SPACE_2_FIXED_OPTIONS: Tuple[str, str] = (
    "upgrade_action_card",
    "activate_association_worker",
)

# 9 bonus tiles pool for CP 5 / 8 setup (base game).
CONSERVATION_BONUS_TILE_POOL: Tuple[str, ...] = (
    "10_money",
    "size_3_enclosure",
    "2_reputation",
    "3_x_tokens",
    "3_cards",
    "partner_zoo",
    "university",
    "x2_multiplier",
    "sponsor_card",
)

TWO_PLAYER_BLOCKED_LEVELS: Tuple[str, str, str] = ("left_level", "middle_level", "right_level")

CONSERVATION_SPACE_10_RULE = (
    "first_player_reaches_10_conservation_then_all_players_discard_1_final_scoring_card"
)
CONSERVATION_FIXED_MONEY_OPTION = "5coins"
CONSERVATION_REWARD_THRESHOLDS: Tuple[int, int, int] = (2, 5, 8)
CONSERVATION_SHARED_BONUS_THRESHOLDS: Tuple[int, int] = (5, 8)
BREAK_TRACK_BY_PLAYERS: Dict[int, int] = {2: 9, 3: 12, 4: 15}
MAX_GAME_ROUNDS: int = 100

MAIN_ACTION_CARDS: Tuple[str, ...] = ("animals", "cards", "build", "association", "sponsors")
_BUILD_ENUM_LEGAL_OPTION_CACHE: Dict[Tuple[str, str, int, Tuple[str, ...], bool, bool], List[Dict[str, Any]]] = {}
ZOO_DECK_CARD_TYPES: Tuple[str, ...] = ("animal", "sponsor", "conservation_project")
CARDS_DRAW_TABLE_BASE: Tuple[int, ...] = (1, 1, 2, 2, 3)
CARDS_DISCARD_TABLE_BASE: Tuple[int, ...] = (1, 0, 1, 0, 1)
ANIMALS_PLAY_LIMIT_BASE: Tuple[int, ...] = (0, 1, 1, 1, 2)
ANIMALS_PLAY_LIMIT_UPGRADED: Tuple[int, ...] = (1, 1, 2, 2, 2)
CARDS_DRAW_TABLE_UPGRADED: Tuple[int, ...] = (1, 2, 2, 3, 4)
CARDS_DISCARD_TABLE_UPGRADED: Tuple[int, ...] = (0, 1, 0, 1, 1)
CONTINENT_BADGE_ALIASES: Dict[str, str] = {
    "africa": "Africa",
    "europe": "Europe",
    "asia": "Asia",
    "america": "America",
    "americas": "America",
    "australia": "Australia",
    "oceania": "Australia",
}
ALL_PARTNER_ZOOS: Tuple[str, ...] = ("africa", "europe", "asia", "america", "australia")
UNIVERSITY_TYPES: Tuple[str, ...] = (
    "reputation_1_hand_limit_5",
    "science_1_reputation_2",
    "science_2",
)
UNIVERSITY_SCIENCE_GAIN: Dict[str, int] = {
    "reputation_1_hand_limit_5": 0,
    "science_1_reputation_2": 1,
    "science_2": 2,
}
UNIVERSITY_REPUTATION_GAIN: Dict[str, int] = {
    "reputation_1_hand_limit_5": 1,
    "science_1_reputation_2": 2,
    "science_2": 0,
}
UNIVERSITY_HAND_LIMIT_SET: Dict[str, int] = {
    "reputation_1_hand_limit_5": 5,
}
DONATION_COST_TRACK: Tuple[int, ...] = (2, 5, 7, 10, 12)
ASSOCIATION_TASK_KINDS: Tuple[str, ...] = ("reputation", "partner_zoo", "university", "conservation_project")
MAX_WORKERS: int = 4
MAX_ASSOCIATION_WORKERS_PER_TASK: int = 3
CONSERVATION_PROJECT_ROW_LIMIT_BY_PLAYERS: Dict[int, int] = {
    2: 3,
    3: 3,
    4: 4,
}
SPONSOR_LEVEL_OVERRIDES: Dict[int, int] = {
    231: 3,
    232: 3,
    233: 3,
    234: 3,
    235: 3,
    236: 4,
}
SPONSOR_REQUIRES_UPGRADED_OVERRIDES: Set[int] = {201, 207, 216, 219, 262, 263}
SPONSOR_MIN_REPUTATION_OVERRIDES: Dict[int, int] = {
    227: 6,
    228: 3,
    243: 3,
    244: 3,
    245: 3,
    263: 6,
}
SPONSOR_MAX_APPEAL_25_OVERRIDES: Set[int] = {207, 222, 258, 259, 260, 264}
SPONSOR_REQUIRED_ICON_OVERRIDES: Dict[int, Tuple[Tuple[str, int], ...]] = {
    231: (("primate", 1),),
    232: (("reptile", 1),),
    233: (("bird", 1),),
    234: (("predator", 1),),
    235: (("herbivore", 1),),
}
SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS: Dict[int, str] = {
    236: "primate",
    237: "reptile",
    238: "bird",
    239: "predator",
    240: "herbivore",
}
SPONSOR_OWN_ICON_APPEAL_TRIGGERS: Dict[int, Tuple[str, int]] = {
    243: ("herbivore", 2),
    244: ("bird", 2),
    245: ("water", 2),
    246: ("rock", 2),
    247: ("primate", 2),
}
SPONSOR_ICON_INCOME_THRESHOLDS: Dict[int, str] = {
    231: "primate",
    232: "reptile",
    233: "bird",
    234: "predator",
    235: "herbivore",
}
SPONSOR_BADGE_OVERRIDES: Dict[int, Tuple[str, ...]] = {
    201: ("Science",),
    202: ("Science",),
    203: ("Science",),
    204: ("Science",),
    205: ("Science",),
    206: ("Science",),
    207: ("Science",),
    208: ("Science",),
    209: ("Science",),
    210: ("America",),
    211: ("Europe",),
    212: ("Australia",),
    213: ("Asia",),
    214: ("Africa",),
    219: ("Science",),
    220: ("Science",),
    221: ("Science",),
    223: ("Science", "Science"),
    224: ("Science",),
    225: ("Science",),
    226: ("Science",),
    236: ("Primate",),
    237: ("Reptile",),
    238: ("Bird",),
    239: ("Predator",),
    240: ("Herbivore",),
    241: ("Water",),
    242: ("Rock",),
    243: ("Herbivore",),
    244: ("Bird",),
    247: ("Primate",),
    248: ("Primate",),
    250: ("Reptile",),
    251: ("Predator", "Bear"),
    252: ("Predator",),
    253: ("Herbivore",),
    258: ("Bird",),
    259: ("Reptile",),
    260: ("Herbivore",),
    264: ("Primate",),
}
SPONSOR_UNIQUE_BUILDING_CARDS: Set[int] = {
    243,
    244,
    245,
    246,
    247,
    248,
    249,
    250,
    251,
    252,
    253,
    254,
    255,
    256,
    257,
}
SPONSOR_UNIQUE_BASE_FOOTPRINTS: Dict[int, Tuple[Tuple[int, int], ...]] = {
    243: ((0, 0), (0, 1), (1, 1)),
    244: ((0, 0), (1, 0), (2, 0), (1, 1)),
    245: ((0, 0), (0, 1), (1, 1), (2, 0)),
    246: ((0, 0), (0, 1), (0, 2), (0, 3)),
    247: ((0, 0), (1, 0), (2, 0), (2, -1)),
    248: ((0, 0), (0, 1), (1, 1), (2, 1)),
    249: ((0, 0), (1, 0), (2, 0)),
    250: ((0, 0), (1, 0), (1, 1), (2, -1)),
    251: ((0, 0), (0, 1), (1, 1), (1, 2)),
    252: ((0, 0), (1, 0), (1, 1), (1, 2)),
    253: ((0, 0), (1, 0), (1, 1), (2, 1)),
    254: ((0, 0), (1, 0), (1, 1)),
    255: ((0, 0), (1, 0)),
    256: ((0, 0), (1, 0)),
    257: ((0, 0), (1, 0)),
}
SPONSOR_UNIQUE_REQUIRED_ROCK: Dict[int, int] = {
    243: 1,
    246: 2,
    247: 1,
    252: 1,
    255: 1,
}
SPONSOR_UNIQUE_REQUIRED_WATER: Dict[int, int] = {
    244: 1,
    245: 2,
    250: 1,
    251: 1,
    256: 1,
}
SPONSOR_UNIQUE_BORDER_MIN: Dict[int, int] = {
    254: 2,
    257: 2,
}
SPONSOR_UNIQUE_IGNORE_ADJACENCY: Set[int] = {257}
BASE_CONSERVATION_PROJECT_LEVELS: Tuple[Tuple[str, int, int], ...] = (
    ("left_level", 5, 5),
    ("middle_level", 4, 4),
    ("right_level", 2, 2),
)
PROJECT_LEVEL_REWARDS_BY_NUMBER: Dict[int, Tuple[Tuple[str, int, int, int], ...]] = {
    101: (
        ("left_level", 5, 5, 0),
        ("middle_level", 4, 3, 0),
        ("right_level", 3, 2, 0),
    ),
    102: (
        ("left_level", 5, 5, 0),
        ("middle_level", 4, 3, 0),
        ("right_level", 3, 2, 0),
    ),
    123: (
        ("left_level", 1, 2, 2),
        ("middle_level", 1, 1, 2),
        ("right_level", 1, 2, 0),
    ),
    124: (
        ("left_level", 1, 2, 2),
        ("middle_level", 1, 1, 2),
        ("right_level", 1, 2, 0),
    ),
    125: (
        ("left_level", 1, 2, 2),
        ("middle_level", 1, 1, 2),
        ("right_level", 1, 2, 0),
    ),
    126: (
        ("left_level", 1, 2, 2),
        ("middle_level", 1, 1, 2),
        ("right_level", 1, 2, 0),
    ),
    127: (
        ("left_level", 1, 2, 2),
        ("middle_level", 1, 1, 2),
        ("right_level", 1, 2, 0),
    ),
    128: (
        ("left_level", 5, 4, 0),
        ("middle_level", 4, 3, 0),
        ("right_level", 3, 2, 0),
    ),
    129: (
        ("left_level", 5, 4, 0),
        ("middle_level", 4, 3, 0),
        ("right_level", 3, 2, 0),
    ),
    130: (
        ("left_level", 8, 4, 0),
        ("middle_level", 5, 3, 0),
        ("right_level", 2, 2, 0),
    ),
    131: (
        ("left_level", 4, 4, 0),
        ("middle_level", 3, 3, 0),
        ("right_level", 2, 2, 0),
    ),
    132: (
        ("left_level", 5, 4, 0),
        ("middle_level", 4, 3, 0),
        ("right_level", 2, 2, 0),
    ),
}
BREEDING_PROJECT_BADGE_BY_NUMBER: Dict[int, str] = {
    123: "bird",
    124: "predator",
    125: "reptile",
    126: "herbivore",
    127: "primate",
}


@dataclass
class PlayerState:
    name: str
    money: int = 25
    appeal: int = 0
    conservation: int = 0
    reputation: int = 1
    workers: int = 1
    x_tokens: int = 0
    hand_limit: int = 3
    action_order: List[str] = field(default_factory=lambda: list(MAIN_ACTION_CARDS))
    action_upgraded: Dict[str, bool] = field(
        default_factory=lambda: {action: False for action in MAIN_ACTION_CARDS}
    )
    zoo_map: Optional[ArkNovaMap] = None
    enclosures: List[Enclosure] = field(default_factory=list)
    enclosure_objects: List[EnclosureObject] = field(default_factory=list)
    hand: List[AnimalCard] = field(default_factory=list)
    zoo_cards: List[AnimalCard] = field(default_factory=list)
    deck: List[AnimalCard] = field(default_factory=list)
    discard: List[AnimalCard] = field(default_factory=list)
    pouched_cards: List[AnimalCard] = field(default_factory=list)
    pouched_cards_by_host: Dict[str, List[AnimalCard]] = field(default_factory=dict)
    final_scoring_cards: List[SetupCardRef] = field(default_factory=list)
    partner_zoos: Set[str] = field(default_factory=set)
    universities: Set[str] = field(default_factory=set)
    supported_conservation_projects: Set[str] = field(default_factory=set)
    multiplier_tokens_on_actions: Dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in MAIN_ACTION_CARDS}
    )
    venom_tokens_on_actions: Dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in MAIN_ACTION_CARDS}
    )
    constriction_tokens_on_actions: Dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in MAIN_ACTION_CARDS}
    )
    extra_actions_granted: Dict[str, int] = field(
        default_factory=lambda: {action: 0 for action in MAIN_ACTION_CARDS}
    )
    extra_any_actions: int = 0
    extra_strength_actions: Dict[int, int] = field(default_factory=dict)
    camouflage_condition_ignores: int = 0
    workers_on_association_board: int = 0
    association_workers_by_task: Dict[str, int] = field(
        default_factory=lambda: {task: 0 for task in ASSOCIATION_TASK_KINDS}
    )
    claimed_conservation_reward_spaces: Set[int] = field(default_factory=set)
    claimed_reputation_milestones: Set[int] = field(default_factory=set)
    opening_draft_drawn: List[AnimalCard] = field(default_factory=list)
    opening_draft_kept_indices: List[int] = field(default_factory=list)
    sponsor_tokens_by_number: Dict[int, int] = field(default_factory=dict)
    sponsor_waza_assignment_mode: str = ""
    sponsor_ignore_large_condition_charges: int = 0
    sponsor_buildings: List[SponsorBuilding] = field(default_factory=list)
    supported_conservation_project_actions: int = 0
    map_left_track_unlocked_count: int = 0
    map_left_track_claimed_indices: Set[int] = field(default_factory=set)
    map_left_track_unlocked_effects: List[str] = field(default_factory=list)
    claimed_partner_zoo_thresholds: Set[int] = field(default_factory=set)
    claimed_university_thresholds: Set[int] = field(default_factory=set)
    map_completion_reward_claimed: bool = False

    def final_score(self) -> int:
        return int(self.appeal) + _bga_conservation_points_impl(int(self.conservation))


@dataclass
class GameState:
    players: List[PlayerState]
    map_image_name: str = "plan1a"
    map_tile_bonuses: Dict[Tuple[int, int], str] = field(default_factory=dict)
    map_tile_tags: Dict[Tuple[int, int], List[str]] = field(default_factory=dict)
    map_rules: Dict[str, Any] = field(default_factory=dict)
    opening_setup: OpeningSetupInfo = field(default_factory=OpeningSetupInfo)
    shared_conservation_bonus_tiles: Dict[int, List[str]] = field(default_factory=dict)
    claimed_conservation_bonus_tiles: Dict[int, List[str]] = field(default_factory=dict)
    zoo_deck: List[AnimalCard] = field(default_factory=list)
    zoo_discard: List[AnimalCard] = field(default_factory=list)
    zoo_display: List[AnimalCard] = field(default_factory=list)
    available_partner_zoos: Set[str] = field(default_factory=set)
    available_universities: Set[str] = field(default_factory=set)
    conservation_project_slots: Dict[str, Dict[str, Optional[int]]] = field(default_factory=dict)
    unused_base_conservation_projects: List[SetupCardRef] = field(default_factory=list)
    final_scoring_deck: List[SetupCardRef] = field(default_factory=list)
    final_scoring_discard: List[SetupCardRef] = field(default_factory=list)
    marked_display_card_ids: Set[str] = field(default_factory=set)
    donation_progress: int = 0
    effect_log: List[str] = field(default_factory=list)
    break_progress: int = 0
    break_max: int = 9
    break_trigger_player: Optional[int] = None
    pending_decision_kind: str = ""
    pending_decision_player_id: Optional[int] = None
    pending_decision_payload: Dict[str, Any] = field(default_factory=dict)
    current_player: int = 0
    turn_index: int = 0
    max_turns_per_player: int = 16
    score_end_threshold: int = 100
    endgame_trigger_player: Optional[int] = None
    endgame_trigger_turn_index: Optional[int] = None
    max_rounds: int = MAX_GAME_ROUNDS
    forced_game_over: bool = False
    forced_game_over_reason: str = ""

    @property
    def total_turn_limit(self) -> int:
        return self.max_turns_per_player * len(self.players)

    def game_over(self) -> bool:
        if bool(self.forced_game_over):
            return True
        if self.endgame_trigger_player is not None:
            trigger_turn = int(self.endgame_trigger_turn_index or 0)
            return (
                int(self.turn_index) > trigger_turn
                and int(self.current_player) == int(self.endgame_trigger_player)
            )
        return False

    def available_conservation_reward_choices(self, player_id: int, threshold: int) -> List[str]:
        player = self.players[player_id]
        if player.conservation < threshold:
            return []
        if threshold in player.claimed_conservation_reward_spaces:
            return []
        if threshold == 2:
            choices: List[str] = []
            if any(not bool(player.action_upgraded.get(action, False)) for action in MAIN_ACTION_CARDS):
                choices.append("upgrade_action_card")
            if int(player.workers) < MAX_WORKERS:
                choices.append("activate_association_worker")
            return choices
        if threshold not in CONSERVATION_SHARED_BONUS_THRESHOLDS:
            raise ValueError("Conservation threshold must be 2, 5, or 8 for reward choices.")
        return [CONSERVATION_FIXED_MONEY_OPTION] + list(
            self.shared_conservation_bonus_tiles.get(threshold, [])
        )

    def claim_conservation_reward(self, player_id: int, threshold: int, reward: str) -> None:
        player = self.players[player_id]
        choices = self.available_conservation_reward_choices(player_id=player_id, threshold=threshold)
        if not choices:
            raise ValueError("Player cannot claim this conservation reward now.")
        if reward not in choices:
            raise ValueError(f"Reward '{reward}' is not available for conservation space {threshold}.")

        if reward == "upgrade_action_card":
            upgraded = _upgrade_one_action_card(player, interactive=False)
            if not upgraded:
                raise ValueError("No action card can be upgraded.")
        elif reward == "activate_association_worker":
            gained = _gain_workers(
                state=self,
                player=player,
                player_id=player_id,
                amount=1,
                source=f"conservation_reward_{threshold}",
                allow_interactive=False,
            )
            if gained <= 0:
                raise ValueError("No inactive association worker is available.")
        elif reward == CONSERVATION_FIXED_MONEY_OPTION:
            player.money += 5
        elif reward == "10_money":
            player.money += 10
        elif reward == "2_reputation":
            _increase_reputation(state=self, player=player, amount=2, allow_interactive=False)
        elif reward == "3_x_tokens":
            player.x_tokens = min(5, int(player.x_tokens) + 3)
        elif reward == "3_cards":
            player.hand.extend(_draw_from_zoo_deck(self, 3))
        else:
            if reward in {"size_3_enclosure", "partner_zoo", "university", "x2_multiplier", "sponsor_card"}:
                raise ValueError(f"Reward '{reward}' requires explicit choice details in the action flow.")
            if threshold not in CONSERVATION_SHARED_BONUS_THRESHOLDS:
                raise ValueError(f"Reward '{reward}' is unsupported for conservation space {threshold}.")

        if threshold in CONSERVATION_SHARED_BONUS_THRESHOLDS and reward != CONSERVATION_FIXED_MONEY_OPTION:
            shared_tiles = self.shared_conservation_bonus_tiles[threshold]
            shared_tiles.remove(reward)
            self.claimed_conservation_bonus_tiles[threshold].append(reward)

        player.claimed_conservation_reward_spaces.add(threshold)


def _validate_observation_viewer_id(state: GameState, viewer_player_id: int) -> None:
    if int(viewer_player_id) < 0 or int(viewer_player_id) >= len(state.players):
        raise ValueError("viewer_player_id is out of range.")


def _serialize_setup_card_ref(card: SetupCardRef) -> Dict[str, str]:
    return {
        "data_id": str(card.data_id),
        "title": str(card.title),
    }


def _serialize_animal_card_public(card: AnimalCard) -> Dict[str, Any]:
    return {
        "number": int(card.number),
        "name": str(card.name),
        "card_type": str(card.card_type),
        "cost": int(card.cost),
        "size": int(card.size),
        "appeal": int(card.appeal),
        "conservation": int(card.conservation),
        "reputation_gain": int(card.reputation_gain),
        "badges": [str(item) for item in tuple(card.badges)],
        "required_water_adjacency": int(card.required_water_adjacency),
        "required_rock_adjacency": int(card.required_rock_adjacency),
        "required_icons": [
            {"icon": str(icon), "count": int(count)}
            for icon, count in tuple(card.required_icons)
        ],
        "ability_title": str(card.ability_title),
        "effects": [
            {"kind": str(kind), "value": str(value)}
            for kind, value in tuple(card.effects)
        ],
        "reptile_house_size": (
            int(card.reptile_house_size)
            if card.reptile_house_size is not None
            else None
        ),
        "large_bird_aviary_size": (
            int(card.large_bird_aviary_size)
            if card.large_bird_aviary_size is not None
            else None
        ),
    }


def _serialize_animal_card_private(card: AnimalCard) -> Dict[str, Any]:
    card_view = _serialize_animal_card_public(card)
    card_view["instance_id"] = str(card.instance_id)
    card_view["ability_text"] = str(card.ability_text)
    return card_view


def _serialize_enclosure_public(enclosure: Enclosure) -> Dict[str, Any]:
    return {
        "size": int(enclosure.size),
        "occupied": bool(enclosure.occupied),
        "origin": (
            [int(enclosure.origin[0]), int(enclosure.origin[1])]
            if enclosure.origin is not None
            else None
        ),
        "rotation": str(enclosure.rotation),
        "enclosure_type": str(enclosure.enclosure_type),
        "used_capacity": int(enclosure.used_capacity),
        "animal_capacity": int(enclosure.animal_capacity),
    }


def _serialize_enclosure_object_public(enclosure_object: EnclosureObject) -> Dict[str, Any]:
    return {
        "size": int(enclosure_object.size),
        "enclosure_type": str(enclosure_object.enclosure_type),
        "adjacent_rock": int(enclosure_object.adjacent_rock),
        "adjacent_water": int(enclosure_object.adjacent_water),
        "animals_inside": int(enclosure_object.animals_inside),
        "origin": [int(enclosure_object.origin[0]), int(enclosure_object.origin[1])],
        "rotation": str(enclosure_object.rotation),
    }


def _serialize_sponsor_building_public(sponsor_building: SponsorBuilding) -> Dict[str, Any]:
    return {
        "sponsor_number": int(sponsor_building.sponsor_number),
        "label": str(sponsor_building.label),
        "cells": [
            [int(cell[0]), int(cell[1])]
            for cell in tuple(sponsor_building.cells)
        ],
    }


def _serialize_map_building_public(building: Building) -> Dict[str, Any]:
    return {
        "building_type": str(building.type.name),
        "subtype": str(building.type.subtype.value),
        "origin": [int(building.origin_hex.x), int(building.origin_hex.y)],
        "rotation": str(building.rotation.name),
        "empty_spaces": int(building.empty_spaces),
        "layout": [
            [int(tile.x), int(tile.y)]
            for tile in sorted(
                building.layout,
                key=lambda item: (int(item.x), int(item.y)),
            )
        ],
    }


_PENDING_PAYLOAD_PUBLIC_KEYS_BY_KIND: Dict[str, Tuple[str, ...]] = {
    "cards_discard": (
        "discard_target",
        "break_triggered",
        "consumed_venom",
    ),
    "break_discard": (
        "discard_target",
        "break_hand_limit_index",
        "resume_turn_player_id",
        "resume_turn_consumed_venom",
    ),
    "opening_draft_keep": (
        "keep_target",
    ),
    "conservation_reward": (
        "threshold",
        "resume_kind",
        "break_triggered",
        "consumed_venom",
        "break_income_index",
        "resume_turn_player_id",
        "resume_turn_consumed_venom",
    ),
}


def _serialize_public_pending_payload(kind: str, payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    kind_key = str(kind or "").strip()
    allowed_keys = _PENDING_PAYLOAD_PUBLIC_KEYS_BY_KIND.get(kind_key, ())
    if not allowed_keys:
        return {}
    serialized: Dict[str, Any] = {}
    for key in allowed_keys:
        if key not in payload:
            continue
        value = payload.get(key)
        if isinstance(value, bool):
            serialized[str(key)] = bool(value)
            continue
        if isinstance(value, int):
            serialized[str(key)] = int(value)
            continue
        if isinstance(value, float):
            serialized[str(key)] = int(value)
            continue
        if isinstance(value, str):
            serialized[str(key)] = str(value)
            continue
        if value is None:
            serialized[str(key)] = None
    if kind_key == "opening_draft_keep":
        draft_ids = payload.get("draft_card_instance_ids")
        if isinstance(draft_ids, list):
            serialized["draft_card_count"] = len(draft_ids)
    return serialized


def _build_public_player_observation(player: PlayerState, *, player_id: int) -> Dict[str, Any]:
    action_upgraded = {
        action: bool(player.action_upgraded.get(action, False))
        for action in MAIN_ACTION_CARDS
    }
    association_workers = {
        task: int(player.association_workers_by_task.get(task, 0))
        for task in ASSOCIATION_TASK_KINDS
    }
    multiplier_tokens_on_actions = {
        action: int(player.multiplier_tokens_on_actions.get(action, 0))
        for action in MAIN_ACTION_CARDS
    }
    venom_tokens_on_actions = {
        action: int(player.venom_tokens_on_actions.get(action, 0))
        for action in MAIN_ACTION_CARDS
    }
    constriction_tokens_on_actions = {
        action: int(player.constriction_tokens_on_actions.get(action, 0))
        for action in MAIN_ACTION_CARDS
    }
    extra_actions_granted = {
        action: int(player.extra_actions_granted.get(action, 0))
        for action in MAIN_ACTION_CARDS
    }
    extra_strength_actions = [
        {
            "strength": int(strength),
            "count": int(count),
        }
        for strength, count in sorted(
            player.extra_strength_actions.items(),
            key=lambda item: int(item[0]),
        )
    ]
    sponsor_tokens_by_number = [
        {
            "number": int(number),
            "count": int(count),
        }
        for number, count in sorted(
            player.sponsor_tokens_by_number.items(),
            key=lambda item: int(item[0]),
        )
    ]
    enclosure_objects = sorted(
        player.enclosure_objects,
        key=lambda item: (
            int(item.origin[0]),
            int(item.origin[1]),
            int(item.size),
            str(item.rotation),
            str(item.enclosure_type),
        ),
    )
    sponsor_buildings = sorted(
        player.sponsor_buildings,
        key=lambda item: (
            int(item.sponsor_number),
            tuple((int(x), int(y)) for x, y in tuple(item.cells)),
            str(item.label),
        ),
    )
    zoo_map_grid: List[List[int]] = []
    zoo_map_buildings: List[Dict[str, Any]] = []
    if player.zoo_map is not None:
        zoo_map_grid = [
            [int(tile.x), int(tile.y)]
            for tile in sorted(
                player.zoo_map.grid,
                key=lambda item: (int(item.x), int(item.y)),
            )
        ]
        map_buildings = sorted(
            player.zoo_map.buildings.values(),
            key=lambda item: (
                str(item.type.name),
                int(item.origin_hex.x),
                int(item.origin_hex.y),
                str(item.rotation.name),
                int(item.empty_spaces),
                tuple((int(tile.x), int(tile.y)) for tile in item.layout),
            ),
        )
        zoo_map_buildings = [
            _serialize_map_building_public(building)
            for building in map_buildings
        ]
    return {
        "player_id": int(player_id),
        "name": str(player.name),
        "money": int(player.money),
        "appeal": int(player.appeal),
        "conservation": int(player.conservation),
        "reputation": int(player.reputation),
        "workers": int(player.workers),
        "x_tokens": int(player.x_tokens),
        "hand_limit": int(player.hand_limit),
        "hand_count": len(player.hand),
        "final_scoring_count": len(player.final_scoring_cards),
        "action_order": [str(action) for action in list(player.action_order)],
        "action_upgraded": action_upgraded,
        "workers_on_association_board": int(player.workers_on_association_board),
        "association_workers_by_task": association_workers,
        "multiplier_tokens_on_actions": multiplier_tokens_on_actions,
        "venom_tokens_on_actions": venom_tokens_on_actions,
        "constriction_tokens_on_actions": constriction_tokens_on_actions,
        "extra_actions_granted": extra_actions_granted,
        "extra_any_actions": int(player.extra_any_actions),
        "extra_strength_actions": extra_strength_actions,
        "camouflage_condition_ignores": int(player.camouflage_condition_ignores),
        "sponsor_tokens_by_number": sponsor_tokens_by_number,
        "sponsor_waza_assignment_mode": str(player.sponsor_waza_assignment_mode),
        "sponsor_ignore_large_condition_charges": int(player.sponsor_ignore_large_condition_charges),
        "partner_zoos": sorted(str(item) for item in player.partner_zoos),
        "universities": sorted(str(item) for item in player.universities),
        "supported_conservation_projects": sorted(
            str(item) for item in player.supported_conservation_projects
        ),
        "supported_conservation_project_actions": int(
            player.supported_conservation_project_actions
        ),
        "claimed_conservation_reward_spaces": sorted(
            int(item) for item in player.claimed_conservation_reward_spaces
        ),
        "claimed_reputation_milestones": sorted(
            int(item) for item in player.claimed_reputation_milestones
        ),
        "map_left_track_unlocked_count": int(player.map_left_track_unlocked_count),
        "map_left_track_claimed_indices": sorted(
            int(item) for item in player.map_left_track_claimed_indices
        ),
        "map_left_track_unlocked_effects": [
            str(item) for item in list(player.map_left_track_unlocked_effects)
        ],
        "claimed_partner_zoo_thresholds": sorted(
            int(item) for item in player.claimed_partner_zoo_thresholds
        ),
        "claimed_university_thresholds": sorted(
            int(item) for item in player.claimed_university_thresholds
        ),
        "map_completion_reward_claimed": bool(player.map_completion_reward_claimed),
        "zoo_cards": [
            _serialize_animal_card_public(card)
            for card in list(player.zoo_cards)
        ],
        "zoo_cards_count": len(player.zoo_cards),
        "pouched_cards_count": len(player.pouched_cards),
        "enclosures": [
            _serialize_enclosure_public(enclosure)
            for enclosure in list(player.enclosures)
        ],
        "enclosure_objects": [
            _serialize_enclosure_object_public(item)
            for item in enclosure_objects
        ],
        "sponsor_buildings": [
            _serialize_sponsor_building_public(item)
            for item in sponsor_buildings
        ],
        "zoo_map_present": bool(player.zoo_map is not None),
        "zoo_map_grid_count": len(zoo_map_grid),
        "zoo_map_grid": zoo_map_grid,
        "zoo_map_building_count": len(zoo_map_buildings),
        "zoo_map_buildings": zoo_map_buildings,
    }


def _build_private_player_observation(player: PlayerState) -> Dict[str, Any]:
    zoo_card_public_indices = {
        str(card.instance_id): idx
        for idx, card in enumerate(list(player.zoo_cards))
        if str(card.instance_id).strip()
    }
    zoo_card_numbers_by_instance = {
        str(card.instance_id): int(card.number)
        for card in list(player.zoo_cards)
        if str(card.instance_id).strip()
    }
    pouched_cards_by_host = {
        str(host): [str(card.instance_id) for card in list(cards)]
        for host, cards in sorted(
            player.pouched_cards_by_host.items(),
            key=lambda item: str(item[0]),
        )
    }
    pouched_cards_by_host_cards = {
        str(host): [
            _serialize_animal_card_private(card)
            for card in list(cards)
        ]
        for host, cards in sorted(
            player.pouched_cards_by_host.items(),
            key=lambda item: str(item[0]),
        )
    }
    pouched_cards_by_host_public: List[Dict[str, Any]] = []
    pouched_host_public_counts = [0 for _ in range(len(player.zoo_cards))]
    for host, cards in sorted(
        player.pouched_cards_by_host.items(),
        key=lambda item: str(item[0]),
    ):
        host_key = str(host)
        host_public_index = zoo_card_public_indices.get(host_key)
        if host_public_index is not None:
            pouched_host_public_counts[int(host_public_index)] += len(list(cards))
        pouched_cards_by_host_public.append(
            {
                "host_public_index": (
                    None
                    if host_public_index is None
                    else int(host_public_index)
                ),
                "host_number": (
                    None
                    if host_key not in zoo_card_numbers_by_instance
                    else int(zoo_card_numbers_by_instance[host_key])
                ),
                "cards": [
                    _serialize_animal_card_private(card)
                    for card in list(cards)
                ],
            }
        )
    return {
        "hand_count": len(player.hand),
        "hand": [
            _serialize_animal_card_private(card)
            for card in list(player.hand)
        ],
        "final_scoring_count": len(player.final_scoring_cards),
        "final_scoring_cards": [
            _serialize_setup_card_ref(card)
            for card in list(player.final_scoring_cards)
        ],
        "opening_draft_drawn_count": len(player.opening_draft_drawn),
        "opening_draft_drawn": [
            _serialize_animal_card_private(card)
            for card in list(player.opening_draft_drawn)
        ],
        "opening_draft_kept_indices": [
            int(idx) for idx in list(player.opening_draft_kept_indices)
        ],
        "pouched_cards": [
            _serialize_animal_card_private(card)
            for card in list(player.pouched_cards)
        ],
        "zoo_cards_public_count": len(player.zoo_cards),
        "pouched_cards_by_host": pouched_cards_by_host,
        "pouched_cards_by_host_cards": pouched_cards_by_host_cards,
        "pouched_cards_by_host_public": pouched_cards_by_host_public,
        "pouched_host_public_counts": [
            int(count)
            for count in pouched_host_public_counts
        ],
        "legacy_private_deck_count": len(player.deck),
        "legacy_private_discard_count": len(player.discard),
    }


def build_public_observation(state: GameState, *, viewer_player_id: int) -> Dict[str, Any]:
    _validate_observation_viewer_id(state, viewer_player_id)
    conservation_project_slots: Dict[str, Dict[str, Optional[int]]] = {}
    for project_id in sorted(state.conservation_project_slots):
        levels_raw = state.conservation_project_slots.get(project_id) or {}
        levels_snapshot: Dict[str, Optional[int]] = {}
        for level_name in sorted(levels_raw):
            owner = levels_raw[level_name]
            levels_snapshot[str(level_name)] = None if owner is None else int(owner)
        conservation_project_slots[str(project_id)] = levels_snapshot

    opening_setup = state.opening_setup
    opening_setup_snapshot = {
        "conservation_space_2_fixed_options": [
            str(option)
            for option in list(opening_setup.conservation_space_2_fixed_options)
        ],
        "conservation_space_5_bonus_tiles": [
            str(tile)
            for tile in list(opening_setup.conservation_space_5_bonus_tiles)
        ],
        "conservation_space_8_bonus_tiles": [
            str(tile)
            for tile in list(opening_setup.conservation_space_8_bonus_tiles)
        ],
        "conservation_space_10_rule": str(opening_setup.conservation_space_10_rule),
        "base_conservation_projects": [
            _serialize_setup_card_ref(project)
            for project in list(opening_setup.base_conservation_projects)
        ],
        "two_player_blocked_project_levels": [
            {
                "project_data_id": str(item.project_data_id),
                "project_title": str(item.project_title),
                "blocked_level": str(item.blocked_level),
            }
            for item in list(opening_setup.two_player_blocked_project_levels)
        ],
    }

    return {
        "viewer_player_id": int(viewer_player_id),
        "map_image_name": str(state.map_image_name),
        "player_count": len(state.players),
        "current_player": int(state.current_player),
        "turn_index": int(state.turn_index),
        "break_progress": int(state.break_progress),
        "break_max": int(state.break_max),
        "break_trigger_player": (
            None if state.break_trigger_player is None else int(state.break_trigger_player)
        ),
        "donation_progress": int(state.donation_progress),
        "pending_decision_kind": str(state.pending_decision_kind),
        "pending_decision_player_id": (
            None
            if state.pending_decision_player_id is None
            else int(state.pending_decision_player_id)
        ),
        "pending_decision_payload_public": _serialize_public_pending_payload(
            state.pending_decision_kind,
            state.pending_decision_payload,
        ),
        "endgame_trigger_player": (
            None if state.endgame_trigger_player is None else int(state.endgame_trigger_player)
        ),
        "endgame_trigger_turn_index": (
            None
            if state.endgame_trigger_turn_index is None
            else int(state.endgame_trigger_turn_index)
        ),
        "forced_game_over": bool(state.forced_game_over),
        "forced_game_over_reason": str(state.forced_game_over_reason),
        "zoo_deck_count": len(state.zoo_deck),
        "zoo_discard_count": len(state.zoo_discard),
        "zoo_display": [
            _serialize_animal_card_public(card)
            for card in list(state.zoo_display)
        ],
        "final_scoring_deck_count": len(state.final_scoring_deck),
        "final_scoring_discard_count": len(state.final_scoring_discard),
        "unused_base_conservation_projects_count": len(state.unused_base_conservation_projects),
        "available_partner_zoos": sorted(str(item) for item in state.available_partner_zoos),
        "available_universities": sorted(str(item) for item in state.available_universities),
        "shared_conservation_bonus_tiles": {
            int(threshold): [str(tile) for tile in list(tiles)]
            for threshold, tiles in sorted(state.shared_conservation_bonus_tiles.items())
        },
        "claimed_conservation_bonus_tiles": {
            int(threshold): [str(tile) for tile in list(tiles)]
            for threshold, tiles in sorted(state.claimed_conservation_bonus_tiles.items())
        },
        "conservation_project_slots": conservation_project_slots,
        "opening_setup": opening_setup_snapshot,
        "players": [
            _build_public_player_observation(player, player_id=player_id)
            for player_id, player in enumerate(state.players)
        ],
    }


def build_private_observation(state: GameState, *, viewer_player_id: int) -> Dict[str, Any]:
    _validate_observation_viewer_id(state, viewer_player_id)
    player = state.players[int(viewer_player_id)]
    return {
        "viewer_player_id": int(viewer_player_id),
        "player": _build_private_player_observation(player),
    }


def build_player_observation(state: GameState, *, viewer_player_id: int) -> Dict[str, Any]:
    return {
        "public": build_public_observation(state, viewer_player_id=viewer_player_id),
        "private": build_private_observation(state, viewer_player_id=viewer_player_id),
    }


def build_deck() -> List[AnimalCard]:
    cards = _load_animal_cards_from_dataset()
    if cards:
        return list(cards)
    dataset_path = _resolve_cards_dataset_path_impl()
    raise RuntimeError(
        f"Card dataset is missing or empty: {dataset_path}. "
        "Please prepare data/cards/cards.base.json before starting the game."
    )


def _canonical_icon_key(raw_name: str) -> str:
    key = raw_name.strip().lower()
    aliases = {
        "americas": "america",
        "oceania": "australia",
        "south america": "america",
        "north america": "america",
        "primates": "primate",
        "reptiles": "reptile",
        "birds": "bird",
        "predators": "predator",
        "herbivores": "herbivore",
    }
    return aliases.get(key, key)


def _sponsor_level(card: AnimalCard) -> int:
    if card.number in SPONSOR_LEVEL_OVERRIDES:
        return SPONSOR_LEVEL_OVERRIDES[card.number]
    return max(0, int(card.cost))


def _sponsor_play_cost(card: AnimalCard) -> int:
    return 0


def _sponsor_requires_upgraded(card: AnimalCard) -> bool:
    if card.number in SPONSOR_REQUIRES_UPGRADED_OVERRIDES:
        return True
    return any(_canonical_icon_key(icon) == "sponsorsii" and int(need) > 0 for icon, need in card.required_icons)


def _sponsor_min_reputation(card: AnimalCard) -> int:
    if card.number in SPONSOR_MIN_REPUTATION_OVERRIDES:
        return SPONSOR_MIN_REPUTATION_OVERRIDES[card.number]
    for icon, need in card.required_icons:
        if _canonical_icon_key(icon) == "reputation":
            return max(0, int(need))
    return 0


def _sponsor_max_appeal(card: AnimalCard) -> Optional[int]:
    if card.number in SPONSOR_MAX_APPEAL_25_OVERRIDES:
        return 25
    return None


def _sponsor_required_icons(card: AnimalCard) -> Dict[str, int]:
    merged: Dict[str, int] = {}
    for icon, need in card.required_icons:
        key = _canonical_icon_key(icon)
        if key in {"sponsorsii", "reputation", "appeal"}:
            continue
        merged[key] = max(merged.get(key, 0), int(need))
    for icon, need in SPONSOR_REQUIRED_ICON_OVERRIDES.get(card.number, ()):
        key = _canonical_icon_key(icon)
        merged[key] = max(merged.get(key, 0), int(need))
    return merged


def _card_badges_for_icons(card: AnimalCard) -> Tuple[str, ...]:
    if card.card_type == "sponsor" and card.number in SPONSOR_BADGE_OVERRIDES:
        return tuple(SPONSOR_BADGE_OVERRIDES[card.number])
    return tuple(card.badges)


def _card_icon_counts(card: AnimalCard) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for badge in _card_badges_for_icons(card):
        normalized = _canonical_icon_key(badge)
        if not normalized:
            continue
        counts[normalized] = counts.get(normalized, 0) + 1
    return counts


def _card_icon_keys(card: AnimalCard) -> Set[str]:
    return set(_card_icon_counts(card).keys())


def _is_small_animal(card: AnimalCard) -> bool:
    return card.card_type == "animal" and int(card.size) <= 2


def _is_large_animal(card: AnimalCard) -> bool:
    return card.card_type == "animal" and int(card.size) >= 4


def _partner_zoo_label(partner: str) -> str:
    normalized = _canonical_icon_key(partner)
    return CONTINENT_BADGE_ALIASES.get(normalized, normalized.title())


def _university_label(university: str) -> str:
    if university == "reputation_1_hand_limit_5":
        return "University(hand_limit=5, +1 reputation)"
    if university == "science_1_reputation_2":
        return "University(+1 science, +2 reputation)"
    if university == "science_2":
        return "University(+2 science)"
    return university


def _refresh_association_market(state: GameState) -> None:
    shared_partner_zoos = set(ALL_PARTNER_ZOOS)
    for player in state.players:
        shared_partner_zoos &= set(player.partner_zoos)
    state.available_partner_zoos = set(ALL_PARTNER_ZOOS) - shared_partner_zoos

    shared_universities = set(UNIVERSITY_TYPES)
    for player in state.players:
        shared_universities &= set(player.universities)
    state.available_universities = set(UNIVERSITY_TYPES) - shared_universities


def _map_rule_left_track_unlocks(state: GameState) -> List[Dict[str, Any]]:
    items = state.map_rules.get("left_track_unlocks", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def _map_rule_partner_thresholds(state: GameState) -> List[Dict[str, Any]]:
    items = state.map_rules.get("partner_zoo_threshold_rewards", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def _map_rule_university_thresholds(state: GameState) -> List[Dict[str, Any]]:
    items = state.map_rules.get("university_threshold_rewards", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def _map_rule_worker_gain_rewards(state: GameState) -> List[Dict[str, Any]]:
    items = state.map_rules.get("worker_gain_rewards", [])
    if not isinstance(items, list):
        return []
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
    return out


def _current_donation_cost(state: GameState) -> int:
    idx = min(state.donation_progress, len(DONATION_COST_TRACK) - 1)
    return DONATION_COST_TRACK[idx]


def _association_task_strength_cost(player: PlayerState, task_kind: str) -> int:
    if task_kind == "reputation":
        return 2
    if task_kind == "partner_zoo":
        return 3
    if task_kind == "university":
        return 4
    if task_kind == "conservation_project":
        return 4 if _player_has_sponsor(player, 203) else 5
    raise ValueError(f"Unsupported association task kind '{task_kind}'.")


def _association_workers_needed(player: PlayerState, task_kind: str) -> int:
    placed = int(player.association_workers_by_task.get(task_kind, 0))
    if placed >= MAX_ASSOCIATION_WORKERS_PER_TASK:
        raise ValueError("This association task cannot be used again before break.")
    if placed == 0:
        return 1
    return 2


def _spend_association_workers(player: PlayerState, task_kind: str, workers_needed: int) -> None:
    if workers_needed <= 0:
        raise ValueError("Association workers spent must be positive.")
    if player.workers < workers_needed:
        raise ValueError("Not enough active association workers.")
    player.workers -= workers_needed
    player.workers_on_association_board += workers_needed
    player.association_workers_by_task[task_kind] = int(player.association_workers_by_task.get(task_kind, 0)) + workers_needed


def _resolve_cards_dataset_path() -> Path:
    return _resolve_cards_dataset_path_impl()


def _cards_data_dir() -> Path:
    return Path(__file__).resolve().parent / "data" / "cards"


def _marine_world_cards_dataset_path() -> Path:
    preferred = _cards_data_dir() / "cards.marine_world.json"
    if preferred.exists():
        return preferred
    return _cards_data_dir() / "cards.json"


@lru_cache(maxsize=1)
def _load_animal_cards_from_dataset() -> Tuple[AnimalCard, ...]:
    return _load_animal_cards_from_dataset_impl(
        card_factory=AnimalCard,
        allowed_card_types=ZOO_DECK_CARD_TYPES,
        canonical_icon_key=_canonical_icon_key,
    )


def _build_map_tile_bonus_map(map_image_name: str) -> Dict[Tuple[int, int], str]:
    return _build_map_tile_bonus_map_impl(map_image_name=map_image_name)


@lru_cache(maxsize=64)
def _load_map_tiles_payload(map_image_name: str) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent
    path = repo_root / "data" / "maps" / "tiles" / f"{map_image_name}.tiles.json"
    if not path.exists():
        raise FileNotFoundError(f"Map tiles file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Map tiles payload must be object: {path}")
    return payload


def _load_map_rules(map_image_name: str) -> Dict[str, Any]:
    payload = _load_map_tiles_payload(map_image_name)
    rules = payload.get("map_rules", {})
    if not isinstance(rules, dict):
        return {}
    return dict(rules)


def _build_map_tile_tags_map(map_image_name: str) -> Dict[Tuple[int, int], List[str]]:
    payload = _load_map_tiles_payload(map_image_name)
    raw_tiles = payload.get("tiles", [])
    if not isinstance(raw_tiles, list):
        return {}
    tags: Dict[Tuple[int, int], List[str]] = {}
    for raw_tile in raw_tiles:
        if not isinstance(raw_tile, dict):
            continue
        x = raw_tile.get("x")
        y = raw_tile.get("y")
        raw_tags = raw_tile.get("tags", [])
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        if not isinstance(raw_tags, list):
            continue
        values = [str(tag).strip().lower() for tag in raw_tags if str(tag).strip()]
        if values:
            tags[(x, y)] = values
    return tags


def _project_number_from_data_id(data_id: str) -> int:
    match = re.search(r"P(\d+)_", data_id)
    if not match:
        return -1
    return int(match.group(1))


def _make_conservation_project_hand_card(project: SetupCardRef, instance_id: str) -> AnimalCard:
    return AnimalCard(
        name=project.title,
        cost=0,
        size=0,
        appeal=0,
        conservation=0,
        card_type="conservation_project",
        number=_project_number_from_data_id(project.data_id),
        instance_id=instance_id,
    )


@lru_cache(maxsize=1)
def _load_conservation_project_catalog() -> Dict[int, SetupCardRef]:
    path = _resolve_cards_dataset_path()
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    cards = payload.get("cards", [])
    if not isinstance(cards, list):
        return {}
    catalog: Dict[int, SetupCardRef] = {}
    for card in cards:
        if not isinstance(card, dict):
            continue
        if str(card.get("type") or "").strip() != "conservation_project":
            continue
        number = card.get("number")
        if not isinstance(number, int):
            continue
        data_id = str(card.get("data_id") or "").strip()
        if not data_id:
            continue
        title = str(card.get("title") or card.get("subtitle") or data_id).strip()
        catalog[number] = SetupCardRef(data_id=data_id, title=title)
    return catalog


def _project_ref_from_number(number: int, fallback_title: str = "") -> SetupCardRef:
    catalog = _load_conservation_project_catalog()
    project = catalog.get(int(number))
    if project is not None:
        return project
    safe_title = re.sub(r"[^A-Za-z0-9]+", "", fallback_title.strip()) or f"Project{int(number)}"
    return SetupCardRef(data_id=f"P{int(number):03d}_{safe_title}", title=fallback_title or safe_title)


def _project_ref_from_card(card: AnimalCard) -> SetupCardRef:
    if card.card_type != "conservation_project":
        raise ValueError("Card is not a conservation project.")
    return _project_ref_from_number(card.number, fallback_title=card.name)


def _is_base_conservation_project(project_id: str) -> bool:
    number = _project_number_from_data_id(project_id)
    return 101 <= number <= 112


def _project_level_rewards(project_id: str) -> Tuple[Tuple[str, int, int, int], ...]:
    number = _project_number_from_data_id(project_id)
    if number in PROJECT_LEVEL_REWARDS_BY_NUMBER:
        return PROJECT_LEVEL_REWARDS_BY_NUMBER[number]
    return tuple((level_name, needed_icons, conservation_gain, 0) for level_name, needed_icons, conservation_gain in BASE_CONSERVATION_PROJECT_LEVELS)


def _ensure_conservation_project_slots(state: GameState, project_id: str) -> None:
    if project_id not in state.conservation_project_slots:
        state.conservation_project_slots[project_id] = {
            "left_level": None,
            "middle_level": None,
            "right_level": None,
        }


def _opening_base_conservation_project_ids(state: GameState) -> List[str]:
    return [project.data_id for project in state.opening_setup.base_conservation_projects]


def _conservation_project_row_ids(state: GameState) -> List[str]:
    base_ids = set(_opening_base_conservation_project_ids(state))
    return [
        project_id
        for project_id in state.conservation_project_slots
        if project_id not in base_ids
    ]


def _set_conservation_project_slot_order(
    state: GameState,
    *,
    row_project_ids: Sequence[str],
) -> None:
    previous = state.conservation_project_slots
    reordered: Dict[str, Dict[str, Optional[int]]] = {}
    for project_id in _opening_base_conservation_project_ids(state):
        slots = previous.get(project_id)
        if slots is not None:
            reordered[project_id] = slots
    for project_id in row_project_ids:
        if project_id in reordered:
            continue
        slots = previous.get(project_id)
        if slots is not None:
            reordered[project_id] = slots
    state.conservation_project_slots = reordered


def _conservation_project_row_limit(state: GameState) -> int:
    return int(
        CONSERVATION_PROJECT_ROW_LIMIT_BY_PLAYERS.get(
            len(state.players),
            4,
        )
    )


def _project_card_for_discard_from_id(
    *,
    project_id: str,
    turn_index: int,
    discard_index: int,
) -> AnimalCard:
    number = _project_number_from_data_id(project_id)
    project_ref = _project_ref_from_number(number, fallback_title=project_id)
    return _make_conservation_project_hand_card(
        project_ref,
        instance_id=f"discard-{project_id}-{turn_index}-{discard_index}",
    )


def _on_new_conservation_project_added_to_row(
    state: GameState,
    *,
    project_id: str,
) -> None:
    base_ids = set(_opening_base_conservation_project_ids(state))
    if project_id in base_ids:
        return

    existing_row_ids = [pid for pid in _conservation_project_row_ids(state) if pid != project_id]
    row_ids = [project_id] + existing_row_ids
    limit = _conservation_project_row_limit(state)
    kept_row_ids = row_ids[:limit]
    discarded_row_ids = row_ids[limit:]
    previous_slots = state.conservation_project_slots
    discarded_with_slots = [
        (discarded_id, dict(previous_slots.get(discarded_id) or {}))
        for discarded_id in discarded_row_ids
    ]
    _set_conservation_project_slot_order(state, row_project_ids=kept_row_ids)
    for discarded_id, slot_owners in discarded_with_slots:
        token_count = 0
        for owner in slot_owners.values():
            if owner is None:
                continue
            token_count += 1
        discard_card = _project_card_for_discard_from_id(
            project_id=discarded_id,
            turn_index=int(state.turn_index),
            discard_index=len(state.zoo_discard),
        )
        state.zoo_discard.append(discard_card)
        state.effect_log.append(
            f"conservation_project_row_discard:{discarded_id}:returned_tokens={token_count}"
        )


def _current_conservation_projects(state: GameState) -> List[SetupCardRef]:
    current: List[SetupCardRef] = []
    seen: Set[str] = set()
    for project in state.opening_setup.base_conservation_projects:
        if project.data_id in seen:
            continue
        current.append(project)
        seen.add(project.data_id)
    catalog_by_number = _load_conservation_project_catalog()
    catalog_by_id = {item.data_id: item for item in catalog_by_number.values()}
    for project_id in state.conservation_project_slots:
        if project_id in seen:
            continue
        project = catalog_by_id.get(project_id)
        if project is None:
            number = _project_number_from_data_id(project_id)
            title = project_id
            project = _project_ref_from_number(number, fallback_title=title)
        current.append(project)
        seen.add(project_id)
    return current


@lru_cache(maxsize=1)
def _load_base_setup_card_pools() -> Tuple[List[SetupCardRef], List[SetupCardRef]]:
    final_cards, base_projects = _load_base_setup_card_pools_impl(
        setup_card_factory=SetupCardRef,
    )
    return list(final_cards), list(base_projects)


def _build_opening_setup_info(rng: random.Random) -> OpeningSetupInfo:
    _, base_projects_pool = _load_base_setup_card_pools()
    return _build_opening_setup_info_impl(
        rng=rng,
        base_projects_pool=base_projects_pool,
        conservation_bonus_tile_pool=CONSERVATION_BONUS_TILE_POOL,
        conservation_space_2_fixed_options=CONSERVATION_SPACE_2_FIXED_OPTIONS,
        conservation_space_10_rule=CONSERVATION_SPACE_10_RULE,
        two_player_blocked_levels=TWO_PLAYER_BLOCKED_LEVELS,
        opening_setup_info_factory=OpeningSetupInfo,
        blocked_project_level_factory=TwoPlayerBlockedProjectLevel,
    )


@lru_cache(maxsize=2)
def _load_final_scoring_card_pool(include_marine_world: bool) -> Tuple[SetupCardRef, ...]:
    base_final_cards, _ = _load_base_setup_card_pools()
    if not include_marine_world:
        return tuple(base_final_cards)

    marine_final_cards = _load_final_scoring_cards_from_dataset_impl(
        setup_card_factory=SetupCardRef,
        dataset_path=_marine_world_cards_dataset_path(),
    )
    merged: Dict[str, SetupCardRef] = {card.data_id: card for card in base_final_cards}
    for card in marine_final_cards:
        merged[card.data_id] = card
    return tuple(sorted(merged.values(), key=lambda card: card.data_id))


def _draw_final_scoring_cards_for_players(
    players: List[PlayerState],
    rng: random.Random,
    *,
    include_marine_world: bool = False,
) -> List[SetupCardRef]:
    final_pool = list(_load_final_scoring_card_pool(include_marine_world))
    return _draw_final_scoring_cards_for_players_impl(
        players=players,
        rng=rng,
        final_pool=final_pool,
    )


def draw_cards(player: PlayerState, n: int = 1) -> None:
    # Legacy helper for older per-player deck tests; shared zoo deck is now primary.
    for _ in range(n):
        if not player.deck:
            break
        player.hand.append(player.deck.pop())


def _draw_from_zoo_deck(state: GameState, n: int) -> List[AnimalCard]:
    if n <= 0:
        return []
    if n > len(state.zoo_deck):
        _trigger_immediate_game_end(
            state,
            reason=f"deck_shortage(requested={int(n)},remaining={len(state.zoo_deck)})",
        )
        return []
    drawn: List[AnimalCard] = []
    for _ in range(n):
        if not state.zoo_deck:
            break
        drawn.append(state.zoo_deck.pop(0))
    return drawn


def _replenish_zoo_display(state: GameState) -> None:
    while len(state.zoo_display) < 6 and state.zoo_deck:
        state.zoo_display.append(state.zoo_deck.pop(0))


def _draw_opening_draft_cards(player: PlayerState, state: GameState) -> List[AnimalCard]:
    return _draw_opening_draft_cards_impl(
        player=player,
        state=state,
        draw_from_zoo_deck=_draw_from_zoo_deck,
    )


def _apply_opening_draft_selection(
    player: PlayerState,
    drafted_cards: List[AnimalCard],
    rng: random.Random,
    kept_indices: Optional[Sequence[int]] = None,
    discard_sink: Optional[List[AnimalCard]] = None,
) -> None:
    _apply_opening_draft_selection_impl(
        player=player,
        drafted_cards=drafted_cards,
        rng=rng,
        kept_indices=kept_indices,
        discard_sink=discard_sink,
    )


def _draft_opening_hand(
    player: PlayerState,
    state: GameState,
    rng: random.Random,
    kept_indices: Optional[Sequence[int]] = None,
) -> None:
    drafted_cards = _draw_opening_draft_cards(player, state)
    _apply_opening_draft_selection(
        player=player,
        drafted_cards=drafted_cards,
        rng=rng,
        kept_indices=kept_indices,
        discard_sink=state.zoo_discard,
    )


def _next_unresolved_opening_draft_player_id(state: GameState) -> Optional[int]:
    for idx, player in enumerate(state.players):
        if player.opening_draft_kept_indices:
            continue
        if not player.opening_draft_drawn:
            continue
        if player.hand:
            continue
        return idx
    return None


def _begin_next_opening_draft_pending_if_needed(state: GameState) -> bool:
    player_id = _next_unresolved_opening_draft_player_id(state)
    if player_id is None:
        return False
    player = state.players[player_id]
    keep_target = min(4, len(player.opening_draft_drawn))
    if keep_target <= 0:
        return False
    _set_pending_decision(
        state,
        kind="opening_draft_keep",
        player_id=player_id,
        payload={
            "keep_target": keep_target,
            "draft_card_instance_ids": [card.instance_id for card in player.opening_draft_drawn],
        },
    )
    return True


def enclosure_build_cost(size: int) -> int:
    return size * 2 + 1


def _action_token_total(token_map: Dict[str, int]) -> int:
    return sum(max(0, int(count)) for count in token_map.values())


def _action_token_line(token_map: Dict[str, int]) -> str:
    return ", ".join(
        f"{action}:{count}" for action, count in token_map.items() if int(count) > 0
    ) or "-"


def _action_base_strength(player: PlayerState, card_name: str) -> int:
    if card_name not in player.action_order:
        raise ValueError(f"Unknown action card '{card_name}'.")
    return player.action_order.index(card_name) + 1


def _constriction_penalty(player: PlayerState, card_name: str) -> int:
    if int(player.constriction_tokens_on_actions.get(card_name, 0)) <= 0:
        return 0
    return 2


def _effective_action_strength(
    player: PlayerState,
    card_name: str,
    *,
    x_spent: int = 0,
    base_strength: Optional[int] = None,
) -> int:
    raw_base = int(base_strength) if base_strength is not None else _action_base_strength(player, card_name)
    return max(0, raw_base + max(0, int(x_spent)) - _constriction_penalty(player, card_name))


def _apply_action_card_token_use(player: PlayerState, card_name: str) -> Tuple[bool, bool]:
    consumed_venom = False
    consumed_constriction = False
    if int(player.venom_tokens_on_actions.get(card_name, 0)) > 0:
        player.venom_tokens_on_actions[card_name] = 0
        consumed_venom = True
    if int(player.constriction_tokens_on_actions.get(card_name, 0)) > 0:
        player.constriction_tokens_on_actions[card_name] = 0
        consumed_constriction = True
    return consumed_venom, consumed_constriction


def _apply_end_of_turn_venom_penalty(player: PlayerState, *, consumed_venom: bool) -> bool:
    if consumed_venom:
        return False
    if _action_token_total(player.venom_tokens_on_actions) <= 0:
        return False
    before = int(player.money)
    player.money = max(0, before - 2)
    return player.money != before


def _eligible_highest_track_target_ids(
    state: GameState,
    player_id: int,
    *,
    value_fn,
) -> List[int]:
    actor = state.players[player_id]
    other_ids = [idx for idx in range(len(state.players)) if idx != player_id]
    if not other_ids:
        return []
    highest_value = max([int(value_fn(actor))] + [int(value_fn(state.players[idx])) for idx in other_ids])
    eligible = [
        idx
        for idx in other_ids
        if int(value_fn(state.players[idx])) == highest_value and not _player_has_sponsor(state.players[idx], 225)
    ]
    if not eligible:
        return []
    if int(value_fn(actor)) == highest_value and not any(
        int(value_fn(state.players[idx])) == highest_value for idx in other_ids
    ):
        return []
    return eligible


def _eligible_hypnosis_target_ids(state: GameState, player_id: int) -> List[int]:
    return _eligible_highest_track_target_ids(
        state,
        player_id,
        value_fn=lambda player: int(player.appeal),
    )


def _pilfering_target_groups(state: GameState, player_id: int, amount: int) -> List[Dict[str, Any]]:
    if amount <= 0:
        return []
    groups: List[Dict[str, Any]] = []
    appeal_targets = _eligible_highest_track_target_ids(
        state,
        player_id,
        value_fn=lambda player: int(player.appeal),
    )
    if appeal_targets:
        groups.append({"track": "appeal", "target_ids": appeal_targets})
    if amount >= 2:
        conservation_targets = _eligible_highest_track_target_ids(
            state,
            player_id,
            value_fn=lambda player: int(player.conservation),
        )
        if conservation_targets:
            groups.append({"track": "conservation", "target_ids": conservation_targets})
    return groups


def _prompt_choose_target_player_for_human(
    state: GameState,
    eligible_target_ids: Sequence[int],
    *,
    effect_name: str,
) -> int:
    if not eligible_target_ids:
        raise ValueError(f"{effect_name} has no legal target players.")
    if len(eligible_target_ids) == 1:
        return int(eligible_target_ids[0])
    print(f"{effect_name}: choose target player.")
    for idx, target_id in enumerate(eligible_target_ids, start=1):
        target_player = state.players[target_id]
        print(f"{idx}. {target_player.name}")
    while True:
        raw = input(f"Select target [1-{len(eligible_target_ids)}]: ").strip()
        if raw.isdigit():
            picked = int(raw)
            if 1 <= picked <= len(eligible_target_ids):
                return int(eligible_target_ids[picked - 1])
        print("Please enter a valid number.")


def _resolve_target_player_id_from_payload(
    state: GameState,
    eligible_target_ids: Sequence[int],
    *,
    effect_name: str,
    payload: Dict[str, Any],
) -> int:
    if len(eligible_target_ids) == 1:
        return int(eligible_target_ids[0])
    raw_candidates = [
        payload.get("target_player_id"),
        payload.get("target_player"),
        payload.get("target_player_name"),
    ]
    for raw in raw_candidates:
        if raw is None:
            continue
        if isinstance(raw, int) and raw in eligible_target_ids:
            return raw
        text = str(raw).strip()
        for target_id in eligible_target_ids:
            target_player = state.players[target_id]
            if text == str(target_id) or text.lower() == target_player.name.lower():
                return int(target_id)
    raise ValueError(f"{effect_name} requires explicit target player when multiple target players are legal.")


def _pop_queued_target_player_id(
    queue: List[Any],
    state: GameState,
    eligible_target_ids: Sequence[int],
    *,
    effect_name: str,
) -> int:
    while queue:
        candidate = queue.pop(0)
        payload = {"target_player": candidate}
        try:
            return _resolve_target_player_id_from_payload(
                state,
                eligible_target_ids,
                effect_name=effect_name,
                payload=payload,
            )
        except ValueError:
            continue
    raise ValueError(f"{effect_name} requires explicit target player when multiple target players are legal.")


def _pop_queued_action_name(
    queue: List[str],
    legal_actions: Sequence[str],
    *,
    effect_name: str,
) -> str:
    while queue:
        candidate = queue.pop(0)
        if candidate in legal_actions:
            return candidate
    raise ValueError(f"{effect_name} requires an explicit legal action choice.")


def _resolve_pilfering_choice_from_details(
    *,
    target_player: PlayerState,
    choice_payload: Dict[str, Any],
    can_take_money: bool,
    can_take_card: bool,
) -> Tuple[str, Optional[int]]:
    choice = str(choice_payload.get("choice") or "").strip().lower()
    if choice == "money":
        if not can_take_money:
            raise ValueError("Pilfering choice 'money' is not legal for this target player.")
        return "money", None
    if choice != "card":
        raise ValueError("Pilfering choice must be 'money' or 'card'.")
    if not can_take_card:
        raise ValueError("Pilfering choice 'card' is not legal for this target player.")
    return "card", None


def _make_concrete_action(
    template_action: Action,
    *,
    label: str = "",
    extra_details: Optional[Dict[str, Any]] = None,
    card_name: Optional[str] = None,
) -> Action:
    details = dict(template_action.details or {})
    if extra_details:
        details.update(extra_details)
    if label:
        details["action_label"] = label
    details["concrete"] = True
    return Action(
        template_action.type,
        value=template_action.value,
        card_name=card_name or template_action.card_name,
        details=details,
    )


def _action_to_slot_1_target_sequences(
    action_names: Sequence[str],
    count: int,
) -> List[List[str]]:
    if count <= 0:
        return [[]]
    return [list(choice) for choice in product(action_names, repeat=count)]


def _build_selection_payload(option: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "building_type": str(option.get("building_type") or ""),
        "cells": [list(cell) for cell in option.get("cells") or []],
    }


def _cells_text(cells: Sequence[Sequence[int] | Tuple[int, int]]) -> str:
    return "[" + ",".join(f"({int(x)},{int(y)})" for x, y in cells) + "]"


def _build_option_label(option: Dict[str, Any], bonus_targets: Sequence[str]) -> str:
    label = f"{option['building_label']} cells={_cells_text(option['cells'])}"
    if bonus_targets:
        label += f" slot1={','.join(str(target) for target in bonus_targets)}"
    return label


def _build_step_action_label(option: Dict[str, Any], bonus_label: str = "") -> str:
    label = _build_option_label(option, [])
    if bonus_label:
        label += f" ; {bonus_label}"
    return label


def _build_sequence_label(
    steps: Sequence[Tuple[Dict[str, Any], Sequence[str]]],
) -> str:
    return " ; then ".join(_build_option_label(option, bonus_targets) for option, bonus_targets in steps)


def _merge_list_detail_fragments(*fragments: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for fragment in fragments:
        for key, value in fragment.items():
            if isinstance(value, list):
                merged.setdefault(key, [])
                merged[key].extend(copy.deepcopy(value))
            else:
                merged[key] = copy.deepcopy(value)
    return merged


def _enumerate_build_bonus_choice_variants(
    state: GameState,
    player_id: int,
    option: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], str]]:
    bonuses = [str(bonus).strip() for bonus in option.get("placement_bonuses") or [] if str(bonus).strip()]
    if not bonuses:
        return [({}, "")]

    player = state.players[player_id]
    variants: List[Tuple[Dict[str, Any], List[str], List[AnimalCard], List[AnimalCard], List[str], int]] = [
        ({}, [], list(state.zoo_display), list(state.zoo_deck), list(player.action_order), int(player.reputation))
    ]
    for bonus in bonuses:
        next_variants: List[
            Tuple[Dict[str, Any], List[str], List[AnimalCard], List[AnimalCard], List[str], int]
        ] = []
        for details, labels, display_cards, deck_cards, action_order, reputation in variants:
            if bonus == "action_to_slot_1":
                for action_name in list(action_order):
                    next_variants.append(
                        (
                            _merge_list_detail_fragments(
                                details,
                                {"bonus_action_to_slot_1_choices": [{"action_name": action_name}]},
                            ),
                            list(labels) + [f"slot1={action_name}"],
                            list(display_cards),
                            list(deck_cards),
                            _move_action_order_to_slot(action_order, action_name, 1),
                            reputation,
                        )
                    )
                continue

            if bonus == "card_in_reputation_range":
                accessible = min(_reputation_display_limit(reputation), len(display_cards))
                for display_index in range(accessible):
                    next_display = list(display_cards)
                    next_deck = list(deck_cards)
                    card = next_display.pop(display_index)
                    if next_deck:
                        next_display.append(next_deck.pop(0))
                    next_variants.append(
                        (
                            _merge_list_detail_fragments(
                                details,
                                {
                                    "build_card_bonus_choices": [
                                        {"draw_source": "display", "display_index": display_index}
                                    ]
                                },
                            ),
                            list(labels) + [f"draw display[{display_index + 1}] #{card.number} {card.name}"],
                            next_display,
                            next_deck,
                            list(action_order),
                            reputation,
                        )
                    )

                next_display = list(display_cards)
                next_deck = list(deck_cards)
                if next_deck:
                    next_deck.pop(0)
                next_variants.append(
                    (
                        _merge_list_detail_fragments(
                            details,
                            {"build_card_bonus_choices": [{"draw_source": "deck"}]},
                        ),
                        list(labels) + ["draw deck"],
                        next_display,
                        next_deck,
                        list(action_order),
                        reputation,
                    )
                )
                continue

            next_variants.append(
                (
                    copy.deepcopy(details),
                    list(labels),
                    list(display_cards),
                    list(deck_cards),
                    list(action_order),
                    reputation,
                )
            )
        variants = next_variants

    deduped: List[Tuple[Dict[str, Any], str]] = []
    seen: Set[str] = set()
    for details, labels, _, _, _, _ in variants:
        key = json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key in seen:
            continue
        seen.add(key)
        deduped.append((copy.deepcopy(details), " ; ".join(label for label in labels if label)))
    return deduped or [({}, "")]


def _cards_action_label(
    *,
    state: GameState,
    snap_display_index: Optional[int],
    from_display_indices: Sequence[int],
    from_deck_count: int,
) -> str:
    if snap_display_index is not None:
        if 0 <= snap_display_index < len(state.zoo_display):
            card = state.zoo_display[snap_display_index]
            return f"snap display[{snap_display_index + 1}] #{card.number} {card.name}"
        return f"snap display[{snap_display_index + 1}]"

    parts: List[str] = []
    if from_display_indices:
        parts.append("display[" + ",".join(str(idx + 1) for idx in from_display_indices) + "]")
    if from_deck_count > 0:
        parts.append(f"deck={from_deck_count}")
    if not parts:
        parts.append("draw=0")
    return " ".join(parts)


def _merge_detail_fragments(*fragments: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {}
    for fragment in fragments:
        for key, value in fragment.items():
            if key in {
                "sponsor_unique_building_selections",
                "sponsor_263_build_details",
                "sponsor_253_plays",
            }:
                merged.setdefault(key, [])
                merged[key].extend(copy.deepcopy(list(value)))
                continue
            merged[key] = copy.deepcopy(value)
    return merged


class _ActionDetailExpansionRequired(Exception):
    def __init__(self, variants: Sequence[Tuple[Dict[str, Any], str]]) -> None:
        super().__init__("action detail expansion required")
        self.variants: List[Tuple[Dict[str, Any], str]] = [
            (copy.deepcopy(details), str(label))
            for details, label in variants
        ]


def _enumerate_sponsor_candidate_detail_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    candidate: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], str]]:
    card = candidate["card"]
    variants: List[Tuple[Dict[str, Any], str]] = [({}, "")]

    def _cross_join(additions: List[Tuple[Dict[str, Any], str]]) -> None:
        nonlocal variants
        joined: List[Tuple[Dict[str, Any], str]] = []
        for base_details, base_label in variants:
            for extra_details, extra_label in additions:
                merged_label_parts = [part for part in (base_label, extra_label) if part]
                joined.append(
                    (
                        _merge_detail_fragments(base_details, extra_details),
                        " ".join(merged_label_parts),
                    )
                )
        variants = joined

    if card.number in SPONSOR_UNIQUE_BUILDING_CARDS:
        legal = _list_legal_sponsor_unique_building_cells(
            state=state,
            player=player,
            sponsor_number=card.number,
        )
        _cross_join(
            [
                (
                    {
                        "sponsor_unique_building_selections": [
                            {
                                "card_instance_id": card.instance_id,
                                "cells": [list(cell) for cell in cells],
                            }
                        ]
                    },
                    f"unique={_cells_text(cells)}",
                )
                for cells in legal
            ]
            or [({}, "")]
        )

    if card.number == 227:
        _cross_join(
            [
                ({"sponsor_227_mode": "small"}, "mode=small"),
                ({"sponsor_227_mode": "large"}, "mode=large"),
            ]
        )

    if card.number == 263:
        size5_options = [
            option
            for option in list_legal_build_options(state=state, player_id=player_id, strength=5)
            if option["building_type"] == "SIZE_5"
        ]
        additions: List[Tuple[Dict[str, Any], str]] = []
        if not size5_options:
            additions.append(({}, "free_size5=none"))
        else:
            for option in size5_options:
                target_choices = _action_to_slot_1_target_sequences(
                    player.action_order,
                    int(option.get("placement_bonuses", []).count("action_to_slot_1")),
                )
                for targets in target_choices:
                    additions.append(
                        (
                            {
                                "sponsor_263_build_details": [
                                    {
                                        "card_instance_id": card.instance_id,
                                        "selections": [_build_selection_payload(option)],
                                        "bonus_action_to_slot_1_targets": list(targets),
                                    }
                                ]
                            },
                            f"free5={_build_option_label(option, targets)}",
                        )
                    )
        _cross_join(additions)

    return variants


def _enumerate_sponsor_253_choice_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    candidates: Sequence[AnimalCard],
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = [
        ({"sponsor_253_plays": [{"skip": True}]}, "253=skip"),
    ]
    for card in candidates:
        source_index = next(
            (idx for idx, candidate in enumerate(player.hand) if candidate.instance_id == card.instance_id),
            -1,
        )
        if source_index < 0:
            continue
        candidate_payload = {
            "card_instance_id": str(card.instance_id),
            "source_index": int(source_index),
        }
        candidate_ref = {
            "card": card,
            "source": "hand",
            "source_index": int(source_index),
            "card_instance_id": str(card.instance_id),
            "level": int(_sponsor_level(card)),
            "playable_now": True,
        }
        for fragment_details, fragment_label in _enumerate_sponsor_candidate_detail_variants(
            state,
            player,
            player_id,
            candidate_ref,
        ):
            label = f"253=>hand[{source_index + 1}] #{card.number} {card.name}"
            if fragment_label:
                label = f"{label} ({fragment_label})"
            variants.append(
                (
                    _merge_detail_fragments(
                        {"sponsor_253_plays": [candidate_payload]},
                        fragment_details,
                    ),
                    label,
                )
            )
    return variants


def _resolve_action_detail_variants_by_simulation(
    *,
    state: GameState,
    player_id: int,
    base_details: Dict[str, Any],
    executor: Callable[[GameState, PlayerState, Dict[str, Any]], None],
    invalid_effect_log_prefixes: Sequence[str] = (),
) -> List[Tuple[Dict[str, Any], str]]:
    resolved: List[Tuple[Dict[str, Any], str]] = []

    def _recurse(details_payload: Dict[str, Any], label_parts: List[str]) -> None:
        sim_state = copy.deepcopy(state)
        sim_player = sim_state.players[player_id]
        sim_details = copy.deepcopy(details_payload)
        sim_details["_allow_interactive"] = False
        sim_details["_expand_implicit_choices"] = True
        effect_log_len_before = len(sim_state.effect_log)
        try:
            executor(sim_state, sim_player, sim_details)
        except _ActionDetailExpansionRequired as pending:
            for extra_details, extra_label in pending.variants:
                next_details = _merge_detail_fragments(details_payload, extra_details)
                next_labels = list(label_parts)
                if extra_label:
                    next_labels.append(extra_label)
                _recurse(next_details, next_labels)
            return
        except ValueError:
            return

        if invalid_effect_log_prefixes:
            recent_effects = sim_state.effect_log[effect_log_len_before:]
            if any(
                any(str(entry).startswith(prefix) for prefix in invalid_effect_log_prefixes)
                for entry in recent_effects
            ):
                return

        label = " ; ".join(part for part in label_parts if part)
        resolved.append((copy.deepcopy(details_payload), label))

    _recurse(copy.deepcopy(base_details), [])
    return resolved or []


def _enumerate_concrete_cards_actions(
    state: GameState,
    player: PlayerState,
    template_action: Action,
) -> List[Action]:
    strength = int((template_action.details or {}).get("effective_strength", 0))
    upgraded = bool(player.action_upgraded["cards"])
    draw_target, discard_target, snap_allowed = _cards_table_values(strength, upgraded)
    actions: List[Action] = []

    if snap_allowed:
        for snap_idx in range(len(state.zoo_display)):
            label = _cards_action_label(
                state=state,
                snap_display_index=snap_idx,
                from_display_indices=(),
                from_deck_count=0,
            )
            actions.append(
                _make_concrete_action(
                    template_action,
                    label=label,
                    extra_details={"snap_display_index": snap_idx},
                )
            )

    accessible = min(_reputation_display_limit(player.reputation), len(state.zoo_display)) if upgraded else 0
    display_subsets = [tuple()] if not upgraded else _enumerate_index_combinations(
        total_items=accessible,
        min_choose=0,
        max_choose=accessible,
    )

    for display_indices in display_subsets:
        if len(display_indices) > draw_target:
            continue
        max_deck_count = draw_target if not upgraded else max(0, draw_target - len(display_indices))
        deck_counts = [draw_target] if not upgraded else list(range(max_deck_count + 1))
        for from_deck_count in deck_counts:
            if from_deck_count > len(state.zoo_deck):
                continue
            preview_hand_size = len(player.hand) + len(display_indices) + from_deck_count
            if discard_target > preview_hand_size:
                continue
            details: Dict[str, Any] = {
                "from_display_indices": list(display_indices),
                "from_deck_count": from_deck_count,
            }
            label = _cards_action_label(
                state=state,
                snap_display_index=None,
                from_display_indices=display_indices,
                from_deck_count=from_deck_count,
            )
            if not snap_allowed and len(display_subsets) == 1 and len(deck_counts) == 1 and discard_target == 0:
                label = ""
            actions.append(_make_concrete_action(template_action, label=label, extra_details=details))

    return actions or [_make_concrete_action(template_action)]


def _clear_pending_decision(state: GameState) -> None:
    state.pending_decision_kind = ""
    state.pending_decision_player_id = None
    state.pending_decision_payload = {}


def _set_pending_decision(
    state: GameState,
    *,
    kind: str,
    player_id: int,
    payload: Optional[Dict[str, Any]] = None,
) -> None:
    state.pending_decision_kind = str(kind)
    state.pending_decision_player_id = int(player_id)
    state.pending_decision_payload = copy.deepcopy(payload or {})


def _card_choice_index_action_label(prefix: str, choice_indices: Sequence[int]) -> str:
    if not choice_indices:
        return f"{prefix} []"
    rendered = " ".join(str(int(choice_index) + 1) for choice_index in choice_indices)
    return f"{prefix} [{rendered}]"


def _cards_pending_discard_action_label(choice_indices: Sequence[int]) -> str:
    return _card_choice_index_action_label("discard", choice_indices)


def _animal_effect_sell_choice_action_label(choice_indices: Sequence[int]) -> str:
    return _card_choice_index_action_label("sell", choice_indices)


def _animal_effect_pouch_choice_action_label(choice_indices: Sequence[int]) -> str:
    return _card_choice_index_action_label("pouch", choice_indices)


def _enumerate_index_combinations(
    *,
    total_items: int,
    min_choose: int,
    max_choose: Optional[int] = None,
) -> List[Tuple[int, ...]]:
    return _enumerate_index_combinations_impl(
        total_items=total_items,
        min_choose=min_choose,
        max_choose=max_choose,
    )


def _enumerate_card_combinations(
    cards: Sequence[AnimalCard],
    *,
    min_choose: int,
    max_choose: Optional[int] = None,
) -> List[Tuple[AnimalCard, ...]]:
    return [
        tuple(cards[idx] for idx in choice)
        for choice in _enumerate_index_combinations(
            total_items=len(cards),
            min_choose=min_choose,
            max_choose=max_choose,
        )
    ]


def _enumerate_animals_effect_choice_variants(
    player: PlayerState,
    plays: Sequence[Dict[str, Any]],
    *,
    state: Optional[GameState] = None,
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, List[Dict[str, Any]]], List[str], List[AnimalCard], List[str]]] = [
        ({}, [], list(player.hand), list(player.action_order))
    ]
    has_choice = False

    for play_index, play in enumerate(plays):
        card_instance_id = str(play.get("card_instance_id") or "").strip()
        if not card_instance_id:
            return [({}, "")]
        reserved_future_ids = {
            str(next_play.get("card_instance_id") or "").strip()
            for next_play in plays[play_index + 1 :]
            if str(next_play.get("card_instance_id") or "").strip()
        }
        next_variants: List[Tuple[Dict[str, List[Dict[str, Any]]], List[str], List[AnimalCard], List[str]]] = []
        for queued_choices, labels, current_hand, current_action_order in variants:
            played_index = next(
                (idx for idx, hand_card in enumerate(current_hand) if hand_card.instance_id == card_instance_id),
                None,
            )
            if played_index is None:
                continue
            hand_after_play = list(current_hand)
            played_card = hand_after_play.pop(played_index)
            effect = resolve_card_effect(played_card)
            if effect.code == "trade_hand_with_display":
                next_hand = list(hand_after_play)
                can_trade = (
                    state is not None
                    and bool(state.zoo_display)
                    and int(_reputation_display_limit(player.reputation)) > 0
                )
                trade_valid = True
                if can_trade:
                    for _ in range(max(0, int(effect.value))):
                        if not next_hand:
                            break
                        if str(next_hand[0].instance_id or "").strip() in reserved_future_ids:
                            trade_valid = False
                            break
                        # Non-interactive trade deterministically exchanges hand slot 0.
                        next_hand.pop(0)
                if not trade_valid:
                    continue
                next_variants.append(
                    (
                        copy.deepcopy(queued_choices),
                        list(labels),
                        next_hand,
                        list(current_action_order),
                    )
                )
                continue
            if effect.code not in {"sell_hand_cards", "pouch_hand_for_appeal", "boost_action_card"}:
                next_variants.append(
                    (
                        copy.deepcopy(queued_choices),
                        list(labels),
                        hand_after_play,
                        list(current_action_order),
                    )
                )
                continue

            has_choice = True
            if effect.code == "boost_action_card":
                target_action = str(effect.target or "").strip()
                if target_action not in current_action_order:
                    next_variants.append(
                        (
                            copy.deepcopy(queued_choices),
                            list(labels),
                            hand_after_play,
                            list(current_action_order),
                        )
                    )
                    continue
                choice_specs = [
                    ("skip", "boost skip", list(current_action_order)),
                    ("slot1", f"boost {target_action}->1", _move_action_order_to_slot(current_action_order, target_action, 1)),
                    ("slot5", f"boost {target_action}->5", _move_action_order_to_slot(current_action_order, target_action, 5)),
                ]
                seen_orders: Set[Tuple[str, ...]] = set()
                for mode, label, next_action_order in choice_specs:
                    order_key = tuple(next_action_order)
                    if order_key in seen_orders:
                        continue
                    seen_orders.add(order_key)
                    updated_choices = copy.deepcopy(queued_choices)
                    updated_choices.setdefault("boost_action_choices", []).append({"mode": mode})
                    next_variants.append(
                        (
                            updated_choices,
                            list(labels) + [label],
                            hand_after_play,
                            list(next_action_order),
                        )
                    )
                continue

            choice_limit = max(0, int(effect.value))
            selectable_cards = [
                hand_card
                for hand_card in hand_after_play
                if hand_card.instance_id not in reserved_future_ids
            ]
            detail_key = "sell_hand_card_choices" if effect.code == "sell_hand_cards" else "pouch_hand_card_choices"
            choice_label_fn = (
                _animal_effect_sell_choice_action_label
                if effect.code == "sell_hand_cards"
                else _animal_effect_pouch_choice_action_label
            )
            capped_limit = min(choice_limit, len(selectable_cards))
            for picked_indices in _enumerate_index_combinations(
                total_items=len(selectable_cards),
                min_choose=0,
                max_choose=capped_limit,
            ):
                picked_cards = tuple(selectable_cards[idx] for idx in picked_indices)
                picked_ids = {card.instance_id for card in picked_cards}
                next_hand = [
                    hand_card for hand_card in hand_after_play if hand_card.instance_id not in picked_ids
                ]
                updated_choices = copy.deepcopy(queued_choices)
                updated_choices.setdefault(detail_key, []).append(
                    {"card_instance_ids": [card.instance_id for card in picked_cards]}
                )
                next_variants.append(
                    (
                        updated_choices,
                        list(labels) + [choice_label_fn(picked_indices)],
                        next_hand,
                        list(current_action_order),
                    )
                )
        variants = next_variants

    if not has_choice:
        return [({}, "")]

    deduped: List[Tuple[Dict[str, Any], str]] = []
    seen_keys: Set[str] = set()
    for queued_choices, labels, _, _ in variants:
        key = json.dumps(queued_choices, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(
            (
                copy.deepcopy(queued_choices),
                " ; ".join(label for label in labels if label),
            )
        )
    return deduped or [({}, "")]


def _enumerate_pending_cards_discard_actions(player: PlayerState, discard_target: int) -> List[Action]:
    if discard_target <= 0:
        return []
    hand_cards = list(player.hand)
    if len(hand_cards) < discard_target:
        return []
    actions: List[Action] = []
    for picked_indices in _enumerate_index_combinations(
        total_items=len(hand_cards),
        min_choose=discard_target,
        max_choose=discard_target,
    ):
        picked_cards = tuple(hand_cards[idx] for idx in picked_indices)
        actions.append(
            Action(
                ActionType.PENDING_DECISION,
                details={
                    "concrete": True,
                    "pending_kind": "cards_discard",
                    "discard_card_instance_ids": [card.instance_id for card in picked_cards],
                    "action_label": _cards_pending_discard_action_label(picked_indices),
                },
            )
        )
    return actions


def _opening_draft_keep_action_label(choice_indices: Sequence[int]) -> str:
    return _card_choice_index_action_label("keep", choice_indices)


def _conservation_bonus_tile_label(tile: str) -> str:
    labels = {
        "10_money": "+10 money",
        "size_3_enclosure": "free enclosure_3",
        "2_reputation": "+2 reputation",
        "3_x_tokens": "+3 x-tokens",
        "3_cards": "draw 3 deck",
        "partner_zoo": "partner zoo",
        "university": "university",
        "x2_multiplier": "x2 multiplier",
        "sponsor_card": "play sponsor<=5",
    }
    return labels.get(str(tile or "").strip(), str(tile or "").strip())


def _conservation_reward_action_label(threshold: int, label: str) -> str:
    return f"cons[{int(threshold)}] | {label}"


def _next_pending_conservation_reward_threshold(player: PlayerState) -> Optional[int]:
    for threshold in CONSERVATION_REWARD_THRESHOLDS:
        if int(player.conservation) < int(threshold):
            continue
        if int(threshold) in player.claimed_conservation_reward_spaces:
            continue
        return int(threshold)
    return None


def _list_legal_conservation_reward_partner_zoos(state: GameState, player: PlayerState) -> List[str]:
    if len(player.partner_zoos) >= 4:
        return []
    if not bool(player.action_upgraded.get("association", False)) and len(player.partner_zoos) >= 2:
        return []
    return [
        partner
        for partner in sorted(state.available_partner_zoos)
        if partner not in player.partner_zoos
    ]


def _list_legal_conservation_reward_universities(state: GameState, player: PlayerState) -> List[str]:
    return [
        university
        for university in sorted(state.available_universities)
        if university not in player.universities
    ]


def _gain_partner_zoo_reward(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    partner: str,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    normalized_partner = str(partner or "").strip().lower()
    if not normalized_partner:
        raise ValueError("Partner zoo reward requires a continent.")
    if len(player.partner_zoos) >= 4:
        raise ValueError("Cannot take more than 4 partner zoos.")
    if not bool(player.action_upgraded.get("association", False)) and len(player.partner_zoos) >= 2:
        raise ValueError("A third partner zoo requires Association side II.")
    if normalized_partner in player.partner_zoos:
        raise ValueError("Selected partner zoo is already owned.")
    if normalized_partner not in state.available_partner_zoos:
        raise ValueError("Selected partner zoo is not currently available.")
    state.available_partner_zoos.remove(normalized_partner)
    player.partner_zoos.add(normalized_partner)
    _apply_map_partner_threshold_rewards(
        state=state,
        player=player,
        player_id=player_id,
        effect_details=effect_details,
        allow_interactive=allow_interactive,
    )


def _gain_university_reward(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    university: str,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    normalized_university = str(university or "").strip().lower()
    if not normalized_university:
        raise ValueError("University reward requires a university type.")
    if normalized_university in player.universities:
        raise ValueError("Selected university is already owned.")
    if normalized_university not in state.available_universities:
        raise ValueError("Selected university is not currently available.")
    state.available_universities.remove(normalized_university)
    player.universities.add(normalized_university)
    hand_limit_target = UNIVERSITY_HAND_LIMIT_SET.get(normalized_university, 0)
    if hand_limit_target > 0:
        player.hand_limit = max(player.hand_limit, hand_limit_target)
    rep_gain = UNIVERSITY_REPUTATION_GAIN.get(normalized_university, 0)
    if rep_gain > 0:
        _apply_reputation_gain_with_details(
            state=state,
            player=player,
            player_id=player_id,
            amount=rep_gain,
            details=effect_details,
            allow_interactive=allow_interactive,
        )
    _apply_map_university_threshold_rewards(
        state=state,
        player=player,
        player_id=player_id,
        effect_details=effect_details,
        allow_interactive=allow_interactive,
    )


def _cross_join_detail_label_variants(
    base_variants: Sequence[Tuple[Dict[str, Any], str]],
    additions: Sequence[Tuple[Dict[str, Any], str]],
) -> List[Tuple[Dict[str, Any], str]]:
    if not additions:
        return [(copy.deepcopy(details), label) for details, label in base_variants]
    joined: List[Tuple[Dict[str, Any], str]] = []
    for base_details, base_label in base_variants:
        for extra_details, extra_label in additions:
            merged_label_parts = [part for part in (base_label, extra_label) if part]
            joined.append(
                (
                    _merge_list_detail_fragments(base_details, extra_details),
                    " ; ".join(merged_label_parts),
                )
            )
    return joined


def _enumerate_upgrade_action_card_choice_variants(
    player: PlayerState,
    *,
    detail_key: str,
    label_prefix: str,
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = []
    for action_name in MAIN_ACTION_CARDS:
        if bool(player.action_upgraded.get(action_name, False)):
            continue
        variants.append(
            (
                {
                    detail_key: [
                        {
                            "upgraded_action": action_name,
                        }
                    ]
                },
                f"{label_prefix}({action_name})",
            )
        )
    return variants or [({}, "")]


def _map_effect_choice_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    effect_code: str,
    *,
    detail_key: str,
    label_prefix: str,
) -> List[Tuple[Dict[str, Any], str]]:
    code = str(effect_code or "").strip().lower()
    if not code:
        return [({}, "")]

    if code == "draw_1_card_deck_or_reputation_range":
        variants: List[Tuple[Dict[str, Any], str]] = [({detail_key: [{"draw_source": "deck"}]}, f"{label_prefix}(deck)")]
        accessible = min(_reputation_display_limit(int(player.reputation)), len(state.zoo_display))
        for idx in range(accessible):
            card = state.zoo_display[idx]
            variants.append(
                (
                    {
                        detail_key: [
                            {
                                "draw_source": "display",
                                "display_index": idx,
                            }
                        ]
                    },
                    f"{label_prefix}(display[{idx + 1}] #{card.number} {card.name})",
                )
            )
        return variants

    if code == "build_free_standard_enclosure_size_2":
        variants: List[Tuple[Dict[str, Any], str]] = []
        build_options = [
            item
            for item in list_legal_build_options(state=state, player_id=player_id, strength=2)
            if item.get("building_type") == "SIZE_2"
        ]
        for build_option in build_options:
            for build_bonus_details, build_bonus_label in _enumerate_build_bonus_choice_variants(state, player_id, build_option):
                variants.append(
                    (
                        {
                            detail_key: [
                                _merge_list_detail_fragments(
                                    {"selection": _build_selection_payload(build_option)},
                                    build_bonus_details,
                                )
                            ]
                        },
                        f"{label_prefix}({_build_step_action_label(build_option, build_bonus_label)})",
                    )
                )
        return variants or [({}, f"{label_prefix}(unavailable)")]

    if code == "play_1_sponsor_by_paying_cost":
        sponsors_upgraded = bool(player.action_upgraded["sponsors"])
        variants: List[Tuple[Dict[str, Any], str]] = []
        for candidate in _list_legal_sponsor_candidates(state, player, sponsors_upgraded):
            card = candidate.get("card")
            if candidate.get("source") != "hand":
                continue
            if not isinstance(card, AnimalCard):
                continue
            if str(candidate.get("reason") or "") != "ok":
                continue
            if int(player.money) < _sponsor_level(card):
                continue
            for fragment_details, fragment_label in _enumerate_sponsor_candidate_detail_variants(state, player, player_id, candidate):
                sponsor_details: Dict[str, Any] = {
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": "hand",
                            "source_index": int(candidate.get("source_index", -1)),
                            "card_instance_id": str(candidate.get("card_instance_id") or ""),
                        }
                    ],
                }
                sponsor_details = _merge_detail_fragments(sponsor_details, fragment_details)
                label = f"{label_prefix}(hand[{int(candidate.get('source_index', -1)) + 1}] #{card.number} {card.name}"
                if fragment_label:
                    label += f" {fragment_label}"
                label += ")"
                variants.append(({detail_key: [{"sponsor_details": sponsor_details}]}, label))
        return variants or [({}, f"{label_prefix}(unavailable)")]

    if code == "upgrade_action_card":
        return _enumerate_upgrade_action_card_choice_variants(
            player,
            detail_key=detail_key,
            label_prefix=label_prefix,
        )

    if code == "gain_worker_1":
        nested_variants: List[Tuple[Dict[str, Any], str]] = [({}, "")]
        if detail_key != "worker_gain_effect_choices":
            nested_variants = _enumerate_worker_gain_reward_variants(
                state=state,
                player=player,
                player_id=player_id,
                gained_workers=1,
                detail_key="worker_gain_effect_choices",
                label_prefix="worker-reward",
            )
        variants: List[Tuple[Dict[str, Any], str]] = []
        for nested_details, nested_label in nested_variants:
            label = f"{label_prefix}(+1 worker)"
            if nested_label:
                label = f"{label} ; {nested_label}"
            variants.append(({detail_key: [copy.deepcopy(nested_details)]}, label))
        return variants or [({detail_key: [{}]}, f"{label_prefix}(+1 worker)")]

    simple_labels = {
        "gain_5_coins": "+5 money",
        "gain_12_coins": "+12 money",
        "gain_3_x_tokens": "+3 x-tokens",
        "gain_conservation_1": "+1 conservation",
        "gain_conservation_2": "+2 conservation",
        "gain_appeal_7": "+7 appeal",
    }
    return [({}, f"{label_prefix}({simple_labels.get(code, code)})")]


def _reputation_gain_target_and_milestones(
    player: PlayerState,
    amount: int,
) -> Tuple[int, int, List[int]]:
    has_association_upgrade = bool(player.action_upgraded.get("association", False))
    reputation_cap = 15 if has_association_upgrade else 9
    old_rep = int(player.reputation)
    target_rep = old_rep + int(amount)
    overflow_appeal = 0
    if reputation_cap == 15 and target_rep > 15:
        overflow_appeal = target_rep - 15
    new_rep = min(reputation_cap, target_rep)
    milestones = [
        milestone
        for milestone in (5, 8, 10, 11, 12, 13, 14, 15)
        if old_rep < milestone <= new_rep and milestone not in player.claimed_reputation_milestones
    ]
    return new_rep, overflow_appeal, milestones


def _enumerate_reputation_gain_followup_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    amount: int,
) -> List[Tuple[Dict[str, Any], str]]:
    new_rep, _, milestones = _reputation_gain_target_and_milestones(player, amount)
    if not milestones:
        return [({}, "")]

    variants: List[Tuple[Dict[str, Any], str]] = []
    simulated_player = copy.deepcopy(player)
    simulated_player.reputation = int(new_rep)

    def _recurse(
        milestone_index: int,
        current_player: PlayerState,
        current_details: Dict[str, Any],
        current_labels: List[str],
    ) -> None:
        if milestone_index >= len(milestones):
            variants.append((copy.deepcopy(current_details), " ; ".join(part for part in current_labels if part)))
            return

        milestone = milestones[milestone_index]
        if milestone == 5:
            upgradeable = [
                action_name
                for action_name in MAIN_ACTION_CARDS
                if not bool(current_player.action_upgraded.get(action_name, False))
            ]
            if not upgradeable:
                _recurse(milestone_index + 1, current_player, current_details, current_labels)
                return
            for action_name in upgradeable:
                next_player = copy.deepcopy(current_player)
                next_player.action_upgraded[action_name] = True
                next_details = _merge_list_detail_fragments(
                    current_details,
                    {"reputation_milestone_reward_choices": [{"upgraded_action": action_name}]},
                )
                _recurse(
                    milestone_index + 1,
                    next_player,
                    next_details,
                    list(current_labels) + [f"rep5 upgrade({action_name})"],
                )
            return

        if milestone == 8:
            additions = _enumerate_worker_gain_reward_variants(
                state=state,
                player=current_player,
                player_id=player_id,
                gained_workers=1,
                detail_key="reputation_milestone_reward_choices",
                label_prefix="rep8 worker",
            )
            if not additions:
                _recurse(milestone_index + 1, current_player, current_details, current_labels)
                return
            for extra_details, extra_label in additions:
                _recurse(
                    milestone_index + 1,
                    copy.deepcopy(current_player),
                    _merge_list_detail_fragments(current_details, extra_details),
                    list(current_labels) + ([extra_label] if extra_label else []),
                )
            return

        _recurse(milestone_index + 1, current_player, current_details, current_labels)

    _recurse(0, simulated_player, {}, [])

    deduped: List[Tuple[Dict[str, Any], str]] = []
    seen: Set[str] = set()
    for details, label in variants:
        key = json.dumps(
            {"details": details, "label": label},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((copy.deepcopy(details), str(label)))
    return deduped or [({}, "")]


def _enumerate_map_threshold_reward_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    *,
    count: int,
    thresholds: Sequence[Dict[str, Any]],
    claimed_thresholds: Set[int],
    detail_key: str,
    label_prefix: str,
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = [({}, "")]
    for item in thresholds:
        threshold = int(item.get("count", 0))
        effect_code = str(item.get("effect") or "").strip().lower()
        if threshold <= 0 or not effect_code:
            continue
        if count < threshold or threshold in claimed_thresholds:
            continue
        additions = _map_effect_choice_variants(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            detail_key=detail_key,
            label_prefix=f"{label_prefix}{threshold}",
        )
        variants = _cross_join_detail_label_variants(variants, additions)
    return variants


def _enumerate_worker_gain_reward_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    *,
    gained_workers: int,
    detail_key: str,
    label_prefix: str,
) -> List[Tuple[Dict[str, Any], str]]:
    variants: List[Tuple[Dict[str, Any], str]] = [({}, "")]
    if gained_workers <= 0:
        return variants
    for reward in _map_rule_worker_gain_rewards(state):
        effect_code = str(reward.get("effect") or "").strip().lower()
        if not effect_code:
            continue
        repeat = max(1, int(reward.get("repeat_per_worker", 1)))
        for _ in range(gained_workers * repeat):
            additions = _map_effect_choice_variants(
                state=state,
                player=player,
                player_id=player_id,
                effect_code=effect_code,
                detail_key=detail_key,
                label_prefix=label_prefix,
            )
            variants = _cross_join_detail_label_variants(variants, additions)
    return variants


def _apply_reputation_gain_with_details(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    amount: int,
    details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    details = details or {}
    if amount <= 0:
        return
    old_rep = int(player.reputation)
    new_rep, overflow_appeal, milestones = _reputation_gain_target_and_milestones(player, amount)
    if overflow_appeal > 0:
        player.appeal += overflow_appeal
    player.reputation = new_rep
    for milestone in milestones:
        player.claimed_reputation_milestones.add(milestone)
        if milestone == 5:
            queue_entry = _pop_detail_queue_entry(details, "reputation_milestone_reward_choices")
            upgraded_action = str((queue_entry or {}).get("upgraded_action") or "").strip()
            if upgraded_action:
                if upgraded_action not in MAIN_ACTION_CARDS:
                    raise ValueError("Selected upgraded_action for reputation milestone is invalid.")
                if bool(player.action_upgraded.get(upgraded_action, False)):
                    raise ValueError("Selected action card is already upgraded.")
                player.action_upgraded[upgraded_action] = True
            else:
                upgraded = _upgrade_one_action_card(player, interactive=allow_interactive)
                if not upgraded:
                    state.effect_log.append(f"reputation_milestone_{milestone}:{player.name}:upgraded=0")
            continue
        if milestone == 8:
            worker_gain_details = _pop_detail_queue_entry(details, "reputation_milestone_reward_choices") or {}
            _gain_workers(
                state=state,
                player=player,
                player_id=player_id,
                amount=1,
                source=f"reputation_milestone_{milestone}",
                allow_interactive=allow_interactive,
                effect_details=worker_gain_details,
            )
            continue
        if milestone in {10, 13}:
            _take_one_reputation_bonus_card(state, player)
            continue
        if milestone in {11, 14}:
            player.conservation += 1
            continue
        if milestone in {12, 15}:
            player.x_tokens = min(5, int(player.x_tokens) + 1)
            continue


def _enumerate_conservation_reward_variants(
    state: GameState,
    player: PlayerState,
    player_id: int,
    threshold: int,
) -> List[Tuple[Dict[str, Any], str]]:
    threshold = int(threshold)
    variants: List[Tuple[Dict[str, Any], str]] = []

    if threshold == 2:
        for action_name in MAIN_ACTION_CARDS:
            if bool(player.action_upgraded.get(action_name, False)):
                continue
            variants.append(
                (
                    {
                        "reward": "upgrade_action_card",
                        "upgraded_action": action_name,
                    },
                    f"upgrade({action_name})",
                )
            )
        if int(player.workers) < MAX_WORKERS:
            base_variant = [({"reward": "activate_association_worker"}, "activate worker")]
            base_variant = _cross_join_detail_label_variants(
                base_variant,
                _enumerate_worker_gain_reward_variants(
                    state=state,
                    player=player,
                    player_id=player_id,
                    gained_workers=1,
                    detail_key="worker_gain_effect_choices",
                    label_prefix="worker-reward",
                ),
            )
            variants.extend(base_variant)
        return variants

    if threshold not in CONSERVATION_SHARED_BONUS_THRESHOLDS:
        return []

    variants.append(({"reward": CONSERVATION_FIXED_MONEY_OPTION}, "+5 money"))
    for reward in list(state.shared_conservation_bonus_tiles.get(threshold, [])):
        if reward in {"10_money", "2_reputation", "3_x_tokens", "3_cards"}:
            base_variant = [({"reward": reward}, _conservation_bonus_tile_label(reward))]
            if reward == "2_reputation":
                base_variant = _cross_join_detail_label_variants(
                    base_variant,
                    _enumerate_reputation_gain_followup_variants(
                        state=state,
                        player=player,
                        player_id=player_id,
                        amount=2,
                    ),
                )
            variants.extend(base_variant)
            continue
        if reward == "size_3_enclosure":
            build_options = [
                item
                for item in list_legal_build_options(state=state, player_id=player_id, strength=3)
                if item.get("building_type") == "SIZE_3"
            ]
            for build_option in build_options:
                for build_bonus_details, build_bonus_label in _enumerate_build_bonus_choice_variants(
                    state,
                    player_id,
                    build_option,
                ):
                    variants.append(
                        (
                            _merge_list_detail_fragments(
                                {
                                    "reward": reward,
                                    "selection": _build_selection_payload(build_option),
                                },
                                build_bonus_details,
                            ),
                            f"{_conservation_bonus_tile_label(reward)} {_build_step_action_label(build_option, build_bonus_label)}",
                        )
                    )
            continue
        if reward == "partner_zoo":
            for partner in _list_legal_conservation_reward_partner_zoos(state, player):
                base_variant = [
                    (
                        {
                            "reward": reward,
                            "partner_zoo": partner,
                        },
                        f"tile(partner({_partner_zoo_label(partner)}))",
                    )
                ]
                simulated_player = copy.deepcopy(player)
                simulated_player.partner_zoos.add(partner)
                base_variant = _cross_join_detail_label_variants(
                    base_variant,
                    _enumerate_map_threshold_reward_variants(
                        state=state,
                        player=simulated_player,
                        player_id=player_id,
                        count=len(simulated_player.partner_zoos),
                        thresholds=_map_rule_partner_thresholds(state),
                        claimed_thresholds=player.claimed_partner_zoo_thresholds,
                        detail_key="map_threshold_effect_choices",
                        label_prefix="partner-threshold-",
                    ),
                )
                variants.extend(base_variant)
            continue
        if reward == "university":
            for university in _list_legal_conservation_reward_universities(state, player):
                base_details = {
                    "reward": reward,
                    "university": university,
                }
                base_label = f"tile({_compact_association_university_label(university)})"

                simulated_player = copy.deepcopy(player)
                simulated_player.universities.add(university)
                hand_limit_target = int(UNIVERSITY_HAND_LIMIT_SET.get(university, 0))
                if hand_limit_target > 0:
                    simulated_player.hand_limit = max(int(simulated_player.hand_limit), hand_limit_target)

                rep_gain = int(UNIVERSITY_REPUTATION_GAIN.get(university, 0))
                rep_followup_variants: List[Tuple[Dict[str, Any], str]] = [({}, "")]
                if rep_gain > 0:
                    rep_followup_variants = _enumerate_reputation_gain_followup_variants(
                        state=state,
                        player=simulated_player,
                        player_id=player_id,
                        amount=rep_gain,
                    )

                for rep_details, rep_label in rep_followup_variants:
                    combined_details = _merge_list_detail_fragments(base_details, rep_details)
                    label_parts = [base_label]
                    if rep_label:
                        label_parts.append(rep_label)

                    simulated_state_after_rep = copy.deepcopy(state)
                    simulated_player_after_rep = simulated_state_after_rep.players[player_id]
                    simulated_player_after_rep.universities.add(university)
                    if hand_limit_target > 0:
                        simulated_player_after_rep.hand_limit = max(
                            int(simulated_player_after_rep.hand_limit),
                            hand_limit_target,
                        )
                    if rep_gain > 0:
                        try:
                            _apply_reputation_gain_with_details(
                                state=simulated_state_after_rep,
                                player=simulated_player_after_rep,
                                player_id=player_id,
                                amount=rep_gain,
                                details=copy.deepcopy(rep_details),
                                allow_interactive=False,
                            )
                        except ValueError:
                            continue

                    map_threshold_variants = _enumerate_map_threshold_reward_variants(
                        state=simulated_state_after_rep,
                        player=simulated_player_after_rep,
                        player_id=player_id,
                        count=len(simulated_player_after_rep.universities),
                        thresholds=_map_rule_university_thresholds(simulated_state_after_rep),
                        claimed_thresholds=simulated_player_after_rep.claimed_university_thresholds,
                        detail_key="map_threshold_effect_choices",
                        label_prefix="university-threshold-",
                    )
                    for threshold_details, threshold_label in map_threshold_variants:
                        merged_details = _merge_list_detail_fragments(combined_details, threshold_details)
                        merged_label_parts = list(label_parts)
                        if threshold_label:
                            merged_label_parts.append(threshold_label)
                        variants.append((merged_details, " ; ".join(part for part in merged_label_parts if part)))
            continue
        if reward == "x2_multiplier":
            for action_name in MAIN_ACTION_CARDS:
                variants.append(
                    (
                        {
                            "reward": reward,
                            "multiplier_action": action_name,
                        },
                        f"tile(x2->{action_name})",
                    )
                )
            continue
        if reward == "sponsor_card":
            sponsors_upgraded = bool(player.action_upgraded["sponsors"])
            candidates = []
            for item in _list_legal_sponsor_candidates(state, player, sponsors_upgraded):
                card = item.get("card")
                if item.get("source") != "hand":
                    continue
                if not isinstance(card, AnimalCard):
                    continue
                if str(item.get("reason") or "") != "ok":
                    continue
                if _sponsor_level(card) > 5:
                    continue
                candidates.append(item)
            for candidate in candidates:
                variant_sets = [_enumerate_sponsor_candidate_detail_variants(state, player, player_id, candidate)]
                for fragment_product in product(*variant_sets):
                    sponsor_details: Dict[str, Any] = {
                        "use_break_ability": False,
                        "sponsor_selections": [
                            {
                                "source": "hand",
                                "source_index": int(candidate.get("source_index", -1)),
                                "card_instance_id": str(candidate.get("card_instance_id") or ""),
                            }
                        ],
                    }
                    label_parts: List[str] = []
                    for selected_candidate, (fragment_details, fragment_label) in zip([candidate], fragment_product):
                        sponsor_card = selected_candidate["card"]
                        source_label = f"hand[{int(selected_candidate['source_index']) + 1}]"
                        card_label = f"{source_label} #{sponsor_card.number} {sponsor_card.name}"
                        if fragment_label:
                            card_label += f" ({fragment_label})"
                        label_parts.append(card_label)
                        sponsor_details = _merge_detail_fragments(sponsor_details, fragment_details)
                    variants.append(
                        (
                            {
                                "reward": reward,
                                "sponsor_details": sponsor_details,
                            },
                            f"{_conservation_bonus_tile_label(reward)} {' + '.join(label_parts)}",
                        )
                    )
            continue

    return variants


def _maybe_begin_conservation_reward_pending(
    state: GameState,
    *,
    player_id: int,
    resume_kind: str,
    break_triggered: bool = False,
    consumed_venom: bool = False,
    break_income_index: int = 0,
    resume_turn_player_id: Optional[int] = None,
    resume_turn_consumed_venom: bool = False,
) -> bool:
    if str(state.pending_decision_kind or "").strip():
        return True
    player = state.players[player_id]
    while True:
        threshold = _next_pending_conservation_reward_threshold(player)
        if threshold is None:
            return False
        variants = _enumerate_conservation_reward_variants(state, player, player_id, threshold)
        if variants:
            payload: Dict[str, Any] = {
                "threshold": threshold,
                "resume_kind": str(resume_kind or "").strip(),
            }
            if resume_kind == "turn_finalize":
                payload["break_triggered"] = bool(break_triggered)
                payload["consumed_venom"] = bool(consumed_venom)
            if resume_kind == "break_remaining":
                payload["break_income_index"] = int(break_income_index)
                if resume_turn_player_id is not None:
                    payload["resume_turn_player_id"] = int(resume_turn_player_id)
                    payload["resume_turn_consumed_venom"] = bool(resume_turn_consumed_venom)
            _set_pending_decision(
                state,
                kind="conservation_reward",
                player_id=player_id,
                payload=payload,
            )
            return True
        player.claimed_conservation_reward_spaces.add(int(threshold))
        state.effect_log.append(f"conservation_reward_{threshold}:{player.name}:skipped_unavailable")


def _enumerate_pending_break_discard_actions(player: PlayerState, discard_target: int) -> List[Action]:
    if discard_target <= 0:
        return []
    hand_cards = list(player.hand)
    if len(hand_cards) < discard_target:
        return []
    return [
        Action(
            ActionType.PENDING_DECISION,
            details={
                "concrete": True,
                "pending_kind": "break_discard",
                "discard_card_instance_ids": [hand_cards[idx].instance_id for idx in picked_indices],
                "action_label": _card_choice_index_action_label("discard", picked_indices),
            },
        )
        for picked_indices in _enumerate_index_combinations(
            total_items=len(hand_cards),
            min_choose=discard_target,
            max_choose=discard_target,
        )
    ]


def _enumerate_pending_opening_draft_actions(player: PlayerState, keep_target: int) -> List[Action]:
    if keep_target <= 0:
        return []
    drafted_cards = list(player.opening_draft_drawn)
    if len(drafted_cards) < keep_target:
        return []
    return [
        Action(
            ActionType.PENDING_DECISION,
            details={
                "concrete": True,
                "pending_kind": "opening_draft_keep",
                "keep_card_instance_ids": [drafted_cards[idx].instance_id for idx in picked_indices],
                "action_label": _opening_draft_keep_action_label(picked_indices),
            },
        )
        for picked_indices in _enumerate_index_combinations(
            total_items=len(drafted_cards),
            min_choose=keep_target,
            max_choose=keep_target,
        )
    ]


def _enumerate_followup_conservation_reward_variants(
    state: GameState,
    *,
    player_id: int,
) -> List[Tuple[List[Dict[str, Any]], str]]:
    player = state.players[player_id]
    while True:
        threshold = _next_pending_conservation_reward_threshold(player)
        if threshold is None:
            return [([], "")]
        threshold_variants = _enumerate_conservation_reward_variants(state, player, player_id, threshold)
        if threshold_variants:
            break
        player.claimed_conservation_reward_spaces.add(int(threshold))

    variants: List[Tuple[List[Dict[str, Any]], str]] = []
    for details, label in threshold_variants:
        next_state = copy.deepcopy(state)
        next_player = next_state.players[player_id]
        try:
            _apply_conservation_reward_choice(
                state=next_state,
                player=next_player,
                player_id=player_id,
                threshold=threshold,
                action=Action(ActionType.PENDING_DECISION, details=copy.deepcopy(details)),
            )
        except ValueError:
            continue
        for tail_choices, tail_label in _enumerate_followup_conservation_reward_variants(next_state, player_id=player_id):
            head_choice = {"reward_threshold": int(threshold), **copy.deepcopy(details)}
            choice_sequence = [head_choice] + copy.deepcopy(tail_choices)
            head_label = _conservation_reward_action_label(threshold, label)
            combined_label = head_label if not tail_label else f"{head_label} ; {tail_label}"
            variants.append((choice_sequence, combined_label))

    if not variants:
        return [([], "")]
    deduped: List[Tuple[List[Dict[str, Any]], str]] = []
    seen: Set[str] = set()
    for detail_list, label in variants:
        key = json.dumps(
            {"details": detail_list, "label": label},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append((copy.deepcopy(detail_list), str(label)))
    return deduped


def _enumerate_pending_conservation_reward_actions(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> List[Action]:
    threshold = int(state.pending_decision_payload.get("threshold", 0))
    if threshold <= 0:
        return []

    actions: List[Action] = []
    for details, label in _enumerate_conservation_reward_variants(
        state=state,
        player=player,
        player_id=player_id,
        threshold=threshold,
    ):
        next_state = copy.deepcopy(state)
        next_player = next_state.players[player_id]
        try:
            _apply_conservation_reward_choice(
                state=next_state,
                player=next_player,
                player_id=player_id,
                threshold=threshold,
                action=Action(ActionType.PENDING_DECISION, details=copy.deepcopy(details)),
            )
        except ValueError:
            continue

        followup_variants = _enumerate_followup_conservation_reward_variants(next_state, player_id=player_id)
        for chained_choices, chained_label in followup_variants:
            action_label = _conservation_reward_action_label(threshold, label)
            if chained_label:
                action_label = f"{action_label} ; {chained_label}"
            action_details: Dict[str, Any] = {
                "concrete": True,
                "pending_kind": "conservation_reward",
                "reward_threshold": threshold,
                "action_label": action_label,
                **copy.deepcopy(details),
            }
            if chained_choices:
                action_details["chained_conservation_reward_choices"] = copy.deepcopy(chained_choices)
            actions.append(Action(ActionType.PENDING_DECISION, details=action_details))

    deduped: List[Action] = []
    seen_keys: Set[str] = set()
    for action in actions:
        key = json.dumps(action.details or {}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(action)
    return deduped


def _append_card_instance_choice_queue(raw_value: Any, queue: List[List[str]]) -> None:
    if isinstance(raw_value, dict):
        ids = [str(item).strip() for item in list(raw_value.get("card_instance_ids") or []) if str(item).strip()]
        queue.append(ids)
        return
    if not isinstance(raw_value, (list, tuple)):
        return
    if raw_value and all(isinstance(item, str) for item in raw_value):
        queue.append([str(item).strip() for item in raw_value if str(item).strip()])
        return
    for item in raw_value:
        if isinstance(item, dict):
            ids = [
                str(raw).strip()
                for raw in list(item.get("card_instance_ids") or [])
                if str(raw).strip()
            ]
            queue.append(ids)
        elif isinstance(item, (list, tuple)):
            queue.append([str(raw).strip() for raw in item if str(raw).strip()])


def _normalize_boost_action_mode(raw_value: Any) -> str:
    value = str(raw_value or "").strip().lower().replace("-", "").replace("_", "")
    if value in {"", "none", "skip", "noop", "no"}:
        return "skip"
    if value in {"1", "slot1", "top"}:
        return "slot1"
    if value in {"5", "slot5", "bottom"}:
        return "slot5"
    return ""


def _normalized_concrete_sponsors_details_for_dedup(details: Dict[str, Any]) -> str:
    normalized = copy.deepcopy(details or {})
    for key in ("base_strength", "effective_strength", "action_label", "concrete"):
        normalized.pop(key, None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized_concrete_association_details_for_dedup(details: Dict[str, Any]) -> str:
    normalized = copy.deepcopy(details or {})
    for key in ("base_strength", "effective_strength", "action_label", "concrete"):
        normalized.pop(key, None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized_concrete_build_details_for_dedup(details: Dict[str, Any]) -> str:
    normalized = copy.deepcopy(details or {})
    for key in ("base_strength", "effective_strength", "action_label", "concrete"):
        normalized.pop(key, None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _normalized_concrete_animals_details_for_dedup(details: Dict[str, Any]) -> str:
    normalized = copy.deepcopy(details or {})
    for key in ("base_strength", "effective_strength", "action_label", "concrete"):
        normalized.pop(key, None)
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _dedupe_dominated_concrete_sponsors_actions(actions: Sequence[Action]) -> List[Action]:
    deduped: List[Action] = []
    kept_indices: Dict[str, int] = {}
    for action in actions:
        if action.type != ActionType.MAIN_ACTION or str(action.card_name or "") != "sponsors":
            deduped.append(action)
            continue
        details = dict(action.details or {})
        if not bool(details.get("concrete")) or bool(details.get("use_break_ability")):
            deduped.append(action)
            continue
        key = _normalized_concrete_sponsors_details_for_dedup(details)
        existing_index = kept_indices.get(key)
        if existing_index is None:
            kept_indices[key] = len(deduped)
            deduped.append(action)
            continue
        existing_action = deduped[existing_index]
        if int(action.value or 0) < int(existing_action.value or 0):
            deduped[existing_index] = action
    return deduped


def _dedupe_dominated_concrete_animals_actions(actions: Sequence[Action]) -> List[Action]:
    deduped: List[Action] = []
    kept_indices: Dict[str, int] = {}
    for action in actions:
        if action.type != ActionType.MAIN_ACTION or str(action.card_name or "") != "animals":
            deduped.append(action)
            continue
        details = dict(action.details or {})
        if not bool(details.get("concrete")):
            deduped.append(action)
            continue
        key = _normalized_concrete_animals_details_for_dedup(details)
        existing_index = kept_indices.get(key)
        if existing_index is None:
            kept_indices[key] = len(deduped)
            deduped.append(action)
            continue
        existing_action = deduped[existing_index]
        if int(action.value or 0) < int(existing_action.value or 0):
            deduped[existing_index] = action
    return deduped


def _dedupe_dominated_concrete_build_actions(actions: Sequence[Action]) -> List[Action]:
    deduped: List[Action] = []
    kept_indices: Dict[str, int] = {}
    for action in actions:
        if action.type != ActionType.MAIN_ACTION or str(action.card_name or "") != "build":
            deduped.append(action)
            continue
        details = dict(action.details or {})
        if not bool(details.get("concrete")):
            deduped.append(action)
            continue
        key = _normalized_concrete_build_details_for_dedup(details)
        existing_index = kept_indices.get(key)
        if existing_index is None:
            kept_indices[key] = len(deduped)
            deduped.append(action)
            continue
        existing_action = deduped[existing_index]
        if int(action.value or 0) < int(existing_action.value or 0):
            deduped[existing_index] = action
    return deduped


def _dedupe_dominated_concrete_association_actions(actions: Sequence[Action]) -> List[Action]:
    deduped: List[Action] = []
    kept_indices: Dict[str, int] = {}
    for action in actions:
        if action.type != ActionType.MAIN_ACTION or str(action.card_name or "") != "association":
            deduped.append(action)
            continue
        details = dict(action.details or {})
        if not bool(details.get("concrete")):
            deduped.append(action)
            continue
        key = _normalized_concrete_association_details_for_dedup(details)
        existing_index = kept_indices.get(key)
        if existing_index is None:
            kept_indices[key] = len(deduped)
            deduped.append(action)
            continue
        existing_action = deduped[existing_index]
        if int(action.value or 0) < int(existing_action.value or 0):
            deduped[existing_index] = action
    return deduped


def _enumerate_pending_decision_actions(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> List[Action]:
    if str(state.pending_decision_kind or "").strip() == "":
        return []
    if state.pending_decision_player_id != player_id:
        return []
    if state.pending_decision_kind == "cards_discard":
        discard_target = int(state.pending_decision_payload.get("discard_target", 0))
        return _enumerate_pending_cards_discard_actions(player, discard_target)
    if state.pending_decision_kind == "break_discard":
        discard_target = int(state.pending_decision_payload.get("discard_target", 0))
        return _enumerate_pending_break_discard_actions(player, discard_target)
    if state.pending_decision_kind == "opening_draft_keep":
        keep_target = int(state.pending_decision_payload.get("keep_target", 0))
        return _enumerate_pending_opening_draft_actions(player, keep_target)
    if state.pending_decision_kind == "conservation_reward":
        return _enumerate_pending_conservation_reward_actions(state, player, player_id)
    raise ValueError(f"Unsupported pending decision kind: {state.pending_decision_kind}")


def _enumerate_concrete_animals_actions(
    state: GameState,
    player: PlayerState,
    player_id: int,
    template_action: Action,
) -> List[Action]:
    strength = int((template_action.details or {}).get("effective_strength", 0))
    options = list_legal_animals_options(state=state, player_id=player_id, strength=strength)
    actions: List[Action] = []

    def _resolve_animals_details(details_payload: Dict[str, Any]) -> List[Tuple[Dict[str, Any], str]]:
        return _resolve_action_detail_variants_by_simulation(
            state=state,
            player_id=player_id,
            base_details=details_payload,
            executor=lambda sim_state, sim_player, sim_details: _perform_animals_action_effect(
                state=sim_state,
                player=sim_player,
                strength=strength,
                details=sim_details,
                player_id=player_id,
            ),
            invalid_effect_log_prefixes=("animals_followup_skipped_missing_hand_card",),
        )

    for option in options:
        base_label = " ; then ".join(_format_animals_play_step_for_human(step, player) for step in option["plays"])
        effect_variants = _enumerate_animals_effect_choice_variants(player, option["plays"], state=state)
        for extra_effect_details, effect_label in effect_variants:
            label = base_label if not effect_label else f"{base_label} ; {effect_label}"
            details = {"animals_sequence_index": int(option["index"]) - 1}
            details.update(copy.deepcopy(extra_effect_details))
            requires_resolution = (
                len(option["plays"]) >= 2
                or _player_has_sponsor(player, 228)
                or int(player.sponsor_tokens_by_number.get(253, 0)) > 0
            )
            resolved_variants = [(copy.deepcopy(details), "")] if not requires_resolution else _resolve_animals_details(details)
            for resolved_details, resolved_label in resolved_variants:
                final_label = label if not resolved_label else f"{label} ; {resolved_label}"
                actions.append(
                    _make_concrete_action(
                        template_action,
                        label=final_label,
                        extra_details=resolved_details,
                    )
                )
    return actions


def _enumerate_concrete_build_actions(
    state: GameState,
    player: PlayerState,
    player_id: int,
    template_action: Action,
) -> List[Action]:
    strength = int((template_action.details or {}).get("effective_strength", 0))
    min_total_size_exclusive = int((template_action.details or {}).get("min_total_size_exclusive", 0))
    upgraded = bool(player.action_upgraded["build"])
    actions: List[Action] = []
    max_build_steps = 2 if upgraded else 1

    def _build_options_cache_key(
        cached_state: GameState,
        remaining_strength: int,
        already_built_types: Set[BuildingType],
    ) -> Tuple[str, str, int, Tuple[str, ...], bool, bool]:
        cached_player = cached_state.players[player_id]
        buildings = ()
        if cached_player.zoo_map is not None:
            buildings = tuple(
                sorted(
                    (
                        building.type.name,
                        building.origin_hex.x,
                        building.origin_hex.y,
                        building.rotation.value,
                    )
                    for building in cached_player.zoo_map.buildings.values()
                )
            )
        capped_strength = min(5, int(remaining_strength))
        return (
            str(cached_state.map_image_name or ""),
            json.dumps(buildings, separators=(",", ":"), ensure_ascii=True),
            capped_strength,
            tuple(sorted(building_type.name for building_type in already_built_types)),
            bool(cached_player.action_upgraded["build"]),
            bool(_player_has_sponsor(cached_player, 219)),
        )

    def _cached_list_legal_build_options(
        cached_state: GameState,
        remaining_strength: int,
        already_built_types: Set[BuildingType],
    ) -> List[Dict[str, Any]]:
        if getattr(list_legal_build_options, "__module__", None) != __name__:
            return list_legal_build_options(
                state=cached_state,
                player_id=player_id,
                strength=min(5, int(remaining_strength)),
                already_built_types=already_built_types,
            )
        key = _build_options_cache_key(cached_state, remaining_strength, already_built_types)
        cached = _BUILD_ENUM_LEGAL_OPTION_CACHE.get(key)
        if cached is None:
            cached = list_legal_build_options(
                state=cached_state,
                player_id=player_id,
                strength=min(5, int(remaining_strength)),
                already_built_types=already_built_types,
            )
            _BUILD_ENUM_LEGAL_OPTION_CACHE[key] = copy.deepcopy(cached)
        return copy.deepcopy(cached)

    def _simulate_build_step_for_enumeration(
        simulated_state: GameState,
        option: Dict[str, Any],
        step_bonus_details: Dict[str, Any],
    ) -> GameState:
        next_state = copy.deepcopy(simulated_state)
        next_player = next_state.players[player_id]
        _ensure_player_map_initialized(next_state, next_player)
        if next_player.zoo_map is None:
            return next_state

        building_type = BuildingType[str(option.get("building_type") or "")]
        origin_raw = list(option.get("origin") or [])
        rotation_name = str(option.get("rotation") or "ROT_0")
        if len(origin_raw) != 2:
            return next_state
        picked_building = Building(
            building_type,
            HexTile(int(origin_raw[0]), int(origin_raw[1])),
            Rotation[rotation_name],
        )
        next_player.money -= int(option.get("cost", 0))
        next_player.zoo_map.add_building(picked_building)

        choice_details = copy.deepcopy(step_bonus_details)
        for bonus_name, _bonus_coord in _building_bonuses(next_state, picked_building):
            if bonus_name == "5coins":
                next_player.money += 5
                continue
            if bonus_name == "reputation":
                next_player.reputation += 1
                continue
            if bonus_name == "x_token":
                next_player.x_tokens = min(5, int(next_player.x_tokens) + 1)
                continue
            if bonus_name == "action_to_slot_1":
                queued_choice = _pop_detail_queue_entry(choice_details, "bonus_action_to_slot_1_choices")
                if queued_choice is not None:
                    chosen_action = str(queued_choice.get("action_name") or "").strip()
                    if chosen_action in next_player.action_order:
                        _rotate_action_card_to_slot_1(next_player, chosen_action)
                continue
            if bonus_name == "card_in_reputation_range":
                queued_choice = _pop_detail_queue_entry(choice_details, "build_card_bonus_choices")
                if queued_choice is not None:
                    _take_one_card_from_deck_or_reputation_range_with_choice(
                        next_state,
                        next_player,
                        draw_source=str(queued_choice.get("draw_source") or ""),
                        display_index=queued_choice.get("display_index"),
                    )

        if _player_has_sponsor(next_player, 241):
            next_player.money += sum(
                1
                for cell in option.get("cells") or []
                if _adjacent_terrain_count_for_cells(next_player, [tuple(cell)], Terrain.WATER) > 0
            )
        if _player_has_sponsor(next_player, 242):
            next_player.money += sum(
                1
                for cell in option.get("cells") or []
                if _adjacent_terrain_count_for_cells(next_player, [tuple(cell)], Terrain.ROCK) > 0
            )

        return next_state

    if not upgraded:
        options = [
            option
            for option in _cached_list_legal_build_options(state, strength, set())
            if int(option.get("cost", 0)) <= int(player.money)
        ]
        if not options:
            return []
        for option in options:
            if int(option.get("size", 0)) <= min_total_size_exclusive:
                continue
            for bonus_details, bonus_label in _enumerate_build_bonus_choice_variants(state, player_id, option):
                actions.append(
                    _make_concrete_action(
                        template_action,
                        label=_build_step_action_label(option, bonus_label),
                        extra_details=_merge_list_detail_fragments(
                            {"selections": [_build_selection_payload(option)]},
                            bonus_details,
                        ),
                    )
                )
        return actions

    seen_sequences: Set[str] = set()

    def _recurse(
        simulated_state: GameState,
        remaining_strength: int,
        built_types: Set[BuildingType],
        built_count: int,
        built_size_total: int,
        selection_sequence: List[Dict[str, Any]],
        accumulated_bonus_details: Dict[str, Any],
        label_steps: List[str],
    ) -> None:
        if built_count >= max_build_steps:
            return
        simulated_player = simulated_state.players[player_id]
        options = [
            option
            for option in _cached_list_legal_build_options(
                simulated_state,
                remaining_strength,
                built_types,
            )
            if int(option.get("cost", 0)) <= int(simulated_player.money)
        ]
        for option in options:
            for step_bonus_details, step_bonus_label in _enumerate_build_bonus_choice_variants(
                simulated_state,
                player_id,
                option,
            ):
                selection_payload = _build_selection_payload(option)
                next_selection_sequence = selection_sequence + [selection_payload]
                next_accumulated_bonus_details = _merge_list_detail_fragments(
                    accumulated_bonus_details,
                    step_bonus_details,
                )
                next_label_steps = label_steps + [_build_step_action_label(option, step_bonus_label)]
                next_total_size = built_size_total + int(option.get("size", 0))
                key = json.dumps(
                    _merge_list_detail_fragments(
                        {"selections": next_selection_sequence},
                        next_accumulated_bonus_details,
                    ),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                )
                if next_total_size > min_total_size_exclusive and key not in seen_sequences:
                    seen_sequences.add(key)
                    actions.append(
                        _make_concrete_action(
                            template_action,
                            label=" ; then ".join(next_label_steps),
                            extra_details=_merge_list_detail_fragments(
                                {"selections": next_selection_sequence},
                                next_accumulated_bonus_details,
                            ),
                        )
                    )

                next_remaining_strength = remaining_strength - int(option.get("size", 0))
                if next_remaining_strength <= 0:
                    continue
                if built_count + 1 >= max_build_steps:
                    continue

                next_state = _simulate_build_step_for_enumeration(
                    simulated_state,
                    option,
                    step_bonus_details,
                )
                _recurse(
                    next_state,
                    next_remaining_strength,
                    built_types | {BuildingType[str(option.get("building_type") or "")]},
                    built_count + 1,
                    next_total_size,
                    next_selection_sequence,
                    next_accumulated_bonus_details,
                    next_label_steps,
                )

    _recurse(copy.deepcopy(state), strength, set(), 0, 0, [], {}, [])
    return actions


def _enumerate_concrete_association_actions(
    state: GameState,
    player: PlayerState,
    player_id: int,
    template_action: Action,
) -> List[Action]:
    strength = int((template_action.details or {}).get("effective_strength", 0))
    options = list_legal_association_options(state=state, player_id=player_id, strength=strength)
    if not options:
        return []

    if bool(player.action_upgraded["association"]):
        actions: List[Action] = []

        def _recurse(
            simulated_state: GameState,
            remaining_strength: int,
            used_task_kinds: Set[str],
            task_sequence: List[Dict[str, Any]],
            label_parts: List[str],
        ) -> None:
            simulated_player = simulated_state.players[player_id]
            next_options = [
                option
                for option in list_legal_association_options(
                    state=simulated_state,
                    player_id=player_id,
                    strength=remaining_strength,
                )
                if str(option.get("task_kind") or "") not in used_task_kinds
            ]
            for option in next_options:
                reward_variants = [({}, "")]
                if str(option.get("task_kind") or "") == "conservation_project":
                    reward_variants = _map_left_track_reward_variants_for_option(
                        state=simulated_state,
                        player=simulated_player,
                        player_id=player_id,
                        option=option,
                    )

                for reward_details, reward_label in reward_variants:
                    next_state = copy.deepcopy(simulated_state)
                    next_player = next_state.players[player_id]
                    task_request = _association_task_request_from_option(option, reward_details)
                    task_label = _association_task_label(option, reward_label)
                    _apply_association_selected_option(
                        state=next_state,
                        player=next_player,
                        player_id=player_id,
                        selected=copy.deepcopy(option),
                        effect_details=task_request.get("map_left_track_choice")
                        if isinstance(task_request.get("map_left_track_choice"), dict)
                        else None,
                        allow_interactive=False,
                    )
                    next_sequence = list(task_sequence) + [task_request]
                    next_labels = list(label_parts) + [task_label]
                    action_label = " + ".join(next_labels)
                    base_details = {
                        "association_task_sequence": copy.deepcopy(next_sequence),
                    }
                    actions.append(
                        _make_concrete_action(
                            template_action,
                            label=action_label,
                            extra_details=base_details,
                        )
                    )
                    donation_cost = _current_donation_cost(next_state)
                    if int(next_player.money) >= donation_cost:
                        donation_details = copy.deepcopy(base_details)
                        donation_details["make_donation"] = True
                        actions.append(
                            _make_concrete_action(
                                template_action,
                                label=f"{action_label} + donate({donation_cost})",
                                extra_details=donation_details,
                            )
                        )
                    strength_cost = int(
                        option.get(
                            "strength_cost",
                            _association_task_strength_cost(
                                simulated_player,
                                str(option.get("task_kind") or ""),
                            ),
                        )
                    )
                    remaining_after = remaining_strength - strength_cost
                    if remaining_after <= 0:
                        continue
                    _recurse(
                        next_state,
                        remaining_after,
                        used_task_kinds | {str(option.get("task_kind") or "")},
                        next_sequence,
                        next_labels,
                    )

        _recurse(
            copy.deepcopy(state),
            strength,
            set(),
            [],
            [],
        )
        return actions

    actions: List[Action] = []
    for option in options:
        reward_variants = [({}, "")]
        if str(option.get("task_kind") or "") == "conservation_project":
            reward_variants = _map_left_track_reward_variants_for_option(
                state=state,
                player=player,
                player_id=player_id,
                option=option,
            )

        for reward_details, reward_label in reward_variants:
            task_request = _association_task_request_from_option(option, reward_details)
            details = {
                "association_task_sequence": [task_request],
            }
            actions.append(
                _make_concrete_action(
                    template_action,
                    label=_association_task_label(option, reward_label),
                    extra_details=details,
                )
            )
    return actions


def _enumerate_concrete_sponsors_actions(
    state: GameState,
    player: PlayerState,
    player_id: int,
    template_action: Action,
) -> List[Action]:
    strength = int((template_action.details or {}).get("effective_strength", 0))
    sponsors_upgraded = bool(player.action_upgraded["sponsors"])
    level_cap = strength + 1 if sponsors_upgraded else strength
    candidates = _list_legal_sponsor_candidates(
        state=state,
        player=player,
        sponsors_upgraded=sponsors_upgraded,
    )
    playable = [
        cand
        for cand in candidates
        if bool(cand.get("playable_now")) and int(cand.get("level", 0)) <= level_cap
    ]

    break_details = {"use_break_ability": True, "sponsor_selections": []}
    if not playable:
        break_details["sponsors_break_only"] = True
        return [_make_concrete_action(template_action, extra_details=break_details)]

    actions: List[Action] = [
        _make_concrete_action(
            template_action,
            label=f"break => +{strength * (2 if sponsors_upgraded else 1)} money",
            extra_details=break_details,
        )
    ]

    sequence_list: List[List[Dict[str, Any]]] = []
    if not sponsors_upgraded:
        sequence_list = [[candidate] for candidate in playable]
    else:
        def _recurse_sequences(available: List[Dict[str, Any]], current: List[Dict[str, Any]], total_level: int) -> None:
            if current:
                sequence_list.append(list(current))
            for candidate in available:
                level = int(candidate.get("level", 0))
                if total_level + level > level_cap:
                    continue
                next_available = [
                    item
                    for item in available
                    if str(item.get("card_instance_id") or "") != str(candidate.get("card_instance_id") or "")
                ]
                _recurse_sequences(next_available, current + [candidate], total_level + level)

        _recurse_sequences(playable, [], 0)

    for sequence in sequence_list:
        variant_sets = [
            _enumerate_sponsor_candidate_detail_variants(state, player, player_id, candidate)
            for candidate in sequence
        ]
        fragment_products = list(product(*variant_sets)) if variant_sets else [tuple()]
        for fragment_product in fragment_products:
            extra_details: Dict[str, Any] = {
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": candidate["source"],
                        "source_index": int(candidate["source_index"]),
                        "card_instance_id": str(candidate["card_instance_id"]),
                    }
                    for candidate in sequence
                ],
            }
            label_parts: List[str] = []
            for candidate, (fragment_details, fragment_label) in zip(sequence, fragment_product):
                card = candidate["card"]
                source = str(candidate["source"])
                source_label = f"{source}[{int(candidate['source_index']) + 1}]"
                card_label = f"{source_label} #{card.number} {card.name}"
                if fragment_label:
                    card_label += f" ({fragment_label})"
                label_parts.append(card_label)
                extra_details = _merge_detail_fragments(extra_details, fragment_details)
            resolved_variants = [(copy.deepcopy(extra_details), "")]
            if int(player.sponsor_tokens_by_number.get(253, 0)) > 0:
                resolved_variants = _resolve_action_detail_variants_by_simulation(
                    state=state,
                    player_id=player_id,
                    base_details=extra_details,
                    executor=lambda sim_state, sim_player, sim_details: _perform_sponsors_action_effect(
                        state=sim_state,
                        player=sim_player,
                        strength=strength,
                        details=sim_details,
                        player_id=player_id,
                    ),
                )
            for resolved_details, resolved_label in resolved_variants:
                final_label = " + ".join(label_parts)
                if resolved_label:
                    final_label = f"{final_label} ; {resolved_label}"
                actions.append(
                    _make_concrete_action(
                        template_action,
                        label=final_label,
                        extra_details=resolved_details,
                    )
                )

    return actions


def legal_actions(
    player: PlayerState,
    state: Optional[GameState] = None,
    player_id: Optional[int] = None,
) -> List[Action]:
    if state is not None and player_id is not None and str(state.pending_decision_kind or "").strip():
        return _enumerate_pending_decision_actions(state, player, player_id)

    actions = _legal_actions_impl(
        player=player,
        action_factory=Action,
        main_action_type=ActionType.MAIN_ACTION,
        x_token_action_type=ActionType.X_TOKEN,
        max_x_tokens=5,
    )
    annotated: List[Action] = []
    for action in actions:
        if action.type != ActionType.MAIN_ACTION:
            annotated.append(action)
            continue
        card_name = str(action.card_name or "")
        if card_name not in player.action_order:
            annotated.append(action)
            continue
        base_strength = _action_base_strength(player, card_name)
        x_spent = int(action.value or 0)
        merged_details = dict(action.details or {})
        merged_details["base_strength"] = base_strength
        effective_strength = _effective_action_strength(
            player,
            card_name,
            x_spent=x_spent,
            base_strength=base_strength,
        )
        if card_name in {"cards", "animals"} and effective_strength > 5:
            continue
        merged_details["effective_strength"] = effective_strength
        annotated.append(
            Action(
                ActionType.MAIN_ACTION,
                value=action.value,
                card_name=action.card_name,
                details=merged_details,
            )
        )

    build_strength_floor_by_x: Dict[int, int] = {}
    prior_build_strength = 0
    for build_action in sorted(
        (
            action
            for action in annotated
            if action.type == ActionType.MAIN_ACTION and str(action.card_name or "") == "build"
        ),
        key=lambda action: (
            int((action.details or {}).get("effective_strength", 0)),
            int(action.value or 0),
        ),
    ):
        x_spent = int(build_action.value or 0)
        effective_strength = int((build_action.details or {}).get("effective_strength", 0))
        build_strength_floor_by_x[x_spent] = prior_build_strength
        prior_build_strength = max(prior_build_strength, effective_strength)

    if build_strength_floor_by_x:
        updated_annotated: List[Action] = []
        for action in annotated:
            if action.type == ActionType.MAIN_ACTION and str(action.card_name or "") == "build":
                details = dict(action.details or {})
                details["min_total_size_exclusive"] = build_strength_floor_by_x.get(int(action.value or 0), 0)
                updated_annotated.append(
                    Action(
                        ActionType.MAIN_ACTION,
                        value=action.value,
                        card_name=action.card_name,
                        details=details,
                    )
                )
                continue
            updated_annotated.append(action)
        annotated = updated_annotated

    if state is None or player_id is None:
        return annotated

    concrete: List[Action] = []
    for action in annotated:
        if action.type == ActionType.X_TOKEN:
            for action_name in player.action_order:
                concrete.append(
                    _make_concrete_action(
                        action,
                        card_name=action_name,
                    )
                )
            continue
        if action.type != ActionType.MAIN_ACTION:
            concrete.append(action)
            continue
        card_name = str(action.card_name or "")
        if card_name == "animals":
            concrete.extend(_enumerate_concrete_animals_actions(state, player, player_id, action))
            continue

        if card_name == "sponsors":
            concrete.extend(_enumerate_concrete_sponsors_actions(state, player, player_id, action))
            continue

        if card_name == "association":
            concrete.extend(_enumerate_concrete_association_actions(state, player, player_id, action))
            continue

        if card_name == "build":
            concrete.extend(_enumerate_concrete_build_actions(state, player, player_id, action))
            continue

        if card_name == "cards":
            concrete.extend(_enumerate_concrete_cards_actions(state, player, action))
            continue

        if card_name:
            concrete.append(_make_concrete_action(action))
    concrete = _dedupe_dominated_concrete_sponsors_actions(concrete)
    concrete = _dedupe_dominated_concrete_animals_actions(concrete)
    concrete = _dedupe_dominated_concrete_build_actions(concrete)
    concrete = _dedupe_dominated_concrete_association_actions(concrete)
    return concrete


def _find_card_in_hand(player: PlayerState, card_name: str) -> Optional[AnimalCard]:
    for card in player.hand:
        if card.name == card_name:
            return card
    return None


def _occupy_smallest_fitting_enclosure(player: PlayerState, required_size: int) -> Optional[Enclosure]:
    candidates: List[Tuple[int, Enclosure]] = []
    for idx, enclosure in enumerate(player.enclosures):
        if not _enclosure_is_standard(enclosure):
            continue
        if _enclosure_has_animals(enclosure):
            continue
        if enclosure.size >= required_size:
            candidates.append((idx, enclosure))
    if not candidates:
        return None
    _, target = min(candidates, key=lambda x: x[1].size)
    target.used_capacity = 1
    _sync_enclosure_occupied(target)
    return target


def _move_action_order_to_slot(
    action_order: Sequence[str],
    chosen_action: str,
    slot_number: int,
) -> List[str]:
    order = list(action_order)
    if chosen_action not in order:
        return order
    idx = order.index(chosen_action)
    card = order.pop(idx)
    target_index = max(0, min(len(order), int(slot_number) - 1))
    order.insert(target_index, card)
    return order


def _rotate_action_card_to_slot(player: PlayerState, chosen_action: str, slot_number: int) -> int:
    if chosen_action not in player.action_order:
        raise ValueError(f"Unknown action card '{chosen_action}'.")
    idx = player.action_order.index(chosen_action)
    player.action_order[:] = _move_action_order_to_slot(player.action_order, chosen_action, slot_number)
    return idx + 1


def _rotate_action_card_to_slot_1(player: PlayerState, chosen_action: str) -> int:
    return _rotate_action_card_to_slot(player, chosen_action, 1)


def _rotate_action_card_to_slot_5(player: PlayerState, chosen_action: str) -> int:
    return _rotate_action_card_to_slot(player, chosen_action, 5)


def _cards_table_values(strength: int, upgraded: bool) -> Tuple[int, int, bool]:
    if strength <= 0:
        return 0, 0, False
    idx = min(5, max(1, strength)) - 1
    if upgraded:
        return (
            CARDS_DRAW_TABLE_UPGRADED[idx],
            CARDS_DISCARD_TABLE_UPGRADED[idx],
            strength >= 3,
        )
    return (
        CARDS_DRAW_TABLE_BASE[idx],
        CARDS_DISCARD_TABLE_BASE[idx],
        strength >= 5,
    )


def _reputation_display_limit(reputation: int) -> int:
    rep = max(0, min(15, reputation))
    if rep <= 0:
        return 0
    if rep == 1:
        return 1
    if rep <= 3:
        return 2
    if rep <= 6:
        return 3
    if rep <= 9:
        return 4
    if rep <= 12:
        return 5
    return 6


def _take_display_cards(state: GameState, indices: List[int]) -> List[AnimalCard]:
    if len(indices) != len(set(indices)):
        raise ValueError("Display indices must be unique.")
    for idx in indices:
        if idx < 0 or idx >= len(state.zoo_display):
            raise ValueError("Display index out of range.")

    taken: List[AnimalCard] = []
    for idx in sorted(indices, reverse=True):
        taken.append(state.zoo_display.pop(idx))
    taken.reverse()
    return taken


def _bga_conservation_points(conservation: int) -> int:
    return _bga_conservation_points_impl(conservation)


def _progress_score(player: PlayerState) -> int:
    return _progress_score_impl(player)


def _sponsor_endgame_bonus(state: GameState, player: PlayerState) -> Tuple[int, int]:
    sponsor_numbers = {card.number for card in _player_played_sponsor_cards(player)}
    if not sponsor_numbers:
        return 0, 0

    inventory = _player_icon_inventory(player)
    snapshot = _player_icon_snapshot(player)
    bonus_appeal = 0
    bonus_conservation = 0

    if 201 in sponsor_numbers:
        science = int(inventory.get("science", 0))
        if science >= 6:
            bonus_conservation += 2
        elif science >= 3:
            bonus_conservation += 1

    if 203 in sponsor_numbers and len(player.universities) >= 3:
        bonus_conservation += 1
    if 208 in sponsor_numbers and len(snapshot["categories"]) >= 5:
        bonus_conservation += 1
    if 209 in sponsor_numbers and len(player.universities) >= 3:
        bonus_conservation += 1
    if 210 in sponsor_numbers and player.zoo_map is not None:
        kiosks = sum(1 for b in player.zoo_map.buildings.values() if b.type == BuildingType.KIOSK)
        if kiosks >= 5:
            bonus_conservation += 1
    if 211 in sponsor_numbers:
        occupied_size_1 = sum(
            1 for e in player.enclosures if _enclosure_is_standard(e) and e.size == 1 and _enclosure_has_animals(e)
        )
        if occupied_size_1 >= 5:
            bonus_conservation += 1
    if 214 in sponsor_numbers:
        bonus_appeal += int(player.x_tokens)
    if 215 in sponsor_numbers and int(player.supported_conservation_project_actions) >= 5:
        bonus_conservation += 1
    if 216 in sponsor_numbers and int(player.reputation) >= 9:
        bonus_conservation += 1
    if 217 in sponsor_numbers and _is_map_completely_covered(player):
        bonus_appeal += 5
    if 218 in sponsor_numbers and int(player.supported_conservation_project_actions) >= 5:
        bonus_conservation += 1
    if 219 in sponsor_numbers:
        pair_count = min(int(inventory.get("water", 0)), int(inventory.get("rock", 0)))
        bonus_appeal += min(6, pair_count * 2)
    if 220 in sponsor_numbers and int(player.reputation) >= 9:
        bonus_conservation += 1
    if 221 in sponsor_numbers and player.zoo_map is not None:
        border = _player_border_coords(player)
        covered = _player_all_covered_cells(player)
        if border and border.issubset(covered):
            bonus_conservation += 1
    if 225 in sponsor_numbers:
        distinct_continents = sum(1 for _, amount in snapshot["continents"].items() if int(amount) > 0)
        if distinct_continents >= 5:
            bonus_conservation += 1
    if 226 in sponsor_numbers:
        distinct_continents = sum(1 for _, amount in snapshot["continents"].items() if int(amount) > 0)
        if distinct_continents >= 5:
            bonus_conservation += 1
    if 241 in sponsor_numbers and _all_terrain_spaces_adjacent_to_buildings(player, Terrain.WATER):
        bonus_conservation += 1
    if 242 in sponsor_numbers and _all_terrain_spaces_adjacent_to_buildings(player, Terrain.ROCK):
        bonus_conservation += 1
    if 243 in sponsor_numbers and int(inventory.get("herbivore", 0)) >= 6:
        bonus_conservation += 1
    if 244 in sponsor_numbers and int(inventory.get("bird", 0)) >= 6:
        bonus_conservation += 1
    if 245 in sponsor_numbers and int(inventory.get("water", 0)) >= 6:
        bonus_conservation += 1
    if 246 in sponsor_numbers and int(inventory.get("rock", 0)) >= 6:
        bonus_conservation += 1
    if 247 in sponsor_numbers and int(inventory.get("primate", 0)) >= 6:
        bonus_conservation += 1
    if 251 in sponsor_numbers:
        bear_icons = int(inventory.get("bear", 0))
        if bear_icons >= 6:
            bonus_conservation += 2
        elif bear_icons >= 3:
            bonus_conservation += 1
    if 257 in sponsor_numbers and _is_map_completely_covered(player):
        bonus_appeal += 5
    if 258 in sponsor_numbers:
        bonus_conservation += _count_terrain_spaces_not_adjacent_to_buildings(player, Terrain.WATER) // 2
    if 259 in sponsor_numbers:
        bonus_conservation += _count_terrain_spaces_not_adjacent_to_buildings(player, Terrain.ROCK) // 2
    if 260 in sponsor_numbers:
        bonus_conservation += _count_empty_fillable_spaces(player) // 6
    if 261 in sponsor_numbers and len(snapshot["categories"]) >= 5:
        bonus_conservation += 1
    if 264 in sponsor_numbers:
        bonus_conservation += _count_placement_bonus_spaces_not_adjacent_to_buildings(state, player) // 2

    return bonus_appeal, bonus_conservation


def _final_scoring_threshold_bonus(
    count: int,
    thresholds: Sequence[Tuple[int, int]],
) -> int:
    bonus = 0
    for threshold, value in thresholds:
        if count >= int(threshold):
            bonus = int(value)
    return bonus


def _all_border_spaces_adjacent_to_buildings(player: PlayerState) -> bool:
    if player.zoo_map is None:
        return False
    adjacent = _spaces_adjacent_to_player_buildings(player)
    covered = _player_all_covered_cells(player)
    border_spaces = {
        pair
        for pair in _player_border_coords(player)
        if pair not in covered
        and player.zoo_map.map_data.terrain.get(HexTile(pair[0], pair[1])) not in {Terrain.ROCK, Terrain.WATER}
    }
    return border_spaces.issubset(adjacent)


def _normalized_shape_key(cells: Sequence[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    if not cells:
        return tuple()
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def _count_distinct_building_shapes(player: PlayerState) -> int:
    shapes: Set[Tuple[Tuple[int, int], ...]] = set()
    if player.zoo_map is not None:
        for building in player.zoo_map.buildings.values():
            layout = tuple((tile.x, tile.y) for tile in building.type.layout)
            shapes.add(_normalized_shape_key(layout))
    for sponsor_building in player.sponsor_buildings:
        shapes.add(_normalized_shape_key(_sponsor_unique_base_footprint(sponsor_building.sponsor_number)))
    return len(shapes)


def _count_card_conditions_for_final_scoring(card: AnimalCard) -> int:
    if card.card_type == "animal":
        total = sum(1 for _, need in card.required_icons if int(need) > 0)
        if int(card.required_rock_adjacency) > 0:
            total += 1
        if int(card.required_water_adjacency) > 0:
            total += 1
        return total
    if card.card_type == "sponsor":
        total = 0
        for icon_name, need in _sponsor_required_icons(card).items():
            if _canonical_icon_key(icon_name) == "sponsorsii":
                continue
            if int(need) > 0:
                total += 1
        if _sponsor_min_reputation(card) > 0:
            total += 1
        if _sponsor_max_appeal(card) is not None:
            total += 1
        return total
    return 0


def _count_conditions_on_cards_in_zoo(player: PlayerState) -> int:
    return sum(_count_card_conditions_for_final_scoring(card) for card in player.zoo_cards)


def _final_scoring_card_id(card_ref: Any) -> str:
    data_id = getattr(card_ref, "data_id", None)
    if isinstance(data_id, str) and data_id:
        return data_id
    if isinstance(card_ref, str):
        return card_ref
    return ""


def _final_scoring_conservation_bonus_for_card(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    card_id: str,
) -> int:
    inventory = _player_icon_inventory(player)
    snapshot = _player_icon_snapshot(player)

    if card_id == "F001_LargeAnimalZoo":
        return _final_scoring_threshold_bonus(_player_large_animals_count(player), ((1, 1), (2, 2), (3, 3), (4, 4)))
    if card_id == "F002_SmallAnimalZoo":
        return _final_scoring_threshold_bonus(_player_small_animals_count(player), ((3, 1), (6, 2), (8, 3), (10, 4)))
    if card_id == "F003_ResearchZoo":
        return _final_scoring_threshold_bonus(int(snapshot["science"]), ((3, 1), (4, 2), (5, 3), (7, 4)))
    if card_id == "F004_ArchitecturalZoo":
        return sum(
            int(flag)
            for flag in (
                _all_terrain_spaces_adjacent_to_buildings(player, Terrain.WATER),
                _all_terrain_spaces_adjacent_to_buildings(player, Terrain.ROCK),
                _all_border_spaces_adjacent_to_buildings(player),
                _is_map_completely_covered(player),
            )
        )
    if card_id == "F005_ConservationZoo":
        return _final_scoring_threshold_bonus(
            int(player.supported_conservation_project_actions),
            ((2, 1), (3, 2), (4, 3), (5, 4)),
        )
    if card_id == "F006_NaturalistsZoo":
        return _final_scoring_threshold_bonus(_count_empty_fillable_spaces(player), ((6, 1), (12, 2), (18, 3), (24, 4)))
    if card_id == "F007_FavoriteZoo":
        return _final_scoring_threshold_bonus(int(player.reputation), ((6, 1), (9, 2), (12, 3), (15, 4)))
    if card_id == "F008_SponsoredZoo":
        return _final_scoring_threshold_bonus(len(_player_played_sponsor_cards(player)), ((3, 1), (5, 2), (7, 3), (9, 4)))
    if card_id == "F009_DiverseSpeciesZoo":
        other = state.players[(player_id + 1) % len(state.players)]
        other_snapshot = _player_icon_snapshot(other)
        categories = set(snapshot["categories"]) | set(other_snapshot["categories"])
        return min(
            4,
            sum(
                1
                for category in categories
                if int(snapshot["categories"].get(category, 0)) > int(other_snapshot["categories"].get(category, 0))
            ),
        )
    if card_id == "F010_ClimbingPark":
        return _final_scoring_threshold_bonus(int(inventory.get("rock", 0)), ((1, 1), (3, 2), (5, 3), (6, 4)))
    if card_id == "F011_AquaticPark":
        return _final_scoring_threshold_bonus(int(inventory.get("water", 0)), ((2, 1), (4, 2), (6, 3), (7, 4)))
    if card_id == "F012_DesignerZoo":
        return _final_scoring_threshold_bonus(_count_distinct_building_shapes(player), ((4, 1), (6, 2), (7, 3), (8, 4)))
    if card_id == "F013_SpecializedHabitatZoo":
        best = 0
        for continent_name, project_id in (
            ("Africa", "P103_Africa"),
            ("America", "P104_Americas"),
            ("Australia", "P105_Australia"),
            ("Asia", "P106_Asia"),
            ("Europe", "P107_Europe"),
        ):
            if project_id in player.supported_conservation_projects:
                continue
            best = max(best, int(snapshot["continents"].get(continent_name, 0)))
        return _final_scoring_threshold_bonus(best, ((3, 1), (4, 2), (5, 3), (6, 4)))
    if card_id == "F014_SpecializedSpeciesZoo":
        best = 0
        for category_name, project_id in (
            ("Primate", "P108_Primates"),
            ("Reptile", "P109_Reptiles"),
            ("Predator", "P110_Predators"),
            ("Herbivore", "P111_Herbivores"),
            ("Bird", "P112_Birds"),
        ):
            if project_id in player.supported_conservation_projects:
                continue
            best = max(best, int(snapshot["categories"].get(category_name, 0)))
        return _final_scoring_threshold_bonus(best, ((3, 1), (4, 2), (5, 3), (6, 4)))
    if card_id == "F015_CateredPicnicAreas":
        kiosk_count = 0
        pavilion_count = 0
        if player.zoo_map is not None:
            kiosk_count = sum(1 for building in player.zoo_map.buildings.values() if building.type == BuildingType.KIOSK)
            pavilion_count = sum(
                1 for building in player.zoo_map.buildings.values() if building.type == BuildingType.PAVILION
            )
        return _final_scoring_threshold_bonus(min(kiosk_count, pavilion_count), ((2, 1), (3, 2), (4, 3), (5, 4)))
    if card_id == "F016_AccessibleZoo":
        return _final_scoring_threshold_bonus(_count_conditions_on_cards_in_zoo(player), ((4, 1), (7, 2), (10, 3), (12, 4)))
    if card_id == "F017_InternationalZoo":
        other = state.players[(player_id + 1) % len(state.players)]
        other_snapshot = _player_icon_snapshot(other)
        continents = {key: int(value) for key, value in snapshot["continents"].items()}
        for partner in player.partner_zoos:
            continent_name = _normalize_continent_badge(partner)
            if continent_name is not None:
                continents[continent_name] = continents.get(continent_name, 0) + 1
        return min(
            4,
            sum(
                1
                for continent_name, amount in continents.items()
                if int(amount) > int(other_snapshot["continents"].get(continent_name, 0))
            ),
        )
    return 0


def _final_scoring_conservation_bonus_from_cards(state: GameState, player: PlayerState) -> int:
    try:
        player_id = state.players.index(player)
    except ValueError:
        player_id = 0
    total = 0
    for card_ref in player.final_scoring_cards:
        card_id = _final_scoring_card_id(card_ref)
        total += _final_scoring_conservation_bonus_for_card(
            state=state,
            player=player,
            player_id=player_id,
            card_id=card_id,
        )
    return total


def _final_score_points(state: GameState, player: PlayerState) -> int:
    bonus_appeal, bonus_conservation = _sponsor_endgame_bonus(state, player)
    final_scoring_bonus = _final_scoring_conservation_bonus_from_cards(state, player)
    total_appeal = int(player.appeal) + int(bonus_appeal)
    total_conservation = int(player.conservation) + int(bonus_conservation) + int(final_scoring_bonus)
    return total_appeal + _bga_conservation_points(total_conservation)


def _rank_scores(scores: Dict[str, int]) -> List[Tuple[int, str, int]]:
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranked: List[Tuple[int, str, int]] = []
    last_score: Optional[int] = None
    current_rank = 0
    for idx, (name, score) in enumerate(ordered, start=1):
        if score != last_score:
            current_rank = idx
            last_score = score
        ranked.append((current_rank, name, score))
    return ranked


def _maybe_trigger_endgame(state: GameState, player_id: int) -> bool:
    if state.endgame_trigger_player is not None:
        return False
    player = state.players[player_id]
    if _progress_score(player) < int(state.score_end_threshold):
        return False
    state.endgame_trigger_player = int(player_id)
    state.endgame_trigger_turn_index = int(state.turn_index) + 1
    state.effect_log.append(
        f"endgame_triggered_by={player.name} score={_progress_score(player)} threshold={state.score_end_threshold}"
    )
    return True


def _trigger_immediate_game_end(state: GameState, *, reason: str) -> bool:
    if bool(state.forced_game_over):
        return False
    state.forced_game_over = True
    state.forced_game_over_reason = str(reason or "").strip() or "forced_game_over"
    _clear_pending_decision(state)
    state.effect_log.append(f"forced_game_over:{state.forced_game_over_reason}")
    return True


def _completed_rounds(state: GameState) -> int:
    players = max(1, len(state.players))
    return int(state.turn_index) // players


def _maybe_trigger_round_limit_endgame(state: GameState) -> bool:
    round_limit = int(getattr(state, "max_rounds", 0) or 0)
    if round_limit <= 0 or bool(state.forced_game_over):
        return False
    if _completed_rounds(state) > round_limit:
        return _trigger_immediate_game_end(
            state,
            reason=f"round_limit_exceeded(completed={_completed_rounds(state)},limit={round_limit})",
        )
    return False


def _building_type_label(building_type: BuildingType) -> str:
    if building_type.name.startswith("SIZE_"):
        return f"enclosure_{building_type.name.split('_', 1)[1]}"
    names = {
        BuildingType.KIOSK: "kiosk",
        BuildingType.PAVILION: "pavilion",
        BuildingType.PETTING_ZOO: "petting_zoo",
        BuildingType.REPTILE_HOUSE: "reptile_house",
        BuildingType.LARGE_BIRD_AVIARY: "large_bird_aviary",
    }
    return names.get(building_type, building_type.name.lower())


def _rotation_name(rotation: Rotation) -> str:
    return rotation.name


def _building_bonuses(state: GameState, building: Building) -> List[Tuple[str, Tuple[int, int]]]:
    bonuses: List[Tuple[str, Tuple[int, int]]] = []
    for tile in building.layout:
        bonus = state.map_tile_bonuses.get((tile.x, tile.y))
        if bonus:
            bonuses.append((bonus, (tile.x, tile.y)))
    return bonuses


def _adjacent_terrain_count_for_building(
    player: PlayerState,
    building: Building,
    terrain: Terrain,
) -> int:
    if player.zoo_map is None:
        return 0
    adjacent: Set[Tuple[int, int]] = set()
    in_layout = {(tile.x, tile.y) for tile in building.layout}
    for tile in building.layout:
        for neighbor in tile.neighbors():
            pair = (neighbor.x, neighbor.y)
            if pair in in_layout:
                continue
            if player.zoo_map.map_data.terrain.get(neighbor) == terrain:
                adjacent.add(pair)
    return len(adjacent)


def _xy_neighbors(cell: Tuple[int, int]) -> List[Tuple[int, int]]:
    x, y = cell
    return [
        (x + 1, y),
        (x, y + 1),
        (x - 1, y + 1),
        (x - 1, y),
        (x, y - 1),
        (x + 1, y - 1),
    ]


def _player_map_grid_coords(player: PlayerState) -> Set[Tuple[int, int]]:
    if player.zoo_map is None:
        return set()
    return {(tile.x, tile.y) for tile in player.zoo_map.grid}


def _player_border_coords(player: PlayerState) -> Set[Tuple[int, int]]:
    if player.zoo_map is None:
        return set()
    return {(tile.x, tile.y) for tile in player.zoo_map.border_tiles(ignore_terrain=True)}


def _building_layout_xy(building: Building) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((tile.x, tile.y) for tile in building.layout))


def _is_standard_enclosure_empty(player: PlayerState, building: Building) -> bool:
    if building.type.subtype != BuildingSubType.ENCLOSURE_BASIC:
        return False
    origin = (building.origin_hex.x, building.origin_hex.y)
    rotation = building.rotation.name
    size = len(building.layout)
    for obj in player.enclosure_objects:
        if obj.origin == origin and obj.rotation == rotation and obj.size == size:
            return obj.animals_inside <= 0
    for enclosure in player.enclosures:
        if enclosure.origin == origin and enclosure.rotation == rotation and enclosure.size == size:
            return _enclosure_is_standard(enclosure) and not _enclosure_has_animals(enclosure)
    return True


def _player_all_covered_cells(player: PlayerState) -> Set[Tuple[int, int]]:
    covered: Set[Tuple[int, int]] = set()
    if player.zoo_map is not None:
        for building in player.zoo_map.buildings.values():
            covered.update((tile.x, tile.y) for tile in building.layout)
    for sponsor_building in player.sponsor_buildings:
        covered.update(sponsor_building.cells)
    return covered


def _adjacent_terrain_count_for_cells(
    player: PlayerState,
    cells: Sequence[Tuple[int, int]],
    terrain: Terrain,
) -> int:
    if player.zoo_map is None:
        return 0
    layout = set(cells)
    adjacent: Set[Tuple[int, int]] = set()
    for cell in layout:
        for neighbor in _xy_neighbors(cell):
            if neighbor in layout:
                continue
            neighbor_hex = HexTile(neighbor[0], neighbor[1])
            if player.zoo_map.map_data.terrain.get(neighbor_hex) == terrain:
                adjacent.add(neighbor)
    return len(adjacent)


def _rotate_relative_cell(cell: Tuple[int, int], steps_right: int) -> Tuple[int, int]:
    tile = HexTile(cell[0], cell[1])
    rotated = tile
    for _ in range(max(0, steps_right) % 6):
        rotated = rotated.rotate_right()
    return rotated.x, rotated.y


def _sponsor_unique_base_footprint(card_number: int) -> Tuple[Tuple[int, int], ...]:
    return SPONSOR_UNIQUE_BASE_FOOTPRINTS.get(card_number, ((0, 0),))


def _sponsor_unique_footprint_variants(card_number: int) -> List[Tuple[Tuple[int, int], ...]]:
    base = _sponsor_unique_base_footprint(card_number)
    variants: Set[Tuple[Tuple[int, int], ...]] = set()
    for step in range(6):
        rotated = tuple(sorted(_rotate_relative_cell(cell, step) for cell in base))
        variants.add(rotated)
    return sorted(variants)


def _list_legal_sponsor_unique_building_cells(
    state: GameState,
    player: PlayerState,
    sponsor_number: int,
) -> List[Tuple[Tuple[int, int], ...]]:
    _ensure_player_map_initialized(state, player)
    if player.zoo_map is None:
        return []
    if sponsor_number not in SPONSOR_UNIQUE_BUILDING_CARDS:
        return []

    grid = _player_map_grid_coords(player)
    border = _player_border_coords(player)
    covered = _player_all_covered_cells(player)
    has_diversity_researcher = _player_has_sponsor(player, 219)
    build_upgraded = bool(player.action_upgraded["build"])
    min_border = int(SPONSOR_UNIQUE_BORDER_MIN.get(sponsor_number, 0))
    rock_need = int(SPONSOR_UNIQUE_REQUIRED_ROCK.get(sponsor_number, 0))
    water_need = int(SPONSOR_UNIQUE_REQUIRED_WATER.get(sponsor_number, 0))
    ignore_adjacency = sponsor_number in SPONSOR_UNIQUE_IGNORE_ADJACENCY
    variants = _sponsor_unique_footprint_variants(sponsor_number)

    legal: Set[Tuple[Tuple[int, int], ...]] = set()
    for origin in sorted(grid):
        for variant in variants:
            placed = tuple(sorted((origin[0] + dx, origin[1] + dy) for dx, dy in variant))
            placed_set = set(placed)
            if not placed_set.issubset(grid):
                continue
            if placed_set & covered:
                continue

            blocked = False
            for x, y in placed:
                tile = HexTile(x, y)
                terrain = player.zoo_map.map_data.terrain.get(tile)
                if terrain in {Terrain.ROCK, Terrain.WATER} and not has_diversity_researcher:
                    blocked = True
                    break
                if terrain == Terrain.BUILD_2_REQUIRED and not build_upgraded:
                    blocked = True
                    break
            if blocked:
                continue

            if min_border > 0 and sum(1 for cell in placed if cell in border) < min_border:
                continue

            if not ignore_adjacency:
                if not covered:
                    if not any(cell in border for cell in placed):
                        continue
                else:
                    touches_existing = any(
                        neighbor in covered
                        for cell in placed
                        for neighbor in _xy_neighbors(cell)
                    )
                    if not touches_existing:
                        continue

            if rock_need > 0 and _adjacent_terrain_count_for_cells(player, placed, Terrain.ROCK) < rock_need:
                continue
            if water_need > 0 and _adjacent_terrain_count_for_cells(player, placed, Terrain.WATER) < water_need:
                continue
            legal.add(placed)

    return sorted(legal)


def _sponsor_unique_building_can_be_placed(
    state: GameState,
    player: PlayerState,
    sponsor_number: int,
) -> bool:
    if sponsor_number not in SPONSOR_UNIQUE_BUILDING_CARDS:
        return True
    return bool(
        _list_legal_sponsor_unique_building_cells(
            state=state,
            player=player,
            sponsor_number=sponsor_number,
        )
    )


def _place_sponsor_unique_building(
    state: GameState,
    player: PlayerState,
    card: AnimalCard,
    player_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    if card.number not in SPONSOR_UNIQUE_BUILDING_CARDS:
        return False
    details = details or {}
    legal = _list_legal_sponsor_unique_building_cells(
        state=state,
        player=player,
        sponsor_number=card.number,
    )
    if not legal:
        return False
    picked: Optional[Tuple[Tuple[int, int], ...]] = None
    selection = _pop_matching_detail_entry(
        details,
        "_validated_sponsor_unique_building_selections",
        card_instance_id=card.instance_id,
    )
    if selection is None:
        selection = _pop_matching_detail_entry(
            details,
            "sponsor_unique_building_selections",
            card_instance_id=card.instance_id,
        )
    if selection is not None:
        parsed = _parse_serialized_cells(selection.get("cells"))
        if parsed is None or parsed not in legal:
            raise ValueError(f"Illegal unique building placement for sponsor #{card.number}.")
        picked = parsed
    elif len(legal) == 1:
        picked = legal[0]
    else:
        raise ValueError(
            f"Sponsor #{card.number} requires explicit sponsor_unique_building_selections when multiple placements exist."
        )
    player.sponsor_buildings.append(
        SponsorBuilding(
            sponsor_number=card.number,
            cells=tuple(sorted(picked)),
            label=card.name,
        )
    )
    if _player_has_sponsor(player, 241):
        gain = sum(
            1
            for cell in picked
            if _adjacent_terrain_count_for_cells(player, [cell], Terrain.WATER) > 0
        )
        if gain > 0:
            player.money += gain
    if _player_has_sponsor(player, 242):
        gain = sum(
            1
            for cell in picked
            if _adjacent_terrain_count_for_cells(player, [cell], Terrain.ROCK) > 0
        )
        if gain > 0:
            player.money += gain
    resolved_player_id = player_id if player_id is not None else state.players.index(player)
    _maybe_apply_map_completion_reward(
        state=state,
        player=player,
        player_id=resolved_player_id,
    )
    return True


def _building_footprints(player: PlayerState) -> List[Dict[str, Any]]:
    footprints: List[Dict[str, Any]] = []
    if player.zoo_map is not None:
        for building in player.zoo_map.buildings.values():
            footprints.append(
                {
                    "kind": "normal",
                    "type": building.type.name,
                    "cells": set((tile.x, tile.y) for tile in building.layout),
                    "empty_standard_enclosure": _is_standard_enclosure_empty(player, building),
                    "sponsor_number": None,
                }
            )
    for sponsor_building in player.sponsor_buildings:
        footprints.append(
            {
                "kind": "sponsor",
                "type": "SPONSOR_UNIQUE",
                "cells": set(sponsor_building.cells),
                "empty_standard_enclosure": False,
                "sponsor_number": sponsor_building.sponsor_number,
            }
        )
    return footprints


def _footprints_adjacent(a_cells: Set[Tuple[int, int]], b_cells: Set[Tuple[int, int]]) -> bool:
    return any(neighbor in b_cells for cell in a_cells for neighbor in _xy_neighbors(cell))


def _count_side_entrance_adjacent_buildings(player: PlayerState) -> int:
    side_entrance_cells = set()
    for sponsor_building in player.sponsor_buildings:
        if sponsor_building.sponsor_number == 257:
            side_entrance_cells.update(sponsor_building.cells)
    if not side_entrance_cells:
        return 0

    count = 0
    for info in _building_footprints(player):
        if info.get("kind") == "sponsor" and int(info.get("sponsor_number") or -1) == 257:
            continue
        if bool(info.get("empty_standard_enclosure")):
            continue
        cells = set(info.get("cells") or set())
        if not cells:
            continue
        if _footprints_adjacent(cells, side_entrance_cells):
            count += 1
    return count


def _spaces_adjacent_to_player_buildings(player: PlayerState) -> Set[Tuple[int, int]]:
    if player.zoo_map is None:
        return set()
    grid = _player_map_grid_coords(player)
    covered = _player_all_covered_cells(player)
    adjacent: Set[Tuple[int, int]] = set()
    for cell in covered:
        for neighbor in _xy_neighbors(cell):
            if neighbor not in grid:
                continue
            if neighbor in covered:
                continue
            adjacent.add(neighbor)
    return adjacent


def _is_map_completely_covered(player: PlayerState) -> bool:
    if player.zoo_map is None:
        return False
    covered = _player_all_covered_cells(player)
    fillable = {
        (tile.x, tile.y)
        for tile in player.zoo_map.grid
        if player.zoo_map.map_data.terrain.get(tile) not in {Terrain.WATER, Terrain.ROCK}
    }
    return fillable.issubset(covered)


def _count_terrain_spaces_adjacent_to_buildings(player: PlayerState, terrain: Terrain) -> int:
    if player.zoo_map is None:
        return 0
    adjacent = _spaces_adjacent_to_player_buildings(player)
    return sum(
        1
        for x, y in adjacent
        if player.zoo_map.map_data.terrain.get(HexTile(x, y)) == terrain
    )


def _count_terrain_spaces_not_adjacent_to_buildings(player: PlayerState, terrain: Terrain) -> int:
    if player.zoo_map is None:
        return 0
    adjacent = _spaces_adjacent_to_player_buildings(player)
    covered = _player_all_covered_cells(player)
    total = 0
    for tile in player.zoo_map.grid:
        pair = (tile.x, tile.y)
        if pair in covered:
            continue
        if player.zoo_map.map_data.terrain.get(tile) != terrain:
            continue
        if pair in adjacent:
            continue
        total += 1
    return total


def _count_empty_border_spaces_adjacent_to_buildings(player: PlayerState) -> int:
    if player.zoo_map is None:
        return 0
    border = _player_border_coords(player)
    covered = _player_all_covered_cells(player)
    adjacent = _spaces_adjacent_to_player_buildings(player)
    total = 0
    for x, y in adjacent:
        pair = (x, y)
        if pair not in border:
            continue
        if pair in covered:
            continue
        terrain = player.zoo_map.map_data.terrain.get(HexTile(x, y))
        if terrain in {Terrain.ROCK, Terrain.WATER}:
            continue
        total += 1
    return total


def _count_empty_fillable_spaces(player: PlayerState) -> int:
    if player.zoo_map is None:
        return 0
    covered = _player_all_covered_cells(player)
    total = 0
    for tile in player.zoo_map.grid:
        pair = (tile.x, tile.y)
        if pair in covered:
            continue
        terrain = player.zoo_map.map_data.terrain.get(tile)
        if terrain in {Terrain.ROCK, Terrain.WATER}:
            continue
        total += 1
    return total


def _count_placement_bonus_spaces_adjacent_to_buildings(
    state: GameState,
    player: PlayerState,
) -> int:
    adjacent = _spaces_adjacent_to_player_buildings(player)
    covered = _player_all_covered_cells(player)
    total = 0
    for pair in state.map_tile_bonuses:
        if pair in covered:
            continue
        if pair in adjacent:
            total += 1
    return total


def _count_placement_bonus_spaces_not_adjacent_to_buildings(
    state: GameState,
    player: PlayerState,
) -> int:
    adjacent = _spaces_adjacent_to_player_buildings(player)
    covered = _player_all_covered_cells(player)
    total = 0
    for pair in state.map_tile_bonuses:
        if pair in covered:
            continue
        if pair in adjacent:
            continue
        total += 1
    return total


def _all_terrain_spaces_adjacent_to_buildings(player: PlayerState, terrain: Terrain) -> bool:
    if player.zoo_map is None:
        return False
    adjacent = _spaces_adjacent_to_player_buildings(player)
    covered = _player_all_covered_cells(player)
    terrain_cells: Set[Tuple[int, int]] = set()
    for tile in player.zoo_map.grid:
        pair = (tile.x, tile.y)
        if pair in covered:
            continue
        if player.zoo_map.map_data.terrain.get(tile) == terrain:
            terrain_cells.add(pair)
    if not terrain_cells:
        return False
    return terrain_cells.issubset(adjacent)


def _apply_cover_money_from_sponsors_241_242(player: PlayerState, building: Building) -> None:
    if _player_has_sponsor(player, 241):
        gain = sum(
            1
            for tile in building.layout
            if _adjacent_terrain_count_for_cells(player, [(tile.x, tile.y)], Terrain.WATER) > 0
        )
        if gain > 0:
            player.money += gain
    if _player_has_sponsor(player, 242):
        gain = sum(
            1
            for tile in building.layout
            if _adjacent_terrain_count_for_cells(player, [(tile.x, tile.y)], Terrain.ROCK) > 0
        )
        if gain > 0:
            player.money += gain


def _increment_enclosure_animals_inside(player: PlayerState, enclosure: Enclosure) -> None:
    if enclosure.origin is None:
        return
    ox, oy = enclosure.origin
    for item in player.enclosure_objects:
        if item.origin == (ox, oy) and item.rotation == enclosure.rotation and item.size == enclosure.size:
            item.animals_inside += 1
            return


def _apply_animal_to_enclosure(player: PlayerState, enclosure: Enclosure, spaces_used: int) -> None:
    used = max(0, int(spaces_used))
    if _enclosure_is_standard(enclosure):
        enclosure.used_capacity = 1
    elif used > 0:
        enclosure.used_capacity = min(
            int(enclosure.animal_capacity),
            _enclosure_used_capacity(enclosure) + used,
        )
    _sync_enclosure_occupied(enclosure)
    _increment_enclosure_animals_inside(player, enclosure)

    if enclosure.origin is None:
        return
    building = _find_zoo_building_by_origin_rotation_size(
        player=player,
        origin=enclosure.origin,
        rotation=enclosure.rotation,
        size=enclosure.size,
    )
    if building is None:
        return
    if building.type.subtype == BuildingSubType.ENCLOSURE_BASIC:
        building.empty_spaces = 0
        return
    if building.type.subtype == BuildingSubType.ENCLOSURE_SPECIAL and used > 0:
        building.empty_spaces = max(0, int(building.empty_spaces) - used)


def _ensure_player_map_initialized(state: GameState, player: PlayerState) -> None:
    if player.zoo_map is not None:
        return
    map_data = load_map_data_by_image_name(state.map_image_name)
    player.zoo_map = ArkNovaMap(map_data=map_data)


def _building_cells(building: Building) -> List[List[int]]:
    return [[x, y] for x, y in sorted((tile.x, tile.y) for tile in building.layout)]


def _building_layout_key(building: Building) -> Tuple[Tuple[int, int], ...]:
    return tuple(sorted((tile.x, tile.y) for tile in building.layout))


def _dedupe_legal_buildings_by_footprint(legal: Sequence[Building]) -> List[Building]:
    dedup_seen: Set[Tuple[str, Tuple[Tuple[int, int], ...]]] = set()
    deduped: List[Building] = []
    for building in legal:
        key = (building.type.name, _building_layout_key(building))
        if key in dedup_seen:
            continue
        dedup_seen.add(key)
        deduped.append(building)
    return deduped


def _building_cells_text(building: Building) -> str:
    cells = sorted((tile.x, tile.y) for tile in building.layout)
    return "[" + ",".join(f"({x},{y})" for x, y in cells) + "]"


def _find_zoo_building_by_origin_rotation_size(
    player: PlayerState,
    origin: Tuple[int, int],
    rotation: str,
    size: int,
) -> Optional[Building]:
    if player.zoo_map is None:
        return None
    ox, oy = origin
    for building in player.zoo_map.buildings.values():
        if (
            building.origin_hex.x == ox
            and building.origin_hex.y == oy
            and building.rotation.name == rotation
            and len(building.layout) == size
        ):
            return building
    return None


def list_legal_build_options(
    state: GameState,
    player_id: int,
    strength: int,
    already_built_types: Optional[Set[BuildingType]] = None,
) -> List[Dict[str, Any]]:
    player = state.players[player_id]
    _ensure_player_map_initialized(state, player)
    if player.zoo_map is None:
        return []
    upgraded = player.action_upgraded["build"]
    has_diversity_researcher = _player_has_sponsor(player, 219)
    max_size = max(0, strength)
    if max_size <= 0:
        return []
    legal = player.zoo_map.legal_building_placements(
        is_build_upgraded=upgraded,
        has_diversity_researcher=has_diversity_researcher,
        max_building_size=max_size,
        already_built_buildings=already_built_types or set(),
    )
    legal = _dedupe_legal_buildings_by_footprint(legal)
    legal.sort(
        key=lambda b: (
            _building_type_label(b.type),
            _building_layout_key(b),
            b.origin_hex.x,
            b.origin_hex.y,
            b.rotation.value,
        )
    )
    options: List[Dict[str, Any]] = []
    for idx, building in enumerate(legal, start=1):
        cells = _building_cells(building)
        bonuses = _building_bonuses(state, building)
        options.append(
            {
                "index": idx,
                "building_type": building.type.name,
                "building_label": _building_type_label(building.type),
                "cells": cells,
                # Compatibility fields; selection should prefer `cells`.
                "origin": [building.origin_hex.x, building.origin_hex.y],
                "rotation": _rotation_name(building.rotation),
                "layout": [tuple(cell) for cell in cells],
                "size": len(cells),
                "cost": len(cells) * 2,
                "placement_bonuses": [bonus for bonus, _ in bonuses],
            }
        )
    return options


def _parse_serialized_cells(selected_cells: Any) -> Optional[Tuple[Tuple[int, int], ...]]:
    if not isinstance(selected_cells, (list, tuple)) or not selected_cells:
        return None
    parsed_cells: List[Tuple[int, int]] = []
    for cell in selected_cells:
        if not isinstance(cell, (list, tuple)) or len(cell) != 2:
            return None
        x_raw, y_raw = cell
        if not isinstance(x_raw, int) or not isinstance(y_raw, int):
            return None
        parsed_cells.append((x_raw, y_raw))
    return tuple(sorted(parsed_cells))


def _find_legal_building_by_serialized_selection(
    legal: Sequence[Building],
    selection: Dict[str, Any],
) -> Optional[Building]:
    selected_type_name = str(selection.get("building_type", "")).strip()
    selected_cells = selection.get("cells")
    if selected_cells is None:
        selected_cells = selection.get("layout")
    target_layout = _parse_serialized_cells(selected_cells)
    if target_layout is not None:
        for building in legal:
            if building.type.name != selected_type_name:
                continue
            if _building_layout_key(building) == target_layout:
                return building
        return None

    selected_origin = selection.get("origin")
    selected_rotation_name = str(selection.get("rotation", "ROT_0")).strip() or "ROT_0"
    if not isinstance(selected_origin, (list, tuple)) or len(selected_origin) != 2:
        return None
    x_raw, y_raw = selected_origin
    if not isinstance(x_raw, int) or not isinstance(y_raw, int):
        return None
    try:
        selected_rotation = Rotation[selected_rotation_name]
    except KeyError:
        return None
    for building in legal:
        if (
            building.type.name == selected_type_name
            and building.origin_hex.x == x_raw
            and building.origin_hex.y == y_raw
            and building.rotation == selected_rotation
        ):
            return building
    return None


def _find_serialized_build_option(
    options: Sequence[Dict[str, Any]],
    selection: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    selected_type_name = str(selection.get("building_type", "")).strip()
    selected_cells = selection.get("cells")
    if selected_cells is None:
        selected_cells = selection.get("layout")
    target_layout = _parse_serialized_cells(selected_cells)
    if target_layout is not None:
        for option in options:
            option_layout = _parse_serialized_cells(option.get("cells"))
            if option.get("building_type") != selected_type_name:
                continue
            if option_layout == target_layout:
                return option
        return None

    selected_origin = selection.get("origin")
    if not isinstance(selected_origin, (list, tuple)) or len(selected_origin) != 2:
        return None
    selected_rotation_name = str(selection.get("rotation", "ROT_0")).strip() or "ROT_0"
    for option in options:
        if option.get("building_type") != selected_type_name:
            continue
        if option.get("origin") != list(selected_origin):
            continue
        if str(option.get("rotation", "ROT_0")).strip() != selected_rotation_name:
            continue
        return option
    return None


def _pop_matching_detail_entry(
    details: Dict[str, Any],
    key: str,
    *,
    card_instance_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    entries = details.get(key)
    if not isinstance(entries, list):
        return None
    wanted_id = str(card_instance_id or "").strip()
    for idx, entry in enumerate(entries):
        if not isinstance(entry, dict):
            continue
        entry_id = str(entry.get("card_instance_id") or "").strip()
        if wanted_id and entry_id and entry_id != wanted_id:
            continue
        return entries.pop(idx)
    return None


def _pop_detail_queue_entry(details: Dict[str, Any], key: str) -> Optional[Dict[str, Any]]:
    entries = details.get(key)
    if not isinstance(entries, list) or not entries:
        return None
    entry = entries.pop(0)
    if not isinstance(entry, dict):
        raise ValueError(f"{key} entries must be objects.")
    return entry


def _apply_build_placement_bonus(
    state: GameState,
    player: PlayerState,
    bonus: str,
    details: Dict[str, Any],
    bonus_index: int,
    bonus_coord: Optional[Tuple[int, int]] = None,
    allow_archaeologist_chain: bool = True,
    allow_interactive: bool = False,
) -> None:
    if bonus == "5coins":
        player.money += 5
    elif bonus == "x_token":
        player.x_tokens = min(5, player.x_tokens + 1)
    elif bonus == "reputation":
        _increase_reputation(state, player, 1, allow_interactive=allow_interactive)
    elif bonus == "card_in_reputation_range":
        queued_choice = _pop_detail_queue_entry(details, "build_card_bonus_choices")
        if queued_choice is not None:
            _take_one_card_from_deck_or_reputation_range_with_choice(
                state,
                player,
                draw_source=str(queued_choice.get("draw_source") or ""),
                display_index=queued_choice.get("display_index"),
            )
        elif allow_interactive:
            accessible = min(_reputation_display_limit(int(player.reputation)), len(state.zoo_display))
            print("Build placement bonus: take 1 card from reputation range or deck.")
            for idx in range(accessible):
                card = state.zoo_display[idx]
                print(f"{idx + 1}. display[{idx + 1}] #{card.number} {card.name}")
            print(f"{accessible + 1}. deck")
            while True:
                raw_pick = input(f"Select bonus card source [1-{accessible + 1}]: ").strip()
                if not raw_pick.isdigit():
                    print("Please enter a number.")
                    continue
                picked = int(raw_pick)
                if 1 <= picked <= accessible:
                    _take_one_card_from_deck_or_reputation_range_with_choice(
                        state,
                        player,
                        draw_source="display",
                        display_index=picked - 1,
                    )
                    break
                if picked == accessible + 1:
                    _take_one_card_from_deck_or_reputation_range_with_choice(
                        state,
                        player,
                        draw_source="deck",
                    )
                    break
                print("Out of range, try again.")
        else:
            _take_one_card_from_deck_or_reputation_range_with_choice(
                state,
                player,
                draw_source="deck",
            )
    elif bonus == "action_to_slot_1":
        queued_choice = _pop_detail_queue_entry(details, "bonus_action_to_slot_1_choices")
        chosen_action = None
        if queued_choice is not None:
            raw = queued_choice.get("action_name")
            if isinstance(raw, str) and raw in player.action_order:
                chosen_action = raw
        if chosen_action is None:
            targets = details.get("bonus_action_to_slot_1_targets")
            if isinstance(targets, list) and bonus_index < len(targets):
                raw = targets[bonus_index]
                if isinstance(raw, str) and raw in player.action_order:
                    chosen_action = raw
        if chosen_action is None:
            if not player.action_order:
                raise ValueError("Cannot resolve action_to_slot_1 placement bonus with empty action order.")
            if allow_interactive:
                chosen_action = _prompt_action_to_slot_1_target_for_human(
                    player,
                    reason="Build placement bonus: move one action card to slot 1.",
                )
            else:
                # Deterministic non-interactive fallback for map/sponsor auto effects.
                chosen_action = player.action_order[-1]
        _rotate_action_card_to_slot_1(player, chosen_action)

    if (
        allow_archaeologist_chain
        and bonus_coord is not None
        and _player_has_sponsor(player, 221)
        and bonus_coord in _player_border_coords(player)
    ):
        covered = _player_all_covered_cells(player)
        extra_candidates = sorted(
            coord
            for coord, coord_bonus in state.map_tile_bonuses.items()
            if coord_bonus and coord not in covered and coord != bonus_coord
        )
        if extra_candidates:
            picked_coord = extra_candidates[0]
            picked_bonus = state.map_tile_bonuses[picked_coord]
            _apply_build_placement_bonus(
                state=state,
                player=player,
                bonus=picked_bonus,
                details=details,
                bonus_index=bonus_index,
                bonus_coord=picked_coord,
                allow_archaeologist_chain=False,
                allow_interactive=allow_interactive,
            )


def _card_identity(card: AnimalCard) -> str:
    if card.instance_id:
        return card.instance_id
    if card.number >= 0:
        return str(card.number)
    return f"{card.name}:{id(card)}"


def _iter_card_zones(state: GameState) -> List[Tuple[str, List[AnimalCard]]]:
    zones: List[Tuple[str, List[AnimalCard]]] = [
        ("deck", state.zoo_deck),
        ("display", state.zoo_display),
        ("discard", state.zoo_discard),
    ]
    for idx, player in enumerate(state.players):
        zones.append((f"p{idx}.hand", player.hand))
        zones.append((f"p{idx}.zoo", player.zoo_cards))
        zones.append((f"p{idx}.pouched", player.pouched_cards))
    return zones


def card_zone_report(state: GameState) -> Dict[str, List[str]]:
    report: Dict[str, List[str]] = {}
    for zone, cards in _iter_card_zones(state):
        report[zone] = [f"{_card_identity(card)}:{card.number}:{card.card_type}:{card.name}" for card in cards]
    return report


def _validate_card_zones(state: GameState) -> None:
    seen: Dict[str, str] = {}
    duplicates: List[str] = []
    for zone, cards in _iter_card_zones(state):
        for card in cards:
            card_id = _card_identity(card)
            prev_zone = seen.get(card_id)
            if prev_zone is not None:
                duplicates.append(f"{card_id} in {prev_zone} and {zone}")
            else:
                seen[card_id] = zone
    if duplicates:
        raise ValueError("Card zone integrity error: " + "; ".join(duplicates))


def _pouch_cards_under_host(
    player: PlayerState,
    *,
    host_card_instance_id: str,
    cards: Sequence[AnimalCard],
) -> None:
    if not cards:
        return
    player.pouched_cards.extend(cards)
    host_key = str(host_card_instance_id or "").strip()
    if not host_key:
        return
    player.pouched_cards_by_host.setdefault(host_key, []).extend(cards)


def _upgrade_one_action_card(player: PlayerState, *, interactive: bool = True) -> bool:
    upgradeable = [action for action in MAIN_ACTION_CARDS if not bool(player.action_upgraded.get(action, False))]
    if not upgradeable:
        return False
    if not interactive:
        player.action_upgraded[upgradeable[0]] = True
        return True

    print(f"{player.name} reached reputation milestone: choose 1 action card to upgrade.")
    for idx, action in enumerate(upgradeable, start=1):
        print(f"{idx}. {action}")

    while True:
        raw_pick = input(f"Select action card [1-{len(upgradeable)}]: ").strip()
        if not raw_pick.isdigit():
            print("Please enter a number.")
            continue
        picked = int(raw_pick)
        if not (1 <= picked <= len(upgradeable)):
            print("Out of range, try again.")
            continue
        player.action_upgraded[upgradeable[picked - 1]] = True
        return True


def _take_one_reputation_bonus_card(state: GameState, player: PlayerState) -> None:
    _take_one_reputation_bonus_card_impl(
        state=state,
        player=player,
        reputation_display_limit_fn=_reputation_display_limit,
        replenish_display_fn=_replenish_zoo_display,
    )


def _apply_map_effect_code(
    state: GameState,
    player: PlayerState,
    player_id: int,
    effect_code: str,
    *,
    source: str,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    code = str(effect_code).strip().lower()
    effect_details = dict(effect_details or {})
    if not code:
        return

    if code == "draw_1_card_deck_or_reputation_range":
        if source.startswith("map_left_track_") or source == "map_break_step5":
            taken = _take_one_card_from_display_any(
                state,
                player,
                display_index=effect_details.get("display_index"),
            )
        else:
            draw_source = str(effect_details.get("draw_source") or "").strip().lower()
            if draw_source:
                taken = _take_one_card_from_deck_or_reputation_range_with_choice(
                    state,
                    player,
                    draw_source=draw_source,
                    display_index=effect_details.get("display_index"),
                )
            else:
                taken = _take_one_card_from_deck_or_reputation_range(state, player)
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "build_free_standard_enclosure_size_2":
        placed = _place_free_building_of_type_if_possible(
            state=state,
            player=player,
            building_type=BuildingType.SIZE_2,
            player_id=player_id,
            details=effect_details,
        )
        state.effect_log.append(f"{source}:{player.name}:{code}:placed={1 if placed else 0}")
        return

    if code == "gain_5_coins":
        player.money += 5
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "play_1_sponsor_by_paying_cost":
        def _try_play_selected_sponsor(selected_card: AnimalCard, sponsor_details: Dict[str, Any]) -> bool:
            try:
                _play_single_sponsor_from_hand_paying_cost(
                    state=state,
                    player=player,
                    player_id=player_id,
                    card=selected_card,
                    details=copy.deepcopy(sponsor_details),
                )
            except ValueError:
                if allow_interactive:
                    raise
                return False
            state.effect_log.append(
                f"{source}:{player.name}:{code}:played={getattr(selected_card, 'number', -1)}"
            )
            return True

        sponsor_details_raw = effect_details.get("sponsor_details")
        if isinstance(sponsor_details_raw, dict):
            sponsor_details = copy.deepcopy(sponsor_details_raw)
            sponsor_selection_cards = _resolve_selected_sponsor_cards_from_details(state, player, sponsor_details)
            if sponsor_selection_cards:
                selected_card = sponsor_selection_cards[0]
                if selected_card in player.hand and int(player.money) >= _sponsor_level(selected_card):
                    if _try_play_selected_sponsor(selected_card, sponsor_details):
                        return
        sponsors_upgraded = bool(player.action_upgraded["sponsors"])
        candidates = _list_legal_sponsor_candidates(state, player, sponsors_upgraded)
        playable_hand = [
            item
            for item in candidates
            if item.get("source") == "hand"
            and bool(item.get("reason") == "ok")
            and isinstance(item.get("card"), AnimalCard)
            and int(player.money) >= _sponsor_level(item["card"])
        ]
        if not playable_hand:
            state.effect_log.append(f"{source}:{player.name}:{code}:played=0")
            return
        playable_hand.sort(
            key=lambda item: (
                int(item.get("level", 0)),
                int(getattr(item.get("card"), "number", -1)),
                str(item.get("card_instance_id", "")),
            )
        )
        for chosen in playable_hand:
            selected_card = chosen.get("card")
            if not isinstance(selected_card, AnimalCard):
                continue
            base_details: Dict[str, Any] = {
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": "hand",
                        "source_index": int(chosen.get("source_index", -1)),
                        "card_instance_id": str(chosen.get("card_instance_id", "")),
                    }
                ],
            }
            for fragment_details, _fragment_label in _enumerate_sponsor_candidate_detail_variants(
                state,
                player,
                player_id,
                chosen,
            ):
                resolved_details = _merge_detail_fragments(base_details, fragment_details)
                if _try_play_selected_sponsor(selected_card, resolved_details):
                    return
        state.effect_log.append(f"{source}:{player.name}:{code}:played=0")
        return

    if code == "gain_worker_1":
        _gain_workers(
            state=state,
            player=player,
            player_id=player_id,
            amount=1,
            source=source,
            effect_details=effect_details,
            allow_interactive=allow_interactive,
        )
        return

    if code == "gain_12_coins":
        player.money += 12
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "gain_3_x_tokens":
        player.x_tokens = min(5, int(player.x_tokens) + 3)
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "gain_conservation_1":
        player.conservation += 1
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "gain_conservation_2":
        player.conservation += 2
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "gain_appeal_7":
        player.appeal += 7
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "upgrade_action_card":
        upgraded_action = str(effect_details.get("upgraded_action") or "").strip()
        if upgraded_action:
            if upgraded_action not in MAIN_ACTION_CARDS:
                raise ValueError("Selected upgraded_action is invalid.")
            if bool(player.action_upgraded.get(upgraded_action, False)):
                raise ValueError("Selected action card is already upgraded.")
            player.action_upgraded[upgraded_action] = True
            upgraded = True
        else:
            upgraded = _upgrade_one_action_card(player, interactive=allow_interactive)
        state.effect_log.append(f"{source}:{player.name}:{code}:upgraded={1 if upgraded else 0}")
        return

    state.effect_log.append(f"{source}:{player.name}:unknown_effect={code}")


def _apply_map_worker_gain_rewards(
    state: GameState,
    player: PlayerState,
    player_id: int,
    gained_workers: int,
    *,
    source: str,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    if gained_workers <= 0:
        return
    effect_details = dict(effect_details or {})
    for reward in _map_rule_worker_gain_rewards(state):
        effect_code = str(reward.get("effect") or "").strip().lower()
        if not effect_code:
            continue
        repeat = int(reward.get("repeat_per_worker", 1))
        repeat = max(1, repeat)
        for _ in range(gained_workers * repeat):
            queued_effect_details = _pop_detail_queue_entry(effect_details, "worker_gain_effect_choices")
            _apply_map_effect_code(
                state=state,
                player=player,
                player_id=player_id,
                effect_code=effect_code,
                source=source,
                effect_details=queued_effect_details,
                allow_interactive=allow_interactive,
            )


def _gain_workers(
    *,
    state: GameState,
    player: PlayerState,
    player_id: int,
    amount: int,
    source: str,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> int:
    if amount <= 0:
        return 0
    before = int(player.workers)
    player.workers = min(MAX_WORKERS, int(player.workers) + int(amount))
    gained = int(player.workers) - before
    if gained > 0:
        _apply_map_worker_gain_rewards(
            state=state,
            player=player,
            player_id=player_id,
            gained_workers=gained,
            source=source,
            effect_details=effect_details,
            allow_interactive=allow_interactive,
        )
    return gained


def _apply_map_threshold_rewards(
    state: GameState,
    player: PlayerState,
    player_id: int,
    *,
    count: int,
    thresholds: Sequence[Dict[str, Any]],
    claimed_thresholds: Set[int],
    source_prefix: str,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    effect_details = dict(effect_details or {})
    for item in thresholds:
        threshold = int(item.get("count", 0))
        effect_code = str(item.get("effect") or "").strip().lower()
        if threshold <= 0 or not effect_code:
            continue
        if count < threshold or threshold in claimed_thresholds:
            continue
        claimed_thresholds.add(threshold)
        queued_effect_details = _pop_detail_queue_entry(effect_details, "map_threshold_effect_choices")
        _apply_map_effect_code(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            source=f"{source_prefix}_{threshold}",
            effect_details=queued_effect_details,
            allow_interactive=allow_interactive,
        )


def _apply_map_partner_threshold_rewards(
    state: GameState,
    player: PlayerState,
    player_id: int,
    *,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    _apply_map_threshold_rewards(
        state=state,
        player=player,
        player_id=player_id,
        count=len(player.partner_zoos),
        thresholds=_map_rule_partner_thresholds(state),
        claimed_thresholds=player.claimed_partner_zoo_thresholds,
        source_prefix="map_partner_threshold",
        effect_details=effect_details,
        allow_interactive=allow_interactive,
    )


def _apply_map_university_threshold_rewards(
    state: GameState,
    player: PlayerState,
    player_id: int,
    *,
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    _apply_map_threshold_rewards(
        state=state,
        player=player,
        player_id=player_id,
        count=len(player.universities),
        thresholds=_map_rule_university_thresholds(state),
        claimed_thresholds=player.claimed_university_thresholds,
        source_prefix="map_university_threshold",
        effect_details=effect_details,
        allow_interactive=allow_interactive,
    )


def _next_map_left_track_unlock(state: GameState, player: PlayerState) -> Optional[Dict[str, Any]]:
    remaining = _remaining_map_left_track_unlocks(state, player)
    if not remaining:
        return None
    _, item = remaining[0]
    return item


def _remaining_map_left_track_unlocks(
    state: GameState,
    player: PlayerState,
) -> List[Tuple[int, Dict[str, Any]]]:
    remaining: List[Tuple[int, Dict[str, Any]]] = []
    for idx, item in enumerate(_map_rule_left_track_unlocks(state)):
        if idx in player.map_left_track_claimed_indices:
            continue
        if isinstance(item, dict):
            remaining.append((idx, item))
    return remaining


def _projected_reputation_after_project_support(player: PlayerState, option: Dict[str, Any]) -> int:
    projected = int(player.reputation) + int(option.get("reputation_gain", 0))
    if not bool(player.action_upgraded.get("association", False)):
        projected = min(projected, 9)
    return max(0, min(15, projected))


def _projected_money_after_project_support(player: PlayerState, option: Dict[str, Any]) -> int:
    projected = int(player.money)
    project_from_display_index = option.get("project_from_display_index")
    if project_from_display_index is not None:
        display_index = int(project_from_display_index)
        display_extra_cost = int(option.get("project_display_extra_cost", display_index + 1))
        projected -= max(0, display_extra_cost)
    return max(0, projected)


def _simulated_display_after_project_support(state: GameState, option: Dict[str, Any]) -> List[AnimalCard]:
    simulated = list(state.zoo_display)
    raw_index = option.get("project_from_display_index")
    if raw_index is None:
        return simulated
    display_index = int(raw_index)
    if 0 <= display_index < len(simulated):
        simulated.pop(display_index)
        deck_index = 0
        if deck_index < len(state.zoo_deck):
            simulated.append(state.zoo_deck[deck_index])
    return simulated


def _take_one_card_from_display_any(
    state: GameState,
    player: PlayerState,
    *,
    display_index: Optional[int] = None,
) -> bool:
    if not state.zoo_display:
        return False
    idx = 0 if display_index is None else int(display_index)
    if idx < 0 or idx >= len(state.zoo_display):
        raise ValueError("Chosen display card is out of range.")
    player.hand.append(state.zoo_display.pop(idx))
    _replenish_zoo_display(state)
    return True


def _map_left_track_reward_variants_for_option(
    state: GameState,
    player: PlayerState,
    player_id: int,
    option: Dict[str, Any],
) -> List[Tuple[Dict[str, Any], str]]:
    remaining_unlocks = _remaining_map_left_track_unlocks(state, player)
    if not remaining_unlocks:
        return [({}, "")]
    simulated_display = _simulated_display_after_project_support(state, option)
    projected_money = _projected_money_after_project_support(player, option)
    all_variants: List[Tuple[Dict[str, Any], str]] = []

    def _wrap(unlock_index: int, effect_code: str, choice_payload: Dict[str, Any], label: str) -> Tuple[Dict[str, Any], str]:
        payload = dict(choice_payload)
        payload["unlock_index"] = int(unlock_index)
        payload["effect_code"] = effect_code
        return ({"map_left_track_choice": payload}, f"unlock[{unlock_index + 1}]: {label}")

    for unlock_index, unlock in remaining_unlocks:
        effect_code = str(unlock.get("effect") or "").strip().lower()
        if not effect_code:
            continue

        if effect_code == "draw_1_card_deck_or_reputation_range":
            if simulated_display:
                for idx, card in enumerate(simulated_display):
                    all_variants.append(
                        _wrap(
                            unlock_index,
                            effect_code,
                            {"display_index": idx},
                            f"draw display[{idx + 1}] #{card.number} {card.name}",
                        )
                    )
            else:
                all_variants.append(_wrap(unlock_index, effect_code, {}, "draw unavailable"))
            continue

        if effect_code == "build_free_standard_enclosure_size_2":
            build_options = [
                item
                for item in list_legal_build_options(state=state, player_id=player_id, strength=2)
                if item.get("building_type") == "SIZE_2"
            ]
            if build_options:
                for build_option in build_options:
                    for build_bonus_details, build_bonus_label in _enumerate_build_bonus_choice_variants(
                        state,
                        player_id,
                        build_option,
                    ):
                        all_variants.append(
                            _wrap(
                                unlock_index,
                                effect_code,
                                _merge_list_detail_fragments(
                                    {"selection": _build_selection_payload(build_option)},
                                    build_bonus_details,
                                ),
                                f"free {_build_step_action_label(build_option, build_bonus_label)}",
                            )
                        )
            else:
                all_variants.append(_wrap(unlock_index, effect_code, {}, "free enclosure_2 unavailable"))
            continue

        if effect_code == "play_1_sponsor_by_paying_cost":
            sponsors_upgraded = bool(player.action_upgraded["sponsors"])
            candidates = []
            for item in _list_legal_sponsor_candidates(state, player, sponsors_upgraded):
                if item.get("source") != "hand":
                    continue
                card = item.get("card")
                if not isinstance(card, AnimalCard):
                    continue
                if not bool(item.get("reason") == "ok"):
                    continue
                if int(projected_money) < _sponsor_level(card):
                    continue
                candidates.append(item)
            if candidates:
                for candidate in candidates:
                    sequence = [candidate]
                    variant_sets = [_enumerate_sponsor_candidate_detail_variants(state, player, player_id, candidate)]
                    for fragment_product in product(*variant_sets):
                        sponsor_details: Dict[str, Any] = {
                            "use_break_ability": False,
                            "sponsor_selections": [
                                {
                                    "source": str(candidate.get("source") or ""),
                                    "source_index": int(candidate.get("source_index", -1)),
                                    "card_instance_id": str(candidate.get("card_instance_id") or ""),
                                }
                            ],
                        }
                        label_parts: List[str] = []
                        for selected_candidate, (fragment_details, fragment_label) in zip(sequence, fragment_product):
                            sponsor_card = selected_candidate["card"]
                            source_label = f"{selected_candidate['source']}[{int(selected_candidate['source_index']) + 1}]"
                            card_label = f"{source_label} #{sponsor_card.number} {sponsor_card.name}"
                            if fragment_label:
                                card_label += f" ({fragment_label})"
                            label_parts.append(card_label)
                            sponsor_details = _merge_detail_fragments(sponsor_details, fragment_details)
                        all_variants.append(
                            _wrap(
                                unlock_index,
                                effect_code,
                                {"sponsor_details": sponsor_details},
                                "play " + " + ".join(label_parts),
                            )
                        )
            else:
                all_variants.append(_wrap(unlock_index, effect_code, {}, "sponsor unavailable"))
            continue

        if effect_code == "gain_5_coins":
            all_variants.append(_wrap(unlock_index, effect_code, {}, "+5 money"))
            continue
        if effect_code == "gain_worker_1":
            all_variants.append(_wrap(unlock_index, effect_code, {}, "+1 worker"))
            continue
        if effect_code == "gain_12_coins":
            all_variants.append(_wrap(unlock_index, effect_code, {}, "+12 money"))
            continue
        if effect_code == "gain_3_x_tokens":
            all_variants.append(_wrap(unlock_index, effect_code, {}, "+3 x-tokens"))
            continue

        all_variants.append(_wrap(unlock_index, effect_code, {}, effect_code))

    return all_variants or [({}, "")]


def _on_map_conservation_project_supported(
    state: GameState,
    player: PlayerState,
    player_id: int,
    effect_details: Optional[Dict[str, Any]] = None,
    *,
    allow_interactive: bool = False,
) -> None:
    remaining_unlocks = _remaining_map_left_track_unlocks(state, player)
    if not remaining_unlocks:
        return
    effect_details = dict(effect_details or {})
    requested_idx_raw = effect_details.get("unlock_index")
    chosen_idx: Optional[int] = None
    if requested_idx_raw is not None:
        requested_idx = int(requested_idx_raw)
        if any(idx == requested_idx for idx, _ in remaining_unlocks):
            chosen_idx = requested_idx
        else:
            raise ValueError("Selected map left-track reward is no longer available.")
    if chosen_idx is None:
        chosen_idx = remaining_unlocks[0][0]
    item = next(
        unlock_item
        for idx, unlock_item in remaining_unlocks
        if idx == chosen_idx
    )
    effect_code = str(item.get("effect") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()
    player.map_left_track_claimed_indices.add(chosen_idx)
    player.map_left_track_unlocked_count = len(player.map_left_track_claimed_indices)
    if category == "purple_recurring_action":
        player.map_left_track_unlocked_effects.append(effect_code)
    if effect_code:
        _apply_map_effect_code(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            source=f"map_left_track_unlock_{chosen_idx + 1}",
            effect_details=effect_details,
            allow_interactive=allow_interactive,
        )


def _apply_map_break_recurring_effects_for_player(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> None:
    for effect_code in list(player.map_left_track_unlocked_effects):
        _apply_map_effect_code(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            source="map_break_step5",
        )


def _map_tower_coords(state: GameState) -> Set[Tuple[int, int]]:
    out: Set[Tuple[int, int]] = set()
    for coord, tags in state.map_tile_tags.items():
        if any(str(tag).strip().lower() == "tower" for tag in tags):
            out.add(coord)
    return out


def _building_for_standard_enclosure(player: PlayerState, enclosure: Enclosure) -> Optional[Building]:
    if player.zoo_map is None or enclosure.origin is None:
        return None
    ox, oy = enclosure.origin
    for building in player.zoo_map.buildings.values():
        if building.type.subtype != BuildingSubType.ENCLOSURE_BASIC:
            continue
        if (building.origin_hex.x, building.origin_hex.y) != (ox, oy):
            continue
        if building.rotation.name != enclosure.rotation:
            continue
        if len(building.layout) != enclosure.size:
            continue
        return building
    return None


def _apply_map_passive_on_standard_enclosure_occupied(
    state: GameState,
    player: PlayerState,
    player_id: int,
    enclosure: Enclosure,
) -> None:
    passive = state.map_rules.get("passive_effects", [])
    if not isinstance(passive, list):
        return
    has_tower_flip_bonus = any(
        isinstance(item, dict)
        and str(item.get("effect") or "").strip().lower() == "gain_appeal_2_on_flip_standard_adjacent_tower"
        for item in passive
    )
    if not has_tower_flip_bonus:
        return
    building = _building_for_standard_enclosure(player, enclosure)
    if building is None:
        return
    tower_coords = _map_tower_coords(state)
    if not tower_coords:
        return
    layout = {(tile.x, tile.y) for tile in building.layout}
    adjacent = set()
    for cell in layout:
        adjacent.update(_xy_neighbors(cell))
    if any(tower in adjacent for tower in tower_coords):
        player.appeal += 2
        state.effect_log.append("map_passive:tower_adjacent_standard_enclosure_occupied:+2_appeal")


def _maybe_apply_map_completion_reward(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> None:
    if player.map_completion_reward_claimed:
        return
    reward = state.map_rules.get("map_completion_reward", {})
    if not isinstance(reward, dict):
        return
    effect_code = str(reward.get("effect") or "").strip().lower()
    if not effect_code:
        return
    if not _is_map_completely_covered(player):
        return
    player.map_completion_reward_claimed = True
    _apply_map_effect_code(
        state=state,
        player=player,
        player_id=player_id,
        effect_code=effect_code,
        source="map_completion",
    )


def _apply_reputation_milestone_reward(
    state: GameState,
    player: PlayerState,
    milestone: int,
    *,
    allow_interactive: bool = False,
) -> None:
    if milestone == 5:
        _upgrade_one_action_card(player, interactive=allow_interactive)
        return
    if milestone == 8:
        player_id = state.players.index(player)
        _gain_workers(
            state=state,
            player=player,
            player_id=player_id,
            amount=1,
            source="reputation_milestone_8",
            allow_interactive=allow_interactive,
        )
        return
    if milestone in {10, 13}:
        _take_one_reputation_bonus_card(state, player)
        return
    if milestone in {11, 14}:
        player.conservation += 1
        return
    if milestone in {12, 15}:
        player.x_tokens = min(5, int(player.x_tokens) + 1)
        return


def _increase_reputation(
    state: GameState,
    player: PlayerState,
    amount: int,
    *,
    allow_interactive: bool = False,
) -> None:
    _increase_reputation_impl(
        state=state,
        player=player,
        amount=amount,
        association_action_key="association",
        reputation_cap_without_upgrade=9,
        reputation_cap_with_upgrade=15,
        milestone_values=[5, 8, 10, 11, 12, 13, 14, 15],
        apply_milestone_reward_fn=lambda milestone_state, milestone_player, milestone: _apply_reputation_milestone_reward(
            milestone_state,
            milestone_player,
            milestone,
            allow_interactive=allow_interactive,
        ),
    )


def _prompt_break_discard_indices(player: PlayerState, overflow: int) -> List[int]:
    if overflow <= 0:
        return []
    return _prompt_card_combination_indices_impl(
        title=(
            f"{player.name} exceeds hand limit ({len(player.hand)}/{player.hand_limit}). "
            f"Choose {overflow} card(s) to discard."
        ),
        cards=player.hand,
        format_card_line=lambda card: _format_card_line_for_player(card, player),
        min_choose=overflow,
        max_choose=overflow,
        action_label="discard",
        combination_header="Discard combinations:",
    )


def _discard_down_to_limit(player: PlayerState, *, allow_interactive: bool = False) -> List[AnimalCard]:
    if len(player.hand) <= int(player.hand_limit):
        return []
    overflow = len(player.hand) - int(player.hand_limit)
    if allow_interactive:
        picked_indices = _prompt_break_discard_indices(player, overflow)
    else:
        # Non-interactive fallback: deterministically discard oldest hand cards.
        picked_indices = list(range(overflow))
    picked_set = set(picked_indices)
    discarded = [card for idx, card in enumerate(player.hand) if idx in picked_set]
    for idx in sorted(picked_indices, reverse=True):
        del player.hand[idx]
    return discarded


def _break_income_from_appeal(appeal: int) -> int:
    return _break_income_from_appeal_impl(appeal)


def _break_income_order(state: GameState) -> List[int]:
    total = len(state.players)
    trigger = state.break_trigger_player
    if trigger is None or trigger < 0 or trigger >= total:
        return list(range(total))
    return [(trigger + offset) % total for offset in range(total)]


def _clear_action_tokens_for_break(player: PlayerState) -> None:
    for action in MAIN_ACTION_CARDS:
        player.multiplier_tokens_on_actions[action] = 0
        player.venom_tokens_on_actions[action] = 0
        player.constriction_tokens_on_actions[action] = 0


def _replenish_display_after_break(state: GameState) -> None:
    for _ in range(min(2, len(state.zoo_display))):
        discarded = state.zoo_display.pop(0)
        state.zoo_discard.append(discarded)
        state.marked_display_card_ids.discard(discarded.instance_id)
    _replenish_zoo_display(state)


def _kiosk_income_for_player(player: PlayerState) -> int:
    footprints = _building_footprints(player)
    kiosks = [footprint for footprint in footprints if str(footprint.get("type")) == "KIOSK"]
    if not kiosks:
        return 0

    income = 0
    for kiosk in kiosks:
        kiosk_cells = set(kiosk.get("cells") or set())
        if not kiosk_cells:
            continue
        for info in footprints:
            if info is kiosk:
                continue
            if bool(info.get("empty_standard_enclosure")):
                continue
            other_cells = set(info.get("cells") or set())
            if not other_cells:
                continue
            if not _footprints_adjacent(kiosk_cells, other_cells):
                continue
            other_type = str(info.get("type") or "")
            if other_type == "PAVILION":
                income += 1
                continue
            if other_type in {"PETTING_ZOO", "REPTILE_HOUSE", "LARGE_BIRD_AVIARY"}:
                income += 1
                continue
            if other_type.startswith("SIZE_"):
                income += 1
                continue
            if other_type in {"SPONSOR_UNIQUE", "SPONSOR_BUILDING", "UNIQUE"}:
                income += 1
                continue
    return income


def _apply_sponsor_break_income_effects_for_player(state: GameState, player: PlayerState) -> None:
    inventory = _player_icon_inventory(player)
    sponsor_numbers = {card.number for card in _player_played_sponsor_cards(player)}
    if not sponsor_numbers:
        return

    if 201 in sponsor_numbers:
        _take_one_card_from_deck_or_reputation_range(state, player)
    if 206 in sponsor_numbers:
        player.conservation += 1
    if 209 in sponsor_numbers:
        player.x_tokens = min(5, player.x_tokens + 1)
    if 220 in sponsor_numbers:
        player.money += 3
    for sponsor_no, icon_name in SPONSOR_ICON_INCOME_THRESHOLDS.items():
        if sponsor_no in sponsor_numbers:
            player.money += _income_by_icon_threshold(int(inventory.get(icon_name, 0)))
    if 257 in sponsor_numbers:
        player.money += _count_side_entrance_adjacent_buildings(player) * 2


def _apply_sponsor_break_income_effects(
    state: GameState,
    *,
    player_order: Optional[Sequence[int]] = None,
) -> None:
    order = list(player_order) if player_order is not None else _break_income_order(state)
    for player_id in order:
        _apply_sponsor_break_income_effects_for_player(state, state.players[player_id])


def _apply_break_pre_income_stages(state: GameState) -> None:
    # 2) Temporary tokens on action cards.
    for player in state.players:
        _clear_action_tokens_for_break(player)

    # 3) Association board: recall workers and refresh market.
    for player in state.players:
        if int(player.workers_on_association_board) > 0:
            player.workers = min(MAX_WORKERS, int(player.workers) + int(player.workers_on_association_board))
            player.workers_on_association_board = 0
        for task in ASSOCIATION_TASK_KINDS:
            player.association_workers_by_task[task] = 0
    _refresh_association_market(state)

    # 4) Replenish display by discarding folders 1 and 2.
    _replenish_display_after_break(state)


def _resolve_break_remaining_stages(
    state: GameState,
    *,
    start_income_index: int = 0,
    preprocessed: bool = False,
    resume_turn_player_id: Optional[int] = None,
    resume_turn_consumed_venom: bool = False,
) -> bool:
    if not preprocessed:
        _apply_break_pre_income_stages(state)

    # 5) Income in break-trigger-player order when needed.
    order = _break_income_order(state)
    for order_index in range(max(0, start_income_index), len(order)):
        player_id = order[order_index]
        player = state.players[player_id]
        player.money += _break_income_from_appeal(int(player.appeal))
        player.money += _kiosk_income_for_player(player)
        _apply_map_break_recurring_effects_for_player(
            state=state,
            player=player,
            player_id=player_id,
        )
        _apply_sponsor_break_income_effects_for_player(
            state=state,
            player=player,
        )
        if _maybe_begin_conservation_reward_pending(
            state,
            player_id=player_id,
            resume_kind="break_remaining",
            break_income_index=order_index + 1,
            resume_turn_player_id=resume_turn_player_id,
            resume_turn_consumed_venom=resume_turn_consumed_venom,
        ):
            return False

    # 6) Reset break track.
    state.break_progress = 0
    state.break_trigger_player = None
    return True


def _set_break_discard_pending(
    state: GameState,
    *,
    player_id: int,
    discard_target: int,
    break_hand_limit_index: int,
    resume_turn_player_id: Optional[int] = None,
    resume_turn_consumed_venom: bool = False,
) -> None:
    payload: Dict[str, Any] = {
        "discard_target": discard_target,
        "break_hand_limit_index": break_hand_limit_index,
    }
    if resume_turn_player_id is not None:
        payload["resume_turn_player_id"] = int(resume_turn_player_id)
        payload["resume_turn_consumed_venom"] = bool(resume_turn_consumed_venom)
    _set_pending_decision(
        state,
        kind="break_discard",
        player_id=player_id,
        payload=payload,
    )


def _resolve_break_hand_limit_stage(
    state: GameState,
    *,
    start_index: int = 0,
    resume_turn_player_id: Optional[int] = None,
    resume_turn_consumed_venom: bool = False,
) -> bool:
    order = _break_income_order(state)
    for order_index in range(max(0, start_index), len(order)):
        player_id = order[order_index]
        player = state.players[player_id]
        overflow = len(player.hand) - int(player.hand_limit)
        if overflow <= 0:
            continue
        _set_break_discard_pending(
            state,
            player_id=player_id,
            discard_target=overflow,
            break_hand_limit_index=order_index,
            resume_turn_player_id=resume_turn_player_id,
            resume_turn_consumed_venom=resume_turn_consumed_venom,
        )
        return False
    return True


def _resolve_break(
    state: GameState,
    *,
    use_pending: bool = False,
    allow_interactive: bool = False,
    resume_turn_player_id: Optional[int] = None,
    resume_turn_consumed_venom: bool = False,
) -> None:
    if use_pending:
        if not _resolve_break_hand_limit_stage(
            state,
            resume_turn_player_id=resume_turn_player_id,
            resume_turn_consumed_venom=resume_turn_consumed_venom,
        ):
            return
        _resolve_break_remaining_stages(
            state,
            resume_turn_player_id=resume_turn_player_id,
            resume_turn_consumed_venom=resume_turn_consumed_venom,
        )
        return

    for player_id in _break_income_order(state):
        state.zoo_discard.extend(
            _discard_down_to_limit(state.players[player_id], allow_interactive=allow_interactive)
        )
    _resolve_break_remaining_stages(state)


def _advance_break_track(state: GameState, steps: int, trigger_player: int) -> bool:
    triggered = _advance_break_track_impl(
        state=state,
        steps=steps,
        trigger_player=trigger_player,
        max_x_tokens=5,
    )
    if triggered:
        state.break_trigger_player = trigger_player
    return triggered


def _perform_cards_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    player_id: int,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    details = details or {}
    upgraded = player.action_upgraded["cards"]
    draw_target, discard_target, snap_allowed = _cards_table_values(strength, upgraded)
    break_triggered = False

    snap_display_index_raw = details.get("snap_display_index")
    snap_display_index = int(snap_display_index_raw) if snap_display_index_raw is not None else None
    from_display_indices = list(details.get("from_display_indices") or [])
    from_deck_count_raw = details.get("from_deck_count")
    from_deck_count = int(from_deck_count_raw) if from_deck_count_raw is not None else None
    discard_hand_indices_raw = details.get("discard_hand_indices")
    discard_hand_indices = list(discard_hand_indices_raw) if discard_hand_indices_raw is not None else None
    discard_card_instance_ids_raw = details.get("discard_card_instance_ids")
    discard_card_instance_ids = (
        [str(item) for item in discard_card_instance_ids_raw]
        if isinstance(discard_card_instance_ids_raw, list)
        else None
    )

    if snap_display_index is not None:
        if not snap_allowed:
            raise ValueError("Snap is not available at this cards strength.")
        if from_display_indices or from_deck_count is not None:
            raise ValueError("Snap cannot be combined with normal draw choices.")
        if snap_display_index < 0 or snap_display_index >= len(state.zoo_display):
            raise ValueError("Snap display index out of range.")
        break_triggered = _advance_break_track(state, steps=2, trigger_player=player_id)
        player.hand.append(state.zoo_display.pop(snap_display_index))
        _replenish_zoo_display(state)
        return break_triggered

    if not upgraded and from_display_indices:
        raise ValueError("Non-upgraded Cards cannot draw from display.")
    if not upgraded:
        if from_deck_count is None:
            from_deck_count = draw_target
        if from_deck_count != draw_target:
            raise ValueError(
                f"Non-upgraded Cards must draw exactly {draw_target} card(s) from deck."
            )

    if upgraded:
        accessible = _reputation_display_limit(player.reputation)
        for idx in from_display_indices:
            if idx < 0 or idx >= len(state.zoo_display):
                raise ValueError("Display index out of range.")
            if idx + 1 > accessible:
                raise ValueError("Selected display card is outside reputation range.")

    if from_deck_count is None:
        from_deck_count = max(0, draw_target - len(from_display_indices))
    if from_deck_count < 0:
        raise ValueError("from_deck_count cannot be negative.")
    total_draw = len(from_display_indices) + from_deck_count
    if total_draw > draw_target:
        raise ValueError(f"Cards action can draw at most {draw_target} card(s).")
    if from_deck_count > len(state.zoo_deck):
        _trigger_immediate_game_end(
            state,
            reason=(
                f"cards_draw_exceeds_deck(player={player.name},"
                f"requested={int(from_deck_count)},remaining={len(state.zoo_deck)})"
            ),
        )
        return break_triggered

    break_triggered = _advance_break_track(state, steps=2, trigger_player=player_id)

    player.hand.extend(_take_display_cards(state, from_display_indices))
    player.hand.extend(_draw_from_zoo_deck(state, from_deck_count))
    if from_display_indices:
        _replenish_zoo_display(state)

    if discard_target > 0:
        if discard_hand_indices is not None and discard_card_instance_ids is not None:
            raise ValueError("Use discard_hand_indices or discard_card_instance_ids, not both.")

        if discard_card_instance_ids is not None:
            if len(discard_card_instance_ids) != discard_target:
                raise ValueError(f"Exactly {discard_target} discard card_instance_id(s) are required.")
            if len(set(discard_card_instance_ids)) != len(discard_card_instance_ids):
                raise ValueError("Discard card_instance_id values must be unique.")
            resolved_indices: List[int] = []
            for card_instance_id in discard_card_instance_ids:
                matching_index = next(
                    (idx for idx, card in enumerate(player.hand) if card.instance_id == card_instance_id),
                    None,
                )
                if matching_index is None:
                    raise ValueError("Discard card_instance_id is not in hand.")
                resolved_indices.append(matching_index)
            discard_hand_indices = resolved_indices

        if discard_hand_indices is None:
            if len(player.hand) < discard_target:
                raise ValueError("Not enough hand cards to discard for cards action.")
            if len(player.hand) == discard_target:
                discard_hand_indices = list(range(len(player.hand)))
            else:
                _set_pending_decision(
                    state,
                    kind="cards_discard",
                    player_id=player_id,
                    payload={"discard_target": discard_target},
                )
                return break_triggered

        if len(discard_hand_indices) != discard_target:
            raise ValueError(f"Exactly {discard_target} discard index(es) are required.")
        if len(set(discard_hand_indices)) != len(discard_hand_indices):
            raise ValueError("Discard indices must be unique.")
        for idx in discard_hand_indices:
            if idx < 0 or idx >= len(player.hand):
                raise ValueError("Discard index out of range.")
        for idx in sorted(discard_hand_indices, reverse=True):
            state.zoo_discard.append(player.hand.pop(idx))

    return break_triggered


def _perform_build_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    player_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    _ensure_player_map_initialized(state, player)
    if player.zoo_map is None:
        return
    details = details or {}
    upgraded = player.action_upgraded["build"]
    allow_interactive = bool(details.get("_allow_interactive", False) or details.get("_interactive", False))
    skip_post_build_effects = bool(details.get("_skip_post_build_effects", False))
    has_diversity_researcher = _player_has_sponsor(player, 219)
    selections_raw = details.get("selections")
    requested_selections: List[Dict[str, Any]]
    if isinstance(selections_raw, list):
        requested_selections = [item for item in selections_raw if isinstance(item, dict)]
    else:
        requested_selections = []

    if not upgraded:
        if requested_selections and len(requested_selections) != 1:
            raise ValueError("Build side I requires exactly one building selection.")
    elif requested_selections and len(requested_selections) > 2:
        raise ValueError("Build side II can include at most two building selections.")

    remaining_size = max(0, strength)
    if remaining_size <= 0:
        return
    built_this_action: Set[BuildingType] = set()
    built_type_sequence: List[BuildingType] = []
    built_count = 0
    bonus_cursor = 0

    def _gain_terrain_adjacent_cover_money(picked_building: Building) -> None:
        if _player_has_sponsor(player, 241):
            gain = sum(
                1
                for tile in picked_building.layout
                if _adjacent_terrain_count_for_cells(player, [(tile.x, tile.y)], Terrain.WATER) > 0
            )
            if gain > 0:
                player.money += gain
        if _player_has_sponsor(player, 242):
            gain = sum(
                1
                for tile in picked_building.layout
                if _adjacent_terrain_count_for_cells(player, [(tile.x, tile.y)], Terrain.ROCK) > 0
            )
            if gain > 0:
                player.money += gain

    def _commit_building(
        picked_building: Building,
        *,
        count_for_action_strength: bool,
        mark_type_for_action: bool,
    ) -> None:
        nonlocal remaining_size, built_count, bonus_cursor
        cost = len(picked_building.layout) * 2
        player.money -= cost
        player.zoo_map.add_building(picked_building)
        if mark_type_for_action:
            built_this_action.add(picked_building.type)
        if count_for_action_strength:
            remaining_size -= len(picked_building.layout)
            built_count += 1
            built_type_sequence.append(picked_building.type)

        if picked_building.type.subtype in {
            BuildingSubType.ENCLOSURE_BASIC,
            BuildingSubType.ENCLOSURE_SPECIAL,
        }:
            _register_enclosure_building(player, picked_building)
        if picked_building.type == BuildingType.PAVILION:
            player.appeal += 1

        bonuses = _building_bonuses(state, picked_building)
        for bonus_name, bonus_coord in bonuses:
            _apply_build_placement_bonus(
                state=state,
                player=player,
                bonus=bonus_name,
                details=details,
                bonus_index=bonus_cursor,
                bonus_coord=bonus_coord,
                allow_interactive=allow_interactive,
            )
            bonus_cursor += 1
        _gain_terrain_adjacent_cover_money(picked_building)
        resolved_player_id = player_id if player_id is not None else state.players.index(player)
        _maybe_apply_map_completion_reward(
            state=state,
            player=player,
            player_id=resolved_player_id,
        )

    while True:
        legal = player.zoo_map.legal_building_placements(
            is_build_upgraded=upgraded,
            has_diversity_researcher=has_diversity_researcher,
            max_building_size=remaining_size,
            already_built_buildings=built_this_action,
        )
        legal = _dedupe_legal_buildings_by_footprint(legal)
        if not legal:
            break
        legal.sort(
            key=lambda b: (
                _building_type_label(b.type),
                _building_layout_key(b),
                b.origin_hex.x,
                b.origin_hex.y,
                b.rotation.value,
            )
        )

        picked: Optional[Building] = None
        if requested_selections:
            requested = requested_selections.pop(0)
            picked = _find_legal_building_by_serialized_selection(legal, requested)
            if picked is None:
                raise ValueError("Illegal build selection.")
        else:
            affordable = [
                candidate
                for candidate in legal
                if player.money >= len(candidate.layout) * 2
            ]
            if not affordable:
                break
            if len(affordable) > 1:
                raise ValueError(
                    "Build action requires explicit selections when multiple affordable legal placements exist."
                )
            picked = affordable[0]

        cost = len(picked.layout) * 2
        if player.money < cost:
            if requested_selections:
                raise ValueError("Insufficient money for requested build selection.")
            break

        _commit_building(
            picked,
            count_for_action_strength=True,
            mark_type_for_action=True,
        )

        if not upgraded:
            break
        if built_count >= 2:
            break
        if remaining_size <= 0:
            break
        if requested_selections:
            continue
        break

    if built_count == 0:
        return

    if not skip_post_build_effects and _player_has_sponsor(player, 217):
        excluded = {
            BuildingType.PETTING_ZOO,
            BuildingType.REPTILE_HOUSE,
            BuildingType.LARGE_BIRD_AVIARY,
        }
        for built_type in built_type_sequence:
            if built_type in excluded:
                continue
            legal_extra = player.zoo_map.legal_building_placements(
                is_build_upgraded=upgraded,
                has_diversity_researcher=has_diversity_researcher,
                max_building_size=len(built_type.layout),
                already_built_buildings=set(),
            )
            legal_extra = _dedupe_legal_buildings_by_footprint(legal_extra)
            legal_extra = [building for building in legal_extra if building.type == built_type]
            legal_extra.sort(
                key=lambda b: (
                    _building_layout_key(b),
                    b.origin_hex.x,
                    b.origin_hex.y,
                    b.rotation.value,
                )
            )
            picked_extra: Optional[Building] = None
            for candidate in legal_extra:
                extra_cost = len(candidate.layout) * 2
                if player.money >= extra_cost:
                    picked_extra = candidate
                    break
            if picked_extra is None:
                continue
            _commit_building(
                picked_extra,
                count_for_action_strength=False,
                mark_type_for_action=False,
            )
            state.effect_log.append(
                f"sponsor_217_extra_build type={picked_extra.type.name} cells={_building_layout_xy(picked_extra)}"
            )
            break


def _animals_play_limit(strength: int, upgraded: bool) -> int:
    if strength <= 0:
        return 0
    idx = min(5, max(1, strength)) - 1
    if upgraded:
        return ANIMALS_PLAY_LIMIT_UPGRADED[idx]
    return ANIMALS_PLAY_LIMIT_BASE[idx]


def _find_enclosure_object(player: PlayerState, enclosure: Enclosure) -> Optional[EnclosureObject]:
    if enclosure.origin is None:
        return None
    ox, oy = enclosure.origin
    for item in player.enclosure_objects:
        if item.origin == (ox, oy) and item.rotation == enclosure.rotation and item.size == enclosure.size:
            return item
    return None


def _find_player_enclosure_by_origin_rotation_size(
    player: PlayerState,
    origin: Tuple[int, int],
    rotation: str,
    size: int,
) -> Optional[Enclosure]:
    for enclosure in player.enclosures:
        if enclosure.origin != origin:
            continue
        if enclosure.rotation != rotation:
            continue
        if enclosure.size != size:
            continue
        return enclosure
    return None


def _enclosure_is_standard(enclosure: Enclosure) -> bool:
    kind = str(getattr(enclosure, "enclosure_type", "standard") or "standard").strip().lower()
    return kind in {"standard", "enclosure_basic", "enclosure_1", "enclosure_2", "enclosure_3", "enclosure_4", "enclosure_5"}


def _enclosure_used_capacity(enclosure: Enclosure) -> int:
    used = max(0, int(getattr(enclosure, "used_capacity", 0) or 0))
    if used > 0:
        return used
    if bool(getattr(enclosure, "occupied", False)):
        if _enclosure_is_standard(enclosure):
            return 1
        return max(0, int(getattr(enclosure, "animal_capacity", 0) or 0))
    return 0


def _enclosure_remaining_capacity(enclosure: Enclosure) -> int:
    if _enclosure_is_standard(enclosure):
        return 0 if _enclosure_used_capacity(enclosure) > 0 else 1
    capacity = max(0, int(getattr(enclosure, "animal_capacity", 0) or 0))
    return max(0, capacity - _enclosure_used_capacity(enclosure))


def _enclosure_has_animals(enclosure: Enclosure) -> bool:
    return _enclosure_used_capacity(enclosure) > 0


def _sync_enclosure_occupied(enclosure: Enclosure) -> None:
    if _enclosure_is_standard(enclosure):
        enclosure.occupied = _enclosure_used_capacity(enclosure) > 0
        return
    enclosure.occupied = _enclosure_remaining_capacity(enclosure) <= 0


def _is_petting_zoo_animal(card: AnimalCard) -> bool:
    ability_title = str(getattr(card, "ability_title", "") or "").strip().lower()
    if ability_title == "petting zoo animal":
        return True
    return any(str(badge).strip().lower() == "pet" for badge in getattr(card, "badges", ()) or ())


def _animal_enclosure_spaces_needed(card: AnimalCard, enclosure: Enclosure) -> Optional[int]:
    kind = str(getattr(enclosure, "enclosure_type", "standard") or "standard").strip().lower()
    if _is_petting_zoo_animal(card):
        if kind != "petting_zoo":
            return None
        return 1
    if kind == "reptile_house":
        if card.reptile_house_size is None:
            return None
        return max(0, int(card.reptile_house_size or 0))
    if kind == "large_bird_aviary":
        if card.large_bird_aviary_size is None:
            return None
        return max(0, int(card.large_bird_aviary_size or 0))
    if kind == "petting_zoo":
        return None
    if enclosure.size < card.size:
        return None
    return 1


def _animal_host_enclosure_from_building(building: Building) -> Optional[Enclosure]:
    origin = (building.origin_hex.x, building.origin_hex.y)
    if building.type.subtype == BuildingSubType.ENCLOSURE_BASIC:
        enclosure_type = "standard"
    elif building.type == BuildingType.REPTILE_HOUSE:
        enclosure_type = "reptile_house"
    elif building.type == BuildingType.LARGE_BIRD_AVIARY:
        enclosure_type = "large_bird_aviary"
    elif building.type == BuildingType.PETTING_ZOO:
        enclosure_type = "petting_zoo"
    else:
        return None
    enclosure = Enclosure(
        size=len(building.layout),
        occupied=False,
        origin=origin,
        rotation=building.rotation.name,
        enclosure_type=enclosure_type,
        used_capacity=0,
        animal_capacity=max(1, int(building.type.max_capacity)),
    )
    _sync_enclosure_occupied(enclosure)
    return enclosure


def _register_enclosure_building(player: PlayerState, building: Building) -> None:
    origin = (building.origin_hex.x, building.origin_hex.y)
    adjacent_rock = _adjacent_terrain_count_for_building(player, building, Terrain.ROCK)
    adjacent_water = _adjacent_terrain_count_for_building(player, building, Terrain.WATER)
    host_enclosure = _animal_host_enclosure_from_building(building)
    if host_enclosure is not None:
        player.enclosures.append(host_enclosure)
    player.enclosure_objects.append(
        EnclosureObject(
            size=len(building.layout),
            enclosure_type=_building_type_label(building.type),
            adjacent_rock=adjacent_rock,
            adjacent_water=adjacent_water,
            animals_inside=0,
            origin=origin,
            rotation=building.rotation.name,
        )
    )


def _enclosure_can_host_animal(player: PlayerState, enclosure: Enclosure, card: AnimalCard) -> bool:
    return _enclosure_can_host_animal_with_ignores(
        player=player,
        enclosure=enclosure,
        card=card,
        ignore_conditions=0,
    )


def _enclosure_can_host_animal_with_ignores(
    player: PlayerState,
    enclosure: Enclosure,
    card: AnimalCard,
    ignore_conditions: int,
    reserved_capacity: int = 0,
) -> bool:
    spaces_needed = _animal_enclosure_spaces_needed(card, enclosure)
    if spaces_needed is None:
        return False
    remaining_capacity = max(0, _enclosure_remaining_capacity(enclosure) - max(0, int(reserved_capacity)))
    if remaining_capacity < spaces_needed:
        return False
    obj = _find_enclosure_object(player, enclosure)
    remaining_ignores = max(0, int(ignore_conditions))
    if not _player_has_sponsor(player, 219):
        if card.required_rock_adjacency > 0:
            if obj is None or obj.adjacent_rock < card.required_rock_adjacency:
                if remaining_ignores <= 0:
                    return False
                remaining_ignores -= 1
        if card.required_water_adjacency > 0:
            if obj is None or obj.adjacent_water < card.required_water_adjacency:
                if remaining_ignores <= 0:
                    return False
                remaining_ignores -= 1
    return True


def _player_icon_inventory(player: PlayerState) -> Dict[str, int]:
    snapshot = _player_icon_snapshot(player)
    inventory: Dict[str, int] = {}
    for continent_name, amount in snapshot["continents"].items():
        normalized = _canonical_icon_key(continent_name)
        inventory[normalized] = inventory.get(normalized, 0) + int(amount)
    inventory["science"] = int(snapshot["science"])
    for category_name, amount in snapshot["categories"].items():
        normalized = _canonical_icon_key(category_name)
        inventory[normalized] = inventory.get(normalized, 0) + int(amount)
    inventory["rock"] = int(snapshot["rock_icons"])
    inventory["water"] = int(snapshot["water_icons"])
    return inventory


def _player_has_sponsor(player: PlayerState, sponsor_number: int) -> bool:
    return any(card.card_type == "sponsor" and card.number == sponsor_number for card in player.zoo_cards)


def _player_played_sponsor_cards(player: PlayerState) -> List[AnimalCard]:
    return [card for card in player.zoo_cards if card.card_type == "sponsor"]


def _player_small_animals_count(player: PlayerState) -> int:
    return sum(1 for card in player.zoo_cards if _is_small_animal(card))


def _player_large_animals_count(player: PlayerState) -> int:
    return sum(1 for card in player.zoo_cards if _is_large_animal(card))


def _breeding_program_requirement_met(player: PlayerState, required_badge: str) -> bool:
    target_badge = _canonical_icon_key(required_badge)
    if not target_badge:
        return False
    for card in player.zoo_cards:
        if card.card_type != "animal":
            continue
        badge_keys = {_canonical_icon_key(badge) for badge in _card_badges_for_icons(card)}
        if target_badge not in badge_keys:
            continue
        for badge in _card_badges_for_icons(card):
            continent_name = _normalize_continent_badge(str(badge))
            if continent_name is None:
                continue
            if _canonical_icon_key(continent_name) in player.partner_zoos:
                return True
    return False


def _blocked_level_by_project(state: GameState) -> Dict[str, str]:
    blocked: Dict[str, str] = {}
    for item in state.opening_setup.two_player_blocked_project_levels:
        blocked[item.project_data_id] = item.blocked_level
    return blocked


def _project_requirement_value(player: PlayerState, project_id: str) -> int:
    snapshot = _player_icon_snapshot(player)
    inventory = _player_icon_inventory(player)
    number = _project_number_from_data_id(project_id)

    if number == 101:  # species diversity: unique animal-category icons
        return len(snapshot["categories"])
    if number == 102:  # habitat diversity: unique continents
        return sum(1 for _, amount in snapshot["continents"].items() if int(amount) > 0)
    if number == 103:
        return int(inventory.get("africa", 0))
    if number == 104:
        return int(inventory.get("america", 0))
    if number == 105:
        return int(inventory.get("australia", 0))
    if number == 106:
        return int(inventory.get("asia", 0))
    if number == 107:
        return int(inventory.get("europe", 0))
    if number == 108:
        return int(inventory.get("primate", 0))
    if number == 109:
        return int(inventory.get("reptile", 0))
    if number == 110:
        return int(inventory.get("predator", 0))
    if number == 111:
        return int(inventory.get("herbivore", 0))
    if number == 112:
        return int(inventory.get("bird", 0))
    if number in BREEDING_PROJECT_BADGE_BY_NUMBER:
        return 1 if _breeding_program_requirement_met(player, BREEDING_PROJECT_BADGE_BY_NUMBER[number]) else 0
    if number == 128:
        return int(inventory.get("water", 0))
    if number == 129:
        return int(inventory.get("rock", 0))
    if number == 130:
        return _player_small_animals_count(player)
    if number == 131:
        return _player_large_animals_count(player)
    if number == 132:
        return int(inventory.get("science", 0))
    return 0


def _is_release_into_wild_project(project_id: str) -> bool:
    return "release" in str(project_id).strip().lower()


def _available_breeding_icon_reduction(player: PlayerState) -> int:
    for sponsor_no in (215, 218):
        if int(player.sponsor_tokens_by_number.get(sponsor_no, 0)) > 0:
            return 1
    return 0


def _consume_breeding_icon_reduction(player: PlayerState) -> bool:
    for sponsor_no in (215, 218):
        tokens = int(player.sponsor_tokens_by_number.get(sponsor_no, 0))
        if tokens > 0:
            player.sponsor_tokens_by_number[sponsor_no] = tokens - 1
            return True
    return False


def _animal_icon_missing_conditions(player: PlayerState, card: AnimalCard) -> int:
    inventory = _player_icon_inventory(player)
    missing = 0
    for raw_icon, need in card.required_icons:
        icon = _canonical_icon_key(raw_icon)
        if icon == "partner_zoo":
            if _animal_matching_partner_zoo_count(player, card) < int(need):
                missing += 1
            continue
        if inventory.get(icon, 0) < int(need):
            missing += 1
    return missing


def _animal_condition_ignore_capacity(player: PlayerState, card: AnimalCard) -> int:
    if _is_large_animal(card) and _player_has_sponsor(player, 263):
        return 1
    return 0


def _animal_size_restriction_met(player: PlayerState, card: AnimalCard) -> bool:
    mode = (player.sponsor_waza_assignment_mode or "").strip().lower()
    if mode == "small" and not _is_small_animal(card):
        return False
    if mode == "large" and not _is_large_animal(card):
        return False
    return True


def _animal_required_icons_met(player: PlayerState, card: AnimalCard, ignore_conditions: int = 0) -> bool:
    return _animal_icon_missing_conditions(player, card) <= max(0, int(ignore_conditions))


def _animal_matching_partner_zoo_count(player: PlayerState, card: AnimalCard) -> int:
    if card.card_type != "animal":
        return 0
    matched: Set[str] = set()
    for badge in _card_badges_for_icons(card):
        continent_name = _normalize_continent_badge(str(badge))
        if continent_name is None:
            continue
        continent_key = _canonical_icon_key(continent_name)
        if continent_key in player.partner_zoos:
            matched.add(continent_key)
    return len(matched)


def _animal_partner_zoo_discount(player: PlayerState, card: AnimalCard) -> int:
    return max(0, _animal_matching_partner_zoo_count(player, card) * 3)


def _animal_size_sponsor_discount(player: PlayerState, card: AnimalCard) -> int:
    discount = 0
    if _is_small_animal(card) and _player_has_sponsor(player, 229):
        discount += 3
    if _is_large_animal(card) and _player_has_sponsor(player, 230):
        discount += 4
    return max(0, discount)


def _animal_discount_breakdown(player: PlayerState, card: AnimalCard) -> Tuple[int, int, int, int, int]:
    base_cost = max(0, int(card.cost))
    partner_discount_raw = _animal_partner_zoo_discount(player, card)
    sponsor_discount_raw = _animal_size_sponsor_discount(player, card)
    applied_partner = min(base_cost, partner_discount_raw)
    remaining = max(0, base_cost - applied_partner)
    applied_sponsor = min(remaining, sponsor_discount_raw)
    total_discount = applied_partner + applied_sponsor
    effective_cost = max(0, base_cost - total_discount)
    return base_cost, applied_partner, applied_sponsor, total_discount, effective_cost


def _animal_play_cost(player: PlayerState, card: AnimalCard) -> int:
    return _animal_discount_breakdown(player, card)[-1]


def _serialize_animal_play_step(
    player: PlayerState,
    card: AnimalCard,
    hand_index: int,
    enclosure: Enclosure,
    enclosure_index: int,
    resolved_cost: int,
) -> Dict[str, Any]:
    base_cost, partner_discount, sponsor_discount, total_discount, _ = _animal_discount_breakdown(player, card)
    spaces_used = _animal_enclosure_spaces_needed(card, enclosure)
    if spaces_used is None:
        raise ValueError("Animal cannot be placed in selected enclosure.")
    return {
        "card_instance_id": card.instance_id,
        "card_number": card.number,
        "card_name": card.name,
        "card_base_cost": base_cost,
        "card_partner_discount": partner_discount,
        "card_sponsor_discount": sponsor_discount,
        "card_discount": total_discount,
        "card_cost": max(0, int(resolved_cost)),
        "card_size": card.size,
        "card_appeal": card.appeal,
        "card_reputation": card.reputation_gain,
        "card_conservation": card.conservation,
        "card_hand_index": hand_index,
        "enclosure_index": enclosure_index,
        "enclosure_size": enclosure.size,
        "enclosure_origin": list(enclosure.origin) if enclosure.origin is not None else None,
        "enclosure_rotation": enclosure.rotation,
        "enclosure_type": enclosure.enclosure_type,
        "enclosure_capacity": int(enclosure.animal_capacity),
        "spaces_used": int(spaces_used),
    }


def list_legal_animals_options(
    state: GameState,
    player_id: int,
    strength: int,
) -> List[Dict[str, Any]]:
    player = state.players[player_id]
    upgraded = player.action_upgraded["animals"]
    play_limit = _animals_play_limit(strength, upgraded)
    if play_limit <= 0:
        return []

    single_steps: List[Dict[str, Any]] = []
    for hand_idx, card in enumerate(player.hand):
        if card.card_type != "animal":
            continue
        if not _animal_size_restriction_met(player, card):
            continue
        ignore_capacity = _animal_condition_ignore_capacity(player, card)
        missing_icon_conditions = _animal_icon_missing_conditions(player, card)
        if missing_icon_conditions > ignore_capacity:
            continue
        remaining_ignores = max(0, ignore_capacity - missing_icon_conditions)
        resolved_cost = _animal_play_cost(player, card)
        if resolved_cost > player.money:
            continue
        for enclosure_idx, enclosure in enumerate(player.enclosures):
            if not _enclosure_can_host_animal_with_ignores(
                player=player,
                enclosure=enclosure,
                card=card,
                ignore_conditions=remaining_ignores,
            ):
                continue
            single_steps.append(
                _serialize_animal_play_step(
                    player=player,
                    card=card,
                    hand_index=hand_idx,
                    enclosure=enclosure,
                    enclosure_index=enclosure_idx,
                    resolved_cost=resolved_cost,
                )
            )

    dedup_keys: Set[Tuple[Tuple[str, int, int], ...]] = set()
    options: List[Dict[str, Any]] = []

    def _append_option(plays: List[Dict[str, Any]]) -> None:
        key = tuple(
            (
                str(play.get("card_instance_id") or ""),
                int(play["card_hand_index"]),
                int(play["enclosure_index"]),
            )
            for play in plays
        )
        if key in dedup_keys:
            return
        dedup_keys.add(key)
        options.append(
            {
                "plays": plays,
                "total_cost": sum(int(play["card_cost"]) for play in plays),
                "total_appeal": sum(int(play["card_appeal"]) for play in plays),
                "total_reputation": sum(int(play["card_reputation"]) for play in plays),
                "total_conservation": sum(int(play["card_conservation"]) for play in plays),
            }
        )

    for first in single_steps:
        _append_option([dict(first)])

    if play_limit >= 2:
        for first in single_steps:
            first_hand_index = int(first["card_hand_index"])
            first_enclosure_index = int(first["enclosure_index"])
            money_left = player.money - int(first["card_cost"])
            if money_left < 0:
                continue

            for second_hand_idx, second_card in enumerate(player.hand):
                if second_hand_idx == first_hand_index:
                    continue
                if second_card.card_type != "animal":
                    continue
                if not _animal_size_restriction_met(player, second_card):
                    continue
                second_ignore_capacity = _animal_condition_ignore_capacity(player, second_card)
                second_missing_icons = _animal_icon_missing_conditions(player, second_card)
                if second_missing_icons > second_ignore_capacity:
                    continue
                second_remaining_ignores = max(0, second_ignore_capacity - second_missing_icons)
                second_cost = _animal_play_cost(player, second_card)
                if second_cost > money_left:
                    continue
                for second_enclosure_idx, second_enclosure in enumerate(player.enclosures):
                    reserved_capacity = 0
                    if second_enclosure_idx == first_enclosure_index:
                        reserved_capacity = int(first.get("spaces_used", 0))
                    if not _enclosure_can_host_animal_with_ignores(
                        player=player,
                        enclosure=second_enclosure,
                        card=second_card,
                        ignore_conditions=second_remaining_ignores,
                        reserved_capacity=reserved_capacity,
                    ):
                        continue
                    second = _serialize_animal_play_step(
                        player=player,
                        card=second_card,
                        hand_index=second_hand_idx,
                        enclosure=second_enclosure,
                        enclosure_index=second_enclosure_idx,
                        resolved_cost=second_cost,
                    )
                    _append_option([dict(first), second])

    options.sort(
        key=lambda item: (
            len(item["plays"]),
            tuple(int(play["card_hand_index"]) for play in item["plays"]),
            tuple(int(play["enclosure_index"]) for play in item["plays"]),
        )
    )

    for idx, option in enumerate(options, start=1):
        option["index"] = idx
    return options


def _perform_animals_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    details: Optional[Dict[str, Any]] = None,
    player_id: int = 0,
) -> None:
    details = details or {}
    upgraded = player.action_upgraded["animals"]
    allow_interactive = bool(details.get("_allow_interactive", False) or details.get("_interactive", False))

    if upgraded and strength >= 5:
        _increase_reputation(
            state=state,
            player=player,
            amount=1,
            allow_interactive=allow_interactive,
        )

    if details.get("skip_animals_action"):
        return

    options = list_legal_animals_options(state=state, player_id=player_id, strength=strength)
    if not options:
        return

    selected: Optional[Dict[str, Any]] = None
    selected_idx_raw = details.get("animals_sequence_index")
    if selected_idx_raw is not None:
        selected_idx = int(selected_idx_raw)
        if selected_idx < 0 or selected_idx >= len(options):
            raise ValueError("animals_sequence_index is out of range.")
        selected = options[selected_idx]
    else:
        if len(options) == 1:
            selected = options[0]
        else:
            raise ValueError(
                "animals_sequence_index is required when multiple legal animal sequences exist."
            )

    if selected is None:
        return

    selected_plays = list(selected["plays"])
    if not selected_plays:
        return

    interactive_effect_prompts = bool(details.get("_interactive"))
    queued_clever_targets: List[str] = []
    queued_hypnosis_targets: List[str] = []
    queued_hypnosis_target_players: List[Any] = []
    queued_pilfering_choices: List[Dict[str, Any]] = []
    queued_sell_hand_card_choices: List[List[str]] = []
    queued_pouch_hand_card_choices: List[List[str]] = []
    raw_clever_targets = details.get("clever_targets")
    if isinstance(raw_clever_targets, (list, tuple)):
        for item in raw_clever_targets:
            target = str(item).strip()
            if target in MAIN_ACTION_CARDS:
                queued_clever_targets.append(target)
    elif isinstance(raw_clever_targets, str):
        target = raw_clever_targets.strip()
        if target in MAIN_ACTION_CARDS:
            queued_clever_targets.append(target)

    raw_hypnosis_targets = details.get("hypnosis_targets")
    if isinstance(raw_hypnosis_targets, (list, tuple)):
        for item in raw_hypnosis_targets:
            target = str(item).strip()
            if target in MAIN_ACTION_CARDS:
                queued_hypnosis_targets.append(target)
    elif isinstance(raw_hypnosis_targets, str):
        target = raw_hypnosis_targets.strip()
        if target in MAIN_ACTION_CARDS:
            queued_hypnosis_targets.append(target)

    raw_hypnosis_target_players = details.get("hypnosis_target_players")
    if isinstance(raw_hypnosis_target_players, (list, tuple)):
        for item in raw_hypnosis_target_players:
            queued_hypnosis_target_players.append(item)
    elif raw_hypnosis_target_players is not None:
        queued_hypnosis_target_players.append(raw_hypnosis_target_players)

    raw_pilfering_choices = details.get("pilfering_choices")
    if isinstance(raw_pilfering_choices, (list, tuple)):
        queued_pilfering_choices = [
            item for item in raw_pilfering_choices if isinstance(item, dict)
        ]
    _append_card_instance_choice_queue(details.get("sell_hand_card_choices"), queued_sell_hand_card_choices)
    _append_card_instance_choice_queue(details.get("pouch_hand_card_choices"), queued_pouch_hand_card_choices)
    queued_boost_action_choices: List[str] = []
    raw_boost_action_choices = details.get("boost_action_choices")
    if isinstance(raw_boost_action_choices, dict):
        normalized = _normalize_boost_action_mode(raw_boost_action_choices.get("mode"))
        if normalized:
            queued_boost_action_choices.append(normalized)
    elif isinstance(raw_boost_action_choices, str):
        normalized = _normalize_boost_action_mode(raw_boost_action_choices)
        if normalized:
            queued_boost_action_choices.append(normalized)
    elif isinstance(raw_boost_action_choices, (list, tuple)):
        for item in raw_boost_action_choices:
            normalized = ""
            if isinstance(item, dict):
                normalized = _normalize_boost_action_mode(item.get("mode"))
            else:
                normalized = _normalize_boost_action_mode(item)
            if normalized:
                queued_boost_action_choices.append(normalized)

    def _choose_action_for_clever() -> str:
        while queued_clever_targets:
            candidate = queued_clever_targets.pop(0)
            if candidate in player.action_order:
                return candidate

        if interactive_effect_prompts:
            print("Clever effect: choose one action card to move to slot 1.")
            for idx, action_card in enumerate(player.action_order, start=1):
                print(f"{idx}. {action_card}")
            while True:
                raw = input(f"Select action card [1-{len(player.action_order)}] (blank=default last): ").strip()
                if raw == "":
                    return player.action_order[-1]
                if not raw.isdigit():
                    print("Please enter a number.")
                    continue
                picked = int(raw)
                if 1 <= picked <= len(player.action_order):
                    return player.action_order[picked - 1]
                print("Out of range, try again.")

        if len(player.action_order) == 1:
            return player.action_order[0]
        # Non-interactive fallback: deterministically pick the rightmost action card.
        return player.action_order[-1]

    initial_hand = list(player.hand)
    resolved_cards: List[AnimalCard] = []
    for play in selected_plays:
        target_card: Optional[AnimalCard] = None
        instance_id = str(play.get("card_instance_id") or "")
        if instance_id:
            target_card = next((card for card in initial_hand if card.instance_id == instance_id), None)
        if target_card is None:
            hand_index = int(play.get("card_hand_index", -1))
            if 0 <= hand_index < len(initial_hand):
                target_card = initial_hand[hand_index]
        if target_card is None:
            raise ValueError("Selected animal card is no longer available.")
        if target_card in resolved_cards:
            raise ValueError("Cannot play the same animal card twice in one animals action.")
        resolved_cards.append(target_card)

    sponsor_228_triggered = (
        _player_has_sponsor(player, 228)
        and bool(resolved_cards)
        and all(_is_small_animal(card_obj) for card_obj in resolved_cards)
    )
    if sponsor_228_triggered:
        used_enclosures = {int(play.get("enclosure_index", -1)) for play in selected_plays}
        projected_money = int(player.money) - sum(int(play.get("card_cost", 0)) for play in selected_plays)
        extra_candidates: List[Tuple[Tuple[int, int, int], Dict[str, Any], AnimalCard]] = []
        for hand_idx, extra_card in enumerate(player.hand):
            if extra_card in resolved_cards:
                continue
            if extra_card.card_type != "animal" or not _is_small_animal(extra_card):
                continue
            if not _animal_size_restriction_met(player, extra_card):
                continue
            ignore_capacity = _animal_condition_ignore_capacity(player, extra_card)
            missing_icons = _animal_icon_missing_conditions(player, extra_card)
            if missing_icons > ignore_capacity:
                continue
            remaining_ignores = max(0, ignore_capacity - missing_icons)
            extra_cost = _animal_play_cost(player, extra_card)
            if extra_cost > projected_money:
                continue
            for enclosure_idx, enclosure in enumerate(player.enclosures):
                reserved_capacity = 0
                if enclosure_idx in used_enclosures:
                    reserved_capacity = sum(
                        int(play.get("spaces_used", 0))
                        for play in selected_plays
                        if int(play.get("enclosure_index", -1)) == enclosure_idx
                    )
                if not _enclosure_can_host_animal_with_ignores(
                    player=player,
                    enclosure=enclosure,
                    card=extra_card,
                    ignore_conditions=remaining_ignores,
                    reserved_capacity=reserved_capacity,
                ):
                    continue
                step = _serialize_animal_play_step(
                    player=player,
                    card=extra_card,
                    hand_index=hand_idx,
                    enclosure=enclosure,
                    enclosure_index=enclosure_idx,
                    resolved_cost=extra_cost,
                )
                ranking = (-int(extra_card.appeal), int(extra_cost), hand_idx)
                extra_candidates.append((ranking, step, extra_card))
        if extra_candidates:
            extra_candidates.sort(key=lambda item: item[0])
            _, chosen_step, chosen_card = extra_candidates[0]
            selected_plays.append(chosen_step)
            resolved_cards.append(chosen_card)
            state.effect_log.append(
                f"sponsor_228_extra_small_animal card={chosen_card.number} enclosure={chosen_step['enclosure_index']}"
            )

    for card, play in zip(resolved_cards, selected_plays):
        if card not in player.hand:
            state.effect_log.append(
                f"animals_followup_skipped_missing_hand_card card={getattr(card, 'number', '?')}"
            )
            continue
        if card.card_type != "animal":
            raise ValueError("Selected card is not an animal card.")
        if not _animal_size_restriction_met(player, card):
            raise ValueError("Selected animal cannot be played due to active sponsor size restriction.")
        resolved_cost = int(play.get("card_cost", _animal_play_cost(player, card)))
        if player.money < resolved_cost:
            raise ValueError("Insufficient money to play selected animal.")

        enclosure_idx = int(play.get("enclosure_index", -1))
        if enclosure_idx < 0 or enclosure_idx >= len(player.enclosures):
            raise ValueError("Selected enclosure index is out of range.")
        enclosure = player.enclosures[enclosure_idx]
        spaces_used = _animal_enclosure_spaces_needed(card, enclosure)
        if spaces_used is None:
            raise ValueError("Selected enclosure cannot host this animal.")

        soft_failures = 0
        icon_missing = _animal_icon_missing_conditions(player, card)
        soft_failures += icon_missing
        enclosure_obj = _find_enclosure_object(player, enclosure)
        if not _player_has_sponsor(player, 219):
            if card.required_rock_adjacency > 0:
                if enclosure_obj is None or enclosure_obj.adjacent_rock < card.required_rock_adjacency:
                    soft_failures += 1
            if card.required_water_adjacency > 0:
                if enclosure_obj is None or enclosure_obj.adjacent_water < card.required_water_adjacency:
                    soft_failures += 1
        sponsor_ignore = _animal_condition_ignore_capacity(player, card)
        soft_failures = max(0, soft_failures - sponsor_ignore)
        if soft_failures > 0:
            if player.camouflage_condition_ignores < soft_failures:
                raise ValueError("Selected animal does not meet required icon/adjacency prerequisites.")
            player.camouflage_condition_ignores -= soft_failures

        if _enclosure_remaining_capacity(enclosure) < spaces_used:
            raise ValueError("Selected enclosure does not have enough remaining capacity.")

        player.money -= resolved_cost
        player.appeal += card.appeal
        _increase_reputation(
            state=state,
            player=player,
            amount=card.reputation_gain,
            allow_interactive=allow_interactive,
        )
        player.conservation += card.conservation
        player.hand.remove(card)
        player.zoo_cards.append(card)
        _apply_animal_to_enclosure(player, enclosure, spaces_used)
        _apply_map_passive_on_standard_enclosure_occupied(
            state=state,
            player=player,
            player_id=player_id,
            enclosure=enclosure,
        )
        _apply_sponsor_passive_triggers_on_card_play(
            state=state,
            played_by_player_id=player_id,
            played_card=card,
            details=details,
            allow_interactive=allow_interactive,
        )

        def _effect_draw_from_deck(count: int) -> Sequence[AnimalCard]:
            drawn_cards = _draw_from_zoo_deck(state, max(0, count))
            player.hand.extend(drawn_cards)
            return drawn_cards

        def _effect_push_to_discard(cards_to_discard: Sequence[AnimalCard]) -> None:
            for discard_card in cards_to_discard:
                if discard_card in player.hand:
                    player.hand.remove(discard_card)
                state.zoo_discard.append(discard_card)

        def _effect_move_action_to_slot_1(action_name: str) -> None:
            if action_name in player.action_order:
                _rotate_action_card_to_slot_1(player, action_name)

        def _effect_boost_action_card(action_name: str) -> str:
            if action_name not in player.action_order:
                return "skip"

            choice_specs: List[Tuple[str, str]] = []
            seen_orders: Set[Tuple[str, ...]] = set()
            for mode, slot_number in (("skip", None), ("slot1", 1), ("slot5", 5)):
                next_order = (
                    list(player.action_order)
                    if slot_number is None
                    else _move_action_order_to_slot(player.action_order, action_name, slot_number)
                )
                order_key = tuple(next_order)
                if order_key in seen_orders:
                    continue
                seen_orders.add(order_key)
                choice_specs.append((mode, f"skip" if slot_number is None else f"slot={slot_number}"))

            available_modes = {mode for mode, _ in choice_specs}
            while queued_boost_action_choices:
                candidate = queued_boost_action_choices.pop(0)
                if candidate in available_modes:
                    if candidate == "slot1":
                        _rotate_action_card_to_slot_1(player, action_name)
                        return "slot=1"
                    if candidate == "slot5":
                        _rotate_action_card_to_slot_5(player, action_name)
                        return "slot=5"
                    return "skip"

            if interactive_effect_prompts:
                print(f"Boost effect: choose where to place {action_name}.")
                for idx, (mode, label) in enumerate(choice_specs, start=1):
                    print(f"{idx}. {label}")
                while True:
                    raw = input(f"Select boost option [1-{len(choice_specs)}] (blank=skip): ").strip()
                    if raw == "":
                        return "skip"
                    if not raw.isdigit():
                        print("Please enter a number.")
                        continue
                    picked = int(raw)
                    if not (1 <= picked <= len(choice_specs)):
                        print("Out of range, try again.")
                        continue
                    chosen_mode = choice_specs[picked - 1][0]
                    if chosen_mode == "slot1":
                        _rotate_action_card_to_slot_1(player, action_name)
                        return "slot=1"
                    if chosen_mode == "slot5":
                        _rotate_action_card_to_slot_5(player, action_name)
                        return "slot=5"
                    return "skip"

            return "skip"

        def _effect_advance_break(steps: int) -> bool:
            triggered = _advance_break_track(state=state, steps=max(0, steps), trigger_player=player_id)
            if triggered:
                _resolve_break(state, allow_interactive=interactive_effect_prompts)
            return triggered

        def _effect_gain_money(amount: int) -> None:
            if amount > 0:
                player.money += amount

        def _effect_take_display_cards(count: int, card_type_filter: str, replenish_each: bool) -> int:
            filter_value = card_type_filter.strip().lower()

            def _matches_filter(display_card: AnimalCard) -> bool:
                if not filter_value:
                    return True
                if filter_value.startswith("badge:"):
                    badge_name = filter_value.split(":", 1)[1].strip().lower()
                    if not badge_name:
                        return True
                    card_badges = tuple(str(item).strip().lower() for item in display_card.badges)
                    return badge_name in card_badges
                return display_card.card_type == filter_value

            taken = 0
            for _ in range(max(0, count)):
                picked_idx = None
                for idx, display_card in enumerate(state.zoo_display):
                    if not _matches_filter(display_card):
                        continue
                    picked_idx = idx
                    break
                if picked_idx is None:
                    break
                player.hand.append(state.zoo_display.pop(picked_idx))
                taken += 1
                if replenish_each:
                    _replenish_zoo_display(state)
            if taken > 0 and not replenish_each:
                _replenish_zoo_display(state)
            return taken

        def _effect_sell_hand_cards(limit: int, money_each: int) -> int:
            max_sell = min(max(0, limit), len(player.hand))
            if max_sell <= 0:
                return 0
            sold_indices: List[int] = []
            if queued_sell_hand_card_choices:
                requested_ids = queued_sell_hand_card_choices.pop(0)
                if interactive_effect_prompts:
                    if len(requested_ids) > max_sell:
                        raise ValueError(f"sell_hand_card_choices can contain at most {max_sell} card(s).")
                    if len(set(requested_ids)) != len(requested_ids):
                        raise ValueError("sell_hand_card_choices card_instance_ids must be unique.")
                    for requested_id in requested_ids:
                        matching_index = next(
                            (idx for idx, hand_card in enumerate(player.hand) if hand_card.instance_id == requested_id),
                            None,
                        )
                        if matching_index is None:
                            raise ValueError("sell_hand_card_choices selected card is not in hand.")
                        sold_indices.append(matching_index)
                else:
                    normalized_ids: List[str] = []
                    seen_ids: Set[str] = set()
                    for requested_id in requested_ids:
                        if requested_id in seen_ids:
                            continue
                        seen_ids.add(requested_id)
                        normalized_ids.append(requested_id)
                    for requested_id in normalized_ids:
                        matching_index = next(
                            (idx for idx, hand_card in enumerate(player.hand) if hand_card.instance_id == requested_id),
                            None,
                        )
                        if matching_index is None:
                            continue
                        sold_indices.append(matching_index)
                    sold_indices = sold_indices[:max_sell]
            elif interactive_effect_prompts:
                sold_indices = _prompt_sell_hand_cards_for_human(
                    player,
                    max_sell=max_sell,
                    money_each=money_each,
                    effect_name="Sell hand cards",
                )
            if len(set(sold_indices)) != len(sold_indices):
                raise ValueError("Selected sell-hand card indices must be unique.")
            if any(idx < 0 or idx >= len(player.hand) for idx in sold_indices):
                raise ValueError("Selected sell-hand card index is out of range.")
            sold_cards = [player.hand[idx] for idx in sorted(sold_indices)]
            for idx in sorted(sold_indices, reverse=True):
                player.hand.pop(idx)
            state.zoo_discard.extend(sold_cards)
            if money_each > 0:
                player.money += len(sold_cards) * money_each
            return len(sold_cards)

        def _effect_pouch_hand_cards(limit: int, appeal_each: int) -> int:
            max_pouch = min(max(0, limit), len(player.hand))
            if max_pouch <= 0:
                return 0
            pouched_indices: List[int] = []
            if queued_pouch_hand_card_choices:
                requested_ids = queued_pouch_hand_card_choices.pop(0)
                if interactive_effect_prompts:
                    if len(requested_ids) > max_pouch:
                        raise ValueError(f"pouch_hand_card_choices can contain at most {max_pouch} card(s).")
                    if len(set(requested_ids)) != len(requested_ids):
                        raise ValueError("pouch_hand_card_choices card_instance_ids must be unique.")
                    for requested_id in requested_ids:
                        matching_index = next(
                            (idx for idx, hand_card in enumerate(player.hand) if hand_card.instance_id == requested_id),
                            None,
                        )
                        if matching_index is None:
                            raise ValueError("pouch_hand_card_choices selected card is not in hand.")
                        pouched_indices.append(matching_index)
                else:
                    normalized_ids: List[str] = []
                    seen_ids: Set[str] = set()
                    for requested_id in requested_ids:
                        if requested_id in seen_ids:
                            continue
                        seen_ids.add(requested_id)
                        normalized_ids.append(requested_id)
                    for requested_id in normalized_ids:
                        matching_index = next(
                            (idx for idx, hand_card in enumerate(player.hand) if hand_card.instance_id == requested_id),
                            None,
                        )
                        if matching_index is None:
                            continue
                        pouched_indices.append(matching_index)
                    pouched_indices = pouched_indices[:max_pouch]
            elif interactive_effect_prompts:
                pouched_indices = _prompt_pouch_hand_cards_for_human(
                    player,
                    max_pouch=max_pouch,
                    appeal_each=appeal_each,
                    effect_name="Pouch cards",
                )
            if len(set(pouched_indices)) != len(pouched_indices):
                raise ValueError("Selected pouch-hand card indices must be unique.")
            if any(idx < 0 or idx >= len(player.hand) for idx in pouched_indices):
                raise ValueError("Selected pouch-hand card index is out of range.")
            pouched_cards = [player.hand[idx] for idx in sorted(pouched_indices)]
            for idx in sorted(pouched_indices, reverse=True):
                player.hand.pop(idx)
            _pouch_cards_under_host(
                player,
                host_card_instance_id=card.instance_id,
                cards=pouched_cards,
            )
            if appeal_each > 0:
                player.appeal += len(pouched_cards) * appeal_each
            return len(pouched_cards)

        def _effect_place_free_kiosk_or_pavilion(times: int) -> int:
            _ensure_player_map_initialized(state, player)
            if player.zoo_map is None:
                return 0
            placed = 0
            bonus_cursor = 0
            for _ in range(max(0, times)):
                legal = player.zoo_map.legal_building_placements(
                    is_build_upgraded=True,
                    has_diversity_researcher=_player_has_sponsor(player, 219),
                    max_building_size=1,
                    already_built_buildings=set(),
                )
                legal = _dedupe_legal_buildings_by_footprint(legal)
                legal = [
                    building for building in legal if building.type in {BuildingType.KIOSK, BuildingType.PAVILION}
                ]
                if not legal:
                    break
                legal.sort(
                    key=lambda building: (
                        0 if building.type == BuildingType.KIOSK else 1,
                        _building_layout_key(building),
                        building.origin_hex.x,
                        building.origin_hex.y,
                        building.rotation.value,
                    )
                )
                picked = legal[0]
                player.zoo_map.add_building(picked)
                if picked.type == BuildingType.PAVILION:
                    player.appeal += 1
                bonuses = _building_bonuses(state, picked)
                for bonus, bonus_coord in bonuses:
                    _apply_build_placement_bonus(
                        state=state,
                        player=player,
                        bonus=bonus,
                        details={},
                        bonus_index=bonus_cursor,
                        bonus_coord=bonus_coord,
                        allow_interactive=False,
                    )
                    bonus_cursor += 1
                _apply_cover_money_from_sponsors_241_242(player, picked)
                _maybe_apply_map_completion_reward(
                    state=state,
                    player=player,
                    player_id=player_id,
                )
                placed += 1
            return placed

        def _effect_add_multiplier_token(action_name: str) -> bool:
            if action_name not in player.multiplier_tokens_on_actions:
                return False
            player.multiplier_tokens_on_actions[action_name] += 1
            return True

        def _effect_apply_venom(amount: int) -> int:
            affected = 0
            for idx, other in enumerate(state.players):
                if idx == player_id:
                    continue
                if _player_has_sponsor(other, 225):
                    continue
                if other.appeal > player.appeal:
                    for action_name in other.action_order[: max(0, amount)]:
                        other.venom_tokens_on_actions[action_name] = 1
                    affected += 1
            return affected

        def _effect_apply_constriction(amount: int) -> int:
            affected = 0
            for idx, other in enumerate(state.players):
                if idx == player_id:
                    continue
                if _player_has_sponsor(other, 225):
                    continue
                tracks_ahead = 0
                if other.appeal > player.appeal:
                    tracks_ahead += 1
                if other.conservation > player.conservation:
                    tracks_ahead += 1
                if tracks_ahead <= 0:
                    continue
                for action_name in other.action_order[-tracks_ahead:]:
                    other.constriction_tokens_on_actions[action_name] = 1
                affected += 1
            return affected

        def _effect_hypnosis(max_slot: int) -> str:
            target_ids = _eligible_hypnosis_target_ids(state, player_id)
            if not target_ids:
                return "no_target"
            if len(target_ids) == 1:
                target_id = target_ids[0]
            elif queued_hypnosis_target_players:
                target_id = _pop_queued_target_player_id(
                    queued_hypnosis_target_players,
                    state,
                    target_ids,
                    effect_name="Hypnosis",
                )
            elif interactive_effect_prompts:
                target_id = _prompt_choose_target_player_for_human(
                    state,
                    target_ids,
                    effect_name="Hypnosis",
                )
            else:
                # Non-interactive fallback: deterministically pick the first legal target player.
                target_id = int(sorted(target_ids)[0])
            target_player = state.players[target_id]
            available_actions = list(target_player.action_order[: max(0, max_slot)])
            if not available_actions:
                return f"target={target_player.name} no_action"

            def _hypnosis_concrete_details_for_action(action_name: str, strength_value: int) -> Optional[Dict[str, Any]]:
                template_action = Action(
                    ActionType.MAIN_ACTION,
                    value=0,
                    card_name=action_name,
                    details={
                        "effective_strength": int(strength_value),
                        "base_strength": int(strength_value),
                    },
                )
                try:
                    if action_name == "animals":
                        concrete = _enumerate_concrete_animals_actions(state, target_player, target_id, template_action)
                    elif action_name == "cards":
                        concrete = _enumerate_concrete_cards_actions(state, target_player, template_action)
                    elif action_name == "build":
                        concrete = _enumerate_concrete_build_actions(state, target_player, target_id, template_action)
                    elif action_name == "association":
                        concrete = _enumerate_concrete_association_actions(state, target_player, target_id, template_action)
                    elif action_name == "sponsors":
                        concrete = _enumerate_concrete_sponsors_actions(state, target_player, target_id, template_action)
                    else:
                        return None
                except ValueError:
                    return None
                if not concrete:
                    return None
                return copy.deepcopy(concrete[0].details or {})

            executable_actions: List[str] = []
            for action_name in available_actions:
                base_strength_for_action = target_player.action_order.index(action_name) + 1
                effective_strength_for_action = _effective_action_strength(
                    target_player,
                    action_name,
                    x_spent=0,
                    base_strength=base_strength_for_action,
                )
                if _hypnosis_concrete_details_for_action(action_name, effective_strength_for_action) is None:
                    continue
                executable_actions.append(action_name)
            if not executable_actions:
                return f"target={target_player.name} no_action"

            chosen_action: Optional[str] = None
            if queued_hypnosis_targets:
                chosen_action = _pop_queued_action_name(
                    queued_hypnosis_targets,
                    executable_actions,
                    effect_name="Hypnosis",
                )
            if interactive_effect_prompts:
                print(f"Hypnosis: choose one action card from {target_player.name} slot 1-{max(0, max_slot)}.")
                for idx, action_name in enumerate(executable_actions, start=1):
                    base_strength = target_player.action_order.index(action_name) + 1
                    effective_strength = _effective_action_strength(
                        target_player,
                        action_name,
                        x_spent=0,
                        base_strength=base_strength,
                    )
                    print(
                        f"{idx}. {action_name} "
                        f"(base={base_strength}, effective={effective_strength}, "
                        f"venom={int(target_player.venom_tokens_on_actions.get(action_name, 0))}, "
                        f"constriction={int(target_player.constriction_tokens_on_actions.get(action_name, 0))})"
                    )
                while True:
                    raw = input(f"Select action [1-{len(executable_actions)}]: ").strip()
                    if raw.isdigit():
                        picked = int(raw)
                        if 1 <= picked <= len(executable_actions):
                            chosen_action = executable_actions[picked - 1]
                            break
                    print("Please enter a valid number.")
            elif chosen_action is None:
                if len(executable_actions) == 1:
                    chosen_action = executable_actions[0]
                else:
                    # Non-interactive fallback: deterministically pick slot-1 action in available range.
                    chosen_action = executable_actions[0]

            if chosen_action is None:
                raise ValueError("Hypnosis action selection failed.")

            x_spent = 0
            if player.x_tokens > 0 and interactive_effect_prompts:
                while True:
                    raw_x = input(f"Spend X-tokens for Hypnosis action [0-{player.x_tokens}] (default 0): ").strip()
                    if raw_x == "":
                        x_spent = 0
                        break
                    if raw_x.isdigit():
                        picked_x = int(raw_x)
                        if 0 <= picked_x <= player.x_tokens:
                            x_spent = picked_x
                            break
                    print("Please enter a valid number.")

            if x_spent > 0:
                player.x_tokens -= x_spent

            base_strength = target_player.action_order.index(chosen_action) + 1
            effective_strength = _effective_action_strength(
                target_player,
                chosen_action,
                x_spent=x_spent,
                base_strength=base_strength,
            )

            hypnosis_details: Optional[Dict[str, Any]] = None
            if interactive_effect_prompts:
                if chosen_action == "animals":
                    hypnosis_details = _prompt_animals_action_details_for_human(
                        state=state,
                        player=player,
                        strength=effective_strength,
                        player_id=player_id,
                    )
                    hypnosis_details = dict(hypnosis_details or {})
                    hypnosis_details["_interactive"] = True
                elif chosen_action == "cards":
                    hypnosis_details = _prompt_cards_action_details_for_human(
                        state=state,
                        player=player,
                        strength=effective_strength,
                    )
                elif chosen_action == "build":
                    hypnosis_details = _prompt_build_action_details_for_human(
                        state=state,
                        player=player,
                        strength=effective_strength,
                        player_id=player_id,
                    )
                elif chosen_action == "association":
                    hypnosis_details = _prompt_association_action_details_for_human(
                        state=state,
                        player=player,
                        strength=effective_strength,
                        player_id=player_id,
                    )
                elif chosen_action == "sponsors":
                    hypnosis_details = _prompt_sponsors_action_details_for_human(
                        state=state,
                        player=player,
                        strength=effective_strength,
                    )
            else:
                hypnosis_details = _hypnosis_concrete_details_for_action(chosen_action, effective_strength)

            skip_dispatch = (
                chosen_action == "build"
                and isinstance(hypnosis_details, dict)
                and isinstance(hypnosis_details.get("selections"), list)
                and len(hypnosis_details.get("selections") or []) == 0
            )
            if not interactive_effect_prompts and hypnosis_details is None:
                skip_dispatch = True
            break_triggered = False
            dispatch_skipped_non_interactive = False
            if not skip_dispatch:
                try:
                    break_triggered = _perform_main_action_dispatch(
                        state=state,
                        player=player,
                        player_id=player_id,
                        chosen=chosen_action,
                        strength=effective_strength,
                        details=hypnosis_details,
                    )
                except ValueError:
                    if interactive_effect_prompts:
                        raise
                    dispatch_skipped_non_interactive = True
            _apply_action_card_token_use(target_player, chosen_action)
            _rotate_action_card_to_slot_1(target_player, chosen_action)
            if break_triggered:
                _resolve_break(state)
            summary = (
                f"target={target_player.name} action={chosen_action} "
                f"x={x_spent} strength={effective_strength}"
            )
            if dispatch_skipped_non_interactive:
                summary += " skipped=invalid"
            return summary

        def _effect_pilfering(amount: int) -> str:
            target_groups = _pilfering_target_groups(state, player_id, amount)
            if not target_groups:
                return "no_target"
            results: List[str] = []
            for target_group in target_groups:
                eligible_target_ids = list(target_group.get("target_ids") or [])
                if not eligible_target_ids:
                    continue
                choice_payload: Optional[Dict[str, Any]] = None
                if len(eligible_target_ids) == 1:
                    target_id = int(eligible_target_ids[0])
                elif interactive_effect_prompts:
                    target_id = _prompt_choose_target_player_for_human(
                        state,
                        eligible_target_ids,
                        effect_name=f"Pilfering ({target_group.get('track', 'track')})",
                    )
                else:
                    if not queued_pilfering_choices:
                        target_id = int(sorted(int(item) for item in eligible_target_ids)[0])
                    else:
                        choice_payload = queued_pilfering_choices.pop(0)
                        target_id = _resolve_target_player_id_from_payload(
                            state,
                            eligible_target_ids,
                            effect_name="Pilfering",
                            payload=choice_payload,
                        )
                target_player = state.players[target_id]
                can_take_money = int(target_player.money) >= 5
                can_take_card = bool(target_player.hand)
                if not can_take_money and not can_take_card:
                    results.append(f"{target_player.name}:none")
                    continue

                choice = "money" if can_take_money else "card"
                chosen_card_index: Optional[int] = None
                if interactive_effect_prompts and can_take_money and can_take_card:
                    print(
                        f"Pilfering: {target_player.name} chooses what to lose."
                    )
                    print("1. Lose 5 money")
                    print("2. Give 1 hand card")
                    while True:
                        raw = input("Select option [1-2]: ").strip()
                        if raw == "1":
                            choice = "money"
                            break
                        if raw == "2":
                            choice = "card"
                            break
                        print("Please enter 1 or 2.")
                elif can_take_money and can_take_card:
                    if choice_payload is None:
                        if not queued_pilfering_choices:
                            choice = "money"
                        else:
                            choice_payload = queued_pilfering_choices.pop(0)
                    if choice_payload is not None:
                        choice, chosen_card_index = _resolve_pilfering_choice_from_details(
                            target_player=target_player,
                            choice_payload=choice_payload,
                            can_take_money=can_take_money,
                            can_take_card=can_take_card,
                        )

                if choice == "money" and can_take_money:
                    target_player.money -= 5
                    player.money += 5
                    results.append(f"{target_player.name}:money")
                    continue
                if can_take_card:
                    if chosen_card_index is None:
                        chosen_card_index = random.randrange(len(target_player.hand))
                    picked = target_player.hand.pop(chosen_card_index)
                    player.hand.append(picked)
                    results.append(f"{target_player.name}:card#{picked.number}")
                    continue
                results.append(f"{target_player.name}:none")
            return "; ".join(results)

        def _effect_scavenge_from_discard(draw_count: int, keep_count: int) -> Tuple[int, int]:
            if draw_count <= 0 or not state.zoo_discard:
                return 0, 0
            random.shuffle(state.zoo_discard)
            drawn: List[AnimalCard] = []
            for _ in range(draw_count):
                if not state.zoo_discard:
                    break
                drawn.append(state.zoo_discard.pop(0))
            kept = min(max(0, keep_count), len(drawn))
            player.hand.extend(drawn[:kept])
            state.zoo_discard.extend(drawn[kept:])
            return len(drawn), kept

        def _effect_draw_final_scoring_keep(draw_count: int, keep_count: int) -> Tuple[int, int]:
            if draw_count <= 0 or not state.final_scoring_deck:
                return 0, 0
            drawn: List[SetupCardRef] = []
            for _ in range(draw_count):
                if not state.final_scoring_deck:
                    break
                drawn.append(state.final_scoring_deck.pop(0))
            kept = min(max(0, keep_count), len(drawn))
            player.final_scoring_cards.extend(drawn[:kept])
            state.final_scoring_discard.extend(drawn[kept:])
            return len(drawn), kept

        def _effect_take_unused_base_project(count: int) -> int:
            if count <= 0:
                return 0
            taken = 0
            for idx in range(count):
                if not state.unused_base_conservation_projects:
                    break
                project = state.unused_base_conservation_projects.pop(0)
                instance_id = (
                    f"assert-{project.data_id}-{state.turn_index}-{player_id}-{len(player.hand)}-{idx}"
                )
                player.hand.append(_make_conservation_project_hand_card(project, instance_id))
                taken += 1
            return taken

        def _effect_gain_x_tokens(amount: int) -> int:
            if amount <= 0:
                return 0
            before = player.x_tokens
            player.x_tokens = min(5, player.x_tokens + amount)
            return player.x_tokens - before

        def _effect_count_icon(icon_name: str) -> int:
            target = icon_name.strip().lower()
            if not target:
                return 0
            snapshot = _player_icon_snapshot(player)
            for key, value in snapshot["continents"].items():
                if key.lower() == target:
                    return int(value)
            for key, value in snapshot["categories"].items():
                if key.lower() == target:
                    return int(value)
            if target == "science":
                return int(snapshot["science"])
            return 0

        def _effect_count_primary_icons() -> int:
            primary = ("bird", "reptile", "primate")
            return sum(_effect_count_icon(name) for name in primary)

        def _effect_mark_extra_action(action_name: str) -> None:
            if action_name in player.extra_actions_granted:
                player.extra_actions_granted[action_name] += 1
                return
            if action_name == "any":
                player.extra_any_actions += 1
                return
            if action_name.startswith("strength:"):
                payload = action_name.split(":", 1)[1]
                if payload.isdigit():
                    strength_value = int(payload)
                    player.extra_strength_actions[strength_value] = (
                        player.extra_strength_actions.get(strength_value, 0) + 1
                    )

        def _effect_adapt_final_scoring(draw_count: int) -> Tuple[int, int]:
            if draw_count <= 0:
                return 0, 0
            drawn: List[SetupCardRef] = []
            for _ in range(draw_count):
                if not state.final_scoring_deck:
                    break
                drawn.append(state.final_scoring_deck.pop(0))
            if drawn:
                player.final_scoring_cards.extend(drawn)
            discard_count = min(draw_count, len(player.final_scoring_cards))
            if discard_count <= 0:
                return len(drawn), 0
            discarded_cards = list(player.final_scoring_cards[:discard_count])
            for card_ref in discarded_cards:
                if card_ref in player.final_scoring_cards:
                    player.final_scoring_cards.remove(card_ref)
            state.final_scoring_discard.extend(discarded_cards)
            return len(drawn), len(discarded_cards)

        def _effect_remove_empty_enclosure_refund(count: int) -> Tuple[int, int]:
            removed = 0
            refunded = 0
            for _ in range(max(0, count)):
                candidates: List[Tuple[int, Enclosure]] = [
                    (idx, enclosure)
                    for idx, enclosure in enumerate(player.enclosures)
                    if _enclosure_is_standard(enclosure)
                    and not _enclosure_has_animals(enclosure)
                    and enclosure.origin is not None
                ]
                if not candidates:
                    break
                candidates.sort(
                    key=lambda item: (
                        -item[1].size,
                        item[1].origin[0] if item[1].origin is not None else 0,
                        item[1].origin[1] if item[1].origin is not None else 0,
                        item[1].rotation,
                    )
                )
                enclosure_idx, enclosure = candidates[0]
                player.enclosures.pop(enclosure_idx)
                remove_origin = enclosure.origin
                if remove_origin is None:
                    continue
                ox, oy = remove_origin
                for obj_idx in range(len(player.enclosure_objects) - 1, -1, -1):
                    obj = player.enclosure_objects[obj_idx]
                    if obj.origin == (ox, oy) and obj.rotation == enclosure.rotation and obj.size == enclosure.size:
                        player.enclosure_objects.pop(obj_idx)
                        break
                if player.zoo_map is not None:
                    matched_origin = None
                    for origin_hex, building in player.zoo_map.buildings.items():
                        if (
                            origin_hex.x == ox
                            and origin_hex.y == oy
                            and building.rotation.name == enclosure.rotation
                            and len(building.layout) == enclosure.size
                            and building.type.subtype == BuildingSubType.ENCLOSURE_BASIC
                        ):
                            matched_origin = origin_hex
                            break
                    if matched_origin is not None:
                        del player.zoo_map.buildings[matched_origin]
                refund = enclosure.size * 2
                player.money += refund
                refunded += refund
                removed += 1
            return removed, refunded

        def _effect_return_association_worker(count: int) -> int:
            returned = 0
            for _ in range(max(0, count)):
                if player.workers_on_association_board <= 0:
                    break
                chosen_task = None
                for task_kind in reversed(ASSOCIATION_TASK_KINDS):
                    if player.association_workers_by_task.get(task_kind, 0) > 0:
                        chosen_task = task_kind
                        break
                if chosen_task is None:
                    break
                player.association_workers_by_task[chosen_task] -= 1
                player.workers_on_association_board -= 1
                player.workers = min(MAX_WORKERS, player.workers + 1)
                returned += 1
            return returned

        def _effect_mark_display_animal(count: int) -> int:
            to_mark = max(0, count)
            if to_mark <= 0:
                return 0
            marked = 0
            accessible = min(_reputation_display_limit(player.reputation), len(state.zoo_display))
            for idx in range(accessible):
                display_card = state.zoo_display[idx]
                if display_card.card_type != "animal":
                    continue
                card_id = _card_identity(display_card)
                if card_id in state.marked_display_card_ids:
                    continue
                state.marked_display_card_ids.add(card_id)
                marked += 1
                if marked >= to_mark:
                    break
            return marked

        def _effect_place_free_large_bird_aviary(times: int) -> int:
            _ensure_player_map_initialized(state, player)
            if player.zoo_map is None:
                return 0
            placed = 0
            bonus_cursor = 0
            max_size = len(BuildingType.LARGE_BIRD_AVIARY.layout)
            for _ in range(max(0, times)):
                legal = player.zoo_map.legal_building_placements(
                    is_build_upgraded=True,
                    has_diversity_researcher=_player_has_sponsor(player, 219),
                    max_building_size=max_size,
                    already_built_buildings=set(),
                )
                legal = _dedupe_legal_buildings_by_footprint(legal)
                legal = [b for b in legal if b.type == BuildingType.LARGE_BIRD_AVIARY]
                if not legal:
                    break
                legal.sort(
                    key=lambda b: (
                        _building_layout_key(b),
                        b.origin_hex.x,
                        b.origin_hex.y,
                        b.rotation.value,
                    )
                )
                picked = legal[0]
                player.zoo_map.add_building(picked)
                _register_enclosure_building(player, picked)
                bonuses = _building_bonuses(state, picked)
                for bonus, bonus_coord in bonuses:
                    _apply_build_placement_bonus(
                        state=state,
                        player=player,
                        bonus=bonus,
                        details={},
                        bonus_index=bonus_cursor,
                        bonus_coord=bonus_coord,
                        allow_interactive=False,
                    )
                    bonus_cursor += 1
                _apply_cover_money_from_sponsors_241_242(player, picked)
                _maybe_apply_map_completion_reward(
                    state=state,
                    player=player,
                    player_id=player_id,
                )
                placed += 1
            return placed

        def _effect_trade_hand_with_display(count: int) -> int:
            traded = 0
            for _ in range(max(0, count)):
                if not player.hand or not state.zoo_display:
                    break
                accessible = min(_reputation_display_limit(player.reputation), len(state.zoo_display))
                if accessible <= 0:
                    break
                display_idx = 0
                hand_idx = 0
                hand_card = player.hand.pop(hand_idx)
                display_card = state.zoo_display[display_idx]
                state.zoo_display[display_idx] = hand_card
                player.hand.append(display_card)
                traded += 1
            return traded

        def _effect_shark_attack(count: int) -> Tuple[int, int]:
            accessible = min(_reputation_display_limit(player.reputation), len(state.zoo_display))
            if accessible <= 0 or count <= 0:
                return 0, 0
            candidate_indices = [
                idx for idx in range(accessible) if state.zoo_display[idx].card_type == "animal"
            ]
            picked_indices = candidate_indices[: max(0, count)]
            if not picked_indices:
                return 0, 0
            discarded_cards: List[AnimalCard] = []
            for idx in sorted(picked_indices, reverse=True):
                discarded_cards.append(state.zoo_display.pop(idx))
            state.zoo_discard.extend(discarded_cards)
            gained_money = sum(max(0, int(card.appeal)) for card in discarded_cards) // 2
            if gained_money > 0:
                player.money += gained_money
            _replenish_zoo_display(state)
            return len(discarded_cards), gained_money

        def _effect_take_specific_base_project(target: str, count: int) -> int:
            wanted = target.strip().lower()
            if not wanted or count <= 0:
                return 0
            taken = 0
            idx = 0
            while idx < len(state.unused_base_conservation_projects) and taken < count:
                project = state.unused_base_conservation_projects[idx]
                project_name = f"{project.data_id} {project.title}".lower()
                if wanted in project_name or (wanted == "primates" and "primate" in project_name):
                    state.unused_base_conservation_projects.pop(idx)
                    instance_id = (
                        f"project-{project.data_id}-{state.turn_index}-{player_id}-{len(player.hand)}-{taken}"
                    )
                    player.hand.append(_make_conservation_project_hand_card(project, instance_id))
                    taken += 1
                    continue
                idx += 1
            return taken

        def _effect_grant_camouflage_ignore(amount: int) -> None:
            if amount <= 0:
                return
            player.camouflage_condition_ignores += amount

        def _effect_symbiosis_copy() -> List[str]:
            candidates: List[AnimalCard] = []
            for zoo_card in reversed(player.zoo_cards):
                if zoo_card is card:
                    continue
                if str(zoo_card.card_type).lower() != "animal":
                    continue
                nested_effect = resolve_card_effect(zoo_card)
                if nested_effect.code in {"none", "symbiosis_copy"}:
                    continue
                if nested_effect.code.startswith("unimplemented:"):
                    continue
                candidates.append(zoo_card)
            if not candidates:
                return []
            nested_card = candidates[0]
            return apply_animal_effect(
                card=nested_card,
                move_action_to_slot_1=_effect_move_action_to_slot_1,
                advance_break=_effect_advance_break,
                draw_from_deck=_effect_draw_from_deck,
                push_to_discard=_effect_push_to_discard,
                choose_action_for_clever=_choose_action_for_clever,
                increase_workers=lambda n: setattr(player, "workers", min(MAX_WORKERS, player.workers + max(0, n))),
                increase_appeal=lambda n: setattr(player, "appeal", player.appeal + max(0, n)),
                gain_money=_effect_gain_money,
                take_display_cards=_effect_take_display_cards,
                sell_hand_cards=_effect_sell_hand_cards,
                pouch_hand_cards=_effect_pouch_hand_cards,
                place_free_kiosk_or_pavilion=_effect_place_free_kiosk_or_pavilion,
                add_multiplier_token=_effect_add_multiplier_token,
                apply_venom=_effect_apply_venom,
                apply_constriction=_effect_apply_constriction,
                perform_hypnosis=_effect_hypnosis,
                perform_pilfering=_effect_pilfering,
                scavenge_from_discard=_effect_scavenge_from_discard,
                mark_extra_action=_effect_mark_extra_action,
                increase_conservation=lambda n: setattr(player, "conservation", player.conservation + max(0, n)),
                draw_final_scoring_keep=_effect_draw_final_scoring_keep,
                take_unused_base_project=_effect_take_unused_base_project,
                gain_x_tokens=_effect_gain_x_tokens,
                count_icon=_effect_count_icon,
                count_primary_icons=_effect_count_primary_icons,
                adapt_final_scoring=_effect_adapt_final_scoring,
                remove_empty_enclosure_refund=_effect_remove_empty_enclosure_refund,
                return_association_worker=_effect_return_association_worker,
                mark_display_animal=_effect_mark_display_animal,
                place_free_large_bird_aviary=_effect_place_free_large_bird_aviary,
                trade_hand_with_display=_effect_trade_hand_with_display,
                shark_attack=_effect_shark_attack,
                take_specific_base_project=_effect_take_specific_base_project,
                symbiosis_copy=None,
                grant_camouflage_ignore=_effect_grant_camouflage_ignore,
                boost_action_card=_effect_boost_action_card,
            )

        effect_messages = apply_animal_effect(
            card=card,
            move_action_to_slot_1=_effect_move_action_to_slot_1,
            advance_break=_effect_advance_break,
            draw_from_deck=_effect_draw_from_deck,
            push_to_discard=_effect_push_to_discard,
            choose_action_for_clever=_choose_action_for_clever,
            increase_workers=lambda n: setattr(player, "workers", min(MAX_WORKERS, player.workers + max(0, n))),
            increase_appeal=lambda n: setattr(player, "appeal", player.appeal + max(0, n)),
            gain_money=_effect_gain_money,
            take_display_cards=_effect_take_display_cards,
            sell_hand_cards=_effect_sell_hand_cards,
            pouch_hand_cards=_effect_pouch_hand_cards,
            place_free_kiosk_or_pavilion=_effect_place_free_kiosk_or_pavilion,
            add_multiplier_token=_effect_add_multiplier_token,
            apply_venom=_effect_apply_venom,
            apply_constriction=_effect_apply_constriction,
            perform_hypnosis=_effect_hypnosis,
            perform_pilfering=_effect_pilfering,
            scavenge_from_discard=_effect_scavenge_from_discard,
            mark_extra_action=_effect_mark_extra_action,
            increase_conservation=lambda n: setattr(player, "conservation", player.conservation + max(0, n)),
            draw_final_scoring_keep=_effect_draw_final_scoring_keep,
            take_unused_base_project=_effect_take_unused_base_project,
            gain_x_tokens=_effect_gain_x_tokens,
            count_icon=_effect_count_icon,
            count_primary_icons=_effect_count_primary_icons,
            adapt_final_scoring=_effect_adapt_final_scoring,
            remove_empty_enclosure_refund=_effect_remove_empty_enclosure_refund,
            return_association_worker=_effect_return_association_worker,
            mark_display_animal=_effect_mark_display_animal,
            place_free_large_bird_aviary=_effect_place_free_large_bird_aviary,
            trade_hand_with_display=_effect_trade_hand_with_display,
            shark_attack=_effect_shark_attack,
            take_specific_base_project=_effect_take_specific_base_project,
            symbiosis_copy=_effect_symbiosis_copy,
            grant_camouflage_ignore=_effect_grant_camouflage_ignore,
            boost_action_card=_effect_boost_action_card,
        )
        for message in effect_messages:
            card_id = card.instance_id or str(card.number)
            state.effect_log.append(f"{card_id}: {message}")

    if sponsor_228_triggered:
        picked_idx = next(
            (
                idx
                for idx, display_card in enumerate(state.zoo_display)
                if display_card.card_type == "animal" and _is_small_animal(display_card)
            ),
            None,
        )
        if picked_idx is not None:
            player.hand.append(state.zoo_display.pop(picked_idx))
            _replenish_zoo_display(state)
            state.effect_log.append("sponsor_228_take_small_animal_from_display=1")


def list_legal_association_options(
    state: GameState,
    player_id: int,
    strength: int,
) -> List[Dict[str, Any]]:
    player = state.players[player_id]
    options: List[Dict[str, Any]] = []

    def _task_workers_or_none(task_kind: str) -> Optional[int]:
        try:
            required = _association_workers_needed(player, task_kind)
        except ValueError:
            return None
        if player.workers < required:
            return None
        return required

    if strength >= 2:
        required = _task_workers_or_none("reputation")
        if required is not None:
            options.append(
                {
                    "task_kind": "reputation",
                    "strength_cost": _association_task_strength_cost(player, "reputation"),
                    "workers_needed": required,
                    "description": f"Gain 2 reputation (workers={required})",
                }
            )

    if strength >= 3:
        required = _task_workers_or_none("partner_zoo")
        if required is not None:
            if len(player.partner_zoos) < 4 and (
                bool(player.action_upgraded.get("association", False)) or len(player.partner_zoos) < 2
            ):
                for partner in sorted(state.available_partner_zoos):
                    if partner in player.partner_zoos:
                        continue
                    options.append(
                        {
                            "task_kind": "partner_zoo",
                            "partner_zoo": partner,
                            "strength_cost": _association_task_strength_cost(player, "partner_zoo"),
                            "workers_needed": required,
                            "description": f"Take partner zoo: {_partner_zoo_label(partner)} (workers={required})",
                        }
                    )

    if strength >= 4:
        required = _task_workers_or_none("university")
        if required is not None:
            for university in sorted(state.available_universities):
                if university in player.universities:
                    continue
                options.append(
                    {
                        "task_kind": "university",
                        "university": university,
                        "strength_cost": _association_task_strength_cost(player, "university"),
                        "workers_needed": required,
                        "description": f"Take {_university_label(university)} (workers={required})",
                    }
                )

    conservation_strength_requirement = _association_task_strength_cost(player, "conservation_project")
    if strength >= conservation_strength_requirement:
        required = _task_workers_or_none("conservation_project")
        if required is not None:
            blocked_by_project = _blocked_level_by_project(state)
            allow_release_repeat = _player_has_sponsor(player, 224)
            icon_reduction = _available_breeding_icon_reduction(player)

            def _append_project_options(
                project: SetupCardRef,
                *,
                project_from_hand_index: Optional[int] = None,
                project_from_display_index: Optional[int] = None,
                display_extra_cost: int = 0,
                source_label: str = "in_play",
            ) -> None:
                if (
                    project.data_id in player.supported_conservation_projects
                    and not (allow_release_repeat and _is_release_into_wild_project(project.data_id))
                ):
                    return
                icon_value = _project_requirement_value(player, project.data_id)
                project_slots = state.conservation_project_slots.get(project.data_id) or {}
                blocked_level = blocked_by_project.get(project.data_id, "")
                project_icon_reduction = icon_reduction if _is_base_conservation_project(project.data_id) else 0
                for level_name, needed_icons, conservation_gain, reputation_gain in _project_level_rewards(project.data_id):
                    if level_name == blocked_level:
                        continue
                    icon_reduction_used = icon_value < needed_icons and project_icon_reduction > 0
                    if icon_value + project_icon_reduction < needed_icons:
                        continue
                    owner = project_slots.get(level_name)
                    if owner is not None:
                        continue
                    reward_parts = [f"+{conservation_gain} CP"]
                    if reputation_gain > 0:
                        reward_parts.append(f"+{reputation_gain} reputation")
                    description = (
                        f"Support conservation project ({source_label}): {project.data_id} | {project.title} "
                        f"[{level_name}: need>={needed_icons} -> {', '.join(reward_parts)}, workers={required}"
                    )
                    if display_extra_cost > 0:
                        description += f", display_cost={display_extra_cost}"
                    if icon_reduction_used:
                        description += ", icon_reduction=1"
                    description += "]"
                    option: Dict[str, Any] = {
                        "task_kind": "conservation_project",
                        "project_id": project.data_id,
                        "project_title": project.title,
                        "project_level": level_name,
                        "strength_cost": conservation_strength_requirement,
                        "required_icons": needed_icons,
                        "icon_value": icon_value,
                        "icon_reduction_used": icon_reduction_used,
                        "conservation_gain": conservation_gain,
                        "reputation_gain": reputation_gain,
                        "workers_needed": required,
                        "description": description,
                    }
                    if project_from_hand_index is not None:
                        option["project_from_hand_index"] = project_from_hand_index
                    if project_from_display_index is not None:
                        option["project_from_display_index"] = project_from_display_index
                        option["project_display_extra_cost"] = display_extra_cost
                    options.append(option)

            for project in _current_conservation_projects(state):
                _append_project_options(project, source_label="in_play")

            for hand_index, hand_card in enumerate(player.hand):
                if hand_card.card_type != "conservation_project":
                    continue
                project = _project_ref_from_card(hand_card)
                if project.data_id in state.conservation_project_slots:
                    continue
                _append_project_options(
                    project,
                    project_from_hand_index=hand_index,
                    source_label=f"hand[{hand_index + 1}]",
                )

            if bool(player.action_upgraded.get("association", False)):
                accessible = min(_reputation_display_limit(player.reputation), len(state.zoo_display))
                for display_index in range(accessible):
                    display_card = state.zoo_display[display_index]
                    if display_card.card_type != "conservation_project":
                        continue
                    project = _project_ref_from_card(display_card)
                    if project.data_id in state.conservation_project_slots:
                        continue
                    display_extra_cost = display_index + 1
                    if player.money < display_extra_cost:
                        continue
                    _append_project_options(
                        project,
                        project_from_display_index=display_index,
                        display_extra_cost=display_extra_cost,
                        source_label=f"display[{display_index + 1}]",
                    )

    for idx, option in enumerate(options, start=1):
        option["index"] = idx
    return options


def _association_task_request_from_option(
    option: Dict[str, Any],
    reward_details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    task_kind = str(option.get("task_kind") or "").strip()
    request: Dict[str, Any] = {"task_kind": task_kind}
    if task_kind == "partner_zoo":
        request["partner_zoo"] = str(option.get("partner_zoo") or "")
    elif task_kind == "university":
        request["university"] = str(option.get("university") or "")
    elif task_kind == "conservation_project":
        request["project_id"] = str(option.get("project_id") or "")
        request["project_level"] = str(option.get("project_level") or "")
        if option.get("project_from_hand_index") is not None:
            request["project_from_hand_index"] = int(option["project_from_hand_index"])
        if option.get("project_from_display_index") is not None:
            request["project_from_display_index"] = int(option["project_from_display_index"])
    if isinstance(reward_details, dict):
        request = _merge_list_detail_fragments(request, reward_details)
    return request


def _compact_association_unlock_label(reward_label: str) -> str:
    label = str(reward_label or "").strip()
    if not label:
        return ""
    match = re.match(r"unlock\[(\d+)\]:\s*(.*)", label, flags=re.IGNORECASE)
    unlock_prefix = "unlock"
    if match:
        unlock_prefix = f"unlock[{match.group(1)}]"
        label = match.group(2).strip()
    else:
        prefix = "unlock: "
        if label.lower().startswith(prefix):
            label = label[len(prefix):].strip()
    label = label.replace("free ", "free:")
    return f"{unlock_prefix}({label})"


def _compact_association_effect_label(reward_label: str) -> str:
    raw = str(reward_label or "").strip()
    if not raw:
        return ""
    compacted_parts: List[str] = []
    for part in [item.strip() for item in raw.split(" ; ") if item.strip()]:
        if part.lower().startswith("unlock"):
            compacted_parts.append(_compact_association_unlock_label(part))
        else:
            compacted_parts.append(part)
    return " + ".join(compacted_parts)


def _compact_association_university_label(university: str) -> str:
    key = str(university or "").strip().lower()
    if key == "reputation_1_hand_limit_5":
        return "uni(hand5,+1rep)"
    if key == "science_1_reputation_2":
        return "uni(+1science,+2rep)"
    if key == "science_2":
        return "uni(+2science)"
    return f"uni({key})"


def _compact_association_project_level(level_name: str) -> str:
    key = str(level_name or "").strip().lower()
    if key == "left_level":
        return "left"
    if key == "middle_level":
        return "mid"
    if key == "right_level":
        return "right"
    return key or "?"


def _association_task_label(option: Dict[str, Any], reward_label: str = "") -> str:
    task_kind = str(option.get("task_kind") or "").strip()
    if task_kind == "reputation":
        label = "rep+2"
    elif task_kind == "partner_zoo":
        label = f"partner({_partner_zoo_label(str(option.get('partner_zoo') or ''))})"
    elif task_kind == "university":
        label = _compact_association_university_label(str(option.get("university") or ""))
    elif task_kind == "conservation_project":
        source = "proj"
        if option.get("project_from_display_index") is not None:
            source = f"proj display[{int(option['project_from_display_index']) + 1}]"
        elif option.get("project_from_hand_index") is not None:
            source = f"proj hand[{int(option['project_from_hand_index']) + 1}]"
        project_number = _project_number_from_data_id(str(option.get("project_id") or ""))
        reward = f"+{int(option.get('conservation_gain', 0))}CP"
        reputation_gain = int(option.get("reputation_gain", 0))
        if reputation_gain > 0:
            reward += f",+{reputation_gain}rep"
        label = (
            f"{source} P{project_number:03d} "
            f"{_compact_association_project_level(str(option.get('project_level') or ''))}"
            f"({reward})"
        )
        if bool(option.get("icon_reduction_used")):
            label += " -1icon"
    else:
        label = str(option.get("description") or task_kind)
    compact_reward = _compact_association_effect_label(reward_label)
    if compact_reward:
        return f"{label} + {compact_reward}"
    return label


def _resolve_association_requested_option(
    options: Sequence[Dict[str, Any]],
    details: Dict[str, Any],
) -> Dict[str, Any]:
    if not options:
        has_explicit_task = (
            details.get("association_option_index") is not None
            or bool(str(details.get("task_kind") or "").strip())
            or bool(details.get("association_task_sequence"))
        )
        if bool(details.get("make_donation")) and not has_explicit_task:
            raise ValueError("Donation must be combined with another association task.")
        if has_explicit_task:
            raise ValueError("Requested association task is not legal.")
        raise ValueError("No legal association tasks are available.")

    selected_idx_raw = details.get("association_option_index")
    if selected_idx_raw is not None:
        selected_idx = int(selected_idx_raw)
        if selected_idx < 0 or selected_idx >= len(options):
            raise ValueError("association_option_index is out of range.")
        return dict(options[selected_idx])

    task_kind = str(details.get("task_kind") or "").strip()
    if not task_kind:
        if len(options) == 1:
            return dict(options[0])
        raise ValueError("Explicit association task details are required when multiple legal options exist.")

    selected = next(
        (
            option
            for option in options
            if option.get("task_kind") == task_kind
            and (
                task_kind != "partner_zoo"
                or option.get("partner_zoo") == details.get("partner_zoo")
            )
            and (
                task_kind != "university"
                or option.get("university") == details.get("university")
            )
            and (
                task_kind != "conservation_project"
                or (
                    option.get("project_id") == details.get("project_id")
                    and (
                        not details.get("project_level")
                        or option.get("project_level") == details.get("project_level")
                    )
                    and (
                        details.get("project_from_hand_index") is None
                        or option.get("project_from_hand_index") == int(details.get("project_from_hand_index"))
                    )
                    and (
                        details.get("project_from_display_index") is None
                        or option.get("project_from_display_index") == int(details.get("project_from_display_index"))
                    )
                )
            )
        ),
        None,
    )
    if (
        selected is None
        and task_kind == "conservation_project"
        and details.get("project_id")
        and not details.get("project_level")
    ):
        project_id = details.get("project_id")
        candidates = [
            option
            for option in options
            if option.get("task_kind") == "conservation_project"
            and option.get("project_id") == project_id
        ]
        if len(candidates) == 1:
            selected = candidates[0]
        elif len(candidates) > 1:
            raise ValueError("project_level is required when multiple legal conservation project levels exist.")
    if selected is None:
        raise ValueError("Requested association task is not legal.")
    return dict(selected)


def _apply_association_selected_option(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    selected: Dict[str, Any],
    effect_details: Optional[Dict[str, Any]] = None,
    allow_interactive: bool = False,
) -> None:
    upgraded = bool(player.action_upgraded["association"])
    task_kind = str(selected.get("task_kind") or "").strip()
    if not task_kind:
        raise ValueError("Association task kind is required.")

    workers_needed = _association_workers_needed(player, task_kind)
    if player.workers < workers_needed:
        raise ValueError("Not enough active association workers.")

    if task_kind == "reputation":
        _apply_reputation_gain_with_details(
            state=state,
            player=player,
            player_id=player_id,
            amount=2,
            details=effect_details,
            allow_interactive=allow_interactive,
        )
    elif task_kind == "partner_zoo":
        _gain_partner_zoo_reward(
            state=state,
            player=player,
            player_id=player_id,
            partner=str(selected.get("partner_zoo") or ""),
            effect_details=effect_details,
            allow_interactive=allow_interactive,
        )
    elif task_kind == "university":
        _gain_university_reward(
            state=state,
            player=player,
            player_id=player_id,
            university=str(selected.get("university") or ""),
            effect_details=effect_details,
            allow_interactive=allow_interactive,
        )
    elif task_kind == "conservation_project":
        project_id = str(selected.get("project_id") or "")
        if not project_id:
            raise ValueError("Missing project_id for conservation project task.")
        release_repeat_allowed = _player_has_sponsor(player, 224) and _is_release_into_wild_project(project_id)
        if project_id in player.supported_conservation_projects and not release_repeat_allowed:
            raise ValueError("Conservation project already supported by this player.")
        project_from_hand_index_raw = selected.get("project_from_hand_index")
        project_from_display_index_raw = selected.get("project_from_display_index")
        if project_from_hand_index_raw is not None and project_from_display_index_raw is not None:
            raise ValueError("Conservation project cannot come from both hand and display.")
        if project_from_hand_index_raw is not None:
            project_from_hand_index = int(project_from_hand_index_raw)
            if project_from_hand_index < 0 or project_from_hand_index >= len(player.hand):
                raise ValueError("Conservation project hand index is out of range.")
            raw_project = player.hand[project_from_hand_index]
            if raw_project.card_type != "conservation_project":
                raise ValueError("Selected hand card is not a conservation project.")
            project_ref = _project_ref_from_card(raw_project)
            if project_ref.data_id != project_id:
                raise ValueError("Selected hand conservation project does not match requested project.")
            player.hand.pop(project_from_hand_index)
            _ensure_conservation_project_slots(state, project_id)
            _on_new_conservation_project_added_to_row(state, project_id=project_id)
        elif project_from_display_index_raw is not None:
            if not upgraded:
                raise ValueError("Only upgraded Association can support conservation projects from display.")
            project_from_display_index = int(project_from_display_index_raw)
            accessible = min(_reputation_display_limit(player.reputation), len(state.zoo_display))
            if project_from_display_index < 0 or project_from_display_index >= accessible:
                raise ValueError("Conservation project display index is out of reputation range.")
            raw_project = state.zoo_display[project_from_display_index]
            if raw_project.card_type != "conservation_project":
                raise ValueError("Selected display card is not a conservation project.")
            project_ref = _project_ref_from_card(raw_project)
            if project_ref.data_id != project_id:
                raise ValueError("Selected display conservation project does not match requested project.")
            display_extra_cost = int(selected.get("project_display_extra_cost", project_from_display_index + 1))
            if player.money < display_extra_cost:
                raise ValueError("Insufficient money for display conservation project additional cost.")
            player.money -= display_extra_cost
            state.zoo_display.pop(project_from_display_index)
            _replenish_zoo_display(state)
            _ensure_conservation_project_slots(state, project_id)
            _on_new_conservation_project_added_to_row(state, project_id=project_id)
        project_level = str(selected.get("project_level") or "")
        if project_level not in {"left_level", "middle_level", "right_level"}:
            raise ValueError("Missing or invalid project_level for conservation project task.")
        slots = state.conservation_project_slots.get(project_id)
        if slots is None:
            raise ValueError("Conservation project slot does not exist.")
        owner = slots.get(project_level)
        if owner is not None:
            raise ValueError("Selected conservation project slot is already occupied.")
        slots[project_level] = player_id
        player.supported_conservation_projects.add(project_id)
        player.conservation += int(selected.get("conservation_gain", 0))
        reputation_gain = int(selected.get("reputation_gain", 0))
        if reputation_gain > 0:
            _increase_reputation(
                state=state,
                player=player,
                amount=reputation_gain,
                allow_interactive=allow_interactive,
            )
        if release_repeat_allowed:
            player.conservation += 1
        if bool(selected.get("icon_reduction_used")):
            _consume_breeding_icon_reduction(player)
        player.supported_conservation_project_actions += 1
        _on_map_conservation_project_supported(
            state=state,
            player=player,
            player_id=player_id,
            effect_details=effect_details,
            allow_interactive=allow_interactive,
        )
    else:
        raise ValueError(f"Unsupported association task kind '{task_kind}'.")

    _spend_association_workers(player=player, task_kind=task_kind, workers_needed=workers_needed)


def _perform_association_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    details: Optional[Dict[str, Any]] = None,
    player_id: int = 0,
) -> None:
    details = details or {}
    allow_interactive = bool(details.get("_allow_interactive", False) or details.get("_interactive", False))
    upgraded = bool(player.action_upgraded["association"])
    make_donation = bool(details.get("make_donation"))
    if make_donation and not upgraded:
        raise ValueError("Donation is only available with upgraded Association action.")

    performed_any_task = False
    sequence_raw = details.get("association_task_sequence")
    if sequence_raw is not None:
        if not isinstance(sequence_raw, list):
            raise ValueError("association_task_sequence must be a list.")
        task_sequence = [dict(item) for item in sequence_raw if isinstance(item, dict)]
        if not task_sequence:
            raise ValueError("association_task_sequence must include at least one task.")
        if len(task_sequence) > 1 and not upgraded:
            raise ValueError("Only upgraded Association can perform multiple tasks.")
        remaining_strength = int(strength)
        used_task_kinds: Set[str] = set()
        for task_details in task_sequence:
            task_kind = str(task_details.get("task_kind") or "").strip()
            if not task_kind:
                raise ValueError("Association task sequence entries require task_kind.")
            if task_kind in used_task_kinds:
                raise ValueError("Upgraded Association tasks must all be different.")
            options = list_legal_association_options(state=state, player_id=player_id, strength=remaining_strength)
            selected = _resolve_association_requested_option(options, task_details)
            strength_cost = int(selected.get("strength_cost", _association_task_strength_cost(player, task_kind)))
            if strength_cost > remaining_strength:
                raise ValueError("Association task sequence exceeds available strength.")
            _apply_association_selected_option(
                state=state,
                player=player,
                player_id=player_id,
                selected=selected,
                effect_details=task_details.get("map_left_track_choice")
                if isinstance(task_details.get("map_left_track_choice"), dict)
                else None,
                allow_interactive=allow_interactive,
            )
            remaining_strength -= strength_cost
            used_task_kinds.add(task_kind)
            performed_any_task = True
    else:
        options = list_legal_association_options(state=state, player_id=player_id, strength=strength)
        selected = _resolve_association_requested_option(options, details)
        _apply_association_selected_option(
            state=state,
            player=player,
            player_id=player_id,
            selected=selected,
            effect_details=details.get("map_left_track_choice")
            if isinstance(details.get("map_left_track_choice"), dict)
            else None,
            allow_interactive=allow_interactive,
        )
        performed_any_task = True

    if make_donation:
        if not performed_any_task:
            raise ValueError("Donation must be combined with another association task.")
        donation_cost = _current_donation_cost(state)
        if player.money < donation_cost:
            raise ValueError("Insufficient money for donation.")
        player.money -= donation_cost
        player.conservation += 1
        state.donation_progress += 1


def _distinct_continent_and_category_types_from_cards(cards: Sequence[AnimalCard]) -> Set[str]:
    types: Set[str] = set()
    for card in cards:
        for badge in _card_icon_keys(card):
            continent = _normalize_continent_badge(badge)
            if continent is not None:
                types.add(_canonical_icon_key(continent))
                continue
            if badge in {"science", "water", "rock", "bear"}:
                continue
            types.add(badge)
    return types


def _take_one_card_from_deck_or_reputation_range(state: GameState, player: PlayerState) -> bool:
    accessible = min(_reputation_display_limit(int(player.reputation)), len(state.zoo_display))
    if accessible > 0:
        player.hand.append(state.zoo_display.pop(0))
        _replenish_zoo_display(state)
        return True
    drawn = _draw_from_zoo_deck(state, 1)
    if drawn:
        player.hand.extend(drawn)
        return True
    return False


def _take_one_card_from_deck_or_reputation_range_with_choice(
    state: GameState,
    player: PlayerState,
    *,
    draw_source: str,
    display_index: Optional[int] = None,
) -> bool:
    choice = str(draw_source).strip().lower()
    accessible = min(_reputation_display_limit(int(player.reputation)), len(state.zoo_display))
    if choice == "display":
        if display_index is None:
            raise ValueError("Display draw choice requires display_index.")
        idx = int(display_index)
        if idx < 0 or idx >= accessible:
            raise ValueError("Chosen display card is outside reputation range.")
        player.hand.append(state.zoo_display.pop(idx))
        _replenish_zoo_display(state)
        return True
    if choice == "deck":
        drawn = _draw_from_zoo_deck(state, 1)
        if drawn:
            player.hand.extend(drawn)
            return True
        return False
    raise ValueError("draw_source must be 'deck' or 'display'.")


def _take_one_card_from_reputation_range(
    state: GameState,
    player: PlayerState,
    *,
    display_index: Optional[int] = None,
) -> bool:
    accessible = min(_reputation_display_limit(int(player.reputation)), len(state.zoo_display))
    if accessible <= 0:
        return False
    idx = 0 if display_index is None else int(display_index)
    if idx < 0 or idx >= accessible:
        raise ValueError("Chosen display card is outside reputation range.")
    player.hand.append(state.zoo_display.pop(idx))
    _replenish_zoo_display(state)
    return True


def _income_by_icon_threshold(count: int) -> int:
    if count >= 5:
        return 9
    if count >= 3:
        return 6
    if count >= 1:
        return 3
    return 0


def _place_free_building_of_type_if_possible(
    state: GameState,
    player: PlayerState,
    building_type: BuildingType,
    player_id: Optional[int] = None,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    _ensure_player_map_initialized(state, player)
    if player.zoo_map is None:
        return False
    details = details or {}
    allow_interactive = bool(details.get("_allow_interactive", False) or details.get("_interactive", False))

    legal = player.zoo_map.legal_building_placements(
        is_build_upgraded=bool(player.action_upgraded["build"]),
        has_diversity_researcher=_player_has_sponsor(player, 219),
        max_building_size=len(building_type.layout),
        already_built_buildings=set(),
    )
    legal = _dedupe_legal_buildings_by_footprint(legal)
    legal = [building for building in legal if building.type == building_type]
    if not legal:
        return False
    selection = details.get("selection")
    picked: Optional[Building] = None
    if isinstance(selection, dict):
        picked = _find_legal_building_by_serialized_selection(legal, selection)
        if picked is None:
            raise ValueError("Selected free building placement is not legal.")
    if picked is None:
        legal.sort(
            key=lambda b: (
                _building_layout_key(b),
                b.origin_hex.x,
                b.origin_hex.y,
                b.rotation.value,
            )
        )
        picked = legal[0]
    player.zoo_map.add_building(picked)
    if picked.type.subtype in {
        BuildingSubType.ENCLOSURE_BASIC,
        BuildingSubType.ENCLOSURE_SPECIAL,
    }:
        _register_enclosure_building(player, picked)
    if picked.type == BuildingType.PAVILION:
        player.appeal += 1

    for bonus_name, bonus_coord in _building_bonuses(state, picked):
        _apply_build_placement_bonus(
            state=state,
            player=player,
            bonus=bonus_name,
            details=details,
            bonus_index=0,
            bonus_coord=bonus_coord,
            allow_interactive=allow_interactive,
        )
    _apply_cover_money_from_sponsors_241_242(player, picked)
    resolved_player_id = player_id if player_id is not None else state.players.index(player)
    _maybe_apply_map_completion_reward(
        state=state,
        player=player,
        player_id=resolved_player_id,
    )
    return True


def _play_sponsor_from_hand_via_253(
    state: GameState,
    player: PlayerState,
    player_id: int,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    details = details or {}
    tokens_left = int(player.sponsor_tokens_by_number.get(253, 0))
    if tokens_left <= 0:
        return False

    sponsors_upgraded = bool(player.action_upgraded["sponsors"])
    candidates: List[AnimalCard] = []
    for card in player.hand:
        if card.card_type != "sponsor":
            continue
        ok, _ = _sponsor_requirements_met(player=player, card=card, sponsors_upgraded=sponsors_upgraded)
        if not ok:
            continue
        if not _sponsor_unique_building_can_be_placed(state, player, card.number):
            continue
        level = _sponsor_level(card)
        if player.money < level:
            continue
        candidates.append(card)
    if not candidates:
        return False

    selection = _pop_detail_queue_entry(details, "sponsor_253_plays")
    if selection is None:
        if len(candidates) == 1:
            selection = {
                "card_instance_id": candidates[0].instance_id,
            }
        elif bool(details.get("_expand_implicit_choices")):
            raise _ActionDetailExpansionRequired(
                _enumerate_sponsor_253_choice_variants(
                    state=state,
                    player=player,
                    player_id=player_id,
                    candidates=candidates,
                )
            )
        else:
            selection = {
                "card_instance_id": candidates[0].instance_id,
            }
            state.effect_log.append(
                f"sponsor_253_default_choice card={candidates[0].number} source_index={player.hand.index(candidates[0])}"
            )
    if bool(selection.get("skip")):
        return False

    card_instance_id = str(selection.get("card_instance_id") or "").strip()
    source_index = int(selection.get("source_index", -1))
    card: Optional[AnimalCard] = None
    if card_instance_id:
        card = next((candidate for candidate in candidates if candidate.instance_id == card_instance_id), None)
    if card is None and 0 <= source_index < len(player.hand):
        indexed = player.hand[source_index]
        if indexed in candidates:
            card = indexed
    if card is None:
        raise ValueError("Sponsor #253 selected sponsor card is not a legal playable hand sponsor.")

    player.sponsor_tokens_by_number[253] = tokens_left - 1
    _play_single_sponsor_from_hand_paying_cost(
        state=state,
        player=player,
        player_id=player_id,
        card=card,
        details=details,
    )
    state.effect_log.append(
        f"sponsor_253_play card={card.number} level={_sponsor_level(card)} cost={_sponsor_level(card)}"
    )
    return True


def _validate_sponsor_card_for_play(
    state: GameState,
    player: PlayerState,
    player_id: int,
    card: AnimalCard,
    *,
    sponsors_upgraded: bool,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    if card.card_type != "sponsor":
        raise ValueError("Selected card is not a sponsor card.")
    ok, reason = _sponsor_requirements_met(
        player=player,
        card=card,
        sponsors_upgraded=sponsors_upgraded,
    )
    if not ok:
        raise ValueError(f"Sponsor requirement not met: {card.name} ({reason})")
    if not _sponsor_unique_building_can_be_placed(state, player, card.number):
        raise ValueError(f"Sponsor requirement not met: {card.name} (no_legal_unique_building_placement)")
    _validate_sponsor_effect_details(
        state=state,
        player=player,
        player_id=player_id,
        card=card,
        details=details,
    )


def _finalize_sponsor_card_play(
    state: GameState,
    player: PlayerState,
    player_id: int,
    card: AnimalCard,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    details = details or {}
    allow_interactive = bool(details.get("_allow_interactive", False) or details.get("_interactive", False))
    player.zoo_cards.append(card)
    messages = _apply_sponsor_immediate_effects(
        state=state,
        player=player,
        player_id=player_id,
        card=card,
        details=details,
        allow_interactive=allow_interactive,
    )
    for message in messages:
        state.effect_log.append(f"{card.instance_id}: sponsor_{message}")
    _apply_sponsor_passive_triggers_on_card_play(
        state=state,
        played_by_player_id=player_id,
        played_card=card,
        details=details,
        allow_interactive=allow_interactive,
    )
    if card.number in {215, 218}:
        player.sponsor_tokens_by_number[card.number] = player.sponsor_tokens_by_number.get(card.number, 0) + 2
    if card.number == 253:
        player.sponsor_tokens_by_number[253] = player.sponsor_tokens_by_number.get(card.number, 0) + 3


def _play_sponsor_card_from_source(
    state: GameState,
    player: PlayerState,
    player_id: int,
    *,
    source: str,
    card: AnimalCard,
    details: Optional[Dict[str, Any]] = None,
    pay_cost: Optional[int] = None,
) -> bool:
    source_key = str(source).strip().lower()
    if source_key == "hand":
        if card not in player.hand:
            raise ValueError("Selected sponsor card disappeared from hand.")
        resolved_pay_cost = _sponsor_play_cost(card) if pay_cost is None else int(pay_cost)
        if player.money < resolved_pay_cost:
            raise ValueError("Insufficient money for sponsor card.")
        player.money -= resolved_pay_cost
        player.hand.remove(card)
        _finalize_sponsor_card_play(
            state=state,
            player=player,
            player_id=player_id,
            card=card,
            details=details,
        )
        return False

    if source_key == "display":
        display_index = next(
            (idx for idx, candidate in enumerate(state.zoo_display) if candidate.instance_id == card.instance_id),
            None,
        )
        if display_index is None:
            raise ValueError("Selected sponsor card disappeared from display.")
        accessible = _reputation_display_limit(int(player.reputation))
        if display_index + 1 > accessible:
            raise ValueError("Selected sponsor card is outside reputation range.")
        resolved_pay_cost = (_sponsor_play_cost(card) + display_index + 1) if pay_cost is None else int(pay_cost)
        if player.money < resolved_pay_cost:
            raise ValueError("Insufficient money for sponsor card and display cost.")
        player.money -= resolved_pay_cost
        state.zoo_display.pop(display_index)
        _finalize_sponsor_card_play(
            state=state,
            player=player,
            player_id=player_id,
            card=card,
            details=details,
        )
        return True

    raise ValueError("Unsupported sponsor source.")


def _validate_sponsor_sequence_affordability(
    state: GameState,
    player: PlayerState,
    resolved_cards: Sequence[Tuple[str, AnimalCard, int, int]],
) -> None:
    simulated_money = int(player.money)
    simulated_display_ids = [card.instance_id for card in state.zoo_display]
    for source, card, _, _ in resolved_cards:
        if source == "display":
            if card.instance_id not in simulated_display_ids:
                raise ValueError("Selected sponsor card from display is not available.")
            display_index = simulated_display_ids.index(card.instance_id)
            pay_cost = _sponsor_play_cost(card) + display_index + 1
            simulated_display_ids.pop(display_index)
        else:
            pay_cost = _sponsor_play_cost(card)
        if simulated_money < pay_cost:
            raise ValueError("Insufficient money for selected sponsor sequence.")
        simulated_money -= pay_cost


def _play_single_sponsor_from_hand_paying_cost(
    state: GameState,
    player: PlayerState,
    player_id: int,
    card: AnimalCard,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    details = details or {}
    sponsors_upgraded = bool(player.action_upgraded["sponsors"])
    _validate_sponsor_card_for_play(
        state=state,
        player=player,
        player_id=player_id,
        card=card,
        sponsors_upgraded=sponsors_upgraded,
        details=details,
    )
    pay_cost = _sponsor_level(card)
    if player.money < pay_cost:
        raise ValueError("Insufficient money to play sponsor by paying cost.")
    _play_sponsor_card_from_source(
        state=state,
        player=player,
        player_id=player_id,
        source="hand",
        card=card,
        details=details,
        pay_cost=pay_cost,
    )


def _apply_sponsor_passive_triggers_on_card_play(
    state: GameState,
    played_by_player_id: int,
    played_card: AnimalCard,
    details: Optional[Dict[str, Any]] = None,
    *,
    allow_interactive: bool = False,
) -> None:
    details = details or {}
    played_icon_counts = _card_icon_counts(played_card)
    played_icon_keys = set(played_icon_counts.keys())
    for owner_id, owner in enumerate(state.players):
        sponsor_numbers = {card.number for card in _player_played_sponsor_cards(owner)}
        if not sponsor_numbers:
            continue

        owner_is_actor = owner_id == played_by_player_id

        science_icons = int(played_icon_counts.get("science", 0))
        if science_icons > 0:
            if owner_is_actor and 202 in sponsor_numbers:
                _increase_reputation(
                    state=state,
                    player=owner,
                    amount=science_icons,
                    allow_interactive=allow_interactive,
                )
            if owner_is_actor and 204 in sponsor_numbers:
                owner.conservation += science_icons
            if 208 in sponsor_numbers:
                owner.money += 2 * science_icons

        for sponsor_no, icon_name in SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS.items():
            icon_count = int(played_icon_counts.get(icon_name, 0))
            if sponsor_no in sponsor_numbers and icon_count > 0:
                owner.money += 3 * icon_count

        bear_icons = int(played_icon_counts.get("bear", 0))
        if 251 in sponsor_numbers and bear_icons > 0:
            owner.appeal += 2 * bear_icons

        for sponsor_no, (icon_name, appeal_gain) in SPONSOR_OWN_ICON_APPEAL_TRIGGERS.items():
            icon_count = int(played_icon_counts.get(icon_name, 0))
            if owner_is_actor and sponsor_no in sponsor_numbers and icon_count > 0:
                owner.appeal += appeal_gain * icon_count

        primate_icons = int(played_icon_counts.get("primate", 0))
        if owner_is_actor and 248 in sponsor_numbers and primate_icons > 0:
            owner.x_tokens = min(5, owner.x_tokens + primate_icons)

        africa_icons = int(played_icon_counts.get("africa", 0))
        if owner_is_actor and 214 in sponsor_numbers and africa_icons > 0 and owner.action_order:
            for _ in range(africa_icons):
                _rotate_action_card_to_slot_1(owner, owner.action_order[-1])
        if owner_is_actor and 212 in sponsor_numbers:
            australia_icons = int(played_icon_counts.get("australia", 0))
            pouch_host = next((card for card in owner.zoo_cards if card.number == 212), None)
            pouched = 0
            for _ in range(australia_icons):
                if not owner.hand:
                    break
                pouched_card = owner.hand.pop(0)
                _pouch_cards_under_host(
                    owner,
                    host_card_instance_id=getattr(pouch_host, "instance_id", ""),
                    cards=[pouched_card],
                )
                owner.appeal += 2
                pouched += 1
            if pouched > 0:
                state.effect_log.append(f"sponsor_212_pouch_card_for_appeal={pouched * 2}")

        if owner_is_actor and 210 in sponsor_numbers:
            america_icons = int(played_icon_counts.get("america", 0))
            placed = 0
            for _ in range(america_icons):
                if _place_free_building_of_type_if_possible(
                    state=state,
                    player=owner,
                    building_type=BuildingType.KIOSK,
                    player_id=owner_id,
                ):
                    placed += 1
            if placed > 0:
                state.effect_log.append(f"sponsor_210_free_kiosk={placed}")
        if owner_is_actor and 211 in sponsor_numbers:
            europe_icons = int(played_icon_counts.get("europe", 0))
            placed = 0
            for _ in range(europe_icons):
                if _place_free_building_of_type_if_possible(
                    state=state,
                    player=owner,
                    building_type=BuildingType.SIZE_1,
                    player_id=owner_id,
                ):
                    placed += 1
            if placed > 0:
                state.effect_log.append(f"sponsor_211_free_size_1_enclosure={placed}")
        if owner_is_actor and 213 in sponsor_numbers:
            asia_icons = int(played_icon_counts.get("asia", 0))
            placed = 0
            for _ in range(asia_icons):
                if _place_free_building_of_type_if_possible(
                    state=state,
                    player=owner,
                    building_type=BuildingType.PAVILION,
                    player_id=owner_id,
                ):
                    placed += 1
            if placed > 0:
                state.effect_log.append(f"sponsor_213_free_pavilion={placed}")

        if owner_is_actor and 227 in sponsor_numbers and played_card.card_type == "animal":
            mode = (owner.sponsor_waza_assignment_mode or "").strip().lower()
            if mode == "small" and _is_small_animal(played_card):
                owner.appeal += 2
            if mode == "large" and _is_large_animal(played_card):
                owner.appeal += 4

        if owner_is_actor and 249 in sponsor_numbers:
            bird_icons = int(played_icon_counts.get("bird", 0))
            for _ in range(bird_icons):
                revealed = _draw_from_zoo_deck(state, 2)
                if revealed:
                    owner.hand.append(revealed[0])
                    state.zoo_discard.extend(revealed[1:])

        if owner_is_actor and 250 in sponsor_numbers:
            reptile_icons = int(played_icon_counts.get("reptile", 0))
            for _ in range(reptile_icons):
                sell_count = min(2, len(owner.hand))
                for _ in range(sell_count):
                    state.zoo_discard.append(owner.hand.pop(0))
                    owner.money += 4

        if owner_is_actor and 252 in sponsor_numbers:
            predator_icons = int(played_icon_counts.get("predator", 0))
            for _ in range(predator_icons):
                predator_count = _player_icon_inventory(owner).get("predator", 0)
                if predator_count <= 0:
                    continue
                revealed = _draw_from_zoo_deck(state, predator_count)
                kept: Optional[AnimalCard] = next(
                    (card for card in revealed if card.card_type == "animal"),
                    None,
                )
                for card in revealed:
                    if kept is not None and card.instance_id == kept.instance_id:
                        continue
                    state.zoo_discard.append(card)
                if kept is not None:
                    owner.hand.append(kept)

        if owner_is_actor and 253 in sponsor_numbers:
            herbivore_icons = int(played_icon_counts.get("herbivore", 0))
            for _ in range(herbivore_icons):
                if not _play_sponsor_from_hand_via_253(
                    state=state,
                    player=owner,
                    player_id=owner_id,
                    details=details,
                ):
                    break

        if owner_is_actor and 262 in sponsor_numbers:
            current_types = _distinct_continent_and_category_types_from_cards(owner.zoo_cards)
            previous_cards: List[AnimalCard] = []
            removed = False
            for card in owner.zoo_cards:
                if not removed and card.instance_id == played_card.instance_id:
                    removed = True
                    continue
                previous_cards.append(card)
            previous_types = _distinct_continent_and_category_types_from_cards(previous_cards)
            gained = len(current_types - previous_types)
            if gained > 0:
                owner.appeal += gained
                owner.money += gained * 2


def _sponsor_requirements_met(
    player: PlayerState,
    card: AnimalCard,
    sponsors_upgraded: bool,
) -> Tuple[bool, str]:
    if card.card_type != "sponsor":
        return False, "not_sponsor"
    if _sponsor_level(card) <= 0:
        return False, "invalid_level"
    if _sponsor_requires_upgraded(card) and not sponsors_upgraded:
        return False, "requires_sponsors_ii"
    min_rep = _sponsor_min_reputation(card)
    if min_rep > int(player.reputation):
        return False, f"min_reputation_{min_rep}"
    max_appeal = _sponsor_max_appeal(card)
    if max_appeal is not None and int(player.appeal) > max_appeal:
        return False, f"appeal_must_be_at_most_{max_appeal}"

    inventory = _player_icon_inventory(player)
    for key, need in _sponsor_required_icons(card).items():
        if key == "partner_zoo":
            if len(player.partner_zoos) < int(need):
                return False, f"partner_zoo_{need}"
            continue
        if inventory.get(key, 0) < int(need):
            return False, f"icon_{key}_{need}"
    return True, "ok"


def _list_legal_sponsor_candidates(
    state: GameState,
    player: PlayerState,
    sponsors_upgraded: bool,
) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    for hand_idx, card in enumerate(player.hand):
        if card.card_type != "sponsor":
            continue
        ok, reason = _sponsor_requirements_met(player=player, card=card, sponsors_upgraded=sponsors_upgraded)
        if ok and not _sponsor_unique_building_can_be_placed(state, player, card.number):
            ok = False
            reason = "no_legal_unique_building_placement"
        level = _sponsor_level(card)
        pay = _sponsor_play_cost(card)
        candidates.append(
            {
                "source": "hand",
                "source_index": hand_idx,
                "card_instance_id": card.instance_id,
                "card": card,
                "level": level,
                "extra_cost": 0,
                "pay_cost": pay,
                "playable_now": ok and player.money >= pay,
                "reason": reason,
            }
        )

    if sponsors_upgraded:
        accessible = min(_reputation_display_limit(int(player.reputation)), len(state.zoo_display))
        for display_idx in range(accessible):
            card = state.zoo_display[display_idx]
            if card.card_type != "sponsor":
                continue
            ok, reason = _sponsor_requirements_met(player=player, card=card, sponsors_upgraded=sponsors_upgraded)
            if ok and not _sponsor_unique_building_can_be_placed(state, player, card.number):
                ok = False
                reason = "no_legal_unique_building_placement"
            level = _sponsor_level(card)
            extra_cost = display_idx + 1
            pay = _sponsor_play_cost(card) + extra_cost
            candidates.append(
                {
                    "source": "display",
                    "source_index": display_idx,
                    "card_instance_id": card.instance_id,
                    "card": card,
                    "level": level,
                    "extra_cost": extra_cost,
                    "pay_cost": pay,
                    "playable_now": ok and player.money >= pay,
                    "reason": reason,
                }
            )
    return candidates


def _validate_sponsor_effect_details(
    state: GameState,
    player: PlayerState,
    player_id: int,
    card: AnimalCard,
    details: Dict[str, Any],
) -> None:
    number = int(card.number)
    if number in SPONSOR_UNIQUE_BUILDING_CARDS:
        legal = _list_legal_sponsor_unique_building_cells(
            state=state,
            player=player,
            sponsor_number=number,
        )
        if len(legal) > 1:
            selection = _pop_matching_detail_entry(
                details,
                "sponsor_unique_building_selections",
                card_instance_id=card.instance_id,
            )
            if selection is None:
                raise ValueError(
                    f"Sponsor #{number} requires explicit sponsor_unique_building_selections when multiple placements exist."
                )
            parsed = _parse_serialized_cells(selection.get("cells"))
            if parsed is None or parsed not in legal:
                raise ValueError(f"Illegal unique building placement for sponsor #{number}.")
            details.setdefault("_validated_sponsor_unique_building_selections", []).append(selection)
    if number == 227:
        selected_mode = str(details.get("sponsor_227_mode") or "").strip().lower()
        if selected_mode not in {"small", "large"}:
            raise ValueError("Sponsor #227 requires explicit sponsor_227_mode: 'small' or 'large'.")
    if number == 263:
        options = [
            option
            for option in list_legal_build_options(state=state, player_id=player_id, strength=5)
            if option["building_type"] == "SIZE_5"
        ]
        if len(options) > 1:
            build_details = _pop_matching_detail_entry(
                details,
                "sponsor_263_build_details",
                card_instance_id=card.instance_id,
            )
            if build_details is None:
                raise ValueError(
                    "Sponsor #263 requires explicit sponsor_263_build_details when multiple size-5 placements exist."
                )
            selections = list(build_details.get("selections") or [])
            if len(selections) != 1:
                raise ValueError("Sponsor #263 build details must contain exactly one size-5 selection.")
            selected = _find_serialized_build_option(options, selections[0])
            if selected is None:
                raise ValueError("Illegal build selection for sponsor #263.")
            details.setdefault("_validated_sponsor_263_build_details", []).append(build_details)
        elif len(options) == 1:
            build_details = _pop_matching_detail_entry(
                details,
                "sponsor_263_build_details",
                card_instance_id=card.instance_id,
            )
            if build_details is not None:
                selections = list(build_details.get("selections") or [])
                if len(selections) != 1:
                    raise ValueError("Sponsor #263 build details must contain exactly one size-5 selection.")
                selected = _find_serialized_build_option(options, selections[0])
                if selected is None:
                    raise ValueError("Illegal build selection for sponsor #263.")
                details.setdefault("_validated_sponsor_263_build_details", []).append(build_details)


def _apply_sponsor_immediate_effects(
    state: GameState,
    player: PlayerState,
    player_id: int,
    card: AnimalCard,
    details: Optional[Dict[str, Any]] = None,
    *,
    allow_interactive: bool = False,
) -> List[str]:
    details = details or {}
    number = int(card.number)
    messages: List[str] = []
    _ensure_player_map_initialized(state, player)
    inventory = _player_icon_inventory(player)

    if number in SPONSOR_UNIQUE_BUILDING_CARDS:
        placed = _place_sponsor_unique_building(
            state=state,
            player=player,
            card=card,
            player_id=player_id,
            details=details,
        )
        if not placed:
            raise ValueError("No legal placement for this sponsor unique building.")
        messages.append("immediate(unique_building_placed)=1")

    if number in {201, 254}:
        taken = _take_one_card_from_deck_or_reputation_range(state, player)
        messages.append(f"immediate(draw_1_from_deck_or_reputation_range)={1 if taken else 0}")
    if number == 254:
        _increase_reputation(
            state=state,
            player=player,
            amount=1,
            allow_interactive=allow_interactive,
        )
        player.conservation += 1
        messages.append("immediate(+1_reputation,+1_conservation)")

    if number == 203:
        universities = len(player.universities)
        gain = 0
        if universities >= 3:
            gain = 10
        elif universities == 2:
            gain = 5
        elif universities == 1:
            gain = 2
        player.money += gain
        messages.append(f"immediate(universities_money)={gain}")

    if number == 204:
        gain = 2 * int(inventory.get("science", 0))
        player.money += gain
        messages.append(f"immediate(science_x2_money)={gain}")

    if number == 205:
        _increase_reputation(
            state=state,
            player=player,
            amount=2,
            allow_interactive=allow_interactive,
        )
        player.conservation += 1
        messages.append("immediate(+2_reputation,+1_conservation)")

    if number == 206:
        gain = 2 * int(player.supported_conservation_project_actions)
        player.appeal += gain
        messages.append(f"immediate(appeal_from_supported_project_actions)={gain}")

    if number == 207:
        distinct_types = len(_distinct_continent_and_category_types_from_cards(player.zoo_cards))
        cp_gain = distinct_types // 2
        if cp_gain > 0:
            player.conservation += cp_gain
            for idx, other in enumerate(state.players):
                if idx == player_id:
                    continue
                other.money += cp_gain * 2
        messages.append(f"immediate(cp_from_distinct_types)={cp_gain}")

    if number == 208:
        gain = int(inventory.get("science", 0))
        player.appeal += gain
        messages.append(f"immediate(science_to_appeal)={gain}")

    if number == 209:
        player.x_tokens = min(5, player.x_tokens + 1)
        messages.append("immediate(+1_x_token)")

    if number in {210, 211, 212, 213, 214}:
        key_map = {
            210: "america",
            211: "europe",
            212: "australia",
            213: "asia",
            214: "africa",
        }
        key = key_map[number]
        gain = int(inventory.get(key, 0))
        player.appeal += gain
        messages.append(f"immediate({key}_icons_to_appeal)={gain}")

    if number == 216:
        _gain_workers(
            state=state,
            player=player,
            player_id=player_id,
            amount=1,
            source="sponsor_216_immediate",
        )
        messages.append("immediate(+1_worker)")

    if number == 219:
        gain = 2 * (int(inventory.get("water", 0)) + int(inventory.get("rock", 0)))
        player.money += gain
        messages.append(f"immediate(water_rock_to_money)={gain}")

    if number == 220:
        player.money += 3
        messages.append("immediate(+3_money)")

    if number == 222:
        cp_gain = min(3, int(inventory.get("science", 0)))
        if cp_gain > 0:
            player.conservation += cp_gain
            for idx, other in enumerate(state.players):
                if idx == player_id:
                    continue
                other.money += cp_gain * 2
        messages.append(f"immediate(cp_from_science_up_to_3)={cp_gain}")

    if number in {224, 225}:
        player.x_tokens = min(5, player.x_tokens + 1)
        messages.append("immediate(+1_x_token)")

    if number == 226:
        _increase_reputation(
            state=state,
            player=player,
            amount=2,
            allow_interactive=allow_interactive,
        )
        messages.append("immediate(+2_reputation)")

    if number == 227:
        selected_mode = str(details.get("sponsor_227_mode") or "").strip().lower()
        if selected_mode not in {"small", "large"}:
            raise ValueError("Sponsor #227 requires explicit sponsor_227_mode: 'small' or 'large'.")
        player.sponsor_waza_assignment_mode = selected_mode
        revealed: List[AnimalCard] = []
        keep: Optional[AnimalCard] = None
        while state.zoo_deck:
            top = state.zoo_deck.pop(0)
            revealed.append(top)
            if top.card_type == "animal":
                if selected_mode == "small" and _is_small_animal(top):
                    keep = top
                    break
                if selected_mode == "large" and _is_large_animal(top):
                    keep = top
                    break
        for candidate in revealed:
            if keep is not None and candidate.instance_id == keep.instance_id:
                continue
            state.zoo_deck.append(candidate)
        if keep is not None:
            player.hand.append(keep)
        messages.append(
            f"immediate(waza_special_assignment mode={selected_mode} revealed={len(revealed)} kept={1 if keep else 0})"
        )

    if number == 228:
        gain = 2 * _player_small_animals_count(player)
        player.money += gain
        messages.append(f"immediate(small_animals_to_money_x2)={gain}")

    if number == 229:
        gain = _player_small_animals_count(player)
        player.appeal += gain
        messages.append(f"immediate(small_animals_to_appeal)={gain}")

    if number == 230:
        gain = 2 * _player_large_animals_count(player)
        player.appeal += gain
        messages.append(f"immediate(large_animals_to_appeal_x2)={gain}")

    if number in {231, 232, 233, 234, 235}:
        icon_map = SPONSOR_ICON_INCOME_THRESHOLDS
        icon_name = icon_map.get(number, "")
        gain = int(inventory.get(icon_name, 0))
        player.appeal += gain
        messages.append(f"immediate({icon_name}_to_appeal)={gain}")

    if number == 241:
        gain = int(inventory.get("water", 0))
        player.appeal += gain
        messages.append(f"immediate(water_icons_to_appeal)={gain}")

    if number == 242:
        gain = (int(inventory.get("rock", 0)) // 2) * 3
        player.appeal += gain
        messages.append(f"immediate(rock_pairs_to_appeal_x3)={gain}")

    if number in {255, 256}:
        player.appeal += 4
        messages.append("immediate(+4_appeal)")

    if number == 261:
        player.conservation += 1
        player.appeal += 1
        messages.append("immediate(+1_conservation,+1_appeal)")

    if number == 262:
        distinct_types = len(_distinct_continent_and_category_types_from_cards(player.zoo_cards))
        gain = distinct_types * 2
        player.money += gain
        messages.append(f"immediate(distinct_types_x2_money)={gain}")

    if number == 258:
        gain = _count_terrain_spaces_adjacent_to_buildings(player, Terrain.WATER)
        player.appeal += gain
        messages.append(f"immediate(adjacent_water_spaces_to_appeal)={gain}")

    if number == 259:
        gain = _count_terrain_spaces_adjacent_to_buildings(player, Terrain.ROCK)
        player.appeal += gain
        messages.append(f"immediate(adjacent_rock_spaces_to_appeal)={gain}")

    if number == 260:
        gain = _count_empty_border_spaces_adjacent_to_buildings(player)
        player.appeal += gain
        messages.append(f"immediate(adjacent_empty_border_spaces_to_appeal)={gain}")

    if number == 264:
        gain = _count_placement_bonus_spaces_adjacent_to_buildings(state, player)
        player.appeal += gain
        messages.append(f"immediate(adjacent_placement_bonus_spaces_to_appeal)={gain}")

    if number == 263:
        options = [
            option
            for option in list_legal_build_options(state=state, player_id=player_id, strength=5)
            if option["building_type"] == "SIZE_5"
        ]
        if options:
            build_details = _pop_matching_detail_entry(
                details,
                "_validated_sponsor_263_build_details",
                card_instance_id=card.instance_id,
            )
            if build_details is None:
                if len(options) > 1:
                    raise ValueError(
                        "Sponsor #263 requires explicit sponsor_263_build_details when multiple size-5 placements exist."
                    )
                build_details = {
                    "selections": [
                        {
                            "building_type": "SIZE_5",
                            "cells": options[0]["cells"],
                        }
                    ],
                    "bonus_action_to_slot_1_targets": [],
                }
            before_money = player.money
            _perform_build_action_effect(
                state=state,
                player=player,
                strength=5,
                player_id=player_id,
                details=build_details,
            )
            spent = max(0, before_money - player.money)
            player.money += spent
            messages.append("immediate(free_size_5_enclosure_placed=1)")
        else:
            messages.append("immediate(free_size_5_enclosure_placed=0)")

    return messages


def _perform_sponsors_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    player_id: int,
    details: Optional[Dict[str, Any]] = None,
) -> bool:
    details = dict(details or {})
    sponsors_upgraded = bool(player.action_upgraded["sponsors"])
    if not details:
        raise ValueError(
            "Sponsors action requires explicit details: choose break ability or sponsor_selections."
        )

    use_break_ability = bool(details.get("use_break_ability", False))
    selections = list(details.get("sponsor_selections") or [])
    if use_break_ability and selections:
        raise ValueError("Sponsors break alternative cannot be combined with sponsor plays.")

    if use_break_ability:
        money_gain = strength * (2 if sponsors_upgraded else 1)
        player.money += money_gain
        return _advance_break_track(state=state, steps=strength, trigger_player=player_id)

    if not selections:
        raise ValueError("Sponsors action must either play sponsor card(s) or use break alternative.")

    if not sponsors_upgraded and len(selections) != 1:
        raise ValueError("Sponsors side I requires exactly one sponsor card.")

    resolved_cards: List[Tuple[str, AnimalCard, int, int]] = []
    # tuple(source, card, current_index, extra_display_cost)
    for raw in selections:
        source = str(raw.get("source") or "").strip().lower()
        card_instance_id = str(raw.get("card_instance_id") or "").strip()
        source_index = int(raw.get("source_index", -1))
        if source == "hand":
            card = None
            index = -1
            if card_instance_id:
                for idx, candidate in enumerate(player.hand):
                    if candidate.instance_id == card_instance_id:
                        card = candidate
                        index = idx
                        break
            if card is None and 0 <= source_index < len(player.hand):
                card = player.hand[source_index]
                index = source_index
            if card is None:
                raise ValueError("Selected sponsor card from hand is not available.")
            if card.card_type != "sponsor":
                raise ValueError("Selected card is not a sponsor card.")
            resolved_cards.append(("hand", card, index, 0))
            continue

        if source == "display":
            if not sponsors_upgraded:
                raise ValueError("Sponsors side I cannot play from display.")
            card = None
            index = -1
            if card_instance_id:
                for idx, candidate in enumerate(state.zoo_display):
                    if candidate.instance_id == card_instance_id:
                        card = candidate
                        index = idx
                        break
            if card is None and 0 <= source_index < len(state.zoo_display):
                card = state.zoo_display[source_index]
                index = source_index
            if card is None:
                raise ValueError("Selected sponsor card from display is not available.")
            if card.card_type != "sponsor":
                raise ValueError("Selected display card is not a sponsor card.")
            accessible = _reputation_display_limit(int(player.reputation))
            if index + 1 > accessible:
                raise ValueError("Selected sponsor card is outside reputation range.")
            resolved_cards.append(("display", card, index, index + 1))
            continue

        raise ValueError("Unsupported sponsor source.")

    total_level = sum(_sponsor_level(card) for _, card, _, _ in resolved_cards)
    level_cap = strength + 1 if sponsors_upgraded else strength
    if total_level > level_cap:
        raise ValueError(
            f"Sponsor levels total {total_level} exceeds allowed maximum {level_cap}."
        )

    for _, card, _, _ in resolved_cards:
        _validate_sponsor_card_for_play(
            state=state,
            player=player,
            player_id=player_id,
            card=card,
            sponsors_upgraded=sponsors_upgraded,
            details=details,
        )

    _validate_sponsor_sequence_affordability(state=state, player=player, resolved_cards=resolved_cards)

    took_from_display = False
    for source, card, _, _ in resolved_cards:
        took_from_display = (
            _play_sponsor_card_from_source(
                state=state,
                player=player,
                player_id=player_id,
                source=source,
                card=card,
                details=details,
            )
            or took_from_display
        )
    if took_from_display:
        _replenish_zoo_display(state)
    return False


def _normalize_continent_badge(badge: str) -> Optional[str]:
    key = badge.strip().lower()
    if not key:
        return None
    return CONTINENT_BADGE_ALIASES.get(key)


def _player_icon_snapshot(player: PlayerState) -> Dict[str, Any]:
    continent_keys = ("Africa", "Europe", "Asia", "America", "Australia")
    continents = {key: 0 for key in continent_keys}
    categories: Dict[str, int] = {}
    science = 0
    rock_icons = 0
    water_icons = 0

    for card in player.zoo_cards:
        if card.card_type == "animal":
            rock_icons += max(0, card.required_rock_adjacency)
            water_icons += max(0, card.required_water_adjacency)
        for badge in _card_badges_for_icons(card):
            normalized_badge = _canonical_icon_key(badge)
            normalized_continent = _normalize_continent_badge(normalized_badge)
            if normalized_continent is not None:
                continents[normalized_continent] += 1
                continue
            if normalized_badge == "science":
                science += 1
                continue
            if normalized_badge == "water":
                water_icons += 1
                continue
            if normalized_badge == "rock":
                rock_icons += 1
                continue
            if normalized_badge:
                display_name = normalized_badge.title()
                categories[display_name] = categories.get(display_name, 0) + 1

    for partner in player.partner_zoos:
        continent_name = _normalize_continent_badge(partner)
        if continent_name is not None:
            continents[continent_name] += 1

    for university in player.universities:
        science += UNIVERSITY_SCIENCE_GAIN.get(university, 0)

    return {
        "continents": continents,
        "categories": dict(sorted(categories.items())),
        "science": science,
        "rock_icons": rock_icons,
        "water_icons": water_icons,
    }


def _animal_release_size_bucket_label(card: AnimalCard) -> str:
    size = int(card.size)
    if size >= 4:
        return "4+"
    if size == 3:
        return "3"
    return "2-"


def _animal_badge_summary(card: AnimalCard) -> Tuple[List[str], List[str]]:
    continents: List[str] = []
    categories: List[str] = []
    for badge in _card_badges_for_icons(card):
        normalized_badge = _canonical_icon_key(badge)
        continent_name = _normalize_continent_badge(normalized_badge)
        if continent_name is not None:
            if continent_name not in continents:
                continents.append(continent_name)
            continue
        if normalized_badge in {"science", "water", "rock", ""}:
            continue
        display_name = normalized_badge.title()
        if display_name not in categories:
            categories.append(display_name)
    return continents, categories


def _format_zoo_animal_line(card: AnimalCard) -> str:
    card_no = f"#{card.number}" if card.number >= 0 else "#?"
    continents, categories = _animal_badge_summary(card)
    continents_text = ", ".join(continents) or "-"
    categories_text = ", ".join(categories) or "-"
    return (
        f"{card_no} {card.name} size={card.size} release={_animal_release_size_bucket_label(card)} "
        f"continents[{continents_text}] categories[{categories_text}]"
    )


def _pouched_card_host_name(player: PlayerState, card: AnimalCard) -> str:
    for host_instance_id, cards in player.pouched_cards_by_host.items():
        if not any(item.instance_id == card.instance_id for item in cards):
            continue
        host_card = next((item for item in player.zoo_cards if item.instance_id == host_instance_id), None)
        if host_card is not None:
            return host_card.name
        if str(host_instance_id).strip():
            return str(host_instance_id)
    return "-"


def _format_card_line(card: AnimalCard) -> str:
    card_no = f"#{card.number}" if card.number >= 0 else "#?"
    if card.card_type == "sponsor":
        req_dict = _sponsor_required_icons(card)
        req_items = [f"{icon}:{need}" for icon, need in sorted(req_dict.items())]
        min_rep = _sponsor_min_reputation(card)
        if min_rep > 0:
            req_items.append(f"reputation>={min_rep}")
        max_appeal = _sponsor_max_appeal(card)
        if max_appeal is not None:
            req_items.append(f"appeal<={max_appeal}")
        if _sponsor_requires_upgraded(card):
            req_items.append("sponsorsii")
        req_icons = ", ".join(req_items)
    else:
        req_icons = ", ".join(f"{icon}:{need}" for icon, need in card.required_icons)
    req_label = req_icons if req_icons else "-"
    effect = resolve_card_effect(card)
    effect_label = effect.code if effect.code != "none" else "-"
    if card.card_type == "sponsor":
        return (
            f"{card_no} [{card.card_type}] {card.name} "
            f"(level={_sponsor_level(card)}, appeal={card.appeal}, rep={card.reputation_gain}, "
            f"cons={card.conservation}, req={req_label}, effect={effect_label}, id={card.instance_id})"
        )
    return (
        f"{card_no} [{card.card_type}] {card.name} "
        f"(cost={card.cost}, size={card.size}, appeal={card.appeal}, rep={card.reputation_gain}, "
        f"cons={card.conservation}, req={req_label}, effect={effect_label}, id={card.instance_id})"
    )


def _format_card_line_for_player(card: AnimalCard, player: PlayerState) -> str:
    base = _format_card_line(card)
    if card.card_type != "animal":
        return base
    _, partner_discount, sponsor_discount, total_discount, effective_cost = _animal_discount_breakdown(player, card)
    if total_discount <= 0:
        return base
    return (
        base[:-1]
        + f", discount={total_discount}, partner_discount={partner_discount}, "
        + f"sponsor_discount={sponsor_discount}, effective_cost={effective_cost})"
    )


def _prompt_opening_draft_indices(player_name: str, drafted_cards: List[AnimalCard]) -> List[int]:
    return _prompt_opening_draft_indices_impl(
        player_name=player_name,
        drafted_cards=drafted_cards,
        format_card_line=_format_card_line,
    )


def _format_animals_play_step_for_human(step: Dict[str, Any], player: PlayerState) -> str:
    return _format_animals_play_step_for_human_impl(
        step=step,
        player=player,
        find_building_fn=_find_zoo_building_by_origin_rotation_size,
        building_cells_text_fn=_building_cells_text,
    )


def _prompt_animals_action_details_for_human(
    state: GameState,
    player: PlayerState,
    strength: int,
    player_id: int,
) -> Dict[str, Any]:
    return _prompt_animals_action_details_for_human_impl(
        state=state,
        player=player,
        strength=strength,
        player_id=player_id,
        animals_play_limit_fn=_animals_play_limit,
        list_legal_animals_options_fn=list_legal_animals_options,
        format_animals_play_step_fn=_format_animals_play_step_for_human,
    )


def _prompt_cards_action_details_for_human(
    state: GameState,
    player: PlayerState,
    strength: int,
) -> Dict[str, Any]:
    return _prompt_cards_action_details_for_human_impl(
        state=state,
        player=player,
        strength=strength,
        cards_table_values_fn=_cards_table_values,
        reputation_display_limit_fn=_reputation_display_limit,
        format_card_line_fn=lambda card: _format_card_line_for_player(card, player),
    )


def _prompt_build_action_details_for_human(
    state: GameState,
    player: PlayerState,
    strength: int,
    player_id: int,
) -> Dict[str, Any]:
    return _prompt_build_action_details_for_human_impl(
        state=state,
        player=player,
        strength=strength,
        player_id=player_id,
        list_legal_build_options_fn=list_legal_build_options,
        building_type_enum=BuildingType,
        reputation_display_limit_fn=_reputation_display_limit,
    )


def _prompt_association_action_details_for_human(
    state: GameState,
    player: PlayerState,
    strength: int,
    player_id: int,
) -> Dict[str, Any]:
    return _prompt_association_action_details_for_human_impl(
        state=state,
        player=player,
        strength=strength,
        player_id=player_id,
        list_legal_association_options_fn=list_legal_association_options,
        partner_zoo_label_fn=_partner_zoo_label,
        university_label_fn=_university_label,
        current_donation_cost_fn=_current_donation_cost,
    )


def _prompt_action_to_slot_1_target_for_human(player: PlayerState, *, reason: str) -> str:
    print(reason)
    for idx, action_name in enumerate(player.action_order, start=1):
        print(f"{idx}. {action_name}")
    while True:
        raw_pick = input(f"Select action card [1-{len(player.action_order)}]: ").strip()
        if raw_pick.isdigit():
            picked = int(raw_pick)
            if 1 <= picked <= len(player.action_order):
                return player.action_order[picked - 1]
        print("Please enter a valid number.")


def _prompt_choose_hand_cards_for_human(
    player: PlayerState,
    *,
    max_choose: int,
    min_choose: int = 0,
    effect_name: str,
    action_label: str,
    reward_label: str,
) -> List[int]:
    if max_choose <= 0 or not player.hand:
        return []
    lower = max(0, min(int(min_choose), max_choose))
    if lower == max_choose:
        choose_label = f"{lower}"
    else:
        choose_label = f"{lower}-{max_choose}"
    return _prompt_card_combination_indices_impl(
        title=f"{effect_name}: choose {choose_label} hand card(s) {reward_label}.",
        cards=player.hand,
        format_card_line=lambda card: _format_card_line_for_player(card, player),
        min_choose=lower,
        max_choose=max_choose,
        action_label=action_label,
        combination_header=f"{action_label.title()} combinations:",
    )


def _prompt_sell_hand_cards_for_human(
    player: PlayerState,
    *,
    max_sell: int,
    money_each: int,
    effect_name: str,
) -> List[int]:
    return _prompt_choose_hand_cards_for_human(
        player,
        max_choose=max_sell,
        effect_name=effect_name,
        action_label="sell",
        reward_label=f"to sell for {money_each} money each",
    )


def _prompt_pouch_hand_cards_for_human(
    player: PlayerState,
    *,
    max_pouch: int,
    appeal_each: int,
    effect_name: str,
) -> List[int]:
    return _prompt_choose_hand_cards_for_human(
        player,
        max_choose=max_pouch,
        effect_name=effect_name,
        action_label="pouch",
        reward_label=f"to place under this card for {appeal_each} appeal each",
    )


def _resolve_selected_sponsor_cards_from_details(
    state: GameState,
    player: PlayerState,
    details: Dict[str, Any],
) -> List[AnimalCard]:
    resolved: List[AnimalCard] = []
    for raw in list(details.get("sponsor_selections") or []):
        source = str(raw.get("source") or "").strip().lower()
        card_instance_id = str(raw.get("card_instance_id") or "").strip()
        source_index = int(raw.get("source_index", -1))
        zone: List[AnimalCard]
        if source == "hand":
            zone = player.hand
        elif source == "display":
            zone = state.zoo_display
        else:
            continue
        chosen: Optional[AnimalCard] = None
        if card_instance_id:
            chosen = next((card for card in zone if card.instance_id == card_instance_id), None)
        if chosen is None and 0 <= source_index < len(zone):
            chosen = zone[source_index]
        if chosen is not None:
            resolved.append(chosen)
    return resolved


def _enrich_sponsor_action_details_for_human(
    state: GameState,
    player: PlayerState,
    details: Dict[str, Any],
) -> Dict[str, Any]:
    if bool(details.get("use_break_ability")):
        return details

    player_id = state.players.index(player)
    selected_cards = _resolve_selected_sponsor_cards_from_details(state, player, details)
    unique_building_selections: List[Dict[str, Any]] = list(details.get("sponsor_unique_building_selections") or [])
    sponsor_263_build_details: List[Dict[str, Any]] = list(details.get("sponsor_263_build_details") or [])

    for card in selected_cards:
        if card.number in SPONSOR_UNIQUE_BUILDING_CARDS:
            legal = _list_legal_sponsor_unique_building_cells(
                state=state,
                player=player,
                sponsor_number=card.number,
            )
            if len(legal) > 1:
                print(f"{card.name} requires choosing a unique building placement.")
                for idx, cells in enumerate(legal, start=1):
                    cells_text = ",".join(f"({x},{y})" for x, y in cells)
                    print(f"{idx}. cells=[{cells_text}]")
                while True:
                    raw_pick = input(f"Select placement [1-{len(legal)}]: ").strip()
                    if raw_pick.isdigit():
                        picked = int(raw_pick)
                        if 1 <= picked <= len(legal):
                            unique_building_selections.append(
                                {
                                    "card_instance_id": card.instance_id,
                                    "cells": [list(cell) for cell in legal[picked - 1]],
                                }
                            )
                            break
                    print("Please enter a valid number.")

        if card.number == 263:
            options = [
                option
                for option in list_legal_build_options(state=state, player_id=player_id, strength=5)
                if option["building_type"] == "SIZE_5"
            ]
            if options:
                selected_option = options[0]
                if len(options) > 1:
                    print(f"{card.name} lets you place one free size-5 enclosure.")
                    for option in options:
                        bonuses = ",".join(option["placement_bonuses"]) if option["placement_bonuses"] else "-"
                        cells_text = ",".join(f"({x},{y})" for x, y in option["cells"])
                        print(
                            f"{option['index']}. cells=[{cells_text}] bonuses={bonuses}"
                        )
                    while True:
                        raw_pick = input(f"Select size-5 enclosure [1-{len(options)}]: ").strip()
                        if raw_pick.isdigit():
                            picked = int(raw_pick)
                            if 1 <= picked <= len(options):
                                selected_option = options[picked - 1]
                                break
                        print("Please enter a valid number.")
                bonus_targets: List[str] = []
                if "action_to_slot_1" in selected_option["placement_bonuses"]:
                    bonus_targets.append(
                        _prompt_action_to_slot_1_target_for_human(
                            player,
                            reason="Free size-5 enclosure grants action_to_slot_1.",
                        )
                    )
                sponsor_263_build_details.append(
                    {
                        "card_instance_id": card.instance_id,
                        "selections": [
                            {
                                "building_type": "SIZE_5",
                                "cells": selected_option["cells"],
                            }
                        ],
                        "bonus_action_to_slot_1_targets": bonus_targets,
                    }
                )

    if unique_building_selections:
        details["sponsor_unique_building_selections"] = unique_building_selections
    if sponsor_263_build_details:
        details["sponsor_263_build_details"] = sponsor_263_build_details
    return details


def _prompt_sponsors_action_details_for_human(
    state: GameState,
    player: PlayerState,
    strength: int,
) -> Dict[str, Any]:
    details = _prompt_sponsors_action_details_for_human_impl(
        state=state,
        player=player,
        strength=strength,
        sponsor_candidates_fn=_list_legal_sponsor_candidates,
        format_card_line_fn=_format_card_line,
    )
    return _enrich_sponsor_action_details_for_human(state, player, details)


def _resolve_manual_opening_drafts(state: GameState, manual_player_names: Set[str]) -> None:
    if not str(state.pending_decision_kind or "").strip():
        _begin_next_opening_draft_pending_if_needed(state)
    while str(state.pending_decision_kind or "").strip() == "opening_draft_keep":
        player_id = int(state.pending_decision_player_id)
        player = state.players[player_id]
        if player.name not in manual_player_names:
            break
        actions = legal_actions(player, state=state, player_id=player_id)
        action = HumanPlayer().choose_action(state, actions)
        apply_action(state, action)


def _perform_main_action_dispatch(
    state: GameState,
    player: PlayerState,
    player_id: int,
    chosen: str,
    strength: int,
    details: Optional[Dict[str, Any]],
) -> bool:
    if chosen == "cards":
        return _perform_cards_action_effect(
            state=state,
            player=player,
            strength=strength,
            player_id=player_id,
            details=details,
        )
    if chosen == "build":
        _perform_build_action_effect(
            state=state,
            player=player,
            strength=strength,
            player_id=player_id,
            details=details,
        )
        return False
    if chosen == "animals":
        _perform_animals_action_effect(
            state=state,
            player=player,
            strength=strength,
            details=details,
            player_id=player_id,
        )
        return False
    if chosen == "association":
        _perform_association_action_effect(
            state=state,
            player=player,
            strength=strength,
            details=details,
            player_id=player_id,
        )
        return False
    if chosen == "sponsors":
        return _perform_sponsors_action_effect(
            state=state,
            player=player,
            strength=strength,
            player_id=player_id,
            details=details,
        )
    raise ValueError(f"Unsupported main action '{chosen}'.")


def _prevalidate_main_action_details(
    state: GameState,
    player: PlayerState,
    player_id: int,
    chosen: str,
    strength: int,
    details: Optional[Dict[str, Any]],
) -> None:
    details = details or {}
    if chosen == "sponsors":
        if not details:
            raise ValueError(
                "Sponsors action requires explicit details: choose break ability or sponsor_selections."
            )
        if not bool(details.get("use_break_ability")):
            validation_details = copy.deepcopy(details)
            for card in _resolve_selected_sponsor_cards_from_details(state, player, details):
                _validate_sponsor_effect_details(
                    state=state,
                    player=player,
                    player_id=player_id,
                    card=card,
                    details=validation_details,
                )
    if chosen == "animals":
        options = list_legal_animals_options(state=state, player_id=player_id, strength=strength)
        if len(options) > 1 and details.get("animals_sequence_index") is None:
            raise ValueError(
                "animals_sequence_index is required when multiple legal animal sequences exist."
            )
    if chosen == "build":
        options = [
            option
            for option in list_legal_build_options(state=state, player_id=player_id, strength=strength)
            if int(option.get("cost", 0)) <= int(player.money)
        ]
        requested = list(details.get("selections") or [])
        if len(options) > 1 and not requested:
            raise ValueError(
                "Build action requires explicit selections when multiple affordable legal placements exist."
            )
    if chosen == "association":
        options = list_legal_association_options(state=state, player_id=player_id, strength=strength)
        has_explicit_selection = (
            bool(details.get("association_task_sequence"))
            or
            details.get("association_option_index") is not None
            or bool(str(details.get("task_kind") or "").strip())
        )
        if len(options) > 1 and not has_explicit_selection:
            raise ValueError(
                "Explicit association task details are required when multiple legal options exist."
            )


def _finalize_turn(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    break_triggered: bool,
    consumed_venom: bool,
) -> None:
    if bool(state.forced_game_over):
        _clear_pending_decision(state)
        _validate_card_zones(state)
        return

    if _maybe_begin_conservation_reward_pending(
        state,
        player_id=player_id,
        resume_kind="turn_finalize",
        break_triggered=break_triggered,
        consumed_venom=consumed_venom,
    ):
        _validate_card_zones(state)
        return

    if break_triggered:
        _resolve_break(
            state,
            use_pending=True,
            resume_turn_player_id=player_id,
            resume_turn_consumed_venom=consumed_venom,
        )
        if str(state.pending_decision_kind or "").strip():
            _validate_card_zones(state)
            return

    _complete_turn_after_break_resolution(
        state,
        player,
        player_id=player_id,
        consumed_venom=consumed_venom,
    )


def _complete_turn_after_break_resolution(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    consumed_venom: bool,
) -> None:
    _apply_end_of_turn_venom_penalty(player, consumed_venom=consumed_venom)
    _maybe_trigger_endgame(state, player_id)
    _validate_card_zones(state)
    state.turn_index = int(state.turn_index) + 1
    state.current_player = (int(state.current_player) + 1) % len(state.players)
    _maybe_trigger_round_limit_endgame(state)


def _resolve_cards_discard_pending_action(
    state: GameState,
    player: PlayerState,
    action: Action,
) -> Tuple[bool, bool]:
    payload = dict(state.pending_decision_payload or {})
    discard_target = int(payload.get("discard_target", 0))
    details = dict(action.details or {})
    discard_card_instance_ids = [str(item) for item in list(details.get("discard_card_instance_ids") or [])]
    if discard_target <= 0:
        raise ValueError("cards_discard pending decision requires a positive discard target.")
    if len(discard_card_instance_ids) != discard_target:
        raise ValueError(f"Exactly {discard_target} discard card_instance_id(s) are required.")
    if len(set(discard_card_instance_ids)) != len(discard_card_instance_ids):
        raise ValueError("Discard card_instance_id values must be unique.")

    resolved_indices: List[int] = []
    for card_instance_id in discard_card_instance_ids:
        matching_index = next(
            (idx for idx, card in enumerate(player.hand) if card.instance_id == card_instance_id),
            None,
        )
        if matching_index is None:
            raise ValueError("Discard card_instance_id is not in hand.")
        resolved_indices.append(matching_index)

    for idx in sorted(resolved_indices, reverse=True):
        state.zoo_discard.append(player.hand.pop(idx))

    break_triggered = bool(payload.get("break_triggered"))
    consumed_venom = bool(payload.get("consumed_venom"))
    _clear_pending_decision(state)
    return break_triggered, consumed_venom


def _resolve_break_discard_pending_action(
    state: GameState,
    player: PlayerState,
    action: Action,
) -> None:
    payload = dict(state.pending_decision_payload or {})
    discard_target = int(payload.get("discard_target", 0))
    details = dict(action.details or {})
    discard_card_instance_ids = [str(item) for item in list(details.get("discard_card_instance_ids") or [])]
    if discard_target <= 0:
        raise ValueError("break_discard pending decision requires a positive discard target.")
    if len(discard_card_instance_ids) != discard_target:
        raise ValueError(f"Exactly {discard_target} discard card_instance_id(s) are required.")
    if len(set(discard_card_instance_ids)) != len(discard_card_instance_ids):
        raise ValueError("Discard card_instance_id values must be unique.")

    resolved_indices: List[int] = []
    for card_instance_id in discard_card_instance_ids:
        matching_index = next(
            (idx for idx, card in enumerate(player.hand) if card.instance_id == card_instance_id),
            None,
        )
        if matching_index is None:
            raise ValueError("Discard card_instance_id is not in hand.")
        resolved_indices.append(matching_index)

    for idx in sorted(resolved_indices, reverse=True):
        state.zoo_discard.append(player.hand.pop(idx))

    break_hand_limit_index = int(payload.get("break_hand_limit_index", 0))
    resume_turn_player_id_raw = payload.get("resume_turn_player_id")
    resume_turn_player_id = int(resume_turn_player_id_raw) if resume_turn_player_id_raw is not None else None
    resume_turn_consumed_venom = bool(payload.get("resume_turn_consumed_venom"))

    _clear_pending_decision(state)
    if not _resolve_break_hand_limit_stage(
        state,
        start_index=break_hand_limit_index + 1,
        resume_turn_player_id=resume_turn_player_id,
        resume_turn_consumed_venom=resume_turn_consumed_venom,
    ):
        _validate_card_zones(state)
        return

    if not _resolve_break_remaining_stages(
        state,
        resume_turn_player_id=resume_turn_player_id,
        resume_turn_consumed_venom=resume_turn_consumed_venom,
    ):
        _validate_card_zones(state)
        return
    if resume_turn_player_id is not None:
        resume_player = state.players[resume_turn_player_id]
        _complete_turn_after_break_resolution(
            state,
            resume_player,
            player_id=resume_turn_player_id,
            consumed_venom=resume_turn_consumed_venom,
        )
    else:
        _validate_card_zones(state)


def _resolve_opening_draft_keep_pending_action(
    state: GameState,
    player: PlayerState,
    action: Action,
) -> None:
    payload = dict(state.pending_decision_payload or {})
    keep_target = int(payload.get("keep_target", 0))
    details = dict(action.details or {})
    keep_card_instance_ids = [str(item) for item in list(details.get("keep_card_instance_ids") or [])]
    draft = list(player.opening_draft_drawn)
    if keep_target <= 0:
        raise ValueError("opening_draft_keep pending decision requires a positive keep target.")
    if len(keep_card_instance_ids) != keep_target:
        raise ValueError(f"Exactly {keep_target} keep card_instance_id(s) are required.")
    if len(set(keep_card_instance_ids)) != len(keep_card_instance_ids):
        raise ValueError("Keep card_instance_id values must be unique.")

    kept_indices: List[int] = []
    for card_instance_id in keep_card_instance_ids:
        matching_index = next(
            (idx for idx, card in enumerate(draft) if card.instance_id == card_instance_id),
            None,
        )
        if matching_index is None:
            raise ValueError("Keep card_instance_id is not in opening draft.")
        kept_indices.append(matching_index)

    _apply_opening_draft_selection(
        player=player,
        drafted_cards=draft,
        rng=random.Random(0),
        kept_indices=sorted(kept_indices),
        discard_sink=state.zoo_discard,
    )
    _clear_pending_decision(state)
    _begin_next_opening_draft_pending_if_needed(state)
    _validate_card_zones(state)


def _apply_conservation_reward_choice(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    threshold: int,
    action: Action,
) -> None:
    details = dict(action.details or {})
    reward = str(details.get("reward") or "").strip()
    if not reward:
        raise ValueError("Conservation reward choice is missing reward.")

    available_choices = state.available_conservation_reward_choices(player_id=player_id, threshold=threshold)
    if reward not in available_choices:
        raise ValueError(f"Reward '{reward}' is not currently available for conservation space {threshold}.")

    if reward == "upgrade_action_card":
        action_name = str(details.get("upgraded_action") or "").strip()
        if action_name not in MAIN_ACTION_CARDS:
            raise ValueError("upgrade_action_card reward requires upgraded_action.")
        if bool(player.action_upgraded.get(action_name, False)):
            raise ValueError("Selected action card is already upgraded.")
        player.action_upgraded[action_name] = True
    elif reward == "activate_association_worker":
        gained = _gain_workers(
            state=state,
            player=player,
            player_id=player_id,
            amount=1,
            source=f"conservation_reward_{threshold}",
            effect_details=details,
        )
        if gained <= 0:
            raise ValueError("No inactive association worker is available.")
    elif reward == CONSERVATION_FIXED_MONEY_OPTION:
        player.money += 5
    else:
        if threshold not in CONSERVATION_SHARED_BONUS_THRESHOLDS:
            raise ValueError(f"Reward '{reward}' is not valid for conservation space {threshold}.")
        shared_tiles = state.shared_conservation_bonus_tiles.get(threshold, [])
        if reward not in shared_tiles:
            raise ValueError(f"Reward '{reward}' is no longer available on conservation space {threshold}.")

        if reward == "10_money":
            player.money += 10
        elif reward == "size_3_enclosure":
            placed = _place_free_building_of_type_if_possible(
                state=state,
                player=player,
                building_type=BuildingType.SIZE_3,
                player_id=player_id,
                details=details,
            )
            if not placed:
                raise ValueError("No legal free size-3 enclosure placement is available.")
        elif reward == "2_reputation":
            _apply_reputation_gain_with_details(
                state=state,
                player=player,
                player_id=player_id,
                amount=2,
                details=details,
            )
        elif reward == "3_x_tokens":
            player.x_tokens = min(5, int(player.x_tokens) + 3)
        elif reward == "3_cards":
            player.hand.extend(_draw_from_zoo_deck(state, 3))
        elif reward == "partner_zoo":
            _gain_partner_zoo_reward(
                state=state,
                player=player,
                player_id=player_id,
                partner=str(details.get("partner_zoo") or ""),
                effect_details=details,
                allow_interactive=False,
            )
        elif reward == "university":
            _gain_university_reward(
                state=state,
                player=player,
                player_id=player_id,
                university=str(details.get("university") or ""),
                effect_details=details,
                allow_interactive=False,
            )
        elif reward == "x2_multiplier":
            action_name = str(details.get("multiplier_action") or "").strip()
            if action_name not in MAIN_ACTION_CARDS:
                raise ValueError("x2_multiplier reward requires multiplier_action.")
            player.multiplier_tokens_on_actions[action_name] += 1
        elif reward == "sponsor_card":
            sponsor_details = dict(details.get("sponsor_details") or {})
            selected_cards = _resolve_selected_sponsor_cards_from_details(state, player, sponsor_details)
            if len(selected_cards) != 1:
                raise ValueError("sponsor_card reward requires exactly one selected sponsor from hand.")
            card = selected_cards[0]
            sponsors_upgraded = bool(player.action_upgraded["sponsors"])
            _validate_sponsor_card_for_play(
                state=state,
                player=player,
                player_id=player_id,
                card=card,
                sponsors_upgraded=sponsors_upgraded,
                details=sponsor_details,
            )
            _play_sponsor_card_from_source(
                state=state,
                player=player,
                player_id=player_id,
                source="hand",
                card=card,
                details=sponsor_details,
                pay_cost=0,
            )
        else:
            raise ValueError(f"Unsupported conservation bonus reward '{reward}'.")

        shared_tiles.remove(reward)
        state.claimed_conservation_bonus_tiles[threshold].append(reward)

    player.claimed_conservation_reward_spaces.add(int(threshold))
    state.effect_log.append(f"conservation_reward_{threshold}:{player.name}:{reward}")


def _resolve_conservation_reward_pending_action(
    state: GameState,
    player: PlayerState,
    *,
    player_id: int,
    action: Action,
) -> None:
    payload = dict(state.pending_decision_payload or {})
    threshold = int(payload.get("threshold", 0))
    if threshold <= 0:
        raise ValueError("conservation_reward pending decision requires a threshold.")

    details = dict(action.details or {})
    action_threshold = int(details.get("reward_threshold", threshold))
    if action_threshold != threshold:
        raise ValueError(f"Expected conservation reward threshold {threshold}.")

    resume_kind = str(payload.get("resume_kind") or "").strip()
    break_triggered = bool(payload.get("break_triggered"))
    consumed_venom = bool(payload.get("consumed_venom"))
    break_income_index = int(payload.get("break_income_index", 0))
    resume_turn_player_id_raw = payload.get("resume_turn_player_id")
    resume_turn_player_id = int(resume_turn_player_id_raw) if resume_turn_player_id_raw is not None else None
    resume_turn_consumed_venom = bool(payload.get("resume_turn_consumed_venom"))
    chained_choices_raw = details.get("chained_conservation_reward_choices")
    chained_choices: List[Dict[str, Any]] = []
    if chained_choices_raw is not None:
        if not isinstance(chained_choices_raw, list):
            raise ValueError("chained_conservation_reward_choices must be a list.")
        for item in chained_choices_raw:
            if not isinstance(item, dict):
                raise ValueError("chained_conservation_reward_choices entries must be objects.")
            chained_choices.append(copy.deepcopy(item))

    _apply_conservation_reward_choice(
        state=state,
        player=player,
        player_id=player_id,
        threshold=threshold,
        action=action,
    )

    _clear_pending_decision(state)
    while _maybe_begin_conservation_reward_pending(
        state,
        player_id=player_id,
        resume_kind=resume_kind,
        break_triggered=break_triggered,
        consumed_venom=consumed_venom,
        break_income_index=break_income_index,
        resume_turn_player_id=resume_turn_player_id,
        resume_turn_consumed_venom=resume_turn_consumed_venom,
    ):
        if not chained_choices:
            _validate_card_zones(state)
            return
        next_threshold = int(state.pending_decision_payload.get("threshold", 0))
        if next_threshold <= 0:
            raise ValueError("Conservation reward chain is missing next threshold.")
        next_choice = dict(chained_choices.pop(0))
        choice_threshold = int(next_choice.get("reward_threshold", next_threshold))
        if choice_threshold != next_threshold:
            raise ValueError(
                f"Conservation reward chain expected threshold {next_threshold}, got {choice_threshold}."
            )
        _apply_conservation_reward_choice(
            state=state,
            player=player,
            player_id=player_id,
            threshold=next_threshold,
            action=Action(
                ActionType.PENDING_DECISION,
                details={
                    "pending_kind": "conservation_reward",
                    "reward_threshold": next_threshold,
                    **next_choice,
                },
            ),
        )
        _clear_pending_decision(state)

    if chained_choices:
        raise ValueError("Unused chained_conservation_reward_choices entries remain after resolution.")

    if resume_kind == "turn_finalize":
        _finalize_turn(
            state,
            player,
            player_id=player_id,
            break_triggered=break_triggered,
            consumed_venom=consumed_venom,
        )
        return

    if resume_kind == "break_remaining":
        if not _resolve_break_remaining_stages(
            state,
            start_income_index=break_income_index,
            preprocessed=True,
            resume_turn_player_id=resume_turn_player_id,
            resume_turn_consumed_venom=resume_turn_consumed_venom,
        ):
            _validate_card_zones(state)
            return
        if resume_turn_player_id is not None:
            resume_player = state.players[resume_turn_player_id]
            _complete_turn_after_break_resolution(
                state,
                resume_player,
                player_id=resume_turn_player_id,
                consumed_venom=resume_turn_consumed_venom,
            )
        else:
            _validate_card_zones(state)
        return

    _validate_card_zones(state)


def apply_action(state: GameState, action: Action) -> None:
    _validate_card_zones(state)
    pending_kind = str(state.pending_decision_kind or "").strip()
    if pending_kind:
        player_id = int(state.pending_decision_player_id)
        player = state.players[player_id]
        if state.pending_decision_player_id != player_id:
            raise ValueError("Pending decision belongs to a different player.")
        if action.type != ActionType.PENDING_DECISION:
            raise ValueError("A pending decision must be resolved before taking another action.")
        action_pending_kind = str((action.details or {}).get("pending_kind") or "").strip()
        if action_pending_kind != pending_kind:
            raise ValueError(f"Expected pending decision kind '{pending_kind}'.")
        if pending_kind == "cards_discard":
            break_triggered, consumed_venom = _resolve_cards_discard_pending_action(state, player, action)
            _finalize_turn(
                state,
                player,
                player_id=player_id,
                break_triggered=break_triggered,
                consumed_venom=consumed_venom,
            )
            return
        if pending_kind == "break_discard":
            _resolve_break_discard_pending_action(state, player, action)
            return
        if pending_kind == "opening_draft_keep":
            _resolve_opening_draft_keep_pending_action(state, player, action)
            return
        if pending_kind == "conservation_reward":
            _resolve_conservation_reward_pending_action(
                state,
                player,
                player_id=player_id,
                action=action,
            )
            return
        raise ValueError(f"Unsupported pending decision kind: {pending_kind}")

    player_id = int(state.current_player)
    player = state.players[player_id]
    break_triggered = False
    consumed_venom = False

    if action.type == ActionType.MAIN_ACTION:
        if not action.card_name:
            raise ValueError("Main action requires card_name.")
        chosen = str(action.card_name)
        x_spent = int(action.value or 0)
        if x_spent < 0:
            raise ValueError("x_spent cannot be negative.")
        if x_spent > int(player.x_tokens):
            raise ValueError("Not enough X-tokens for selected action.")
        base_strength = player.action_order.index(chosen) + 1
        strength = _effective_action_strength(
            player,
            chosen,
            x_spent=x_spent,
            base_strength=base_strength,
        )
        _prevalidate_main_action_details(
            state=state,
            player=player,
            player_id=player_id,
            chosen=chosen,
            strength=strength,
            details=action.details,
        )
        base_strength = _rotate_action_card_to_slot_1(player, chosen)
        player.x_tokens -= x_spent
        strength = _effective_action_strength(
            player,
            chosen,
            x_spent=x_spent,
            base_strength=base_strength,
        )
        break_triggered = bool(
            _perform_main_action_dispatch(state, player, player_id, chosen, strength, action.details)
        )
        consumed_venom, _ = _apply_action_card_token_use(player, chosen)
        if str(state.pending_decision_kind or "").strip():
            state.pending_decision_payload["break_triggered"] = break_triggered
            state.pending_decision_payload["consumed_venom"] = consumed_venom
            _validate_card_zones(state)
            return
    elif action.type == ActionType.X_TOKEN:
        if int(player.x_tokens) >= 5:
            raise ValueError("Cannot take X action at max X-token limit.")
        chosen = action.card_name or player.action_order[-1]
        _rotate_action_card_to_slot_1(player, chosen)
        player.x_tokens = min(5, int(player.x_tokens) + 1)
    else:
        raise ValueError("Unsupported action type in this runner.")

    _finalize_turn(
        state,
        player,
        player_id=player_id,
        break_triggered=break_triggered,
        consumed_venom=consumed_venom,
    )


class PlayerAgent:
    def choose_action(self, state: GameState, actions: List[Action]) -> Action:
        raise NotImplementedError


class HumanPlayer(PlayerAgent):
    def _print_player_snapshot(self, state: GameState, player: PlayerState) -> None:
        free_enclosures = [
            e.size for e in player.enclosures if _enclosure_is_standard(e) and not _enclosure_has_animals(e)
        ]
        occupied_count = sum(1 for e in player.enclosures if _enclosure_has_animals(e))
        icons = _player_icon_snapshot(player)
        continents_line = ", ".join(
            f"{name}:{value}" for name, value in icons["continents"].items()
        )
        category_line = ", ".join(
            f"{name}:{value}" for name, value in icons["categories"].items()
        ) or "-"
        score_now = _progress_score(player)
        next_income = _break_income_from_appeal(player.appeal)
        print(
            f"\nT{state.turn_index + 1:02d} | {player.name}\n"
            f"score={score_now} money={player.money} x={player.x_tokens} rep={player.reputation} "
            f"cons={player.conservation} appeal={player.appeal} next_income={next_income} "
            f"hand={len(player.hand)}/{player.hand_limit} final_scoring_cards={len(player.final_scoring_cards)} "
            f"worker={player.workers} worker_on_board={player.workers_on_association_board} "
            f"venom={_action_token_total(player.venom_tokens_on_actions)} "
            f"constriction={_action_token_total(player.constriction_tokens_on_actions)} "
            f"zoo={len(player.zoo_cards)} "
            f"enclosures={len(player.enclosures)} occupied={occupied_count} "
            f"break={state.break_progress}/{state.break_max} "
            f"deck={len(state.zoo_deck)} discard={len(state.zoo_discard)}"
        )
        print(
            f"Icons | continents[{continents_line}] science={icons['science']} "
            f"rock={icons['rock_icons']} water={icons['water_icons']} categories[{category_line}]"
        )
        print("Action cards (slot 1 -> 5): " + ", ".join(player.action_order))
        print(
            "Action upgraded: "
            + ", ".join(f"{action}:{'II' if player.action_upgraded[action] else 'I'}" for action in MAIN_ACTION_CARDS)
        )
        multiplier_line = ", ".join(
            f"{action}:{count}" for action, count in player.multiplier_tokens_on_actions.items() if count > 0
        ) or "-"
        extra_action_line = ", ".join(
            f"{action}:{count}" for action, count in player.extra_actions_granted.items() if count > 0
        ) or "-"
        extra_any = player.extra_any_actions
        extra_strength_line = ", ".join(
            f"{strength}:{count}" for strength, count in sorted(player.extra_strength_actions.items()) if count > 0
        ) or "-"
        venom_line = _action_token_line(player.venom_tokens_on_actions)
        constriction_line = _action_token_line(player.constriction_tokens_on_actions)
        print(
            f"Action tokens | multiplier[{multiplier_line}] extra_actions[{extra_action_line}] "
            f"extra_any={extra_any} extra_strength[{extra_strength_line}] "
            f"venom[{venom_line}] constriction[{constriction_line}]"
        )
        market_partners = ", ".join(_partner_zoo_label(partner) for partner in sorted(state.available_partner_zoos)) or "-"
        market_universities = ", ".join(_university_label(uni) for uni in sorted(state.available_universities)) or "-"
        owned_partners = ", ".join(_partner_zoo_label(partner) for partner in sorted(player.partner_zoos)) or "-"
        owned_universities = ", ".join(_university_label(uni) for uni in sorted(player.universities)) or "-"
        print(
            f"Association market | partner[{market_partners}] universities[{market_universities}] "
            f"donation_cost={_current_donation_cost(state)}"
        )
        print(
            f"Association owned | partner[{owned_partners}] universities[{owned_universities}] "
            f"supported_projects={len(player.supported_conservation_projects)}"
        )
        if state.conservation_project_slots:
            slot_items = []
            for project_id, level_owners in state.conservation_project_slots.items():
                blocked = _blocked_level_by_project(state).get(project_id, "")
                level_items: List[str] = []
                for level_name, _, _ in BASE_CONSERVATION_PROJECT_LEVELS:
                    if level_name == blocked:
                        level_items.append(f"{level_name}=blocked")
                        continue
                    owner = (level_owners or {}).get(level_name)
                    owner_text = "-" if owner is None else state.players[owner].name
                    level_items.append(f"{level_name}={owner_text}")
                slot_items.append(f"{project_id}[{', '.join(level_items)}]")
            print("Conservation project slots | " + ", ".join(slot_items))
        if state.zoo_display:
            print("Display:")
            for idx, card in enumerate(state.zoo_display, start=1):
                in_range = idx <= _reputation_display_limit(player.reputation)
                marker = "*" if in_range else ""
                print(f"- [{idx}{marker}] {_format_card_line_for_player(card, player)}")
        print("Hand:")
        if not player.hand:
            print("- (empty)")
        else:
            for idx, card in enumerate(player.hand, start=1):
                print(f"- [{idx}] {_format_card_line_for_player(card, player)}")
        if player.opening_draft_drawn and not player.opening_draft_kept_indices and not player.hand:
            print("Opening draft cards:")
            for idx, card in enumerate(player.opening_draft_drawn, start=1):
                print(f"- [{idx}] {_format_card_line(card)}")
        zoo_animals = [card for card in player.zoo_cards if card.card_type == "animal"]
        print("Zoo animals:")
        if not zoo_animals:
            print("- (none)")
        else:
            for idx, card in enumerate(zoo_animals, start=1):
                print(f"- [{idx}] {_format_zoo_animal_line(card)}")
        if player.pouched_cards:
            print("Pouched cards (not in zoo):")
            for idx, card in enumerate(player.pouched_cards, start=1):
                host_name = _pouched_card_host_name(player, card)
                print(f"- [{idx}] #{card.number} {card.name} under[{host_name}]")
        print(f"Free enclosure sizes: {free_enclosures if free_enclosures else 'none'}")
        if player.enclosures:
            print("Animal host enclosures:")
            for idx, enclosure in enumerate(player.enclosures, start=1):
                if enclosure.origin is None:
                    cells_text = "[]"
                else:
                    building = _find_zoo_building_by_origin_rotation_size(
                        player=player,
                        origin=enclosure.origin,
                        rotation=enclosure.rotation,
                        size=enclosure.size,
                    )
                    cells_text = _building_cells_text(building) if building is not None else "[]"
                used_capacity = _enclosure_used_capacity(enclosure)
                remaining_capacity = _enclosure_remaining_capacity(enclosure)
                print(
                    f"- E{idx} type={enclosure.enclosure_type} size={enclosure.size} cells={cells_text} "
                    f"used={used_capacity}/{enclosure.animal_capacity} remaining={remaining_capacity}"
                )
        if player.enclosure_objects:
            print("Enclosure objects (terrain metadata):")
            for obj in sorted(
                player.enclosure_objects,
                key=lambda item: (item.origin[0], item.origin[1], item.size, item.rotation),
            ):
                building = _find_zoo_building_by_origin_rotation_size(
                    player=player,
                    origin=obj.origin,
                    rotation=obj.rotation,
                    size=obj.size,
                )
                cells_text = _building_cells_text(building) if building is not None else "[]"
                print(
                    f"- type={obj.enclosure_type} size={obj.size} cells={cells_text} "
                    f"rock={obj.adjacent_rock} water={obj.adjacent_water} animals_inside={obj.animals_inside}"
                )
        if player.zoo_map and player.zoo_map.buildings:
            print("Zoo buildings (actual map buildings):")
            for building in sorted(
                player.zoo_map.buildings.values(),
                key=lambda b: (_building_type_label(b.type), b.origin_hex.x, b.origin_hex.y),
            ):
                print(
                    f"- {_building_type_label(building.type)} cells={_building_cells_text(building)}"
                )
        if player.sponsor_buildings:
            print("Sponsor unique buildings:")
            for sponsor_building in sorted(
                player.sponsor_buildings,
                key=lambda item: (item.sponsor_number, item.cells),
            ):
                cells_text = "[" + ",".join(f"({x},{y})" for x, y in sponsor_building.cells) + "]"
                label = sponsor_building.label or f"#{sponsor_building.sponsor_number}"
                print(f"- #{sponsor_building.sponsor_number} {label} cells={cells_text}")

    def choose_action(self, state: GameState, actions: List[Action]) -> Action:
        actor_id = (
            int(state.pending_decision_player_id)
            if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None
            else int(state.current_player)
        )
        player = state.players[actor_id]
        self._print_player_snapshot(state, player)
        print("Legal actions:")
        for idx, action in enumerate(actions, start=1):
            print(f"{idx}. {action}")

        while True:
            raw = input(f"Select action [1-{len(actions)}] (or q to quit): ").strip().lower()
            if raw in {"q", "quit", "exit"}:
                raise SystemExit(0)
            if not raw.isdigit():
                print("Please enter a number.")
                continue
            picked = int(raw)
            if 1 <= picked <= len(actions):
                selected = actions[picked - 1]
                if bool((selected.details or {}).get("concrete")):
                    if selected.type == ActionType.MAIN_ACTION:
                        details = dict(selected.details or {})
                        details["_interactive"] = True
                        return Action(
                            ActionType.MAIN_ACTION,
                            value=selected.value,
                            card_name=selected.card_name,
                            details=details,
                        )
                    return selected
                if selected.type == ActionType.MAIN_ACTION and selected.card_name == "animals":
                    strength = _effective_action_strength(player, "animals", x_spent=int(selected.value or 0))
                    details = _prompt_animals_action_details_for_human(
                        state=state,
                        player=player,
                        strength=strength,
                        player_id=state.current_player,
                    )
                    details = dict(details or {})
                    details["_interactive"] = True
                    return Action(
                        ActionType.MAIN_ACTION,
                        value=selected.value,
                        card_name="animals",
                        details=details,
                    )
                if selected.type == ActionType.MAIN_ACTION and selected.card_name == "cards":
                    strength = _effective_action_strength(player, "cards", x_spent=int(selected.value or 0))
                    details = _prompt_cards_action_details_for_human(
                        state=state,
                        player=player,
                        strength=strength,
                    )
                    details = dict(details or {})
                    details["_interactive"] = True
                    return Action(
                        ActionType.MAIN_ACTION,
                        value=selected.value,
                        card_name="cards",
                        details=details,
                    )
                if selected.type == ActionType.MAIN_ACTION and selected.card_name == "build":
                    strength = _effective_action_strength(player, "build", x_spent=int(selected.value or 0))
                    details = _prompt_build_action_details_for_human(
                        state=state,
                        player=player,
                        strength=strength,
                        player_id=state.current_player,
                    )
                    details = dict(details or {})
                    details["_interactive"] = True
                    return Action(
                        ActionType.MAIN_ACTION,
                        value=selected.value,
                        card_name="build",
                        details=details,
                    )
                if selected.type == ActionType.MAIN_ACTION and selected.card_name == "association":
                    strength = _effective_action_strength(player, "association", x_spent=int(selected.value or 0))
                    details = _prompt_association_action_details_for_human(
                        state=state,
                        player=player,
                        strength=strength,
                        player_id=state.current_player,
                    )
                    details = dict(details or {})
                    details["_interactive"] = True
                    return Action(
                        ActionType.MAIN_ACTION,
                        value=selected.value,
                        card_name="association",
                        details=details,
                    )
                if selected.type == ActionType.MAIN_ACTION and selected.card_name == "sponsors":
                    strength = _effective_action_strength(player, "sponsors", x_spent=int(selected.value or 0))
                    details = _prompt_sponsors_action_details_for_human(
                        state=state,
                        player=player,
                        strength=strength,
                    )
                    details = dict(details or {})
                    details["_interactive"] = True
                    return Action(
                        ActionType.MAIN_ACTION,
                        value=selected.value,
                        card_name="sponsors",
                        details=details,
                    )
                if selected.type == ActionType.X_TOKEN:
                    print("Choose which action card to move to slot 1 for X-token:")
                    for i, action_card in enumerate(player.action_order, start=1):
                        print(f"{i}. {action_card}")
                    while True:
                        raw_card = input(f"Select card [1-{len(player.action_order)}]: ").strip()
                        if not raw_card.isdigit():
                            print("Please enter a number.")
                            continue
                        card_idx = int(raw_card)
                        if 1 <= card_idx <= len(player.action_order):
                            return Action(ActionType.X_TOKEN, card_name=player.action_order[card_idx - 1])
                        print("Out of range, try again.")
                return selected
            print("Out of range, try again.")


def setup_game(
    seed: int = 42,
    player_names: Optional[List[str]] = None,
    manual_opening_draft_player_names: Optional[Set[str]] = None,
    include_marine_world: bool = False,
) -> GameState:
    state = _setup_game_state_impl(
        seed=seed,
        player_names=player_names,
        manual_opening_draft_player_names=manual_opening_draft_player_names,
        build_deck=build_deck,
        map_image_name="plan1a",
        load_map_data=load_map_data_by_image_name,
        build_map_tile_bonus_map=_build_map_tile_bonus_map,
        player_state_factory=PlayerState,
        game_state_factory=GameState,
        ark_map_factory=ArkNovaMap,
        main_action_names=MAIN_ACTION_CARDS,
        draw_final_scoring_cards_for_players_fn=lambda players, rng: _draw_final_scoring_cards_for_players(
            players,
            rng,
            include_marine_world=include_marine_world,
        ),
        build_opening_setup_info_fn=_build_opening_setup_info,
        load_base_setup_card_pools_fn=_load_base_setup_card_pools,
        refresh_association_market=_refresh_association_market,
        draw_opening_draft_cards_fn=_draw_opening_draft_cards,
        apply_opening_draft_selection_fn=_apply_opening_draft_selection,
        replenish_zoo_display=_replenish_zoo_display,
        validate_card_zones=_validate_card_zones,
        break_track_by_players=BREAK_TRACK_BY_PLAYERS,
        max_turns_per_player=MAX_GAME_ROUNDS,
    )
    state.map_rules = _load_map_rules(state.map_image_name)
    state.map_tile_tags = _build_map_tile_tags_map(state.map_image_name)
    _begin_next_opening_draft_pending_if_needed(state)
    return state


def play_game(
    agents: Dict[str, PlayerAgent],
    player_names: List[str],
    seed: int = 42,
    verbose: bool = True,
    include_marine_world: bool = False,
) -> Dict[str, int]:
    manual_draft_players = {name for name in player_names if isinstance(agents.get(name), HumanPlayer)}
    state = setup_game(
        seed=seed,
        player_names=player_names,
        manual_opening_draft_player_names=manual_draft_players,
        include_marine_world=include_marine_world,
    )

    missing_agents = [p.name for p in state.players if p.name not in agents]
    if missing_agents:
        raise ValueError(f"Missing agents for players: {', '.join(missing_agents)}")

    if verbose:
        opening = state.opening_setup
        all_zoo_cards: List[AnimalCard] = list(state.zoo_deck) + list(state.zoo_display) + list(state.zoo_discard)
        for p in state.players:
            all_zoo_cards.extend(p.hand)
            all_zoo_cards.extend(p.zoo_cards)
        effect_coverage = build_effect_coverage(all_zoo_cards)
        print("Opening setup:")
        print(
            "Card effects coverage (zoo deck): "
            f"total={effect_coverage['total']} supported={effect_coverage['supported']} "
            f"unsupported={effect_coverage['unsupported']} no_effect={effect_coverage['no_effect']}"
        )
        print(f"- CP 2 fixed options: {opening.conservation_space_2_fixed_options}")
        print(
            f"- CP 5 rewards: random_once={opening.conservation_space_5_bonus_tiles} "
            f"+ fixed_repeatable={CONSERVATION_FIXED_MONEY_OPTION}"
        )
        print(
            f"- CP 8 rewards: random_once={opening.conservation_space_8_bonus_tiles} "
            f"+ fixed_repeatable={CONSERVATION_FIXED_MONEY_OPTION}"
        )
        print(f"- CP 10 rule: {opening.conservation_space_10_rule}")
        print("- Base conservation projects:")
        for idx, project in enumerate(opening.base_conservation_projects, start=1):
            blocked_level = opening.two_player_blocked_project_levels[idx - 1].blocked_level
            print(f"  {idx}. {project.data_id} | {project.title} | blocked_in_2p={blocked_level}")
        print("- Final scoring cards (setup draw):")
        for player in state.players:
            cards = [card.data_id for card in player.final_scoring_cards]
            print(f"  {player.name}: {cards}")
        print()

    if verbose:
        print("- Opening 8-card draft per player:")
        for player in state.players:
            print(f"  {player.name}:")
            for idx, card in enumerate(player.opening_draft_drawn, start=1):
                mark = " [KEEP]" if (idx - 1) in set(player.opening_draft_kept_indices) else ""
                print(f"    {idx}. {_format_card_line(card)}{mark}")
        print()

    while str(state.pending_decision_kind or "").strip() or not state.game_over():
        actor_id = (
            int(state.pending_decision_player_id)
            if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None
            else int(state.current_player)
        )
        player = state.players[actor_id]
        actions = legal_actions(player, state=state, player_id=actor_id)
        agent = agents[player.name]
        action = agent.choose_action(state, actions)
        effect_log_cursor = len(state.effect_log)
        apply_action(state, action)
        new_effect_logs = state.effect_log[effect_log_cursor:]

        if verbose:
            print(
                f"T{state.turn_index:02d} | {player.name:11s} -> {action!s:18s} | "
                f"money={player.money:2d} appeal={player.appeal:2d} cons={player.conservation:2d}"
            )
            for line in new_effect_logs:
                print(f"  effect: {line}")

    scores: Dict[str, int] = {}
    for p in state.players:
        scores[p.name] = _final_score_points(state, p)

    if verbose:
        ranking = _rank_scores(scores)
        print("\nFinal ranking:")
        for rank, name, score in ranking:
            print(f"{rank}. {name}: {score}")
        top_score = ranking[0][2]
        winners = [name for _, name, score in ranking if score == top_score]
        print(f"Winner: {', '.join(winners)}")

    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified Ark Nova PVP prototype runner.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--marine-world",
        action="store_true",
        help="Include Marine World final scoring cards (F012-F017) in the setup pool.",
    )
    parser.add_argument("--quiet", action="store_true", help="Reduce turn-by-turn output")
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    verbose = not args.quiet
    play_game(
        seed=args.seed,
        verbose=verbose,
        agents={"Player1": HumanPlayer(), "Player2": HumanPlayer()},
        player_names=["Player1", "Player2"],
        include_marine_world=bool(args.marine_world),
    )


if __name__ == "__main__":
    main_cli()
# python main.py --seed 7
# python tools/rl/train_self_play.py --algo masked_ppo --updates 100 --episodes-per-update 8 --output-dir runs/self_play_masked --resume-from runs/self_play_masked/checkpoint_0020.pt

"""Simplified Ark Nova-like AI prototype.

Run:
    python main.py
    python main.py --mode pvp

This is a compact baseline you can extend:
- simplified game state
- legal action generation
- heuristic two-step lookahead AI
- random baseline AI
- optional human-vs-human turn selection loop
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from enum import Enum
import copy
from functools import lru_cache
import json
from pathlib import Path
import random
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

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
    prompt_opening_draft_indices as _prompt_opening_draft_indices_impl,
    resolve_manual_opening_drafts as _resolve_manual_opening_drafts_impl,
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
                    return f"{base}(strength={strength})"
                return f"{base}(x={x_spent}, strength={strength})"
            if x_spent <= 0:
                return base
            return f"{base}(x={x_spent})"
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
CONSERVATION_SHARED_BONUS_THRESHOLDS: Tuple[int, int] = (5, 8)
BREAK_TRACK_BY_PLAYERS: Dict[int, int] = {2: 9, 3: 12, 4: 15}

MAIN_ACTION_CARDS: Tuple[str, ...] = ("animals", "cards", "build", "association", "sponsors")
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
SPONSOR_LEVEL_OVERRIDES: Dict[int, int] = {
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
    231: ("Primate",),
    232: ("Reptile",),
    233: ("Bird",),
    234: ("Predator",),
    235: ("Herbivore",),
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
    current_player: int = 0
    turn_index: int = 0
    max_turns_per_player: int = 16

    @property
    def total_turn_limit(self) -> int:
        return self.max_turns_per_player * len(self.players)

    def game_over(self) -> bool:
        return self.turn_index >= self.total_turn_limit

    def available_conservation_reward_choices(self, player_id: int, threshold: int) -> List[str]:
        if threshold not in CONSERVATION_SHARED_BONUS_THRESHOLDS:
            raise ValueError("Conservation threshold must be 5 or 8 for shared bonus rewards.")
        player = self.players[player_id]
        if player.conservation < threshold:
            return []
        if threshold in player.claimed_conservation_reward_spaces:
            return []
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

        if reward == CONSERVATION_FIXED_MONEY_OPTION:
            player.money += 5
        else:
            shared_tiles = self.shared_conservation_bonus_tiles[threshold]
            shared_tiles.remove(reward)
            self.claimed_conservation_bonus_tiles[threshold].append(reward)

        player.claimed_conservation_reward_spaces.add(threshold)


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


def _draw_final_scoring_cards_for_players(players: List[PlayerState], rng: random.Random) -> List[SetupCardRef]:
    final_pool, _ = _load_base_setup_card_pools()
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


def _eligible_hypnosis_target_ids(state: GameState, player_id: int) -> List[int]:
    actor = state.players[player_id]
    other_ids = [idx for idx in range(len(state.players)) if idx != player_id]
    if not other_ids:
        return []
    max_other_appeal = max(int(state.players[idx].appeal) for idx in other_ids)
    if int(actor.appeal) >= max_other_appeal:
        return []
    return [
        idx
        for idx in other_ids
        if int(state.players[idx].appeal) == max_other_appeal and not _player_has_sponsor(state.players[idx], 225)
    ]


def _pilfering_target_ids(state: GameState, player_id: int, amount: int) -> List[int]:
    actor = state.players[player_id]
    other_ids = [idx for idx in range(len(state.players)) if idx != player_id]
    if not other_ids or amount <= 0:
        return []
    targets: List[int] = []
    max_other_appeal = max(int(state.players[idx].appeal) for idx in other_ids)
    if int(actor.appeal) < max_other_appeal:
        for idx in other_ids:
            if int(state.players[idx].appeal) == max_other_appeal and not _player_has_sponsor(state.players[idx], 225):
                targets.append(idx)
                break
    if amount >= 2:
        max_other_conservation = max(int(state.players[idx].conservation) for idx in other_ids)
        if int(actor.conservation) < max_other_conservation:
            for idx in other_ids:
                if (
                    int(state.players[idx].conservation) == max_other_conservation
                    and idx not in targets
                    and not _player_has_sponsor(state.players[idx], 225)
                ):
                    targets.append(idx)
                    break
    return targets


def legal_actions(
    player: PlayerState,
    state: Optional[GameState] = None,
    player_id: Optional[int] = None,
) -> List[Action]:
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
        merged_details["effective_strength"] = _effective_action_strength(
            player,
            card_name,
            x_spent=x_spent,
            base_strength=base_strength,
        )
        annotated.append(
            Action(
                ActionType.MAIN_ACTION,
                value=action.value,
                card_name=action.card_name,
                details=merged_details,
            )
        )

    if state is None or player_id is None:
        return annotated

    filtered: List[Action] = []
    for action in annotated:
        if action.type != ActionType.MAIN_ACTION:
            filtered.append(action)
            continue
        card_name = str(action.card_name or "")
        strength_raw = (action.details or {}).get("effective_strength")
        strength = int(strength_raw) if isinstance(strength_raw, int) else _effective_action_strength(
            player,
            card_name,
            x_spent=int(action.value or 0),
        )

        if card_name == "animals":
            if list_legal_animals_options(state=state, player_id=player_id, strength=strength):
                filtered.append(action)
            continue

        if card_name == "sponsors":
            sponsors_upgraded = bool(player.action_upgraded["sponsors"])
            level_cap = strength + 1 if sponsors_upgraded else strength
            candidates = _list_legal_sponsor_candidates(
                state=state,
                player=player,
                sponsors_upgraded=sponsors_upgraded,
            )
            has_playable_under_cap = any(
                bool(cand.get("playable_now")) and int(cand.get("level", 0)) <= level_cap
                for cand in candidates
            )
            new_details = dict(action.details or {})
            if has_playable_under_cap:
                new_details.pop("sponsors_break_only", None)
            else:
                new_details["sponsors_break_only"] = True
            filtered.append(
                Action(
                    ActionType.MAIN_ACTION,
                    value=action.value,
                    card_name=action.card_name,
                    details=new_details,
                )
            )
            continue

        if card_name == "association":
            if list_legal_association_options(state=state, player_id=player_id, strength=strength):
                filtered.append(action)
            continue

        if card_name:
            filtered.append(action)
    return filtered


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


def _rotate_action_card_to_slot_1(player: PlayerState, chosen_action: str) -> int:
    if chosen_action not in player.action_order:
        raise ValueError(f"Unknown action card '{chosen_action}'.")
    idx = player.action_order.index(chosen_action)
    card = player.action_order.pop(idx)
    player.action_order.insert(0, card)
    return idx + 1


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


def _final_score_points(state: GameState, player: PlayerState) -> int:
    bonus_appeal, bonus_conservation = _sponsor_endgame_bonus(state, player)
    total_appeal = int(player.appeal) + int(bonus_appeal)
    total_conservation = int(player.conservation) + int(bonus_conservation)
    return total_appeal + _bga_conservation_points(total_conservation)


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
) -> bool:
    if card.number not in SPONSOR_UNIQUE_BUILDING_CARDS:
        return False
    legal = _list_legal_sponsor_unique_building_cells(
        state=state,
        player=player,
        sponsor_number=card.number,
    )
    if not legal:
        return False
    picked = legal[0]
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


def _find_legal_building_by_serialized_selection(
    legal: Sequence[Building],
    selection: Dict[str, Any],
) -> Optional[Building]:
    selected_type_name = str(selection.get("building_type", "")).strip()
    selected_cells = selection.get("cells")
    if selected_cells is None:
        selected_cells = selection.get("layout")
    if isinstance(selected_cells, (list, tuple)) and selected_cells:
        parsed_cells: List[Tuple[int, int]] = []
        for cell in selected_cells:
            if not isinstance(cell, (list, tuple)) or len(cell) != 2:
                return None
            x_raw, y_raw = cell
            if not isinstance(x_raw, int) or not isinstance(y_raw, int):
                return None
            parsed_cells.append((x_raw, y_raw))
        target_layout = tuple(sorted(parsed_cells))
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


def _apply_build_placement_bonus(
    state: GameState,
    player: PlayerState,
    bonus: str,
    details: Dict[str, Any],
    bonus_index: int,
    bonus_coord: Optional[Tuple[int, int]] = None,
    allow_archaeologist_chain: bool = True,
) -> None:
    if bonus == "5coins":
        player.money += 5
    elif bonus == "x_token":
        player.x_tokens = min(5, player.x_tokens + 1)
    elif bonus == "reputation":
        _increase_reputation(state, player, 1)
    elif bonus == "card_in_reputation_range":
        # Simplified: take top deck card if possible.
        draw = _draw_from_zoo_deck(state, 1)
        if draw:
            player.hand.extend(draw)
    elif bonus == "action_to_slot_1":
        targets = details.get("bonus_action_to_slot_1_targets")
        chosen_action = None
        if isinstance(targets, list) and bonus_index < len(targets):
            raw = targets[bonus_index]
            if isinstance(raw, str) and raw in player.action_order:
                chosen_action = raw
        if chosen_action is None:
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


def _upgrade_one_action_card(player: PlayerState) -> bool:
    upgradeable = [action for action in MAIN_ACTION_CARDS if not bool(player.action_upgraded.get(action, False))]
    if not upgradeable:
        return False

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
) -> None:
    code = str(effect_code).strip().lower()
    if not code:
        return

    if code == "draw_1_card_deck_or_reputation_range":
        _take_one_card_from_deck_or_reputation_range(state, player)
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "build_free_standard_enclosure_size_2":
        placed = _place_free_building_of_type_if_possible(
            state=state,
            player=player,
            building_type=BuildingType.SIZE_2,
            player_id=player_id,
        )
        state.effect_log.append(f"{source}:{player.name}:{code}:placed={1 if placed else 0}")
        return

    if code == "gain_5_coins":
        player.money += 5
        state.effect_log.append(f"{source}:{player.name}:{code}")
        return

    if code == "play_1_sponsor_by_paying_cost":
        sponsors_upgraded = bool(player.action_upgraded["sponsors"])
        candidates = _list_legal_sponsor_candidates(state, player, sponsors_upgraded)
        playable_hand = [item for item in candidates if item.get("source") == "hand" and bool(item.get("playable_now"))]
        if not playable_hand:
            state.effect_log.append(f"{source}:{player.name}:{code}:played=0")
            return
        playable_hand.sort(
            key=lambda item: (
                int(item.get("pay_cost", 0)),
                int(item.get("level", 0)),
                int(getattr(item.get("card"), "number", -1)),
                str(item.get("card_instance_id", "")),
            )
        )
        chosen = playable_hand[0]
        selected_card = chosen.get("card")
        details = {
            "use_break_ability": False,
            "sponsor_selections": [
                {
                    "source": "hand",
                    "source_index": int(chosen.get("source_index", -1)),
                    "card_instance_id": str(chosen.get("card_instance_id", "")),
                }
            ],
        }
        min_strength = max(1, int(chosen.get("level", 1)))
        _perform_sponsors_action_effect(
            state=state,
            player=player,
            strength=min_strength,
            player_id=player_id,
            details=details,
        )
        state.effect_log.append(
            f"{source}:{player.name}:{code}:played={getattr(selected_card, 'number', -1)}"
        )
        return

    if code == "gain_worker_1":
        _gain_workers(state=state, player=player, player_id=player_id, amount=1, source=source)
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
        upgraded = _upgrade_one_action_card(player)
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
) -> None:
    if gained_workers <= 0:
        return
    for reward in _map_rule_worker_gain_rewards(state):
        effect_code = str(reward.get("effect") or "").strip().lower()
        if not effect_code:
            continue
        repeat = int(reward.get("repeat_per_worker", 1))
        repeat = max(1, repeat)
        for _ in range(gained_workers * repeat):
            _apply_map_effect_code(
                state=state,
                player=player,
                player_id=player_id,
                effect_code=effect_code,
                source=source,
            )


def _gain_workers(
    *,
    state: GameState,
    player: PlayerState,
    player_id: int,
    amount: int,
    source: str,
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
        )
    return gained


def _apply_map_partner_threshold_rewards(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> None:
    count = len(player.partner_zoos)
    for item in _map_rule_partner_thresholds(state):
        threshold = int(item.get("count", 0))
        effect_code = str(item.get("effect") or "").strip().lower()
        if threshold <= 0 or not effect_code:
            continue
        if count < threshold:
            continue
        if threshold in player.claimed_partner_zoo_thresholds:
            continue
        player.claimed_partner_zoo_thresholds.add(threshold)
        _apply_map_effect_code(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            source=f"map_partner_threshold_{threshold}",
        )


def _apply_map_university_threshold_rewards(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> None:
    count = len(player.universities)
    for item in _map_rule_university_thresholds(state):
        threshold = int(item.get("count", 0))
        effect_code = str(item.get("effect") or "").strip().lower()
        if threshold <= 0 or not effect_code:
            continue
        if count < threshold:
            continue
        if threshold in player.claimed_university_thresholds:
            continue
        player.claimed_university_thresholds.add(threshold)
        _apply_map_effect_code(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            source=f"map_university_threshold_{threshold}",
        )


def _on_map_conservation_project_supported(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> None:
    unlocks = _map_rule_left_track_unlocks(state)
    idx = int(player.map_left_track_unlocked_count)
    if idx < 0 or idx >= len(unlocks):
        return
    item = unlocks[idx]
    effect_code = str(item.get("effect") or "").strip().lower()
    category = str(item.get("category") or "").strip().lower()
    player.map_left_track_unlocked_count = idx + 1
    if category == "purple_recurring_action":
        player.map_left_track_unlocked_effects.append(effect_code)
    if effect_code:
        _apply_map_effect_code(
            state=state,
            player=player,
            player_id=player_id,
            effect_code=effect_code,
            source=f"map_left_track_unlock_{idx + 1}",
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


def _apply_reputation_milestone_reward(state: GameState, player: PlayerState, milestone: int) -> None:
    if milestone == 5:
        _upgrade_one_action_card(player)
        return
    if milestone == 8:
        player_id = state.players.index(player)
        _gain_workers(
            state=state,
            player=player,
            player_id=player_id,
            amount=1,
            source="reputation_milestone_8",
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


def _increase_reputation(state: GameState, player: PlayerState, amount: int) -> None:
    _increase_reputation_impl(
        state=state,
        player=player,
        amount=amount,
        association_action_key="association",
        reputation_cap_without_upgrade=9,
        reputation_cap_with_upgrade=15,
        milestone_values=[5, 8, 10, 11, 12, 13, 14, 15],
        apply_milestone_reward_fn=_apply_reputation_milestone_reward,
    )


def _prompt_break_discard_indices(player: PlayerState, overflow: int) -> List[int]:
    if overflow <= 0:
        return []
    prompt = f"Select {overflow} card index{'es' if overflow > 1 else ''} to discard: "
    print(
        f"{player.name} exceeds hand limit ({len(player.hand)}/{player.hand_limit}). "
        f"Choose {overflow} card(s) to discard."
    )
    while True:
        for idx, card in enumerate(player.hand, start=1):
            print(f"{idx}. {_format_card_line_for_player(card, player)}")
        raw = input(prompt).strip()
        parts = raw.replace(",", " ").split()
        if len(parts) != overflow:
            print(f"Please enter exactly {overflow} index{'es' if overflow > 1 else ''}.")
            continue
        try:
            picked = [int(part) for part in parts]
        except ValueError:
            print("Please enter valid numbers.")
            continue
        if any(index < 1 or index > len(player.hand) for index in picked):
            print("Index out of range.")
            continue
        if len(set(picked)) != len(picked):
            print("Please choose distinct cards.")
            continue
        return [index - 1 for index in picked]


def _discard_down_to_limit(player: PlayerState) -> List[AnimalCard]:
    if len(player.hand) <= int(player.hand_limit):
        return []
    overflow = len(player.hand) - int(player.hand_limit)
    picked_indices = _prompt_break_discard_indices(player, overflow)
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


def _resolve_break(state: GameState) -> None:
    # 1) Hand card limit.
    for player_id in _break_income_order(state):
        state.zoo_discard.extend(_discard_down_to_limit(state.players[player_id]))

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

    # 5) Income in break-trigger-player order when needed.
    order = _break_income_order(state)
    for player_id in order:
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

    # 6) Reset break track.
    state.break_progress = 0
    state.break_trigger_player = None


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

    break_triggered = _advance_break_track(state, steps=2, trigger_player=player_id)

    snap_display_index_raw = details.get("snap_display_index")
    snap_display_index = int(snap_display_index_raw) if snap_display_index_raw is not None else None
    from_display_indices = list(details.get("from_display_indices") or [])
    from_deck_count_raw = details.get("from_deck_count")
    from_deck_count = int(from_deck_count_raw) if from_deck_count_raw is not None else None
    discard_hand_indices_raw = details.get("discard_hand_indices")
    discard_hand_indices = list(discard_hand_indices_raw) if discard_hand_indices_raw is not None else None

    if snap_display_index is not None:
        if not snap_allowed:
            raise ValueError("Snap is not available at this cards strength.")
        if from_display_indices or from_deck_count is not None:
            raise ValueError("Snap cannot be combined with normal draw choices.")
        if snap_display_index < 0 or snap_display_index >= len(state.zoo_display):
            raise ValueError("Snap display index out of range.")
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

    player.hand.extend(_take_display_cards(state, from_display_indices))
    player.hand.extend(_draw_from_zoo_deck(state, from_deck_count))
    if from_display_indices:
        _replenish_zoo_display(state)

    if discard_target > 0:
        if discard_hand_indices is None:
            if len(player.hand) < discard_target:
                raise ValueError("Not enough hand cards to discard for cards action.")
            discard_hand_indices = list(range(len(player.hand) - discard_target, len(player.hand)))
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
            for candidate in legal:
                candidate_cost = len(candidate.layout) * 2
                if player.money >= candidate_cost:
                    picked = candidate
                    break
            if picked is None:
                break

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
        if remaining_size <= 0:
            break
        if requested_selections:
            continue
        break

    if built_count == 0:
        return

    if _player_has_sponsor(player, 217):
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


def _animal_enclosure_spaces_needed(card: AnimalCard, enclosure: Enclosure) -> Optional[int]:
    kind = str(getattr(enclosure, "enclosure_type", "standard") or "standard").strip().lower()
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


def _pick_animals_option_for_ai(options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not options:
        return None
    return max(
        options,
        key=lambda item: (
            len(item["plays"]),
            int(item["total_appeal"]) + int(item["total_conservation"]) * 3,
            -int(item["total_cost"]),
        ),
    )


def _perform_animals_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    details: Optional[Dict[str, Any]] = None,
    player_id: int = 0,
) -> None:
    details = details or {}
    upgraded = player.action_upgraded["animals"]

    if upgraded and strength >= 5:
        _increase_reputation(state=state, player=player, amount=1)

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
        selected = _pick_animals_option_for_ai(options)

    if selected is None:
        return

    selected_plays = list(selected["plays"])
    if not selected_plays:
        return

    interactive_effect_prompts = bool(details.get("_interactive"))
    queued_clever_targets: List[str] = []
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
            raise ValueError("Selected animal card is no longer in hand.")
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
        _increase_reputation(state=state, player=player, amount=card.reputation_gain)
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

        def _effect_advance_break(steps: int) -> bool:
            triggered = _advance_break_track(state=state, steps=max(0, steps), trigger_player=player_id)
            if triggered:
                _resolve_break(state)
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
            sell_count = min(max(0, limit), len(player.hand))
            if sell_count <= 0:
                return 0
            sold_cards = list(player.hand[-sell_count:])
            del player.hand[-sell_count:]
            state.zoo_discard.extend(sold_cards)
            if money_each > 0:
                player.money += sell_count * money_each
            return sell_count

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
            target_id = target_ids[0]
            target_player = state.players[target_id]
            available_actions = list(target_player.action_order[: max(0, max_slot)])
            if not available_actions:
                return f"target={target_player.name} no_action"

            chosen_action = available_actions[0]
            if interactive_effect_prompts:
                print(f"Hypnosis: choose one action card from {target_player.name} slot 1-{len(available_actions)}.")
                for idx, action_name in enumerate(available_actions, start=1):
                    base_strength = idx
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
                    raw = input(f"Select action [1-{len(available_actions)}]: ").strip()
                    if raw.isdigit():
                        picked = int(raw)
                        if 1 <= picked <= len(available_actions):
                            chosen_action = available_actions[picked - 1]
                            break
                    print("Please enter a valid number.")

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

            break_triggered = _perform_main_action_dispatch(
                state=state,
                player=player,
                player_id=player_id,
                chosen=chosen_action,
                strength=effective_strength,
                details=hypnosis_details,
            )
            _apply_action_card_token_use(target_player, chosen_action)
            _rotate_action_card_to_slot_1(target_player, chosen_action)
            if break_triggered:
                _resolve_break(state)
            return (
                f"target={target_player.name} action={chosen_action} "
                f"x={x_spent} strength={effective_strength}"
            )

        def _effect_pilfering(amount: int) -> str:
            targets = _pilfering_target_ids(state, player_id, amount)
            if not targets:
                return "no_target"
            results: List[str] = []
            for target_id in targets:
                target_player = state.players[target_id]
                can_take_money = int(target_player.money) >= 5
                can_take_card = bool(target_player.hand)
                if not can_take_money and not can_take_card:
                    results.append(f"{target_player.name}:none")
                    continue

                choice = "money" if can_take_money else "card"
                if interactive_effect_prompts and can_take_money and can_take_card:
                    print(
                        f"Pilfering: {target_player.name} chooses what to lose."
                    )
                    print("1. Lose 5 money")
                    print("2. Give 1 random hand card")
                    while True:
                        raw = input("Select option [1-2]: ").strip()
                        if raw == "1":
                            choice = "money"
                            break
                        if raw == "2":
                            choice = "card"
                            break
                        print("Please enter 1 or 2.")

                if choice == "money" and can_take_money:
                    target_player.money -= 5
                    player.money += 5
                    results.append(f"{target_player.name}:money")
                    continue
                if can_take_card:
                    picked = random.choice(target_player.hand)
                    target_player.hand.remove(picked)
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
                place_free_kiosk_or_pavilion=_effect_place_free_kiosk_or_pavilion,
                add_multiplier_token=_effect_add_multiplier_token,
                apply_venom=_effect_apply_venom,
                apply_constriction=_effect_apply_constriction,
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
            place_free_kiosk_or_pavilion=_effect_place_free_kiosk_or_pavilion,
            add_multiplier_token=_effect_add_multiplier_token,
            apply_venom=_effect_apply_venom,
            apply_constriction=_effect_apply_constriction,
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
                    "workers_needed": required,
                    "description": f"Gain 2 reputation (workers={required})",
                }
            )

    if strength >= 3:
        required = _task_workers_or_none("partner_zoo")
        if required is not None:
            if bool(player.action_upgraded.get("association", False)) or len(player.partner_zoos) < 2:
                for partner in sorted(state.available_partner_zoos):
                    if partner in player.partner_zoos:
                        continue
                    options.append(
                        {
                            "task_kind": "partner_zoo",
                            "partner_zoo": partner,
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
                        "workers_needed": required,
                        "description": f"Take {_university_label(university)} (workers={required})",
                    }
                )

    conservation_strength_requirement = 4 if _player_has_sponsor(player, 203) else 5
    if strength >= conservation_strength_requirement:
        required = _task_workers_or_none("conservation_project")
        if required is not None:
            blocked_by_project = _blocked_level_by_project(state)
            allow_release_repeat = _player_has_sponsor(player, 224)
            icon_reduction = _available_breeding_icon_reduction(player)
            for project in state.opening_setup.base_conservation_projects:
                if (
                    project.data_id in player.supported_conservation_projects
                    and not (allow_release_repeat and _is_release_into_wild_project(project.data_id))
                ):
                    continue
                icon_value = _project_requirement_value(player, project.data_id)
                project_slots = state.conservation_project_slots.get(project.data_id) or {}
                blocked_level = blocked_by_project.get(project.data_id, "")
                for level_name, needed_icons, conservation_gain in BASE_CONSERVATION_PROJECT_LEVELS:
                    if level_name == blocked_level:
                        continue
                    icon_reduction_used = icon_value < needed_icons
                    if icon_value + icon_reduction < needed_icons:
                        continue
                    owner = project_slots.get(level_name)
                    if owner is not None:
                        continue
                    options.append(
                        {
                            "task_kind": "conservation_project",
                            "project_id": project.data_id,
                            "project_title": project.title,
                            "project_level": level_name,
                            "required_icons": needed_icons,
                            "icon_value": icon_value,
                            "icon_reduction_used": icon_reduction_used,
                            "conservation_gain": conservation_gain,
                            "workers_needed": required,
                            "description": (
                                f"Support conservation project: {project.data_id} | {project.title} "
                                f"[{level_name}: icons>={needed_icons} -> +{conservation_gain} CP, workers={required}"
                                + (", icon_reduction=1" if icon_reduction_used else "")
                                + "]"
                            ),
                        }
                    )

    for idx, option in enumerate(options, start=1):
        option["index"] = idx
    return options


def _pick_association_option_for_ai(options: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not options:
        return None
    priority = {
        "conservation_project": 5,
        "university": 4,
        "partner_zoo": 3,
        "reputation": 2,
    }
    return max(
        options,
        key=lambda option: (
            priority.get(str(option.get("task_kind")), 0),
            int(option.get("conservation_gain", 0)),
            str(option.get("description", "")),
        ),
    )


def _perform_association_action_effect(
    state: GameState,
    player: PlayerState,
    strength: int,
    details: Optional[Dict[str, Any]] = None,
    player_id: int = 0,
) -> None:
    details = details or {}
    upgraded = player.action_upgraded["association"]
    options = list_legal_association_options(state=state, player_id=player_id, strength=strength)
    make_donation = bool(details.get("make_donation"))
    donation_cost: Optional[int] = None
    if make_donation:
        if not upgraded:
            raise ValueError("Donation is only available with upgraded Association action.")
        donation_cost = _current_donation_cost(state)
        if player.money < donation_cost:
            raise ValueError("Insufficient money for donation.")

    selected: Optional[Dict[str, Any]] = None
    selected_idx_raw = details.get("association_option_index")
    if selected_idx_raw is not None:
        selected_idx = int(selected_idx_raw)
        if selected_idx < 0 or selected_idx >= len(options):
            raise ValueError("association_option_index is out of range.")
        selected = options[selected_idx]
    else:
        task_kind = str(details.get("task_kind") or "").strip()
        if task_kind:
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
                if candidates:
                    selected = max(
                        candidates,
                        key=lambda option: int(option.get("conservation_gain", 0)),
                    )
            if selected is None:
                raise ValueError("Requested association task is not legal.")
        else:
            selected = _pick_association_option_for_ai(options)

    workers_needed = 0
    if selected is not None:
        task_kind = str(selected.get("task_kind"))
        workers_needed = _association_workers_needed(player, task_kind)
        if player.workers < workers_needed:
            raise ValueError("Not enough active association workers.")

        if task_kind == "reputation":
            _increase_reputation(state=state, player=player, amount=2)
        elif task_kind == "partner_zoo":
            partner = str(selected.get("partner_zoo") or "")
            if not bool(player.action_upgraded.get("association", False)) and len(player.partner_zoos) >= 2:
                raise ValueError("Association side I can hold at most 2 partner zoos.")
            if partner not in state.available_partner_zoos:
                raise ValueError("Selected partner zoo is not currently available.")
            state.available_partner_zoos.remove(partner)
            player.partner_zoos.add(partner)
            _apply_map_partner_threshold_rewards(
                state=state,
                player=player,
                player_id=player_id,
            )
        elif task_kind == "university":
            university = str(selected.get("university") or "")
            if university not in state.available_universities:
                raise ValueError("Selected university is not currently available.")
            state.available_universities.remove(university)
            player.universities.add(university)
            hand_limit_target = UNIVERSITY_HAND_LIMIT_SET.get(university, 0)
            if hand_limit_target > 0:
                player.hand_limit = max(player.hand_limit, hand_limit_target)
            rep_gain = UNIVERSITY_REPUTATION_GAIN.get(university, 0)
            if rep_gain > 0:
                _increase_reputation(state=state, player=player, amount=rep_gain)
            _apply_map_university_threshold_rewards(
                state=state,
                player=player,
                player_id=player_id,
            )
        elif task_kind == "conservation_project":
            project_id = str(selected.get("project_id") or "")
            if not project_id:
                raise ValueError("Missing project_id for conservation project task.")
            release_repeat_allowed = _player_has_sponsor(player, 224) and _is_release_into_wild_project(project_id)
            if project_id in player.supported_conservation_projects and not release_repeat_allowed:
                raise ValueError("Conservation project already supported by this player.")
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
            if release_repeat_allowed:
                player.conservation += 1
            if bool(selected.get("icon_reduction_used")):
                _consume_breeding_icon_reduction(player)
            player.supported_conservation_project_actions += 1
            _on_map_conservation_project_supported(
                state=state,
                player=player,
                player_id=player_id,
            )
        else:
            raise ValueError(f"Unsupported association task kind '{task_kind}'.")

        _spend_association_workers(player=player, task_kind=task_kind, workers_needed=workers_needed)

    if donation_cost is not None:
        if selected is None:
            raise ValueError("Donation must be combined with another association task.")
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
) -> bool:
    _ensure_player_map_initialized(state, player)
    if player.zoo_map is None:
        return False

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
            details={},
            bonus_index=0,
            bonus_coord=bonus_coord,
        )
    _apply_cover_money_from_sponsors_241_242(player, picked)
    resolved_player_id = player_id if player_id is not None else state.players.index(player)
    _maybe_apply_map_completion_reward(
        state=state,
        player=player,
        player_id=resolved_player_id,
    )
    return True


def _auto_play_sponsor_from_hand_via_253(
    state: GameState,
    player: PlayerState,
    player_id: int,
) -> bool:
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
    candidates.sort(key=lambda c: (_sponsor_level(c), c.number, c.instance_id))
    card = candidates[0]

    player.sponsor_tokens_by_number[253] = tokens_left - 1
    player.money -= _sponsor_level(card)
    player.hand.remove(card)
    player.zoo_cards.append(card)

    messages = _apply_sponsor_immediate_effects(
        state=state,
        player=player,
        player_id=player_id,
        card=card,
        details={},
    )
    for message in messages:
        state.effect_log.append(f"{card.instance_id}: sponsor_{message}")
    _apply_sponsor_passive_triggers_on_card_play(
        state=state,
        played_by_player_id=player_id,
        played_card=card,
    )
    if card.number in {215, 218}:
        player.sponsor_tokens_by_number[card.number] = player.sponsor_tokens_by_number.get(card.number, 0) + 2
    if card.number == 253:
        player.sponsor_tokens_by_number[253] = player.sponsor_tokens_by_number.get(253, 0) + 3
    state.effect_log.append(f"sponsor_253_auto_play card={card.number} level={_sponsor_level(card)}")
    return True


def _apply_sponsor_passive_triggers_on_card_play(
    state: GameState,
    played_by_player_id: int,
    played_card: AnimalCard,
) -> None:
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
                _increase_reputation(state=state, player=owner, amount=science_icons)
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
            pouched = 0
            for _ in range(australia_icons):
                if not owner.hand:
                    break
                owner.pouched_cards.append(owner.hand.pop(0))
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
                if not _auto_play_sponsor_from_hand_via_253(
                    state=state,
                    player=owner,
                    player_id=owner_id,
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
        pay = level
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
            pay = level + extra_cost
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


def _apply_sponsor_immediate_effects(
    state: GameState,
    player: PlayerState,
    player_id: int,
    card: AnimalCard,
    details: Optional[Dict[str, Any]] = None,
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
        )
        if not placed:
            raise ValueError("No legal placement for this sponsor unique building.")
        messages.append("immediate(unique_building_placed)=1")

    if number in {201, 254}:
        taken = _take_one_card_from_deck_or_reputation_range(state, player)
        messages.append(f"immediate(draw_1_from_deck_or_reputation_range)={1 if taken else 0}")
    if number == 254:
        _increase_reputation(state=state, player=player, amount=1)
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
        _increase_reputation(state=state, player=player, amount=2)
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
        _increase_reputation(state=state, player=player, amount=2)
        messages.append("immediate(+2_reputation)")

    if number == 227:
        selected_mode = str(details.get("sponsor_227_mode") or "").strip().lower()
        if selected_mode not in {"small", "large"}:
            small_hand = sum(1 for c in player.hand if _is_small_animal(c))
            large_hand = sum(1 for c in player.hand if _is_large_animal(c))
            selected_mode = "large" if large_hand >= small_hand else "small"
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
            state.zoo_discard.append(candidate)
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
            before_money = player.money
            _perform_build_action_effect(
                state=state,
                player=player,
                strength=5,
                player_id=player_id,
                details={
                    "selections": [
                        {
                            "building_type": "SIZE_5",
                            "cells": options[0]["cells"],
                        }
                    ],
                    "bonus_action_to_slot_1_targets": [],
                },
            )
            spent = max(0, before_money - player.money)
            player.money += spent
            messages.append("immediate(free_size_5_enclosure_placed=1)")
        else:
            messages.append("immediate(free_size_5_enclosure_placed=0)")

    return messages


def _auto_select_sponsors_for_ai(
    state: GameState,
    player: PlayerState,
    strength: int,
    sponsors_upgraded: bool,
) -> Dict[str, Any]:
    candidates = _list_legal_sponsor_candidates(state=state, player=player, sponsors_upgraded=sponsors_upgraded)
    playable = [cand for cand in candidates if cand["playable_now"]]
    if not playable:
        return {"use_break_ability": True, "sponsor_selections": []}

    cap = strength + 1 if sponsors_upgraded else strength
    selected: List[Dict[str, Any]] = []
    total_level = 0
    for cand in sorted(playable, key=lambda item: (int(item["level"]), -int(item["pay_cost"]))):
        level = int(cand["level"])
        if total_level + level > cap:
            continue
        selected.append(
            {
                "source": cand["source"],
                "source_index": int(cand["source_index"]),
                "card_instance_id": str(cand["card_instance_id"]),
            }
        )
        total_level += level
        if not sponsors_upgraded:
            break
    if not selected:
        return {"use_break_ability": True, "sponsor_selections": []}
    return {"use_break_ability": False, "sponsor_selections": selected}


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
        details = _auto_select_sponsors_for_ai(
            state=state,
            player=player,
            strength=strength,
            sponsors_upgraded=sponsors_upgraded,
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
        ok, reason = _sponsor_requirements_met(
            player=player,
            card=card,
            sponsors_upgraded=sponsors_upgraded,
        )
        if not ok:
            raise ValueError(f"Sponsor requirement not met: {card.name} ({reason})")
        if not _sponsor_unique_building_can_be_placed(state, player, card.number):
            raise ValueError(f"Sponsor requirement not met: {card.name} (no_legal_unique_building_placement)")

    simulated_money = int(player.money)
    simulated_display_ids = [card.instance_id for card in state.zoo_display]
    for source, card, _, _ in resolved_cards:
        if source == "display":
            if card.instance_id not in simulated_display_ids:
                raise ValueError("Selected sponsor card from display is not available.")
            display_index = simulated_display_ids.index(card.instance_id)
            pay_cost = _sponsor_level(card) + display_index + 1
            simulated_display_ids.pop(display_index)
        else:
            pay_cost = _sponsor_level(card)
        if simulated_money < pay_cost:
            raise ValueError("Insufficient money for selected sponsor sequence.")
        simulated_money -= pay_cost

    took_from_display = False
    for source, card, _, _ in resolved_cards:
        if source == "display":
            for idx, candidate in enumerate(state.zoo_display):
                if candidate.instance_id == card.instance_id:
                    display_index = idx
                    break
            else:
                raise ValueError("Selected sponsor card disappeared from display.")
            accessible = _reputation_display_limit(int(player.reputation))
            if display_index + 1 > accessible:
                raise ValueError("Selected sponsor card is outside reputation range.")
            extra_cost = display_index + 1
            pay_cost = _sponsor_level(card) + extra_cost
            if player.money < pay_cost:
                raise ValueError("Insufficient money for sponsor card and display cost.")
            player.money -= pay_cost
            state.zoo_display.pop(display_index)
            took_from_display = True
        else:
            for idx, candidate in enumerate(player.hand):
                if candidate.instance_id == card.instance_id:
                    hand_index = idx
                    break
            else:
                raise ValueError("Selected sponsor card disappeared from hand.")
            pay_cost = _sponsor_level(card)
            if player.money < pay_cost:
                raise ValueError("Insufficient money for sponsor card.")
            player.money -= pay_cost
            player.hand.pop(hand_index)

        player.zoo_cards.append(card)
        effect_messages = _apply_sponsor_immediate_effects(
            state=state,
            player=player,
            player_id=player_id,
            card=card,
            details=details,
        )
        for message in effect_messages:
            state.effect_log.append(f"{card.instance_id}: sponsor_{message}")
        _apply_sponsor_passive_triggers_on_card_play(
            state=state,
            played_by_player_id=player_id,
            played_card=card,
        )

        if card.number in {215, 218}:
            player.sponsor_tokens_by_number[card.number] = player.sponsor_tokens_by_number.get(card.number, 0) + 2
        if card.number == 253:
            player.sponsor_tokens_by_number[253] = player.sponsor_tokens_by_number.get(253, 0) + 3

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
    cost = _sponsor_level(card) if card.card_type == "sponsor" else card.cost
    return (
        f"{card_no} [{card.card_type}] {card.name} "
        f"(cost={cost}, size={card.size}, appeal={card.appeal}, rep={card.reputation_gain}, "
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


def _prompt_sponsors_action_details_for_human(
    state: GameState,
    player: PlayerState,
    strength: int,
) -> Dict[str, Any]:
    return _prompt_sponsors_action_details_for_human_impl(
        state=state,
        player=player,
        strength=strength,
        sponsor_candidates_fn=_list_legal_sponsor_candidates,
        format_card_line_fn=_format_card_line,
    )


def _resolve_manual_opening_drafts(state: GameState, manual_player_names: Set[str]) -> None:
    _resolve_manual_opening_drafts_impl(
        state=state,
        manual_player_names=manual_player_names,
        draw_opening_draft_cards_fn=_draw_opening_draft_cards,
        apply_opening_draft_selection_fn=_apply_opening_draft_selection,
        prompt_opening_draft_indices_fn=_prompt_opening_draft_indices,
    )


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


def apply_action(state: GameState, action: Action) -> None:
    _validate_card_zones(state)
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
    elif action.type == ActionType.X_TOKEN:
        if int(player.x_tokens) >= 5:
            raise ValueError("Cannot take X action at max X-token limit.")
        chosen = action.card_name or player.action_order[-1]
        _rotate_action_card_to_slot_1(player, chosen)
        player.x_tokens = min(5, int(player.x_tokens) + 1)
    else:
        raise ValueError("Unsupported action type in this runner.")

    if break_triggered:
        _resolve_break(state)

    _apply_end_of_turn_venom_penalty(player, consumed_venom=consumed_venom)
    _validate_card_zones(state)
    state.turn_index = int(state.turn_index) + 1
    state.current_player = (int(state.current_player) + 1) % len(state.players)


def evaluate_player(player: PlayerState) -> float:
    free_capacity = 0
    for enclosure in player.enclosures:
        if _enclosure_is_standard(enclosure):
            if not _enclosure_has_animals(enclosure):
                free_capacity += enclosure.size
            continue
        if enclosure.enclosure_type in {"reptile_house", "large_bird_aviary"}:
            free_capacity += _enclosure_remaining_capacity(enclosure)
    best_hand_value = 0.0
    for card in player.hand:
        value = card.appeal + card.conservation * 4 - card.cost * 0.2
        best_hand_value = max(best_hand_value, value)
    return (
        player.appeal
        + player.conservation * 4.5
        + player.money * 0.25
        + free_capacity * 0.4
        + best_hand_value * 0.3
    )


def evaluate_state_for_player(state: GameState, player_index: int) -> float:
    me = state.players[player_index]
    others = [p for idx, p in enumerate(state.players) if idx != player_index]
    my_score = evaluate_player(me)
    opp_score = sum(evaluate_player(o) for o in others) / max(len(others), 1)
    return my_score - 0.35 * opp_score


def _turns_remaining_for_player(state: GameState, player_index: int) -> int:
    # Estimate how many actions the same player will still take.
    total_players = len(state.players)
    turns_left = state.total_turn_limit - state.turn_index
    return max(0, (turns_left + total_players - 1 - player_index) // total_players)


def _action_bonus_for_heuristic(state: GameState, action: Action, player_index: int) -> float:
    player = state.players[player_index]
    turns_left = _turns_remaining_for_player(state, player_index)

    if action.type == ActionType.MAIN_ACTION and action.card_name:
        if action.card_name == "cards":
            playable_now = any(
                _animal_size_restriction_met(player, card)
                and _animal_play_cost(player, card) <= player.money
                and (
                    _animal_icon_missing_conditions(player, card)
                    <= _animal_condition_ignore_capacity(player, card)
                )
                and any(
                    _enclosure_can_host_animal_with_ignores(
                        player=player,
                        enclosure=e,
                        card=card,
                        ignore_conditions=max(
                            0,
                            _animal_condition_ignore_capacity(player, card)
                            - _animal_icon_missing_conditions(player, card),
                        ),
                    )
                    for e in player.enclosures
                )
                for card in player.hand
                if card.card_type == "animal"
            )
            return 0.2 if playable_now else 0.8
        if action.card_name == "build":
            return 0.6
        if action.card_name == "animals":
            return 0.7
        if action.card_name == "association":
            return 0.5 if turns_left > 2 else 0.2
        if action.card_name == "sponsors":
            return 0.6
    if action.type == ActionType.X_TOKEN:
        return 0.4 if player.x_tokens < 5 else -10.0

    return 0.0


class AIPlayer:
    def choose_action(self, state: GameState, actions: List[Action]) -> Action:
        raise NotImplementedError


class HumanPlayer(AIPlayer):
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
        player = state.players[state.current_player]
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


class RandomAI(AIPlayer):
    def choose_action(self, state: GameState, actions: List[Action]) -> Action:
        return random.choice(actions)


class HeuristicAI(AIPlayer):
    @staticmethod
    def _default_opponent_action(state: GameState) -> Action:
        player = state.players[state.current_player]
        actions = legal_actions(player, state=state, player_id=state.current_player)
        return actions[0]

    def _two_ply_score(self, state: GameState, action: Action, player_index: int) -> float:
        sim = copy.deepcopy(state)
        apply_action(sim, action)
        immediate = evaluate_state_for_player(sim, player_index) + _action_bonus_for_heuristic(
            state, action, player_index
        )

        # Advance to this AI player's next turn with a simple opponent policy.
        while not sim.game_over() and sim.current_player != player_index:
            opp_action = self._default_opponent_action(sim)
            apply_action(sim, opp_action)

        if sim.game_over():
            return immediate

        follow_actions = legal_actions(sim.players[player_index], state=sim, player_id=player_index)
        if not follow_actions:
            return immediate

        best_follow = float("-inf")
        for follow_action in follow_actions:
            sim2 = copy.deepcopy(sim)
            apply_action(sim2, follow_action)
            score = evaluate_state_for_player(sim2, player_index) + _action_bonus_for_heuristic(
                sim, follow_action, player_index
            )
            if score > best_follow:
                best_follow = score

        return immediate * 0.45 + best_follow * 0.55

    def choose_action(self, state: GameState, actions: List[Action]) -> Action:
        player_index = state.current_player
        best_score = float("-inf")
        best_actions: List[Action] = []

        for action in actions:
            score = self._two_ply_score(state, action, player_index)
            if score > best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        return random.choice(best_actions)


def setup_game(
    seed: int = 42,
    player_names: Optional[List[str]] = None,
    manual_opening_draft_player_names: Optional[Set[str]] = None,
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
        draw_final_scoring_cards_for_players_fn=_draw_final_scoring_cards_for_players,
        build_opening_setup_info_fn=_build_opening_setup_info,
        load_base_setup_card_pools_fn=_load_base_setup_card_pools,
        refresh_association_market=_refresh_association_market,
        draw_opening_draft_cards_fn=_draw_opening_draft_cards,
        apply_opening_draft_selection_fn=_apply_opening_draft_selection,
        replenish_zoo_display=_replenish_zoo_display,
        validate_card_zones=_validate_card_zones,
        break_track_by_players=BREAK_TRACK_BY_PLAYERS,
        max_turns_per_player=16,
    )
    state.map_rules = _load_map_rules(state.map_image_name)
    state.map_tile_tags = _build_map_tile_tags_map(state.map_image_name)
    return state


def play_game(
    seed: int = 42,
    verbose: bool = True,
    agents: Optional[Dict[str, AIPlayer]] = None,
    player_names: Optional[List[str]] = None,
) -> Dict[str, int]:
    if agents is None:
        agents = {
            "HeuristicAI": HeuristicAI(),
            "RandomAI": RandomAI(),
        }
    if player_names is None:
        player_names = ["HeuristicAI", "RandomAI"]
    manual_draft_players = {name for name in player_names if isinstance(agents.get(name), HumanPlayer)}
    state = setup_game(
        seed=seed,
        player_names=player_names,
        manual_opening_draft_player_names=manual_draft_players,
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

    if manual_draft_players:
        _resolve_manual_opening_drafts(state, manual_draft_players)

    if verbose:
        print("- Opening 8-card draft per player:")
        for player in state.players:
            print(f"  {player.name}:")
            for idx, card in enumerate(player.opening_draft_drawn, start=1):
                mark = " [KEEP]" if (idx - 1) in set(player.opening_draft_kept_indices) else ""
                print(f"    {idx}. {_format_card_line(card)}{mark}")
        print()

    while not state.game_over():
        player = state.players[state.current_player]
        actions = legal_actions(player, state=state, player_id=state.current_player)
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
        print("\nFinal score:")
        for name, score in scores.items():
            print(f"- {name}: {score}")
        winner = max(scores.items(), key=lambda x: x[1])
        print(f"Winner: {winner[0]}")

    return scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simplified Ark Nova prototype runner.")
    parser.add_argument(
        "--mode",
        choices=("ai", "pvp"),
        default="ai",
        help="ai: HeuristicAI vs RandomAI; pvp: Player1 vs Player2 interactive",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--quiet", action="store_true", help="Reduce turn-by-turn output")
    return parser.parse_args()


def main_cli() -> None:
    args = parse_args()
    verbose = not args.quiet
    if args.mode == "pvp":
        play_game(
            seed=args.seed,
            verbose=verbose,
            agents={"Player1": HumanPlayer(), "Player2": HumanPlayer()},
            player_names=["Player1", "Player2"],
        )
    else:
        play_game(seed=args.seed, verbose=verbose)


if __name__ == "__main__":
    main_cli()

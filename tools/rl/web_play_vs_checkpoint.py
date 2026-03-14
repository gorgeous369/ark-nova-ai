"""Serve a local browser UI for human-vs-checkpoint Ark Nova play."""

from __future__ import annotations

import argparse
import copy
from functools import lru_cache
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse
import sys
from threading import Lock
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main

from arknova_rl.config import SUPPORTED_EXPLICIT_DEVICE_HELP, normalize_torch_device_spec
from arknova_rl.evaluator import PolicyBundle, load_policy_bundle_from_checkpoint


CARD_IMAGE_ROOT = REPO_ROOT / "data" / "cards" / "images" / "cards"
CARD_IMAGE_MANIFEST_PATH = CARD_IMAGE_ROOT / "manifest.json"
SETUP_CARD_DATA_ID_RE = re.compile(r"^([FP])(\d{3})")
CARD_IMAGE_CATEGORY_ALIASES = {
    "animal": ("animal", "animals"),
    "animals": ("animal", "animals"),
    "sponsor": ("sponsor", "sponsors"),
    "sponsors": ("sponsor", "sponsors"),
    "project": ("project", "projects", "conservation_project", "conservation_projects"),
    "projects": ("project", "projects", "conservation_project", "conservation_projects"),
    "conservation_project": ("project", "projects", "conservation_project", "conservation_projects"),
    "conservation_projects": ("project", "projects", "conservation_project", "conservation_projects"),
    "endgame": ("endgame",),
}
MAP_IMAGE_ROOT = REPO_ROOT / "data" / "maps" / "images"
MAP_TILE_ROOT = REPO_ROOT / "data" / "maps" / "tiles"
MAP_IMAGE_WIDTH = 4028
MAP_IMAGE_HEIGHT = 2126
MAP_X0 = 270.0
MAP_Y0 = 320.0
MAP_STEP_X = 430.0
MAP_STEP_Y = 210.0
MAP_HEX_RADIUS = 105.0


def _parse_device_arg(raw_value: str) -> str:
    try:
        return normalize_torch_device_spec(raw_value, allow_auto=False)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


class CheckpointPolicy:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: torch.device,
        deterministic: bool,
        name: str,
    ) -> None:
        self.device = device
        self.deterministic = bool(deterministic)
        self.bundle: PolicyBundle = load_policy_bundle_from_checkpoint(
            Path(checkpoint_path),
            device=device,
            name=name,
        )
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def reset_hidden(self) -> None:
        self._hidden = None

    def _ensure_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._hidden is None:
            self._hidden = self.bundle.model.init_hidden(1, device=self.device)
        return self._hidden

    def choose_action(self, state: main.GameState, actions: Sequence[main.Action]) -> main.Action:
        actor_id = (
            int(state.pending_decision_player_id)
            if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None
            else int(state.current_player)
        )
        if not actions:
            raise RuntimeError("CheckpointPolicy received no legal actions.")

        state_vec = self.bundle.obs_encoder.encode_from_state(state, actor_id)
        action_features = self.bundle.action_encoder.encode_many(actions)
        action_mask = np.ones((len(actions),), dtype=np.float32)

        state_t = torch.as_tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
        action_t = torch.as_tensor(action_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.as_tensor(action_mask, dtype=torch.float32, device=self.device).unsqueeze(0)

        hidden = self._ensure_hidden()
        with torch.no_grad():
            logits, _, next_hidden = self.bundle.model.forward_step(
                state_vec=state_t,
                action_features=action_t,
                action_mask=mask_t,
                hidden=hidden,
            )
        self._hidden = (
            next_hidden[0].detach(),
            next_hidden[1].detach(),
        )

        if self.deterministic:
            chosen_index = int(torch.argmax(logits[0]).item())
        else:
            distribution = torch.distributions.Categorical(logits=logits[0])
            chosen_index = int(distribution.sample().item())
        return actions[chosen_index]


def _current_actor_id(state: main.GameState) -> int:
    if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None:
        return int(state.pending_decision_player_id)
    return int(state.current_player)


def _serialize_action(action: main.Action, *, index: int) -> Dict[str, Any]:
    details = dict(action.details or {})
    return {
        "index": int(index),
        "type": str(action.type.value),
        "card_name": str(action.card_name or ""),
        "value": 0 if action.value is None else int(action.value),
        "rendered": str(action),
        "action_label": str(details.get("action_label") or ""),
        "concrete": bool(details.get("concrete")),
        "pending_kind": str(details.get("pending_kind") or ""),
    }


@lru_cache(maxsize=1)
def _card_image_manifest_index() -> Dict[Tuple[str, int], Path]:
    if not CARD_IMAGE_MANIFEST_PATH.is_file():
        return {}
    try:
        payload = json.loads(CARD_IMAGE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    index: Dict[Tuple[str, int], Path] = {}
    for item in payload.get("cards", []):
        if not isinstance(item, dict):
            continue
        number_text = str(item.get("number") or "").strip()
        local_path = str(item.get("local_path") or "").strip()
        if not number_text.isdigit() or not local_path:
            continue
        image_path = CARD_IMAGE_ROOT / local_path
        if not image_path.is_file():
            continue
        number = int(number_text)
        category_names = {
            str(item.get("category") or "").strip().lower(),
            str(item.get("output_category") or "").strip().lower(),
        }
        for category_name in category_names:
            if not category_name:
                continue
            index[(category_name, number)] = image_path
    return index


def _resolve_card_image_file(category: str, number: Any) -> Optional[Path]:
    try:
        card_number = int(number)
    except (TypeError, ValueError):
        return None
    if card_number <= 0:
        return None
    category_name = str(category or "").strip().lower()
    if not category_name:
        return None
    index = _card_image_manifest_index()
    for alias in CARD_IMAGE_CATEGORY_ALIASES.get(category_name, (category_name,)):
        image_path = index.get((alias, card_number))
        if image_path is not None:
            return image_path
    return None


def _resolve_setup_card_image_file(data_id: str) -> Optional[Path]:
    match = SETUP_CARD_DATA_ID_RE.match(str(data_id or "").strip())
    if match is None:
        return None
    prefix = str(match.group(1))
    number = int(match.group(2))
    if prefix == "F":
        return _resolve_card_image_file("endgame", number)
    if prefix == "P":
        return _resolve_card_image_file("project", number)
    return None


@lru_cache(maxsize=64)
def _load_map_tiles_payload_local(map_image_name: str) -> Dict[str, Any]:
    image_name = str(map_image_name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]+", image_name):
        raise ValueError(f"Unsupported map image name: {map_image_name!r}")
    path = MAP_TILE_ROOT / f"{image_name}.tiles.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Map tile payload must be an object: {path}")
    return payload


def _resolve_map_image_file(map_image_name: str) -> Optional[Path]:
    image_name = str(map_image_name or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9]+", image_name):
        return None
    image_path = MAP_IMAGE_ROOT / f"{image_name}.jpg"
    if not image_path.is_file():
        return None
    return image_path


def _resolve_map_image_request_path(path: str) -> Optional[Path]:
    normalized_path = str(urlparse(path).path)
    prefix = "/assets/maps/"
    if not normalized_path.startswith(prefix):
        return None
    filename = normalized_path[len(prefix):]
    if not filename.endswith(".jpg"):
        return None
    image_name = filename[:-4]
    return _resolve_map_image_file(image_name)


def _map_tile_center(x: int, y: int) -> Tuple[float, float]:
    return (
        MAP_X0 + (int(x) + int(y)) * MAP_STEP_X,
        MAP_Y0 + (int(x) - int(y)) * (MAP_STEP_Y / 2.0),
    )


def _map_hex_points(cx: float, cy: float) -> str:
    points: List[str] = []
    for corner_index in range(6):
        angle = math.radians(30 + corner_index * 60)
        px = cx + MAP_HEX_RADIUS * math.cos(angle)
        py = cy + MAP_HEX_RADIUS * math.sin(angle)
        points.append(f"{px:.2f},{py:.2f}")
    return " ".join(points)


def _map_bonus_short_label(bonus: str) -> str:
    value = str(bonus or "").strip()
    if not value:
        return ""
    known = {
        "x_token": "X",
        "5coins": "+5",
        "reputation": "Rep",
        "card_in_reputation_range": "Card",
        "action_to_slot_1": "Slot1",
    }
    return known.get(value, value.replace("_", " ")[:8])


def _map_building_short_label(building_type: str, subtype: str) -> str:
    type_name = str(building_type or "").strip().upper()
    subtype_name = str(subtype or "").strip().lower()
    if type_name.startswith("SIZE_"):
        return f"E{type_name.split('_', 1)[1]}"
    if type_name == "PETTING_ZOO":
        return "PZ"
    if type_name == "REPTILE_HOUSE":
        return "RH"
    if type_name == "LARGE_BIRD_AVIARY":
        return "BA"
    if type_name == "KIOSK":
        return "K"
    if type_name == "PAVILION":
        return "P"
    if subtype_name == "sponsor_building":
        return "SB"
    return type_name[:3] or "B"


def _normalize_coord_pair(value: Any) -> Optional[Tuple[int, int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x_raw, y_raw = value
    try:
        return int(x_raw), int(y_raw)
    except (TypeError, ValueError):
        return None


def _build_map_view_model(public_observation: Dict[str, Any], *, human_player_id: int) -> Dict[str, Any]:
    players = public_observation.get("players", [])
    if not isinstance(players, list) or int(human_player_id) < 0 or int(human_player_id) >= len(players):
        return {}

    player = players[int(human_player_id)]
    if not isinstance(player, dict) or not bool(player.get("zoo_map_present")):
        return {}

    map_image_name = str(public_observation.get("map_image_name") or "").strip()
    if not map_image_name:
        return {}

    payload = _load_map_tiles_payload_local(map_image_name)
    tiles_raw = payload.get("tiles", [])
    if not isinstance(tiles_raw, list):
        return {}

    enclosure_lookup: Dict[Tuple[int, int, str], Dict[str, Any]] = {}
    for enclosure in player.get("enclosures", []) or []:
        if not isinstance(enclosure, dict):
            continue
        origin = _normalize_coord_pair(enclosure.get("origin"))
        if origin is None:
            continue
        key = (int(origin[0]), int(origin[1]), str(enclosure.get("rotation") or ""))
        enclosure_lookup[key] = enclosure

    coverage_by_coord: Dict[Tuple[int, int], Dict[str, Any]] = {}
    covered_coords: set[Tuple[int, int]] = set()
    for building in player.get("zoo_map_buildings", []) or []:
        if not isinstance(building, dict):
            continue
        building_type = str(building.get("building_type") or "")
        subtype = str(building.get("subtype") or "")
        origin = _normalize_coord_pair(building.get("origin"))
        rotation = str(building.get("rotation") or "")
        enclosure = None
        if origin is not None:
            enclosure = enclosure_lookup.get((int(origin[0]), int(origin[1]), rotation))
        occupied = bool(enclosure.get("occupied")) if isinstance(enclosure, dict) else False
        used_capacity = int(enclosure.get("used_capacity", 0)) if isinstance(enclosure, dict) else 0
        animal_capacity = int(enclosure.get("animal_capacity", 0)) if isinstance(enclosure, dict) else 0
        label = _map_building_short_label(building_type, subtype)
        for cell in building.get("layout", []) or []:
            coord = _normalize_coord_pair(cell)
            if coord is None:
                continue
            coord_key = (int(coord[0]), int(coord[1]))
            covered_coords.add(coord_key)
            coverage_by_coord[coord_key] = {
                "covered": True,
                "covered_by": building_type,
                "covered_subtype": subtype,
                "covered_label": label,
                "occupied": occupied,
                "used_capacity": used_capacity,
                "animal_capacity": animal_capacity,
            }

    tiles: List[Dict[str, Any]] = []
    for raw_tile in tiles_raw:
        if not isinstance(raw_tile, dict):
            continue
        x = raw_tile.get("x")
        y = raw_tile.get("y")
        if not isinstance(x, int) or not isinstance(y, int):
            continue
        cx, cy = _map_tile_center(x, y)
        coverage = coverage_by_coord.get((int(x), int(y)), {})
        placement_bonus = str(raw_tile.get("placement_bonus") or "")
        tags = [
            str(tag)
            for tag in (raw_tile.get("tags") or [])
            if str(tag).strip()
        ]
        tiles.append(
            {
                "x": int(x),
                "y": int(y),
                "center_x": float(cx),
                "center_y": float(cy),
                "points": _map_hex_points(cx, cy),
                "terrain": str(raw_tile.get("terrain") or "plain"),
                "build2_required": bool(raw_tile.get("build2_required")),
                "placement_bonus": placement_bonus,
                "placement_bonus_short": _map_bonus_short_label(placement_bonus),
                "tags": tags,
                "covered": bool(coverage.get("covered")),
                "covered_by": str(coverage.get("covered_by") or ""),
                "covered_subtype": str(coverage.get("covered_subtype") or ""),
                "covered_label": str(coverage.get("covered_label") or ""),
                "occupied": bool(coverage.get("occupied")),
                "used_capacity": int(coverage.get("used_capacity", 0)),
                "animal_capacity": int(coverage.get("animal_capacity", 0)),
            }
        )

    return {
        "map_name": str(payload.get("map_name") or map_image_name),
        "image_name": map_image_name,
        "image_url": f"/assets/maps/{map_image_name}.jpg",
        "image_width": MAP_IMAGE_WIDTH,
        "image_height": MAP_IMAGE_HEIGHT,
        "map_effects": [
            str(item)
            for item in (payload.get("map_effects") or [])
            if str(item).strip()
        ],
        "tiles": tiles,
        "covered_count": len(covered_coords),
    }


def _resolve_card_image_request_path(path: str) -> Optional[Path]:
    normalized_path = str(urlparse(path).path)
    prefix = "/assets/cards/"
    if not normalized_path.startswith(prefix):
        return None
    remainder = normalized_path[len(prefix):]
    parts = remainder.split("/")
    if len(parts) != 2:
        return None
    category_name, filename = parts
    if not filename.endswith(".jpg"):
        return None
    number_text = filename[:-4]
    if not number_text.isdigit():
        return None
    return _resolve_card_image_file(category_name, int(number_text))


class HumanVsCheckpointSession:
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: torch.device,
        human_seat: int,
        deterministic: bool,
        seed: int,
        include_marine_world: bool,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.human_player_id = int(human_seat)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.include_marine_world = bool(include_marine_world)
        self.human_name = "You"
        self.ai_name = "AI"
        self.lock = Lock()
        self.ai = CheckpointPolicy(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            deterministic=self.deterministic,
            name=self.ai_name,
        )
        self.state: Optional[main.GameState] = None
        self.action_log: List[Dict[str, Any]] = []
        self.restart(seed=self.seed)

    def restart(self, *, seed: Optional[int] = None) -> Dict[str, Any]:
        with self.lock:
            if seed is not None:
                self.seed = int(seed)
            self.ai.reset_hidden()
            if self.human_player_id == 0:
                player_names = [self.human_name, self.ai_name]
            else:
                player_names = [self.ai_name, self.human_name]
            self.state = main.setup_game(
                seed=self.seed,
                player_names=player_names,
                manual_opening_draft_player_names={self.human_name},
                include_marine_world=self.include_marine_world,
            )
            self.action_log = []
            self._advance_until_human_turn_locked()
            return self._snapshot_locked()

    def submit_action(self, action_index: int) -> Dict[str, Any]:
        with self.lock:
            state = self._require_state()
            if state.game_over():
                raise ValueError("Game is already over.")
            actor_id = _current_actor_id(state)
            if actor_id != self.human_player_id:
                raise ValueError("It is not the human player's turn.")
            actions = main.legal_actions(state.players[actor_id], state=state, player_id=actor_id)
            picked_index = int(action_index)
            if picked_index < 0 or picked_index >= len(actions):
                raise ValueError("Selected action index is out of range.")
            chosen = actions[picked_index]
            self._apply_action_locked(actor_id=actor_id, action=chosen)
            self._advance_until_human_turn_locked()
            return self._snapshot_locked()

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return self._snapshot_locked()

    def _require_state(self) -> main.GameState:
        if self.state is None:
            raise RuntimeError("Session state is not initialized.")
        return self.state

    def _apply_action_locked(self, *, actor_id: int, action: main.Action) -> None:
        state = self._require_state()
        effect_log_cursor = len(state.effect_log)
        player = state.players[int(actor_id)]
        main.apply_action(state, action)
        self.action_log.append(
            {
                "turn_index": int(state.turn_index),
                "player_id": int(actor_id),
                "player_name": str(player.name),
                "action": str(action),
                "effects": [str(item) for item in state.effect_log[effect_log_cursor:]],
            }
        )
        if len(self.action_log) > 80:
            self.action_log = self.action_log[-80:]

    def _advance_until_human_turn_locked(self) -> None:
        state = self._require_state()
        while not state.game_over():
            actor_id = _current_actor_id(state)
            if actor_id == self.human_player_id:
                return
            actor = state.players[actor_id]
            actions = main.legal_actions(actor, state=state, player_id=actor_id)
            chosen = self.ai.choose_action(state, actions)
            self._apply_action_locked(actor_id=actor_id, action=chosen)

    def _snapshot_locked(self) -> Dict[str, Any]:
        state = self._require_state()
        actor_id = _current_actor_id(state) if not state.game_over() else None
        observation = main.build_player_observation(state, viewer_player_id=self.human_player_id)
        map_view = _build_map_view_model(
            observation.get("public", {}),
            human_player_id=self.human_player_id,
        )
        display_hidden = (
            str(observation.get("public", {}).get("pending_decision_kind") or "").strip()
            == "opening_draft_keep"
        )
        if display_hidden:
            observation = copy.deepcopy(observation)
            public_observation = observation.setdefault("public", {})
            public_observation["zoo_display"] = []
        legal_actions: List[Dict[str, Any]] = []
        if actor_id == self.human_player_id and not state.game_over():
            actions = main.legal_actions(
                state.players[self.human_player_id],
                state=state,
                player_id=self.human_player_id,
            )
            legal_actions = [
                _serialize_action(action, index=index)
                for index, action in enumerate(actions)
            ]
        scores = None
        if state.game_over():
            scores = {
                str(player.name): int(main._final_score_points(state, player))
                for player in state.players
            }
        return {
            "seed": int(self.seed),
            "checkpoint": str(self.checkpoint_path),
            "human_player_id": int(self.human_player_id),
            "human_name": str(self.human_name),
            "ai_name": str(self.ai_name),
            "game_over": bool(state.game_over()),
            "human_turn": bool(actor_id == self.human_player_id and not state.game_over()),
            "display_hidden": bool(display_hidden),
            "current_actor_id": None if actor_id is None else int(actor_id),
            "current_actor_name": (
                ""
                if actor_id is None
                else str(state.players[int(actor_id)].name)
            ),
            "observation": observation,
            "map_view": map_view,
            "legal_actions": legal_actions,
            "scores": scores,
            "recent_actions": list(self.action_log[-24:]),
        }


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Ark Nova Arena</title>
  <style>
    :root {
      --bg: #10271e;
      --bg-soft: #173529;
      --bg-ink: #0a1612;
      --panel: rgba(12, 29, 23, 0.82);
      --panel-strong: rgba(7, 18, 14, 0.9);
      --line: rgba(140, 212, 176, 0.2);
      --line-strong: rgba(165, 232, 198, 0.45);
      --text: #edf7ef;
      --muted: #a9cdb8;
      --accent: #f5b94c;
      --accent-2: #76d0a4;
      --danger: #ee7d73;
      --shadow: 0 20px 60px rgba(0, 0, 0, 0.35);
      --radius: 20px;
      --card-radius: 18px;
      --chip: rgba(255, 255, 255, 0.08);
      --hex-size: 34px;
      font-family: "Avenir Next", "Trebuchet MS", "Segoe UI", sans-serif;
      color: var(--text);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at top left, rgba(118, 208, 164, 0.25), transparent 22rem),
        radial-gradient(circle at top right, rgba(245, 185, 76, 0.18), transparent 18rem),
        linear-gradient(160deg, #0c1915 0%, #10271e 45%, #132d23 100%);
      color: var(--text);
    }
    body::before {
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(255,255,255,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(255,255,255,0.025) 1px, transparent 1px);
      background-size: 32px 32px;
      pointer-events: none;
      opacity: 0.28;
    }
    .app {
      position: relative;
      padding: 20px;
      display: grid;
      gap: 18px;
    }
    .hero {
      display: grid;
      grid-template-columns: 1.3fr 0.9fr;
      gap: 18px;
      align-items: stretch;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      box-shadow: var(--shadow);
      backdrop-filter: blur(18px);
    }
    .hero-main {
      padding: 20px 22px;
      display: grid;
      gap: 16px;
    }
    .hero-title {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: start;
    }
    .title-block h1 {
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      font-size: clamp(30px, 4vw, 46px);
      letter-spacing: 0.02em;
    }
    .title-block p {
      margin: 6px 0 0;
      color: var(--muted);
    }
    .hero-controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
    }
    .hero-controls input,
    .hero-controls button,
    .action-button {
      border-radius: 999px;
      border: 1px solid var(--line-strong);
      background: rgba(255, 255, 255, 0.06);
      color: var(--text);
      padding: 10px 14px;
      font: inherit;
    }
    .hero-controls button,
    .action-button {
      cursor: pointer;
      transition: transform 140ms ease, background 140ms ease, border-color 140ms ease;
    }
    .hero-controls button:hover,
    .action-button:hover {
      transform: translateY(-1px);
      background: rgba(118, 208, 164, 0.14);
    }
    .hero-controls button.primary,
    .action-button.primary {
      background: linear-gradient(135deg, rgba(245,185,76,0.32), rgba(118,208,164,0.28));
      border-color: rgba(245, 185, 76, 0.5);
    }
    .status-strip {
      display: flex;
      gap: 14px;
      flex-wrap: wrap;
    }
    .status-chip {
      display: inline-flex;
      gap: 8px;
      align-items: center;
      background: var(--chip);
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--muted);
    }
    .status-chip strong {
      color: var(--text);
      font-weight: 700;
    }
    .break-track {
      height: 16px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.08);
      overflow: hidden;
      border: 1px solid var(--line);
      position: relative;
    }
    .break-fill {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), #ffd970);
      transition: width 180ms ease;
    }
    .hero-side {
      padding: 18px;
      display: grid;
      gap: 12px;
    }
    .player-stack {
      display: grid;
      gap: 12px;
    }
    .player-card {
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      display: grid;
      gap: 10px;
    }
    .player-card.active {
      border-color: rgba(245, 185, 76, 0.55);
      box-shadow: inset 0 0 0 1px rgba(245, 185, 76, 0.18);
    }
    .player-row {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      align-items: baseline;
    }
    .player-name {
      font-size: 18px;
      font-weight: 700;
    }
    .player-role {
      color: var(--muted);
      font-size: 13px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }
    .stat {
      padding: 10px;
      border-radius: 14px;
      background: rgba(255, 255, 255, 0.04);
      border: 1px solid rgba(255, 255, 255, 0.05);
    }
    .stat-label {
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .layout {
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(300px, 0.85fr);
      gap: 18px;
    }
    .main-column,
    .side-column {
      display: grid;
      gap: 18px;
      align-content: start;
    }
    .section {
      padding: 18px;
      display: grid;
      gap: 14px;
    }
    .section-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }
    .section-title {
      margin: 0;
      font-size: 18px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
    }
    .section-subtitle {
      color: var(--muted);
      font-size: 13px;
    }
    .card-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
      gap: 12px;
    }
    .mini-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(128px, 1fr));
      gap: 10px;
    }
    .game-card {
      padding: 12px;
      border-radius: var(--card-radius);
      border: 1px solid rgba(255, 255, 255, 0.12);
      background:
        linear-gradient(180deg, rgba(255,255,255,0.09), rgba(255,255,255,0.04)),
        rgba(11, 24, 19, 0.78);
      display: grid;
      gap: 8px;
      min-height: 152px;
    }
    .card-art {
      width: 100%;
      aspect-ratio: 0.72;
      object-fit: cover;
      border-radius: 14px;
      border: 1px solid rgba(255, 255, 255, 0.08);
      background: rgba(255, 255, 255, 0.04);
      box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.04);
    }
    .game-card .topline {
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: start;
    }
    .badge-row {
      display: flex;
      gap: 6px;
      flex-wrap: wrap;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      border-radius: 999px;
      padding: 4px 8px;
      background: rgba(255, 255, 255, 0.08);
      color: var(--muted);
      font-size: 12px;
    }
    .game-card h3 {
      margin: 0;
      font-size: 16px;
      line-height: 1.2;
    }
    .game-card p {
      margin: 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .resource-line {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 12px;
    }
    .draft-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(210px, 1fr));
      gap: 12px;
    }
    .draft-note {
      color: var(--muted);
      font-size: 13px;
    }
    .map-shell {
      overflow: auto;
      padding: 12px;
      border-radius: 20px;
      background:
        radial-gradient(circle at top, rgba(118,208,164,0.16), transparent 18rem),
        rgba(7, 17, 14, 0.85);
      border: 1px solid var(--line);
      min-height: 320px;
    }
    .map-canvas {
      min-width: 720px;
    }
    .map-svg {
      display: block;
      width: 100%;
      height: auto;
      min-width: 720px;
      border-radius: 16px;
      background:
        radial-gradient(circle at top, rgba(118, 208, 164, 0.12), transparent 28rem),
        linear-gradient(180deg, rgba(10, 25, 20, 0.96), rgba(6, 14, 11, 0.98));
      overflow: hidden;
    }
    .map-tile {
      cursor: pointer;
      transition: fill 120ms ease, stroke 120ms ease, stroke-width 120ms ease, opacity 120ms ease;
      stroke-linejoin: round;
    }
    .map-tile.terrain-plain {
      fill: rgba(255, 255, 255, 0.03);
      stroke: rgba(243, 248, 245, 0.26);
      stroke-width: 4;
    }
    .map-tile.terrain-rock {
      fill: rgba(139, 94, 60, 0.28);
      stroke: rgba(255, 240, 224, 0.26);
      stroke-width: 4;
    }
    .map-tile.terrain-water {
      fill: rgba(29, 155, 240, 0.24);
      stroke: rgba(220, 243, 255, 0.26);
      stroke-width: 4;
    }
    .map-tile.build2 {
      stroke: rgba(234, 82, 147, 0.95);
      stroke-width: 8;
    }
    .map-tile.covered {
      fill: rgba(11, 20, 16, 0.52);
    }
    .map-tile.covered.subtype-kiosk {
      fill: rgba(118, 208, 164, 0.42);
    }
    .map-tile.covered.subtype-pavilion {
      fill: rgba(112, 173, 255, 0.42);
    }
    .map-tile.covered.subtype-enclosure_special {
      fill: rgba(233, 120, 176, 0.36);
    }
    .map-tile.covered.occupied {
      fill: rgba(245, 185, 76, 0.34);
    }
    .map-tile.selected {
      stroke: rgba(245, 185, 76, 0.98);
      stroke-width: 10;
    }
    .map-label {
      pointer-events: none;
      fill: rgba(242, 247, 244, 0.92);
      font-family: "Avenir Next", "Trebuchet MS", sans-serif;
      text-anchor: middle;
    }
    .map-label.covered {
      fill: rgba(255, 251, 240, 0.98);
      font-weight: 700;
    }
    .map-coord {
      font-size: 24px;
    }
    .map-bonus {
      font-size: 24px;
      fill: rgba(255, 244, 196, 0.98);
    }
    .map-build2 {
      font-size: 22px;
      fill: rgba(255, 140, 190, 0.98);
      font-weight: 700;
    }
    .map-cover {
      font-size: 26px;
    }
    .map-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .legend-chip {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.05);
      color: var(--muted);
      font-size: 12px;
    }
    .legend-swatch {
      width: 12px;
      height: 12px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.22);
    }
    .map-details {
      display: grid;
      gap: 8px;
      padding: 14px;
      border-radius: 18px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
    }
    .map-details strong {
      font-size: 16px;
      color: var(--text);
    }
    .map-details .detail-line {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
    }
    .map-caption {
      color: var(--muted);
      font-size: 13px;
    }
    .actions-panel {
      display: grid;
      gap: 12px;
    }
    .actions-toolbar {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .actions-toolbar input {
      width: 100%;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.05);
      color: var(--text);
      padding: 12px 14px;
      font: inherit;
    }
    .action-group {
      display: grid;
      gap: 10px;
    }
    .action-group h4 {
      margin: 0;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    .action-list {
      display: grid;
      gap: 10px;
      max-height: 62vh;
      overflow: auto;
      padding-right: 4px;
    }
    .action-button {
      width: 100%;
      text-align: left;
      border-radius: 18px;
      padding: 13px 14px;
      background: rgba(255, 255, 255, 0.05);
    }
    .action-button strong {
      display: block;
      margin-bottom: 4px;
      font-size: 14px;
    }
    .action-meta {
      color: var(--muted);
      font-size: 12px;
    }
    .log-list {
      display: grid;
      gap: 10px;
      max-height: 46vh;
      overflow: auto;
      padding-right: 4px;
    }
    .log-entry {
      padding: 12px 14px;
      border-radius: 16px;
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.04);
      display: grid;
      gap: 7px;
    }
    .log-entry header {
      display: flex;
      justify-content: space-between;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.07em;
    }
    .empty-state {
      padding: 16px;
      border: 1px dashed var(--line);
      border-radius: 16px;
      color: var(--muted);
      text-align: center;
    }
    .loading,
    .error-banner {
      padding: 12px 14px;
      border-radius: 16px;
      background: rgba(255,255,255,0.06);
      border: 1px solid var(--line);
      color: var(--muted);
    }
    .error-banner {
      background: rgba(238, 125, 115, 0.14);
      border-color: rgba(238, 125, 115, 0.45);
      color: #ffd9d5;
    }
    @media (max-width: 1160px) {
      .hero,
      .layout {
        grid-template-columns: 1fr;
      }
      .action-list {
        max-height: none;
      }
    }
  </style>
</head>
<body>
  <div class="app">
    <section class="hero">
      <div class="panel hero-main">
        <div class="hero-title">
          <div class="title-block">
            <h1>Ark Nova Arena</h1>
            <p>Local browser UI for human vs checkpoint play.</p>
          </div>
          <div class="hero-controls">
            <input id="seed-input" type="number" step="1" placeholder="Seed">
            <button id="restart-button" class="primary">Restart</button>
            <button id="refresh-button">Refresh</button>
          </div>
        </div>
        <div class="status-strip" id="status-strip"></div>
        <div>
          <div class="break-track"><div id="break-fill" class="break-fill"></div></div>
        </div>
        <div id="message-banner" class="loading">Loading game state...</div>
      </div>
      <aside class="panel hero-side">
        <div class="section-header">
          <h2 class="section-title">Players</h2>
          <span id="turn-pill" class="section-subtitle"></span>
        </div>
        <div id="player-stack" class="player-stack"></div>
      </aside>
    </section>

    <section class="layout">
      <div class="main-column">
        <section class="panel section">
          <div class="section-header">
            <h2 class="section-title">Opening Draft / Hand</h2>
            <span class="section-subtitle">Your private cards and setup information</span>
          </div>
          <div id="private-cards"></div>
        </section>

        <section class="panel section">
          <div class="section-header">
            <h2 class="section-title">Display</h2>
            <span id="display-subtitle" class="section-subtitle"></span>
          </div>
          <div id="display-cards" class="card-grid"></div>
        </section>

        <section class="panel section">
          <div class="section-header">
            <h2 class="section-title">Your Zoo Map</h2>
            <span class="section-subtitle">Schematic tile map, tile meaning, and live coverage state</span>
          </div>
          <div class="map-shell">
            <div id="map-canvas" class="map-canvas"></div>
          </div>
          <div id="map-legend" class="map-legend"></div>
          <div id="map-details" class="map-details"></div>
          <div id="map-caption" class="map-caption"></div>
        </section>

        <section class="panel section">
          <div class="section-header">
            <h2 class="section-title">Zoo Cards</h2>
            <span class="section-subtitle">Public cards already played in both zoos</span>
          </div>
          <div id="zoo-cards"></div>
        </section>
      </div>

      <div class="side-column">
        <section class="panel section actions-panel">
          <div class="section-header">
            <h2 class="section-title">Legal Actions</h2>
            <span id="legal-count" class="section-subtitle"></span>
          </div>
          <div class="actions-toolbar">
            <input id="action-filter" type="search" placeholder="Filter actions by text">
          </div>
          <div id="actions-panel" class="action-list"></div>
        </section>

        <section class="panel section">
          <div class="section-header">
            <h2 class="section-title">Recent Actions</h2>
            <span class="section-subtitle">Newest first</span>
          </div>
          <div id="action-log" class="log-list"></div>
        </section>
      </div>
    </section>
  </div>

  <script>
    const state = {
      snapshot: null,
      loading: false,
      pendingActionIndex: null,
      filterText: "",
      mapSelectedKey: "",
    };

    const els = {
      statusStrip: document.getElementById("status-strip"),
      breakFill: document.getElementById("break-fill"),
      messageBanner: document.getElementById("message-banner"),
      playerStack: document.getElementById("player-stack"),
      turnPill: document.getElementById("turn-pill"),
      privateCards: document.getElementById("private-cards"),
      displayCards: document.getElementById("display-cards"),
      displaySubtitle: document.getElementById("display-subtitle"),
      mapCanvas: document.getElementById("map-canvas"),
      mapLegend: document.getElementById("map-legend"),
      mapDetails: document.getElementById("map-details"),
      mapCaption: document.getElementById("map-caption"),
      zooCards: document.getElementById("zoo-cards"),
      actionsPanel: document.getElementById("actions-panel"),
      legalCount: document.getElementById("legal-count"),
      actionLog: document.getElementById("action-log"),
      seedInput: document.getElementById("seed-input"),
      restartButton: document.getElementById("restart-button"),
      refreshButton: document.getElementById("refresh-button"),
      actionFilter: document.getElementById("action-filter"),
    };
    const SVG_NS = "http://www.w3.org/2000/svg";

    function setBanner(text, kind = "loading") {
      els.messageBanner.textContent = text;
      els.messageBanner.className = kind === "error" ? "error-banner" : "loading";
    }

    function mapTileKey(tile) {
      return `${tile.x},${tile.y}`;
    }

    function terrainLabel(terrain) {
      const value = String(terrain || "").trim().toLowerCase();
      if (value === "rock") return "Rock";
      if (value === "water") return "Water";
      return "Plain";
    }

    function prettyLabel(value) {
      return String(value || "")
        .replaceAll("_", " ")
        .replaceAll("-", " ")
        .trim()
        .replace(/\\b\\w/g, (ch) => ch.toUpperCase());
    }

    function mapCoverageLabel(tile) {
      if (!tile.covered) return "Uncovered";
      const base = prettyLabel(tile.covered_by || tile.covered_subtype || "building");
      if (tile.occupied) {
        return `${base} · occupied`;
      }
      return `${base} · covered`;
    }

    function renderMapLegend() {
      const chips = [
        ["Plain", "rgba(255,255,255,0.16)"],
        ["Rock", "rgba(139,94,60,0.7)"],
        ["Water", "rgba(29,155,240,0.7)"],
        ["Build II", "rgba(234,82,147,0.85)"],
        ["Covered", "rgba(15,22,19,0.86)"],
        ["Occupied", "rgba(245,185,76,0.82)"],
      ];
      els.mapLegend.replaceChildren(
        ...chips.map(([label, color]) => {
          const chip = document.createElement("div");
          chip.className = "legend-chip";
          const swatch = document.createElement("span");
          swatch.className = "legend-swatch";
          swatch.style.background = color;
          const text = document.createElement("span");
          text.textContent = label;
          chip.append(swatch, text);
          return chip;
        })
      );
    }

    function renderMapDetails(mapView, tile) {
      if (!tile) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "Hover or click a map hex to inspect it.";
        els.mapDetails.replaceChildren(empty);
        return;
      }
      const title = document.createElement("strong");
      title.textContent = `${mapView.map_name} · (${tile.x}, ${tile.y})`;
      const lines = [
        `Terrain: ${terrainLabel(tile.terrain)}`,
        `Coverage: ${mapCoverageLabel(tile)}`,
      ];
      if (tile.build2_required) {
        lines.push("Rule: requires Build II");
      }
      if (tile.placement_bonus) {
        lines.push(`Bonus: ${prettyLabel(tile.placement_bonus)}`);
      }
      if ((tile.tags || []).length) {
        lines.push(`Tags: ${(tile.tags || []).map((item) => prettyLabel(item)).join(", ")}`);
      }
      if (tile.covered && tile.animal_capacity > 0) {
        lines.push(`Capacity: ${tile.used_capacity}/${tile.animal_capacity}`);
      }
      els.mapDetails.replaceChildren(
        title,
        ...lines.map((line) => {
          const row = document.createElement("div");
          row.className = "detail-line";
          row.textContent = line;
          return row;
        })
      );
    }

    function createCardCard(card, extra = {}) {
      const article = document.createElement("article");
      article.className = "game-card";

      const imageUrl = extra.imageUrl || cardImageUrl(card);
      if (imageUrl) {
        const image = document.createElement("img");
        image.className = "card-art";
        image.loading = "lazy";
        image.src = imageUrl;
        image.alt = card.name || card.title || "Card image";
        image.addEventListener("error", () => {
          image.remove();
        }, { once: true });
        article.appendChild(image);
      }

      const topLine = document.createElement("div");
      topLine.className = "topline";
      const title = document.createElement("h3");
      title.textContent = `${card.number ? "#" + card.number + " " : ""}${card.name || card.title || "Card"}`;
      const pill = document.createElement("span");
      pill.className = "pill";
      pill.textContent = extra.primaryPill || card.card_type || "card";
      topLine.append(title, pill);

      const badges = document.createElement("div");
      badges.className = "badge-row";
      const badgeItems = extra.badges || card.badges || [];
      if (badgeItems.length) {
        badgeItems.slice(0, 4).forEach((item) => {
          const span = document.createElement("span");
          span.className = "pill";
          span.textContent = item;
          badges.appendChild(span);
        });
      }

      const resources = document.createElement("div");
      resources.className = "resource-line";
      const chunks = [];
      if (card.cost !== undefined) chunks.push(`Cost ${card.cost}`);
      if (card.size !== undefined) chunks.push(`Size ${card.size}`);
      if (card.appeal !== undefined) chunks.push(`Appeal ${card.appeal}`);
      if (card.conservation !== undefined) chunks.push(`Cons ${card.conservation}`);
      if (card.reputation_gain !== undefined && Number(card.reputation_gain) > 0) {
        chunks.push(`Rep +${card.reputation_gain}`);
      }
      if (extra.resourceText) chunks.push(extra.resourceText);
      resources.textContent = chunks.join(" · ");

      const body = document.createElement("p");
      body.textContent = extra.body || card.ability_title || card.title || "No text";

      article.append(topLine);
      if (badges.children.length) article.append(badges);
      if (resources.textContent) article.append(resources);
      article.append(body);
      return article;
    }

    function cardImageUrl(card) {
      const cardType = String(card.card_type || "").trim().toLowerCase();
      const number = Number.parseInt(card.number, 10);
      if (cardType === "animal" || cardType === "sponsor") {
        if (!Number.isFinite(number)) return "";
        return `/assets/cards/${cardType}/${number}.jpg`;
      }
      if (cardType === "conservation_project") {
        if (!Number.isFinite(number)) return "";
        return `/assets/cards/project/${number}.jpg`;
      }
      const dataId = String(card.data_id || "").trim();
      const setupMatch = dataId.match(/^([FP])(\\d{3})/);
      if (!setupMatch) return "";
      if (setupMatch[1] === "F") {
        return `/assets/cards/endgame/${Number.parseInt(setupMatch[2], 10)}.jpg`;
      }
      return `/assets/cards/project/${Number.parseInt(setupMatch[2], 10)}.jpg`;
    }

    function renderPlayers(snapshot) {
      const publicObs = snapshot.observation.public;
      const players = publicObs.players || [];
      els.playerStack.innerHTML = "";
      players.forEach((player) => {
        const card = document.createElement("article");
        card.className = "player-card";
        if (player.player_id === snapshot.current_actor_id) {
          card.classList.add("active");
        }

        const top = document.createElement("div");
        top.className = "player-row";
        const name = document.createElement("div");
        name.className = "player-name";
        const isHuman = player.player_id === snapshot.human_player_id;
        name.textContent = player.name;
        const role = document.createElement("div");
        role.className = "player-role";
        role.textContent = isHuman ? "Human" : "Checkpoint";
        top.append(name, role);

        const stats = document.createElement("div");
        stats.className = "stat-grid";
        [
          ["Money", player.money],
          ["Appeal", player.appeal],
          ["Conservation", player.conservation],
          ["X", player.x_tokens],
          ["Rep", player.reputation],
          ["Hand", player.hand_count],
          ["Zoo", player.zoo_cards_count],
          ["Final", player.final_scoring_count],
        ].forEach(([label, value]) => {
          const box = document.createElement("div");
          box.className = "stat";
          const l = document.createElement("span");
          l.className = "stat-label";
          l.textContent = label;
          const v = document.createElement("strong");
          v.textContent = String(value);
          box.append(l, v);
          stats.appendChild(box);
        });

        const order = document.createElement("div");
        order.className = "resource-line";
        order.textContent = `Actions: ${(player.action_order || []).join(" · ")}`;

        card.append(top, stats, order);
        els.playerStack.appendChild(card);
      });
    }

    function renderPrivate(snapshot) {
      const privatePlayer = snapshot.observation.private.player;
      const sections = [];

      if ((privatePlayer.opening_draft_drawn || []).length) {
        const wrapper = document.createElement("section");
        wrapper.style.display = "grid";
        wrapper.style.gap = "10px";
        const note = document.createElement("div");
        note.className = "draft-note";
        note.textContent = `Opening draft: keep 4 cards. Currently kept indices: ${(privatePlayer.opening_draft_kept_indices || []).map((n) => n + 1).join(", ") || "none"}`;
        const grid = document.createElement("div");
        grid.className = "draft-grid";
        (privatePlayer.opening_draft_drawn || []).forEach((card, index) => {
          const kept = (privatePlayer.opening_draft_kept_indices || []).includes(index);
          grid.appendChild(
            createCardCard(card, {
              primaryPill: kept ? "kept" : "draft",
              body: card.ability_title || "Draft card",
            })
          );
        });
        wrapper.append(note, grid);
        sections.push(wrapper);
      }

      if ((privatePlayer.hand || []).length) {
        const grid = document.createElement("div");
        grid.className = "card-grid";
        (privatePlayer.hand || []).forEach((card) => {
          grid.appendChild(createCardCard(card, { primaryPill: "hand" }));
        });
        sections.push(grid);
      }

      if ((privatePlayer.final_scoring_cards || []).length) {
        const wrap = document.createElement("div");
        wrap.className = "mini-grid";
        (privatePlayer.final_scoring_cards || []).forEach((card) => {
          wrap.appendChild(
            createCardCard(card, {
              primaryPill: "final scoring",
              body: card.data_id || card.title,
              resourceText: "",
            })
          );
        });
        sections.push(wrap);
      }

      if (!sections.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No private cards to show right now.";
        els.privateCards.replaceChildren(empty);
        return;
      }
      els.privateCards.replaceChildren(...sections);
    }

    function renderDisplay(snapshot) {
      const publicObs = snapshot.observation.public;
      const display = publicObs.zoo_display || [];
      if (snapshot.display_hidden) {
        els.displaySubtitle.textContent = "Hidden during opening draft";
        els.displayCards.innerHTML = "";
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "Display is hidden until opening draft is finished.";
        els.displayCards.appendChild(empty);
        return;
      }
      els.displaySubtitle.textContent = `Deck ${publicObs.zoo_deck_count} · Discard ${publicObs.zoo_discard_count}`;
      els.displayCards.innerHTML = "";
      if (!display.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "Display is empty.";
        els.displayCards.appendChild(empty);
        return;
      }
      display.forEach((card) => els.displayCards.appendChild(createCardCard(card, { primaryPill: "display" })));
    }

    function renderZooCards(snapshot) {
      const publicObs = snapshot.observation.public;
      const wrapper = document.createElement("div");
      wrapper.style.display = "grid";
      wrapper.style.gap = "16px";
      (publicObs.players || []).forEach((player) => {
        const section = document.createElement("section");
        section.style.display = "grid";
        section.style.gap = "10px";
        const header = document.createElement("div");
        header.className = "section-subtitle";
        header.textContent = `${player.name} · ${player.zoo_cards_count} public zoo cards`;
        section.appendChild(header);
        if (!(player.zoo_cards || []).length) {
          const empty = document.createElement("div");
          empty.className = "empty-state";
          empty.textContent = "No zoo cards yet.";
          section.appendChild(empty);
        } else {
          const grid = document.createElement("div");
          grid.className = "mini-grid";
          (player.zoo_cards || []).forEach((card) => {
            grid.appendChild(createCardCard(card, { primaryPill: "zoo" }));
          });
          section.appendChild(grid);
        }
        wrapper.appendChild(section);
      });
      els.zooCards.replaceChildren(wrapper);
    }

    function renderMap(snapshot) {
      const mapView = snapshot.map_view;
      els.mapCanvas.innerHTML = "";
      els.mapLegend.innerHTML = "";
      if (!mapView || !(mapView.tiles || []).length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "Map data unavailable.";
        els.mapCanvas.appendChild(empty);
        els.mapDetails.replaceChildren(empty.cloneNode(true));
        els.mapCaption.textContent = "";
        return;
      }

      renderMapLegend();

      const tileByKey = new Map((mapView.tiles || []).map((tile) => [mapTileKey(tile), tile]));
      if (!tileByKey.has(state.mapSelectedKey)) {
        state.mapSelectedKey = mapTileKey(mapView.tiles[0]);
      }

      const svg = document.createElementNS(SVG_NS, "svg");
      svg.setAttribute("class", "map-svg");
      svg.setAttribute("viewBox", `0 0 ${mapView.image_width} ${mapView.image_height}`);
      svg.setAttribute("aria-label", `${mapView.map_name} zoo map`);

      (mapView.tiles || []).forEach((tile) => {
        const tileKey = mapTileKey(tile);
        const group = document.createElementNS(SVG_NS, "g");
        const polygon = document.createElementNS(SVG_NS, "polygon");
        const classes = [
          "map-tile",
          `terrain-${String(tile.terrain || "plain").toLowerCase()}`,
        ];
        if (tile.build2_required) classes.push("build2");
        if (tile.covered) classes.push("covered");
        if (tile.covered_subtype) classes.push(`subtype-${String(tile.covered_subtype)}`);
        if (tile.occupied) classes.push("occupied");
        if (tileKey === state.mapSelectedKey) classes.push("selected");
        polygon.setAttribute("class", classes.join(" "));
        polygon.setAttribute("points", String(tile.points || ""));
        polygon.addEventListener("mouseenter", () => {
          renderMapDetails(mapView, tile);
        });
        polygon.addEventListener("click", () => {
          state.mapSelectedKey = tileKey;
          renderMap(snapshot);
        });
        const title = document.createElementNS(SVG_NS, "title");
        title.textContent = `${mapView.map_name} (${tile.x}, ${tile.y}) | ${terrainLabel(tile.terrain)} | ${mapCoverageLabel(tile)}${tile.placement_bonus ? ` | Bonus ${prettyLabel(tile.placement_bonus)}` : ""}`;
        polygon.appendChild(title);
        group.appendChild(polygon);

        const coord = document.createElementNS(SVG_NS, "text");
        coord.setAttribute("class", "map-label map-coord");
        coord.setAttribute("x", String(tile.center_x));
        coord.setAttribute("y", String(tile.center_y));
        coord.setAttribute("dominant-baseline", "middle");
        coord.textContent = `${tile.x},${tile.y}`;
        group.appendChild(coord);

        if (tile.build2_required) {
          const build2 = document.createElementNS(SVG_NS, "text");
          build2.setAttribute("class", "map-label map-build2");
          build2.setAttribute("x", String(tile.center_x));
          build2.setAttribute("y", String(tile.center_y - 34));
          build2.setAttribute("dominant-baseline", "middle");
          build2.textContent = "B2";
          group.appendChild(build2);
        }

        if (tile.placement_bonus_short) {
          const bonus = document.createElementNS(SVG_NS, "text");
          bonus.setAttribute("class", "map-label map-bonus");
          bonus.setAttribute("x", String(tile.center_x));
          bonus.setAttribute("y", String(tile.center_y + 38));
          bonus.setAttribute("dominant-baseline", "middle");
          bonus.textContent = String(tile.placement_bonus_short);
          group.appendChild(bonus);
        }

        if (tile.covered_label) {
          const cover = document.createElementNS(SVG_NS, "text");
          cover.setAttribute("class", "map-label map-cover covered");
          cover.setAttribute("x", String(tile.center_x));
          cover.setAttribute("y", String(tile.center_y - (tile.build2_required ? 10 : 0)));
          cover.setAttribute("dominant-baseline", "middle");
          cover.textContent = String(tile.covered_label);
          group.appendChild(cover);
        }

        svg.appendChild(group);
      });

      els.mapCanvas.appendChild(svg);
      renderMapDetails(mapView, tileByKey.get(state.mapSelectedKey) || mapView.tiles[0]);
      els.mapCaption.textContent = `${mapView.map_name} · covered ${mapView.covered_count}/${mapView.tiles.length} tiles${(mapView.map_effects || []).length ? ` · ${mapView.map_effects[0]}` : ""}`;
    }

    function renderActions(snapshot) {
      const actions = snapshot.legal_actions || [];
      els.legalCount.textContent = snapshot.human_turn ? `${actions.length} legal actions` : "Waiting on AI";
      els.actionsPanel.innerHTML = "";

      if (!snapshot.human_turn) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = snapshot.game_over ? "Game over." : `AI is acting as ${snapshot.current_actor_name}.`;
        els.actionsPanel.appendChild(empty);
        return;
      }

      const filterText = state.filterText.trim().toLowerCase();
      const grouped = new Map();
      actions.forEach((action) => {
        const haystack = `${action.rendered} ${action.card_name} ${action.type}`.toLowerCase();
        if (filterText && !haystack.includes(filterText)) {
          return;
        }
        const key = action.card_name || action.pending_kind || action.type;
        if (!grouped.has(key)) grouped.set(key, []);
        grouped.get(key).push(action);
      });

      if (!grouped.size) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No legal actions match the current filter.";
        els.actionsPanel.appendChild(empty);
        return;
      }

      grouped.forEach((items, key) => {
        const group = document.createElement("section");
        group.className = "action-group";
        const title = document.createElement("h4");
        title.textContent = `${key} · ${items.length}`;
        const list = document.createElement("div");
        list.className = "action-group-list";
        list.style.display = "grid";
        list.style.gap = "10px";
        items.forEach((action) => {
          const button = document.createElement("button");
          button.className = "action-button";
          if (state.pendingActionIndex === action.index) {
            button.classList.add("primary");
          }
          button.disabled = state.loading;
          button.addEventListener("click", () => submitAction(action.index));

          const strong = document.createElement("strong");
          strong.textContent = action.rendered;
          const meta = document.createElement("div");
          meta.className = "action-meta";
          meta.textContent = `${action.type}${action.value ? " · x=" + action.value : ""}${action.concrete ? " · concrete" : ""}`;
          button.append(strong, meta);
          list.appendChild(button);
        });
        group.append(title, list);
        els.actionsPanel.appendChild(group);
      });
    }

    function renderLogs(snapshot) {
      const logs = [...(snapshot.recent_actions || [])].reverse();
      els.actionLog.innerHTML = "";
      if (!logs.length) {
        const empty = document.createElement("div");
        empty.className = "empty-state";
        empty.textContent = "No actions yet.";
        els.actionLog.appendChild(empty);
        return;
      }
      logs.forEach((entry) => {
        const article = document.createElement("article");
        article.className = "log-entry";
        const header = document.createElement("header");
        const who = document.createElement("span");
        who.textContent = entry.player_name;
        const turn = document.createElement("span");
        turn.textContent = `Turn ${entry.turn_index}`;
        header.append(who, turn);
        const action = document.createElement("strong");
        action.textContent = entry.action;
        article.append(header, action);
        if ((entry.effects || []).length) {
          const list = document.createElement("div");
          list.className = "resource-line";
          list.style.display = "grid";
          list.style.gap = "6px";
          (entry.effects || []).forEach((line) => {
            const row = document.createElement("div");
            row.textContent = line;
            list.appendChild(row);
          });
          article.appendChild(list);
        }
        els.actionLog.appendChild(article);
      });
    }

    function renderStatus(snapshot) {
      const publicObs = snapshot.observation.public;
      const currentName = snapshot.current_actor_name || "-";
      els.turnPill.textContent = snapshot.game_over ? "Finished" : `Current actor: ${currentName}`;
      const breakPercent = publicObs.break_max > 0 ? (publicObs.break_progress / publicObs.break_max) * 100 : 0;
      els.breakFill.style.width = `${Math.max(0, Math.min(100, breakPercent))}%`;
      els.statusStrip.replaceChildren(
        makeChip("Seed", snapshot.seed),
        makeChip("Turn", publicObs.turn_index),
        makeChip("Break", `${publicObs.break_progress}/${publicObs.break_max}`),
        makeChip("Display", snapshot.display_hidden ? "hidden" : publicObs.zoo_display.length),
        makeChip("Pending", publicObs.pending_decision_kind || "none"),
        makeChip("Checkpoint", snapshot.checkpoint.split("/").slice(-1)[0] || snapshot.checkpoint),
      );
      els.seedInput.value = String(snapshot.seed);

      if (snapshot.game_over && snapshot.scores) {
        const pairs = Object.entries(snapshot.scores).map(([name, score]) => `${name} ${score}`).join(" · ");
        setBanner(`Game over. Final scores: ${pairs}`, "loading");
      } else if (snapshot.human_turn) {
        setBanner(`Your move as ${snapshot.current_actor_name}. Choose one legal action.`, "loading");
      } else {
        setBanner(`AI is acting as ${snapshot.current_actor_name}.`, "loading");
      }
    }

    function makeChip(label, value) {
      const chip = document.createElement("div");
      chip.className = "status-chip";
      const l = document.createElement("span");
      l.textContent = label;
      const v = document.createElement("strong");
      v.textContent = String(value);
      chip.append(l, v);
      return chip;
    }

    function render(snapshot) {
      state.snapshot = snapshot;
      renderStatus(snapshot);
      renderPlayers(snapshot);
      renderPrivate(snapshot);
      renderDisplay(snapshot);
      renderMap(snapshot);
      renderZooCards(snapshot);
      renderActions(snapshot);
      renderLogs(snapshot);
    }

    async function requestJson(url, options = {}) {
      const response = await fetch(url, {
        headers: { "Content-Type": "application/json" },
        ...options,
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.error || `HTTP ${response.status}`);
      }
      return payload;
    }

    async function loadState() {
      state.loading = true;
      try {
        const payload = await requestJson("/api/state");
        render(payload);
      } catch (error) {
        console.error(error);
        setBanner(error.message || "Failed to load state.", "error");
      } finally {
        state.loading = false;
      }
    }

    async function submitAction(index) {
      state.loading = true;
      state.pendingActionIndex = index;
      renderActions(state.snapshot || { legal_actions: [], human_turn: false, game_over: false });
      try {
        const payload = await requestJson("/api/action", {
          method: "POST",
          body: JSON.stringify({ action_index: index }),
        });
        render(payload);
      } catch (error) {
        console.error(error);
        setBanner(error.message || "Failed to submit action.", "error");
      } finally {
        state.pendingActionIndex = null;
        state.loading = false;
      }
    }

    async function restartGame() {
      state.loading = true;
      try {
        const seedValue = Number.parseInt(els.seedInput.value, 10);
        const payload = await requestJson("/api/restart", {
          method: "POST",
          body: JSON.stringify(Number.isFinite(seedValue) ? { seed: seedValue } : {}),
        });
        render(payload);
      } catch (error) {
        console.error(error);
        setBanner(error.message || "Failed to restart game.", "error");
      } finally {
        state.loading = false;
      }
    }

    els.refreshButton.addEventListener("click", loadState);
    els.restartButton.addEventListener("click", restartGame);
    els.actionFilter.addEventListener("input", (event) => {
      state.filterText = event.target.value || "";
      if (state.snapshot) renderActions(state.snapshot);
    });

    loadState();
  </script>
</body>
</html>
"""


class SessionRequestHandler(BaseHTTPRequestHandler):
    session: HumanVsCheckpointSession

    def log_message(self, format: str, *args: Any) -> None:
        del format, args

    def do_GET(self) -> None:
        image_path = _resolve_card_image_request_path(self.path)
        if image_path is not None:
            self._send_file(image_path, content_type="image/jpeg")
            return
        map_image_path = _resolve_map_image_request_path(self.path)
        if map_image_path is not None:
            self._send_file(map_image_path, content_type="image/jpeg")
            return
        if self.path in {"/", "/index.html"}:
            self._send_html(HTML_PAGE)
            return
        if self.path == "/api/state":
            self._send_json(HTTPStatus.OK, self.session.snapshot())
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        if self.path not in {"/api/action", "/api/restart"}:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            return
        try:
            payload = self._read_json_body()
            if self.path == "/api/action":
                snapshot = self.session.submit_action(int(payload.get("action_index", -1)))
            else:
                seed_value = payload.get("seed")
                snapshot = self.session.restart(
                    seed=None if seed_value in {None, ""} else int(seed_value)
                )
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
            return
        self._send_json(HTTPStatus.OK, snapshot)

    def _read_json_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or 0)
        raw = self.rfile.read(length) if length > 0 else b"{}"
        if not raw:
            return {}
        parsed = json.loads(raw.decode("utf-8"))
        if not isinstance(parsed, dict):
            raise ValueError("Request body must be a JSON object.")
        return parsed

    def _send_html(self, payload: str) -> None:
        encoded = payload.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_file(self, path: Path, *, content_type: str) -> None:
        blob = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(blob)))
        self.send_header("Cache-Control", "public, max-age=3600")
        self.end_headers()
        self.wfile.write(blob)

    def _send_json(self, status: HTTPStatus, payload: Dict[str, Any]) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve a local browser UI for Ark Nova vs checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path for the AI player")
    parser.add_argument("--seed", type=int, default=42, help="Initial game seed")
    parser.add_argument(
        "--device",
        type=_parse_device_arg,
        default="cpu",
        help=f"Checkpoint inference device: {SUPPORTED_EXPLICIT_DEVICE_HELP}",
    )
    parser.add_argument(
        "--human-seat",
        choices=("1", "2"),
        default="1",
        help="Seat number controlled by the human player",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample from the policy instead of using greedy argmax",
    )
    parser.add_argument(
        "--marine-world",
        action="store_true",
        help="Include Marine World final scoring cards in setup pool",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=8765, help="Bind port")
    return parser.parse_args(argv)


def main_cli(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(str(args.device))
    session = HumanVsCheckpointSession(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        human_seat=int(args.human_seat) - 1,
        deterministic=not bool(args.stochastic),
        seed=int(args.seed),
        include_marine_world=bool(args.marine_world),
    )

    handler_class = type(
        "ArkNovaWebUIHandler",
        (SessionRequestHandler,),
        {"session": session},
    )
    server = ThreadingHTTPServer((str(args.host), int(args.port)), handler_class)
    print(
        f"serving Ark Nova web UI on http://{args.host}:{args.port} "
        f"(checkpoint={args.checkpoint}, human_seat={args.human_seat}, device={device.type})"
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down web UI.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main_cli()

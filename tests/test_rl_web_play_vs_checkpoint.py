import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import main


def _load_web_play_module():
    module_path = Path("tools/rl/web_play_vs_checkpoint.py")
    spec = importlib.util.spec_from_file_location("web_play_vs_checkpoint_test_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeModel:
    def init_hidden(self, batch_size, *, device):
        return (
            torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
            torch.zeros((1, int(batch_size), 1), dtype=torch.float32, device=device),
        )

    def forward_step(self, *, state_vec, action_features, action_mask, hidden=None):
        del state_vec, action_features, action_mask
        logits = torch.tensor([[1.0]], dtype=torch.float32, device=hidden[0].device)
        values = torch.zeros((1,), dtype=torch.float32, device=hidden[0].device)
        return logits, values, hidden


def _fake_bundle():
    return SimpleNamespace(
        name="AI",
        model=_FakeModel(),
        obs_encoder=SimpleNamespace(
            encode_from_state=lambda state, actor_id: np.zeros((4,), dtype=np.float32),
        ),
        action_encoder=SimpleNamespace(
            encode_many=lambda actions: np.zeros((len(actions), 3), dtype=np.float32),
        ),
    )


class _FakePlayer:
    def __init__(self, name: str) -> None:
        self.name = name


class _FakeState:
    def __init__(self, player_names):
        self.players = [_FakePlayer(name) for name in player_names]
        self.current_player = 0
        self.pending_decision_kind = ""
        self.pending_decision_player_id = None
        self.turn_index = 1
        self.effect_log = []
        self._game_over = False

    def game_over(self):
        return bool(self._game_over)


def test_web_play_cli_defaults_are_stable():
    module = _load_web_play_module()
    args = module.parse_args(["--checkpoint", "runs/self_play_masked/checkpoint_0190.pt"])

    assert args.checkpoint == "runs/self_play_masked/checkpoint_0190.pt"
    assert args.seed == 42
    assert args.device == "cpu"
    assert args.human_seat == "1"
    assert args.stochastic is False
    assert args.marine_world is False
    assert args.host == "127.0.0.1"
    assert args.port == 8765


def test_card_image_resolution_uses_local_cards_manifest_exports():
    module = _load_web_play_module()

    animal_path = module._resolve_card_image_file("animal", 401)
    sponsor_path = module._resolve_card_image_file("sponsor", 201)
    endgame_path = module._resolve_card_image_file("endgame", 1)
    project_path = module._resolve_card_image_file("project", 104)
    conservation_project_path = module._resolve_card_image_file("conservation_project", 117)

    assert animal_path is not None
    assert animal_path.name.startswith("401_")
    assert animal_path.parent.name == "animals"
    assert sponsor_path is not None
    assert sponsor_path.name.startswith("201_")
    assert sponsor_path.parent.name == "sponsors"
    assert endgame_path is not None
    assert endgame_path.name.startswith("001_")
    assert endgame_path.parent.name == "endgame"
    assert project_path is not None
    assert project_path.name.startswith("104_")
    assert project_path.parent.name == "projects"
    assert conservation_project_path is not None
    assert conservation_project_path.name.startswith("117_")
    assert conservation_project_path.parent.name == "projects"
    assert module._resolve_card_image_file("animal", "not-a-number") is None
    assert module._resolve_card_image_request_path("/assets/cards/animal/401.jpg") == animal_path
    assert module._resolve_card_image_request_path("/assets/cards/sponsor/201.jpg") == sponsor_path
    assert module._resolve_card_image_request_path("/assets/cards/endgame/1.jpg") == endgame_path
    assert module._resolve_card_image_request_path("/assets/cards/project/104.jpg") == project_path
    assert module._resolve_card_image_request_path("/assets/cards/animal/../401.jpg") is None
    assert module._resolve_card_image_request_path("/assets/cards/animal/not-a-number.jpg") is None


def test_html_page_references_local_card_asset_route():
    module = _load_web_play_module()

    assert "/assets/cards/${cardType}/${number}.jpg" in module.HTML_PAGE
    assert "/assets/cards/project/${number}.jpg" in module.HTML_PAGE
    assert "/assets/cards/endgame/${Number.parseInt(setupMatch[2], 10)}.jpg" in module.HTML_PAGE


def test_map_view_model_uses_real_map_image_and_marks_covered_tiles():
    module = _load_web_play_module()

    map_view = module._build_map_view_model(
        {
            "map_image_name": "plan1a",
            "players": [
                {
                    "zoo_map_present": True,
                    "zoo_map_buildings": [
                        {
                            "building_type": "SIZE_1",
                            "subtype": "enclosure_basic",
                            "origin": [0, 0],
                            "rotation": "ROT_0",
                            "layout": [[0, 0]],
                            "empty_spaces": 0,
                        }
                    ],
                    "enclosures": [
                        {
                            "size": 1,
                            "occupied": True,
                            "origin": [0, 0],
                            "rotation": "ROT_0",
                            "enclosure_type": "standard",
                            "used_capacity": 1,
                            "animal_capacity": 1,
                        }
                    ],
                }
            ],
        },
        human_player_id=0,
    )

    assert map_view["image_url"] == "/assets/maps/plan1a.jpg"
    assert map_view["image_width"] == 4028
    assert map_view["image_height"] == 2126
    assert module._resolve_map_image_request_path("/assets/maps/plan1a.jpg").name == "plan1a.jpg"

    tile = next(item for item in map_view["tiles"] if item["x"] == 0 and item["y"] == 0)
    assert tile["terrain"] == "plain"
    assert tile["placement_bonus"] == "x_token"
    assert tile["covered"] is True
    assert tile["covered_by"] == "SIZE_1"
    assert tile["covered_label"] == "E1"
    assert tile["occupied"] is True
    assert "," in tile["points"]


def test_session_advances_ai_turn_and_accepts_human_action(monkeypatch):
    module = _load_web_play_module()
    monkeypatch.setattr(module, "load_policy_bundle_from_checkpoint", lambda *args, **kwargs: _fake_bundle())

    def _fake_setup_game(*, seed, player_names, manual_opening_draft_player_names, include_marine_world):
        del seed, manual_opening_draft_player_names, include_marine_world
        return _FakeState(player_names)

    def _fake_legal_actions(player, *, state, player_id):
        del player
        if int(player_id) == 0:
            return [
                main.Action(
                    main.ActionType.MAIN_ACTION,
                    card_name="cards",
                    details={"concrete": True, "action_label": "AI opening"},
                )
            ]
        return [
            main.Action(
                main.ActionType.PENDING_DECISION,
                details={"pending_kind": "opening_draft_keep", "action_label": "keep [1 2 3 4]"},
            ),
            main.Action(
                main.ActionType.PENDING_DECISION,
                details={"pending_kind": "opening_draft_keep", "action_label": "keep [1 2 3 5]"},
            ),
        ]

    def _fake_apply_action(state, action):
        state.effect_log.append(f"effect:{action}")
        if int(state.current_player) == 0:
            state.current_player = 1
            state.pending_decision_kind = "opening_draft_keep"
            state.pending_decision_player_id = 1
            return
        state.pending_decision_kind = ""
        state.pending_decision_player_id = None
        state._game_over = True

    def _fake_build_player_observation(state, *, viewer_player_id):
        return {
            "public": {
                "players": [
                    {
                        "player_id": idx,
                        "name": player.name,
                        "money": 25,
                        "appeal": 0,
                        "conservation": 0,
                        "x_tokens": 0,
                        "reputation": 1,
                        "hand_count": 0,
                        "zoo_cards_count": 0,
                        "final_scoring_count": 2,
                        "action_order": ["animals", "build", "cards", "association", "sponsors"],
                    }
                    for idx, player in enumerate(state.players)
                ],
                "break_progress": 0,
                "break_max": 9,
                "turn_index": state.turn_index,
                "pending_decision_kind": state.pending_decision_kind,
                "zoo_display": [
                    {
                        "number": 114,
                        "name": "YOSEMITE NATIONAL PARK",
                        "card_type": "conservation_project",
                        "cost": 0,
                        "size": 0,
                        "appeal": 0,
                        "conservation": 0,
                        "reputation_gain": 0,
                        "badges": [],
                    }
                ],
                "zoo_deck_count": 200,
                "zoo_discard_count": 0,
            },
            "private": {
                "viewer_player_id": int(viewer_player_id),
                "player": {
                    "opening_draft_drawn": [],
                    "opening_draft_kept_indices": [],
                    "hand": [],
                    "final_scoring_cards": [],
                },
            },
        }

    monkeypatch.setattr(module.main, "setup_game", _fake_setup_game)
    monkeypatch.setattr(module.main, "legal_actions", _fake_legal_actions)
    monkeypatch.setattr(module.main, "apply_action", _fake_apply_action)
    monkeypatch.setattr(module.main, "build_player_observation", _fake_build_player_observation)
    monkeypatch.setattr(
        module.main,
        "_final_score_points",
        lambda state, player: 12 if player.name == "AI" else 17,
    )

    session = module.HumanVsCheckpointSession(
        checkpoint_path=Path("fake_checkpoint.pt"),
        device=torch.device("cpu"),
        human_seat=1,
        deterministic=True,
        seed=42,
        include_marine_world=False,
    )

    initial = session.snapshot()
    assert initial["human_turn"] is True
    assert initial["current_actor_name"] == "You"
    assert initial["display_hidden"] is True
    assert initial["observation"]["public"]["zoo_display"] == []
    assert [entry["player_name"] for entry in initial["recent_actions"]] == ["AI"]
    assert len(initial["legal_actions"]) == 2
    assert initial["legal_actions"][0]["pending_kind"] == "opening_draft_keep"

    finished = session.submit_action(1)
    assert finished["game_over"] is True
    assert finished["display_hidden"] is False
    assert finished["scores"] == {"AI": 12, "You": 17}
    assert [entry["player_name"] for entry in finished["recent_actions"]] == ["AI", "You"]

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

import main


def _load_play_vs_checkpoint_module():
    module_path = Path("tools/rl/play_vs_checkpoint.py")
    spec = importlib.util.spec_from_file_location("play_vs_checkpoint_cli_test_module", module_path)
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
        logits = torch.tensor([[0.0, 2.0, -1.0]], dtype=torch.float32, device=hidden[0].device)
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


def test_play_vs_checkpoint_cli_defaults_are_stable():
    module = _load_play_vs_checkpoint_module()
    args = module.parse_args(["--checkpoint", "runs/self_play_masked/checkpoint_0190.pt"])

    assert args.checkpoint == "runs/self_play_masked/checkpoint_0190.pt"
    assert args.seed == 42
    assert args.device == "cpu"
    assert args.human_seat == "1"
    assert args.stochastic is False
    assert args.marine_world is False
    assert args.quiet is False


def test_checkpoint_player_chooses_argmax_legal_action(monkeypatch):
    module = _load_play_vs_checkpoint_module()
    monkeypatch.setattr(module, "load_policy_bundle_from_checkpoint", lambda *args, **kwargs: _fake_bundle())

    player = module.CheckpointPlayer(
        checkpoint_path=Path("fake_checkpoint.pt"),
        device=torch.device("cpu"),
        deterministic=True,
        name="AI",
    )
    state = SimpleNamespace(
        pending_decision_kind="",
        pending_decision_player_id=None,
        current_player=0,
    )
    actions = [
        main.Action(main.ActionType.MAIN_ACTION, card_name="cards", details={"concrete": True}),
        main.Action(main.ActionType.MAIN_ACTION, card_name="build", details={"concrete": True}),
        main.Action(main.ActionType.MAIN_ACTION, card_name="animals", details={"concrete": True}),
    ]

    chosen = player.choose_action(state, actions)

    assert chosen is actions[1]


def test_play_vs_checkpoint_cli_wires_human_and_ai_seats(monkeypatch):
    module = _load_play_vs_checkpoint_module()
    monkeypatch.setattr(module, "load_policy_bundle_from_checkpoint", lambda *args, **kwargs: _fake_bundle())

    captured = {}

    def _fake_play_game(*, agents, player_names, seed, verbose, include_marine_world, private_viewer_names):
        captured["agents"] = agents
        captured["player_names"] = list(player_names)
        captured["seed"] = int(seed)
        captured["verbose"] = bool(verbose)
        captured["include_marine_world"] = bool(include_marine_world)
        captured["private_viewer_names"] = set(private_viewer_names)
        return {}

    monkeypatch.setattr(module.main, "play_game", _fake_play_game)

    module.main_cli(
        [
            "--checkpoint",
            "runs/self_play_masked/checkpoint_0190.pt",
            "--human-seat",
            "2",
            "--quiet",
        ]
    )

    assert captured["player_names"] == ["AI", "You"]
    assert isinstance(captured["agents"]["You"], module.main.HumanPlayer)
    assert isinstance(captured["agents"]["AI"], module.CheckpointPlayer)
    assert captured["seed"] == 42
    assert captured["verbose"] is False
    assert captured["include_marine_world"] is False
    assert captured["private_viewer_names"] == {"You"}

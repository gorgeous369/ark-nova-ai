"""Play a local human-vs-checkpoint Ark Nova game."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main

from arknova_rl.config import SUPPORTED_EXPLICIT_DEVICE_HELP, normalize_torch_device_spec
from arknova_rl.evaluator import PolicyBundle, load_policy_bundle_from_checkpoint


def _parse_device_arg(raw_value: str) -> str:
    try:
        return normalize_torch_device_spec(raw_value, allow_auto=False)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def _resolve_device(device_name: str) -> torch.device:
    return torch.device(str(device_name))


class CheckpointPlayer(main.PlayerAgent):
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: torch.device,
        deterministic: bool = True,
        name: Optional[str] = None,
    ) -> None:
        self.device = device
        self.deterministic = bool(deterministic)
        self.bundle: PolicyBundle = load_policy_bundle_from_checkpoint(
            Path(checkpoint_path),
            device=device,
            name=name,
        )
        self._hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None

    def _ensure_hidden(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._hidden is None:
            self._hidden = self.bundle.model.init_hidden(1, device=self.device)
        return self._hidden

    def choose_action(self, state: main.GameState, actions: list[main.Action]) -> main.Action:
        actor_id = (
            int(state.pending_decision_player_id)
            if str(state.pending_decision_kind or "").strip() and state.pending_decision_player_id is not None
            else int(state.current_player)
        )
        if not actions:
            raise RuntimeError("CheckpointPlayer received no legal actions.")

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


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Play Ark Nova against a trained checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path for the AI player")
    parser.add_argument("--seed", type=int, default=42, help="Game seed")
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
    parser.add_argument("--quiet", action="store_true", help="Reduce turn-by-turn output")
    return parser.parse_args(argv)


def main_cli(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = _resolve_device(args.device)

    human_name = "You"
    ai_name = "AI"
    ai_player = CheckpointPlayer(
        checkpoint_path=Path(args.checkpoint),
        device=device,
        deterministic=not bool(args.stochastic),
        name=ai_name,
    )

    if str(args.human_seat) == "1":
        player_names = [human_name, ai_name]
    else:
        player_names = [ai_name, human_name]

    agents = {
        human_name: main.HumanPlayer(),
        ai_name: ai_player,
    }

    print(
        f"starting human vs checkpoint: checkpoint={Path(args.checkpoint)} "
        f"device={device.type} human_seat={args.human_seat} "
        f"policy={'stochastic' if args.stochastic else 'greedy'}"
    )
    main.play_game(
        agents=agents,
        player_names=player_names,
        seed=int(args.seed),
        verbose=not bool(args.quiet),
        include_marine_world=bool(args.marine_world),
        private_viewer_names={human_name},
    )


if __name__ == "__main__":
    main_cli()

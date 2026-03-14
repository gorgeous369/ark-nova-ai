import pytest

import main


class _QuitHuman(main.HumanPlayer):
    def choose_action(self, state, actions):
        del state, actions
        raise SystemExit(0)


class _QuitAgent(main.PlayerAgent):
    def choose_action(self, state, actions):
        del state, actions
        raise SystemExit(0)


def test_play_game_hides_opponent_private_setup_info_for_viewer(capsys):
    with pytest.raises(SystemExit):
        main.play_game(
            agents={"You": _QuitHuman(), "AI": _QuitAgent()},
            player_names=["You", "AI"],
            seed=42,
            verbose=True,
            private_viewer_names={"You"},
        )

    output = capsys.readouterr().out

    assert "You: hidden" not in output
    assert "AI: hidden (2 cards)" in output
    assert "AI: hidden (8 draft cards, kept=4)" in output

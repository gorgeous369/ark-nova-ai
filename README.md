# ark-nova-ai

Ark Nova prototype focused on:

- rules-faithful 2-player game flow in Python
- explicit action-detail payloads (for agent integration)
- RL self-play training baseline (masked PPO / recurrent PPO / MAPPO)

## Project status

- Playable local 2-player loop exists (`main.py`).
- Core game logic is split between `main.py` and `arknova_engine/`.
- Automated test suite covers turn structure and many rule branches (`tests/`).
- RL pipeline exists but is baseline-grade and still evolving.

## Repository layout

- `main.py`: primary runtime, CLI entry, legal action generation, action application.
- `arknova_engine/`: extracted engine modules (setup, map model, scoring, card effects, action modules).
- `arknova_rl/`: observation/action encoding, model, PPO-style trainer.
- `tools/rl/train_self_play.py`: RL training entry point.
- `tools/`: dataset and utility scripts (cards/maps fetch, map tile tools, effect coverage report).
- `tests/`: pytest-based coverage.
- `docs/`: rule extracts, implementation checklists, and workflows.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` does not include PyTorch. Install `torch` separately if you want RL training.

## Run local game

```bash
.venv/bin/python main.py --seed 7
```

Useful flags:

- `--quiet`: reduce turn-by-turn logs
- `--marine-world`: include Marine World final scoring cards in setup pool

## Run tests

```bash
.venv/bin/pytest -q
```

Run a focused file:

```bash
.venv/bin/pytest -q tests/test_main_association_action.py
```

## RL self-play training

Entry script:

```bash
.venv/bin/python tools/rl/train_self_play.py \
  --algo masked_ppo \
  --updates 100 \
  --episodes-per-update 8 \
  --output-dir runs/self_play_masked
```

Available algorithms:

- `masked_ppo`
- `recurrent_ppo`
- `mappo`

Key model-size flags:

- `--hidden-size` (default `256`)
- `--lstm-size` (default `128`)
- `--action-hidden-size` (default `128`)

Key reward-shaping flags:

- `--step-reward-scale` (default `0.2`)
- `--terminal-reward-scale` (default `1.0`)
- `--endgame-trigger-reward` (default `2.0`)
- `--endgame-speed-bonus` (default `2.0`)
- `--terminal-win-bonus` (default `3.0`)
- `--terminal-loss-penalty` (default `3.0`)

Checkpoints are written to `--output-dir` as `checkpoint_XXXX.pt`.

## Data and utility scripts

Fetch cards dataset:

```bash
.venv/bin/python tools/fetch_cards.py --output data/cards/cards.json
```

Fetch maps dataset:

```bash
.venv/bin/python tools/fetch_maps.py --output data/maps/maps.json
```

Fetch maps + images:

```bash
.venv/bin/python tools/fetch_maps.py --download-images --images-dir data/maps/images
```

Generate effect coverage TSV:

```bash
.venv/bin/python tools/report_card_effect_coverage.py
```

Create map tile template:

```bash
.venv/bin/python tools/create_map_tile_template.py \
  --map-id m1a \
  --map-name "Observation Tower (Alternative)" \
  --image-name plan1a \
  --image-path data/maps/images/plan1a.jpg \
  --output data/maps/tiles/plan1a.tiles.json
```

See also: `docs/map_tile_annotation_workflow.md`.

## Scope caveat

This is an actively developed prototype. Use `docs/base_game_requirements.md` as the current correctness checklist for remaining rule gaps and validation status.

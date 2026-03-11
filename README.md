# ark-nova-ai

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

## Card data and audits

- `data/cards/cards.base.json`: base-game zoo card dataset used by the runtime.
- `data/cards/cards.marine_world.json`: Marine World card data, loaded when `--marine-world` is enabled.
- `data/cards/cards.promo.json`: promo card data, loaded by the dataset loader when present.
- `data/cards/card_effect_coverage.tsv`: generated effect-coverage report for zoo cards.
- `ark_nova_card_effect_audit.md`: review-oriented audit notes for dataset/runtime mismatches and missing branches.

When editing sponsor or animal card data, keep in mind that some cards are partly data-driven and partly runtime-driven:

- top-level card fields such as `appeal`, `conservation`, `reputation`, `required_icons`, and `max_appeal` are applied generically during play
- effect text in `effects` is descriptive data, but some cards also rely on hardcoded runtime hooks in `main.py`
- sponsor badges can come either from the dataset itself or from runtime overrides in `main.py`

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

Common maintenance commands:

```bash
# validate edited card JSON
python3 -c 'import json; json.load(open("data/cards/cards.base.json")); print("JSON OK")'

# dataset-focused regression checks
.venv/bin/python -m pytest -q tests/test_site_cards_dataset.py

# sponsor-rule regressions
.venv/bin/python -m pytest -q tests/test_main_sponsors_rules.py

# regenerate zoo-card effect coverage TSV
.venv/bin/python tools/report_card_effect_coverage.py
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

## Scope caveat

This is an actively developed prototype. Use `docs/base_game_requirements.md` as the current correctness checklist for remaining rule gaps and validation status.

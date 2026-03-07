# ark-nova-ai

Current focus: **Ark Nova base game PVP rules/runtime** with explicit action details for future RL integration.

## What is in this repository now

- `main.py`: Python playable PVP prototype with explicit per-action detail payloads.
- `arknova_engine/base_game.py`: animals+cards+build+association+sponsors+x-token+break official-flow engine.
- `arknova_engine/actions/`: separated implementations for 6 actions (`cards`, `build`, `animals`, `association`, `sponsors`, `x-token`).
- `third_party/tabletopgames_arknova/`: integrated open-source Ark Nova baseline from `jbargu/TabletopGames` (MIT).
- `docs/base_game_requirements.md`: implementation checklist for full base-game correctness.
- `tools/extract_rulebook_text.py`: utility to extract text from `ark-nova-rulebook.pdf`.
- `docs/open_source_engines.md`: candidate repository comparison for rules/reference engines.
- `arknova_engine/site_cards.py`: parser/fetcher for public card data.
- `tools/fetch_cards.py`: generates local card dataset JSON.
- `arknova_engine/site_maps.py`: parser/fetcher for public map data.
- `tools/fetch_maps.py`: generates local map dataset JSON.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Run

```bash
python3 main.py
```

## Test

```bash
.venv/bin/pytest -q
```

## Create map tile template (machine-readable)

```bash
.venv/bin/python tools/create_map_tile_template.py \
  --map-id m1a \
  --map-name "Observation Tower (Alternative)" \
  --image-name plan1a \
  --image-path data/maps/images/plan1a.jpg \
  --output data/maps/tiles/plan1a.tiles.json
```

See full workflow in `docs/map_tile_annotation_workflow.md`.

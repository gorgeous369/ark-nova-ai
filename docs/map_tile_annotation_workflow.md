# Map Tile Annotation Workflow (Program-Readable)

Goal: convert a map image (e.g. `plan1a`) into a machine-readable tile file.

## 1. Create a tile template

```bash
.venv/bin/python tools/create_map_tile_template.py \
  --map-id m1a \
  --map-name "Observation Tower (Alternative)" \
  --image-name plan1a \
  --image-path data/maps/images/plan1a.jpg \
  --effect "Gain 2 Appeal when flipping a standard enclosure adjacent to Observation Tower." \
  --effect "Observation Tower area itself is not a building." \
  --output data/maps/tiles/plan1a.tiles.json
```

This generates all 58 `(x,y)` grid tiles with default `terrain=plain`.

## 2. Fill tile properties

Edit `data/maps/tiles/plan1a.tiles.json`:

- `terrain`: one of
  - `plain`
  - `rock`
  - `water`
- `build2_required`: `true|false`
- `placement_bonus`: `null` or
  - `"<bonus_kind>"`
- `tags`: special labels, for this map use `["tower"]` on Observation Tower space.

Allowed `placement_bonus.kind`:

- `x_token`
- `5coins`
- `card_in_reputation_range`
- `reputation`
- `action_to_slot_1`

Coordinate axes:

- `x` positive direction: right-down
- `y` positive direction: right-up
- origin `(0,0)`: top-left tile

Recommended rule:

- Use only `rock`/`water` for non-buildable terrain.
- Mark build-upgrade restrictions via `build2_required=true`.
- Keep map-level continuous effects in `map_effects`.

## 3. Generate coordinate overlay

```bash
.venv/bin/python tools/render_map_overlay_svg.py \
  data/maps/tiles/plan1a.tiles.json \
  --output data/maps/tiles/plan1a.overlay.svg
```

If the grid does not line up, tune:

- `--x0` / `--y0` (origin for tile `(0,0)`)
- `--step-x` / `--step-y`
- `--hex-radius`

## 4. Validate

```bash
.venv/bin/python tools/validate_map_tiles.py data/maps/tiles/plan1a.tiles.json
```

Validation checks:

- full 58-tile coverage
- unique coordinates
- valid terrain values
- placement bonus schema

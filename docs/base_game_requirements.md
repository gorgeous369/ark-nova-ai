# Ark Nova Base Game Requirements (Official Rules Baseline)

Source: `ark-nova-rulebook.pdf` (uploaded in this repo, 19 pages).  
Scope: **Base game only** (no Marine Worlds / map packs / promos).

This file is the implementation checklist for building a correct rules engine.

## 1. Setup

- [ ] Global setup: board, Break token at player-count start, bonus tiles, deck/display, supplies.
- [ ] Personal setup: random Map A, 5 action cards side I, starting resources/counters/workers/tokens, starting cards.
- [ ] 2-player setup adjustments (blocking spaces/tokens).

## 2. Core Turn Structure

- [ ] Exactly one main action per turn (or X-token action).
- [ ] Action strength equals card slot value (1..5), plus spent X-tokens.
- [ ] After action: chosen action card moves to slot 1, others shift right.
- [ ] Card upgrades (side II) and their gating conditions.

## 3. Cards Action

- [ ] Side I draw/discard table by strength.
- [ ] Side I snapping allowed only at strength 5.
- [ ] Side II draw from display within reputation range and improved table.
- [ ] Side II snapping threshold reduced (strength >= 3).
- [ ] Always advance Break by 2 before draw/snap.
- [ ] Proper display refill timing/order and discard behavior.

## 4. Build Action

- [ ] Side I: build exactly one building, max size = strength, pay 2 money/hex.
- [ ] Legal placement rules: adjacency, first-build border rule, no overlap/out-of-map.
- [ ] Terrain restrictions: water/rock blocked, build-2 spaces require upgraded Build.
- [ ] Kiosk distance rule (minimum distance 3).
- [ ] Placement bonuses resolved immediately.
- [ ] Pavilion immediate appeal bonus.
- [ ] Side II: multiple different buildings up to total size strength.
- [ ] Side II unlocks Reptile House and Large Bird Aviary.
- [ ] Special enclosure uniqueness (one each where applicable).
- [ ] Full-map cover bonus (+7 appeal).

## 5. Animals Action

- [x] Side I/II play-count limits by strength.
- [x] Validate play conditions and icon requirements.
- [x] Cost payment and partner-zoo discounts.
- [x] Enclosure fit rules (standard/special), water/rock adjacency requirements.
- [x] Petting Zoo-only animals and special enclosure occupancy rules.
- [x] Appeal/reputation/conservation gains from card.
- [ ] Immediate vs “after finishing” timing.
- [ ] Proper card placement/discard for played/released animals.

## 6. Association Action

- [x] Worker placement limits per task slot (1/2 workers escalation).
- [x] Side I: exactly one task.
- [x] Side II: multiple different tasks, total value <= strength.
- [x] Tasks: reputation, partner zoo, partner university, conservation project work.
- [ ] Conservation project flow: play project card, support levels, token placement, rewards.
- [ ] Release-to-wild project behavior (discard animal, free enclosure capacity, appeal rollback).
- [x] Donation rule (side II, max 1 per action, at least one task performed first).
- [ ] Project row limits by player count and overflow discard handling.

## 7. Sponsors Action

- [x] Side I: play exactly one sponsor card up to strength, or Break-X gain-X.
- [x] Side II: play multiple sponsors with total level <= strength+1.
- [x] Side II: sponsor play from display with reputation-range + folder-cost rules.
- [x] Side II Break alternative doubles money gain.
- [x] Sponsor card condition checks and effect timing (immediate/ongoing/end-game).

## 8. X-token Action

- [x] Alternative action: gain exactly 1 X-token, still rotate action cards.
- [x] X-token cap (max 5).

## 9. Break Phase

- [ ] Trigger logic when Break reaches final space.
- [ ] Triggering player gains 1 X-token.
- [ ] Break sequence order:
- [ ] 1) hand-limit discard,
- [ ] 2) clear multiplier/venom/constriction tokens,
- [ ] 3) recall association workers + refill zoo/university offers,
- [ ] 4) discard display folders 1-2, shift/refill,
- [ ] 5) income (appeal, kiosk adjacency, recurring income icons),
- [ ] 6) reset Break token.

## 10. End Game & Scoring

- [ ] End trigger when appeal and conservation markers cross/meet in same scoring area.
- [ ] Finish current turn and one final turn for all others.
- [ ] Final scoring cards handling (including 10-conservation discard rule).
- [ ] End-game sponsor/final-score effects.
- [ ] Victory-point computation and tie-breakers per rulebook.

## 11. AI Requirements (for this project)

- [ ] Full legal action generation for current game state.
- [ ] Deterministic rules engine stepping for simulation.
- [ ] Hidden-information support (deck/hand uncertainty) for search.
- [ ] Baseline AI (heuristic/MCTS) against random and scripted bots.
- [ ] Regression tests from fixed seeds/game logs.

## Current status in this repository

- [x] Upstream open-source baseline integrated (`third_party/tabletopgames_arknova`).
- [x] Rulebook extraction tooling (`tools/extract_rulebook_text.py`).
- [x] Build + Break core engine module (`arknova_engine/base_game.py`) with tests.
- [x] Map/build legality module (`arknova_engine/map_model.py`) with tests.
- [x] Full Cards action rules (draw/snap/reputation range/discard/display refill flow).
- [ ] Full Animals action rules.
- [x] Animals core rules (play limits, conditions, discounts, enclosure validation, display play on side II).
- [ ] Full Association action rules.
- [x] Association core rules (task-value/worker limits, partner zoo/university/reputation, donation, display project cost/range).
- [ ] Full Sponsors action rules.
- [x] Sponsors core rules (side I/II play and break modes, level cap, display cost/range, conditions).
- [x] X-token alternative action rules.
- [ ] Full base-game rules compliance.

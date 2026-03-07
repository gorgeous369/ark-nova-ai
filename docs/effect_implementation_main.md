# Ark Nova Effect Implementation Matrix (main.py)

## Scope
- Runtime path: [`main.py`](/Users/seishoazuma/PycharmProjects/ark-nova-ai/main.py)
- Effect resolver/executor: [`arknova_engine/card_effects.py`](/Users/seishoazuma/PycharmProjects/ark-nova-ai/arknova_engine/card_effects.py)
- Row-level effect snapshot (generated): [`data/cards/card_effect_coverage.tsv`](/Users/seishoazuma/PycharmProjects/ark-nova-ai/data/cards/card_effect_coverage.tsv)

## Animal Effects Implemented In `main.py`

Notes:
- `effect_code` is from `resolve_card_effect(...)`.
- `base_count` is current count in `cards.base.json` (can be 0 for expansion-only or not-yet-collected cards).

| effect_code | ability mapping (resolver) | implemented behavior (executor) | base_count |
|---|---|---|---:|
| draw_from_deck | `Sprint N` | draw N cards from deck | 4 |
| hunter | `Hunter N` | reveal N from deck, keep first animal, discard rest | 5 |
| take_display_cards | `Snapping N` | take display cards (with/without replenish each pick) | 5 |
| draw_keep_from_deck | `Perception N` | draw N, keep K, discard rest | 3 |
| scavenge_from_discard | `Scavenging N` | shuffle discard, draw N, keep K, rest back to discard | 6 |
| move_action_to_slot_1 | `Boost: <Action>` | move specific action card to slot 1 | 7 |
| move_any_action_to_slot_1 | `Clever` | choose any action card and move to slot 1 | 6 |
| hire_worker | `Full-throated` | gain workers | 1 |
| gain_appeal | `Pack`, `Petting Zoo Animal` | gain appeal | 11 |
| advance_break | `Glide N` | advance break token by N | 0 |
| jumping_break_and_money | `Jumping N` | advance break and gain money | 4 |
| sell_hand_cards | `Sun Bathing N` | sell/discard hand cards for money | 7 |
| discard_hand_for_money | `Pouch N` | discard hand cards for money | 5 |
| digging_cycle | `Digging N` | iterative draw/sell loop N times | 5 |
| pilfering_or_money | `Pilfering N` | gain fixed money branch | 3 |
| free_kiosk_or_pavilion | `Posturing N` | place free kiosk/pavilion N times | 5 |
| venom_tokens | `Venom N` | apply venom tokens to leading opponents | 3 |
| constriction_tokens | `Constriction` | apply constriction tokens by tracks-ahead | 3 |
| take_display_sponsors | `Sponsor Magnet` | take sponsor cards from display | 2 |
| multiplier_token | `Multiplier: <Action>` | add multiplier token on action | 6 |
| extra_action_granted | `Action: <Action>` | grant extra action of that type | 4 |
| gain_conservation | `Iconic Animal: <...>` | gain conservation points | 5 |
| draw_final_scoring_keep | `Resistance` | draw final scoring cards, keep K | 2 |
| take_unused_base_project | `Assertion` | take unused base conservation project card | 2 |
| extra_action_any | `Determination` | grant 1 extra action (any) | 2 |
| extra_action_strength | `Hypnosis N` | grant extra action with strength N | 2 |
| gain_x_tokens | `Inventive` | gain X tokens | 2 |
| gain_x_tokens_from_icon | `Inventive: Bear` | gain X by icon count | 1 |
| gain_x_tokens_from_primary | `Inventive: Primary` | gain X by primary icons (cap by value) | 2 |
| flock_optional | `Flock Animal N` | recognized and logged (no extra board mutation) | 3 |
| symbiosis_copy | `Symbiosis` | copy latest valid animal effect in own zoo | 0 |
| camouflage_grant | `Camouflage` | grant ignore-prerequisite charges | 0 |
| scuba_dive_x | `Scuba Dive X` | reveal X and keep sponsor candidate | 0 |
| reveal_until_badge | `Monkey Gang` | reveal until target badge, keep hit | 0 |
| adapt_final_scoring | `Adapt N` | draw/discard final scoring adaptation | 0 |
| remove_empty_enclosure_refund | `Cut Down` | remove empty enclosure(s) and refund | 0 |
| return_association_worker | `Extra Shift` | return worker(s) from association board | 0 |
| mark_display_animal | `Mark` | mark display animal(s) within reputation range | 0 |
| place_free_large_bird_aviary | `Peacocking` | place free Large Bird Aviary | 1 |
| take_display_by_badge | `Sea Animal Magnet` | take display cards filtered by badge | 0 |
| shark_attack | `Shark Attack N` | discard display animals, gain money from appeal | 0 |
| trade_hand_with_display | `Trade` | trade hand card with display card | 0 |
| take_specific_base_project | `Dominance` | take specific base project from unused pool | 1 |

## Sponsor Effects In `main.py` (Current Status)

Sponsors are no longer stubbed in `main.py`:
- Side I / Side II sponsor play flow exists.
- Break alternative exists (with side-II double-money behavior).
- Prerequisite checks and many base-card effects are wired.

See detailed sponsor matrix:
- [`docs/sponsor_effect_implementation_main.md`](/Users/seishoazuma/PycharmProjects/ark-nova-ai/docs/sponsor_effect_implementation_main.md)

Remaining gaps are mostly unique-building and map-topology heavy sponsor effects plus some advanced passive chains.

## Where richer sponsor action logic already exists
- Separate engine module (not wired to `main.py` runtime):  
  [`arknova_engine/actions/sponsors_action.py`](/Users/seishoazuma/PycharmProjects/ark-nova-ai/arknova_engine/actions/sponsors_action.py)

## Quick Verification Commands

```bash
.venv/bin/python tools/report_card_effect_coverage.py --output data/cards/card_effect_coverage.tsv
.venv/bin/python -m pytest -q
```

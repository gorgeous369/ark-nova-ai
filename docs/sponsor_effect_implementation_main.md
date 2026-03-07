# Sponsor Effect Implementation Status (main.py)

Scope:
- Runtime: `main.py`
- Source reference for sponsor intent: `https://www.sonia-g.co.jp/boardgame/?アークノヴァ/後援者カード`
- Card pool used by runtime: `data/cards/cards.base.json`

Legend:
- `OK`: implemented in current runtime.
- `PARTIAL`: some parts implemented, some parts still TODO.
- `TODO`: not implemented yet (or only logged as note).

## Action Layer
- `OK` Sponsors action now supports:
  - Play sponsor cards from hand (side I).
  - Play sponsor cards from hand/display-in-reputation-range (side II).
  - Side I level cap = strength, side II level cap = strength+1.
  - Display extra cost (folder index).
  - Break alternative (money + break advance; side II doubles money).
- `OK` Sponsor prerequisites checked:
  - `sponsorsII` requirement.
  - `reputation >= N` (with key overrides from Sonia page).
  - `appeal <= 25` condition cards.
  - icon/partner-zoo requirements.

## Break Income Layer
- `OK` #201: income draw 1 card (deck or reputation range).
- `OK` #206: income +1 conservation.
- `OK` #209: income +1 X-token.
- `OK` #220: income +3 money.
- `OK` #231-#235: income 3/6/9 money by 1/3/5 matching icons.
- `OK` #257 income by adjacent buildings to Side Entrance (excluding empty standard enclosures).

## Passive Trigger Layer (card-play triggers)
- `OK` #202, #204, #208 science triggers.
  - Trigger resolution is by icon count on the played card (not only once per card).
- `OK` #214 africa trigger (move one action card to slot 1; auto picks current last card).
- `OK` #227 small/large assignment:
  - play restriction applied to animals.
  - +2/+4 appeal trigger on matching small/large animal play.
- `OK` #236-#240 global category triggers (+3 money).
- `OK` #243-#247 own-zoo icon triggers (+2 appeal).
- `OK` #248 primate trigger (+1 X-token).
- `OK` #249 bird trigger (Perception 2 style draw2 keep1).
- `OK` #250 reptile trigger (Sun Bathing 2 style sell up to 2 hand cards for 4 each).
- `OK` #251 bear global trigger (+2 appeal).
- `OK` #252 predator trigger (Hunter N using current predator icon count).
- `OK` #262 “new type gained” trigger (+2 money, +1 appeal per newly introduced continent/category type).
- `OK` #263 passive condition-ignore for large animals (1 condition).
- `OK` #210/#211/#213 free building placements.
- `OK` #217 extra same-type build after Build action.
- `OK` #228 extra small-animal flow.
- `OK` #253 token-spend sponsor chaining.
- `OK` #224 release-into-wild special repeat/extra CP behavior.
- `PARTIAL` #225 immunity:
  - implemented: venom/constriction immunity.
  - not modeled in this runtime: Hypnosis/Pilfering opponent-targeting branch (core simplified engine path has no such targeting).

## Immediate Effects (play-time)
- `OK` #201, #254 draw 1 card (deck or reputation range).
- `OK` #203 universities -> money (2/5/10).
- `OK` #204 science x2 money.
- `OK` #205 +2 reputation, +1 conservation.
- `OK` #206 appeal from supported base projects x2.
- `OK` #207 CP from distinct types / opponents gain money.
- `OK` #208 science -> appeal.
- `OK` #209 +1 X-token.
- `OK` #210-#214 continent icons -> appeal.
- `OK` #216 +1 worker.
- `OK` #219 water+rock icons -> money.
- `OK` #220 +3 money.
- `OK` #222 science -> CP (cap 3), opponents gain money.
- `OK` #224/#225 +1 X-token.
- `OK` #226 +2 reputation.
- `OK` #227 choose small/large, reveal-until-target-size and keep.
- `OK` #228 small animals -> money (x2).
- `OK` #229 small animals -> appeal.
- `OK` #230 large animals -> appeal (x2).
- `OK` #231-#235 matching category icons -> appeal.
- `OK` #241 water icons -> appeal.
- `OK` #242 rock pairs -> appeal.
- `OK` #255/#256 +4 appeal.
- `OK` #261 +1 conservation, +1 appeal.
- `OK` #262 distinct types -> money.
- `OK` #263 free size-5 enclosure placement (first legal placement, cost refunded).
- `OK` #243-#253/#257/#258/#259/#260/#264 unique-building/map-topology effects are executed in runtime.

## Data Fixes Applied
- Base dataset re-split from full card dataset:
  - `data/cards/cards.base.json` regenerated from `data/cards/cards.json`.
- Runtime overrides applied for known base sponsor data gaps:
  - level overrides: #232-#236.
  - requirement overrides: side-II, rep thresholds, appeal<=25, sponsorship category requirements.
  - badge/icon overrides for cards with incomplete icon metadata.
  - sponsor badge counting uses override values as source-of-truth when present (prevents dataset+override double count).

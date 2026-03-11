# Ark Nova Card Effect Audit

Generated from `data/cards/cards.base.json` + `data/cards/cards.marine_world.json` against current runtime code.

## Summary

- Animals: 161 cards. 149 have executable effects, 12 have no effect, 0 parsed animal effects are currently unimplemented.
- Sponsors: 82 cards. 64 look implemented in code, 17 have dataset effect text but no current code branch (`265-276, 278-282`), 1 has no explicit effect data (`223`).
- Final scoring: 17 cards, all 17 are hardcoded in `main.py:5914 _final_scoring_conservation_bonus_for_card`.
- Conservation projects: 32 cards. 22 have explicit requirement logic, 10 (`P113-P122`) currently fall through default requirement value `0` and should be treated as not wired.

## Animal Cards

Unified execution path: `arknova_engine/card_effects.py:76 resolve_card_effect -> arknova_engine/card_effects.py:316 apply_animal_effect -> main.py:8983 _perform_animals_action_effect`.

- `action` (4): #409 SUN BEAR, #430 PYGMY HIPPOPOTAMUS, #453 COLLARED MANGABEY, #515 AUSTRALIAN PELICAN
  status: implemented
- `adapt` (2): #553 TAMBAQUI, #554 AFRICAN PENGUIN
  status: implemented
- `assertion` (2): #427 WHITE RHINOCEROS, #432 INDIAN RHINOCEROS
  status: implemented
- `boost` (9): #408 SLOTH BEAR, #415 RACCOON, #419 EUROPEAN BADGER, #428 GIRAFFE, #429 GREVY'S ZEBRA, #466 BOLIVIAN RED HOWLER, #502 GREAT HORNBILL, #536 LONGHORN COWFISH, #556 WOLVERINE
  status: implemented
- `camouflage` (2): #548 LINED SEAHORSE, #549 COMMON OCTOPUS
  status: implemented
- `clever` (6): #405 FENNEC FOX, #455 MANTLED GUEREZA, #460 DUSKY-LEAF MONKEY, #465 GOLDEN LION TAMARIN, #467 ECUADORIAN SQUIRELL MONKEY, #487 EUROPEAN GRASS SNAKE
  status: implemented
- `constriction` (4): #474 INDIAN ROCK PYTHON, #482 ANACONDA, #483 BOA CONSTRICTOR, #537 BLACKBAR TRIGGERFISH
  status: implemented
- `cut_down` (1): #545 LONGCOMB SAWFISH
  status: implemented
- `determination` (2): #505 BALD EAGLE, #509 GOLDEN EAGLE
  status: implemented
- `digging` (6): #435 MALAYAN TAPIR, #440 MOUNTAIN TAPIR, #445 CRESTED PORCUPINE, #446 DUGONG, #524 MANGALICA, #541 BLUESPOTTED RIBBONTAIL RAY
  status: implemented
- `dominance` (1): #451 PROBOSCIS MONKEY
  status: implemented
- `extra_shift` (1): #542 HUMPHEAD WRASSE
  status: implemented
- `flock_animal` (3): #438 REINDEER, #439 LAMA, #443 RED DEER
  status: implemented
- `full_throated` (1): #421 NEW ZEALAND FUR SEAL
  status: implemented
- `glide` (2): #543 COASTAL MANTA RAY, #550 COMPASS JELLYFISH
  status: implemented
- `hunter` (6): #403 LEOPARD, #404 CARACAL, #410 YELLOW-THROATED MARTEN, #412 JAGUAR, #420 STOAT, #560 BRAHMINY KITE
  status: implemented
- `hypnosis` (2): #475 KING COBRA, #485 COMMON EUROPEAN ADDER
  status: implemented
- `iconic_animal` (5): #418 EURASIAN LYNX, #436 AMERICAN BISON, #452 SENEGAL BUSHBABY, #476 KOMODO DRAGON, #517 LAUGHING KOOKABURRA
  status: implemented
- `inventive` (2): #414 SOUTH AMERICAN COATI, #522 DONKEY
  status: implemented
- `inventive_bear` (1): #411 GRIZZLY BEAR
  status: implemented
- `inventive_primary` (2): #459 RED-SHANKED DOUC, #464 BROWN SPIDER MONKEY
  status: implemented
- `jumping` (4): #413 COUGAR, #444 ALPINE IBEX, #461 HORSFIELD'S TARSIER, #462 NORTHERN PLAINS GRAY LANGUR
  status: implemented
- `mark` (1): #529 MAGNIFICENT SEA ANEMONE
  status: implemented
- `monkey_gang` (2): #558 NORTHERN MURIQUI, #559 COQUERELS SIFAKA
  status: implemented
- `multiplier` (6): #416 EURASIAN BROWN BEAR, #434 RED PANDA, #442 MOOSE, #457 MANDRILL, #510 WHITE STORK, #516 NORTHERN CASSOWARY
  status: implemented
- `none` (12): #406 SIBERIAN TIGER, #407 SUMATRAN TIGER, #433 GIANT PANDA, #468 COTTON-TOP TAMARIN, #484 EUROPEAN POND TURTLE, #486 COMMON WALL LIZARD, #488 SLOW WORM, #493 THORNY DEVIL, #495 SECRETARY BIRD, #498 SHOEBILL, #530 ORANGE CLOWNFISH, #539 AMERICAN WHITESPOTTED FILEFISH
  status: no_effect
- `pack` (4): #402 LION, #417 WOLF, #423 NEW ZEALAND SEA LION, #424 AUSTRALIAN DINGO
  status: implemented
- `peacocking` (1): #514 EMU
  status: implemented
- `perception` (3): #503 SNOWY OWL, #512 EURASIAN EAGLE-OWL, #513 BARN OWL
  status: implemented
- `petting_zoo_animal` (9): #341 CAPYBARA, #519 (DOMESTIC) GOAT, #520 SHEEP, #521 HORSE, #523 DOMESTIC RABBIT, #525 GUINEA PIG, #526 ALPACA, #527 COCONUT LORIKEET, #557 Vietnamese Pot-Bellied Pig
  status: implemented
- `pilfering` (4): #456 BARBARY MACAQUE, #458 JAPANESE MACAQUE, #463 PANAMANIAN WHITE-FACED CAPUCHIN, #555 GOLDEN SNUB-NOSED MONKEY
  status: implemented
- `posturing` (6): #497 LESSER FLAMINGO, #501 INDIAN PEAFOWL, #508 SCARLET MACAW, #511 GREATER FLAMINGO, #518 LESSER BIRD-OF-PARADISE, #533 BLACKSIDE HAWKFISH
  status: implemented
- `pouch` (5): #425 TASMANIAN DEVIL, #447 RED KANGAROO, #448 KOALA, #450 COMMON WOMBAT, #528 BENNETT'S WALLABY
  status: implemented
- `resistance` (2): #426 AFRICAN BUSH ELEPHANT, #431 ASIAN ELEPHANT
  status: implemented
- `scavenging` (6): #490 GOULD'S MONITOR, #496 MARABOU, #499 CINEREOUS VULTURE, #500 LONG-BILLED VULTURE, #504 ANDEAN CONDOR, #506 KING VULTURE
  status: implemented
- `scuba_dive_x` (2): #551 LOGGERHEAD SEA TURTLE, #552 GREEN SEA TURTLE
  status: implemented
- `sea_animal_magnet` (1): #532 ZOOPLANKTON
  status: implemented
- `shark_attack` (2): #544 CARIBBEAN REEF SHARK, #546 SAND TIGER SHARK
  status: implemented
- `snapping` (5): #469 NILE CROCODILE, #477 VEILED CHAMELEON, #479 AMERICAN ALLIGATOR, #480 BROAD-SNOUTED CAIMAN, #489 SALTWATER CROCODILE
  status: implemented
- `sponsor_magnet` (2): #437 MUSKOX, #441 EUROPEAN BISON
  status: implemented
- `sprint` (4): #401 CHEETAH, #491 FRILLED LIZARD, #494 AFRICAN OSTRICH, #507 GREATER RHEA
  status: implemented
- `sun_bathing` (7): #422 AUSTRALIAN SEA LION, #454 RING-TAILED LEMUR, #471 AFRICAN SPURRED TORTOISE, #472 ROCK MONITOR, #473 COMMON AGAMA, #478 CHINESE WATER DRAGON, #481 GALAPAGOS GIANT TORTOISE
  status: implemented
- `symbiosis` (2): #535 SHARKNOSE GOBY, #547 MEDITERRANEAN RAINBOW
  status: implemented
- `trade` (1): #531 PALETTE SURGEONFISH
  status: implemented
- `venom` (6): #449 PLATYPUS, #470 WESTERN GREEN MAMBA, #492 INLAND TAIPAN, #534 SOUTHERN BLUE-RINGED OCTOPUS, #538 DEVIL FIREFISH, #540 GUINEAFOWL PUFFER
  status: implemented

## Sponsor Cards

Runtime pipeline: `main.py:12587 _finalize_sponsor_card_play -> main.py:13362 _apply_sponsor_immediate_effects immediate / main.py:12727 _apply_sponsor_passive_triggers_on_card_play passive / main.py:7874 _apply_sponsor_break_income_effects_for_player_after_card_draw income / main.py:5737 _sponsor_endgame_bonus endgame`.

### Implemented / wired sponsor cards

- #201 SCIENCE LAB | dataset kinds: endgame,immediate,income | code: immediate,income,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:7896 _apply_sponsor_break_income_effects_for_player | main.py:5737 _sponsor_endgame_bonus
- #202 SPOKESPERSON | dataset kinds: passive | code: passive | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play
- #203 VETERINARIAN | dataset kinds: immediate,passive | code: immediate,endgame,rule_hook | refs: main.py:14282 _apply_sponsor_immediate_effects | main.py:6057 _sponsor_endgame_bonus | main.py:1510 _association_task_strength_cost
- #204 SCIENCE MUSEUM | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play
- #205 GORILLA FIELD RESEARCH | dataset kinds: - | code: immediate | refs: main.py:13362 _apply_sponsor_immediate_effects
- #206 MEDICAL BREAKTHROUGH | dataset kinds: immediate,income | code: immediate,income | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:7874 _apply_sponsor_break_income_effects_for_player_after_card_draw
- #207 BASIC RESEARCH | dataset kinds: immediate | code: immediate | refs: main.py:13362 _apply_sponsor_immediate_effects
- #208 SCIENCE LIBRARY | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:5737 _sponsor_endgame_bonus
- #209 TECHNOLOGY INSTITUTE | dataset kinds: endgame,immediate,income | code: immediate,income,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:7874 _apply_sponsor_break_income_effects_for_player_after_card_draw | main.py:5737 _sponsor_endgame_bonus
- #210 EXPERT ON THE AMERICAS | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:5737 _sponsor_endgame_bonus
- #211 EXPERT ON EUROPE | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:5737 _sponsor_endgame_bonus
- #212 EXPERT ON AUSTRALIA | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play
- #213 EXPERT ON ASIA | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play
- #214 EXPERT ON AFRICA | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:5737 _sponsor_endgame_bonus
- #215 BREEDING COOPERATION | dataset kinds: endgame,passive | code: endgame,rule_hook | refs: main.py:5737 _sponsor_endgame_bonus | main.py:12587 _finalize_sponsor_card_play | main.py:8725 _available_breeding_icon_reduction | main.py:8732 _consume_breeding_icon_reduction
- #216 TALENTED COMMUNICATOR | dataset kinds: endgame,immediate | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #217 ENGINEER | dataset kinds: endgame,passive | code: endgame,rule_hook | refs: main.py:5737 _sponsor_endgame_bonus | main.py:8238 _perform_build_action_effect
- #218 BREEDING PROGRAM | dataset kinds: endgame,passive | code: endgame,rule_hook | refs: main.py:5737 _sponsor_endgame_bonus | main.py:12587 _finalize_sponsor_card_play | main.py:8725 _available_breeding_icon_reduction | main.py:8732 _consume_breeding_icon_reduction
- #219 DIVERSITY RESEARCHER | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:5737 _sponsor_endgame_bonus | main.py:8238 _perform_build_action_effect | main.py:8983 _perform_animals_action_effect
- #220 FEDERAL GRANTS | dataset kinds: endgame,immediate,income | code: immediate,income,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:7874 _apply_sponsor_break_income_effects_for_player_after_card_draw | main.py:5737 _sponsor_endgame_bonus
- #221 ARCHEOLOGIST | dataset kinds: endgame,passive | code: endgame,rule_hook | refs: main.py:5737 _sponsor_endgame_bonus | main.py:6859 _apply_build_placement_bonus
- #222 RELEASE OF PATENTS | dataset kinds: immediate | code: immediate | refs: main.py:13362 _apply_sponsor_immediate_effects
- #224 MIGRATION RECORDING | dataset kinds: immediate,passive | code: immediate,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:11697 list_legal_association_options | main.py:12060 _apply_association_selected_option
- #225 QUARANTINE LAB | dataset kinds: endgame,immediate,passive | code: immediate,endgame,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus | main.py:8983 _perform_animals_action_effect | main.py:1945 _eligible_highest_track_target_ids
- #226 FOREIGN INSTITUTE | dataset kinds: endgame | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #227 WAZA SPECIAL ASSIGNMENT | dataset kinds: immediate,passive | code: immediate,passive,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:13291 _validate_sponsor_effect_details
- #228 WAZA SMALL ANIMAL PROGRAM | dataset kinds: immediate,passive | code: immediate,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:8983 _perform_animals_action_effect
- #229 EXPERT IN SMALL ANIMALS | dataset kinds: immediate,passive | code: immediate,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:8792 _animal_size_sponsor_discount
- #230 EXPERT IN LARGE ANIMALS | dataset kinds: immediate,passive | code: immediate,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:8792 _animal_size_sponsor_discount
- #231 SPONSORSHIP: PRIMATES | dataset kinds: immediate,income | code: immediate,income | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:338 SPONSOR_ICON_INCOME_THRESHOLDS
- #232 SPONSORSHIP: REPTILES | dataset kinds: immediate,income | code: immediate,income | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:338 SPONSOR_ICON_INCOME_THRESHOLDS
- #233 SPONSORSHIP: VULTURES | dataset kinds: immediate,income | code: immediate,income | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:338 SPONSOR_ICON_INCOME_THRESHOLDS
- #234 SPONSORSHIP: LIONS | dataset kinds: immediate,income | code: immediate,income | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:338 SPONSOR_ICON_INCOME_THRESHOLDS
- #235 SPONSORSHIP: ELEPHANTS | dataset kinds: immediate,income | code: immediate,income | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:338 SPONSOR_ICON_INCOME_THRESHOLDS
- #236 PRIMATOLOGIST | dataset kinds: passive | code: passive | refs: main.py:324 SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS
- #237 HERPETOLOGIST | dataset kinds: passive | code: passive | refs: main.py:324 SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS
- #238 ORNITHOLOGIST | dataset kinds: passive | code: passive | refs: main.py:324 SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS
- #239 EXPERT IN PREDATORS | dataset kinds: passive | code: passive | refs: main.py:324 SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS
- #240 EXPERT IN HERBIVORES | dataset kinds: passive | code: passive | refs: main.py:324 SPONSOR_GLOBAL_ICON_MONEY_TRIGGERS
- #241 HYDROLOGIST | dataset kinds: endgame,immediate,passive | code: immediate,endgame,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus | main.py:6593 _apply_cover_money_from_sponsors_241_242 | main.py:8238 _perform_build_action_effect | main.py:6328 _place_sponsor_unique_building
- #242 GEOLOGIST | dataset kinds: endgame,immediate,passive | code: immediate,endgame,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus | main.py:6593 _apply_cover_money_from_sponsors_241_242 | main.py:8238 _perform_build_action_effect | main.py:6328 _place_sponsor_unique_building
- #243 MEERKAT DEN | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:5737 _sponsor_endgame_bonus | main.py:331 SPONSOR_OWN_ICON_APPEAL_TRIGGERS | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #244 PENGUIN POOL | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:5737 _sponsor_endgame_bonus | main.py:331 SPONSOR_OWN_ICON_APPEAL_TRIGGERS | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #245 AQUARIUM | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:5737 _sponsor_endgame_bonus | main.py:331 SPONSOR_OWN_ICON_APPEAL_TRIGGERS | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #246 CABLE CAR | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:5737 _sponsor_endgame_bonus | main.py:331 SPONSOR_OWN_ICON_APPEAL_TRIGGERS | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #247 BABOON ROCK | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:5737 _sponsor_endgame_bonus | main.py:331 SPONSOR_OWN_ICON_APPEAL_TRIGGERS | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #248 RHESUS MONKEY PARK | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #249 BARRED OWL HUT | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #250 SEA TURTLE TANK | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #251 POLAR BEAR EXHIBIT | dataset kinds: endgame,immediate,passive | code: immediate,passive,endgame | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:5737 _sponsor_endgame_bonus | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #252 SPOTTED HYENA COMPOUND | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #253 OKAPI STABLE | dataset kinds: immediate,passive | code: immediate,passive,rule_hook | refs: main.py:12727 _apply_sponsor_passive_triggers_on_card_play | main.py:12479 _play_sponsor_from_hand_via_253 | main.py:12587 _finalize_sponsor_card_play | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #254 ZOO SCHOOL | dataset kinds: immediate | code: immediate | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #255 ADVENTURE PLAYGROUND | dataset kinds: immediate | code: immediate | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #256 WATER PLAYGROUND | dataset kinds: immediate | code: immediate | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #257 SIDE ENTRANCE | dataset kinds: endgame,immediate,income | code: immediate,income,endgame | refs: main.py:7874 _apply_sponsor_break_income_effects_for_player_after_card_draw | main.py:5737 _sponsor_endgame_bonus | main.py:387 SPONSOR_UNIQUE_BUILDING_CARDS
- #258 NATIVE SEABIRDS | dataset kinds: endgame,immediate | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #259 NATIVE LIZARDS | dataset kinds: endgame,immediate | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #260 NATIVE FARM ANIMALS | dataset kinds: endgame,immediate | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #261 GUIDED SCHOOL TOURS | dataset kinds: endgame | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #262 EXPLORER | dataset kinds: immediate,passive | code: immediate,passive | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:12727 _apply_sponsor_passive_triggers_on_card_play
- #263 WAZA LARGE ANIMAL PROGRAM | dataset kinds: immediate,passive | code: immediate,rule_hook | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:8755 _animal_condition_ignore_capacity | main.py:13291 _validate_sponsor_effect_details
- #264 FREE-RANGE NEW WORLD MONKEYS | dataset kinds: endgame,immediate | code: immediate,endgame | refs: main.py:13362 _apply_sponsor_immediate_effects | main.py:5737 _sponsor_endgame_bonus
- #277 FIELD RESEARCH TYPE D ORCAS | dataset kinds: - | code: immediate | refs: main.py:13430 _finalize_sponsor_card_play

### Sponsor cards with dataset effect text but no current code branch

- #265 FRANCHISE BUSINESS | dataset kinds: endgame,immediate,income
- #266 MARINE BIOLOGIST | dataset kinds: endgame,immediate,income,passive
- #267 FARM CAT | dataset kinds: endgame
- #268 CONFERENCE ON EUROPE | dataset kinds: endgame,passive
- #269 CONFERENCE ON AUSTRALIA | dataset kinds: endgame,passive
- #270 MARINE RESEARCH EXPEDITION | dataset kinds: passive
- #271 EXCAVATION SITE | dataset kinds: endgame,immediate
- #272 EXPANSION AREA | dataset kinds: endgame,passive
- #273 PUBLICATIONS | dataset kinds: immediate,passive
- #274 MASCOT STATUE | dataset kinds: endgame,income
- #275 HORSE WHISPERER | dataset kinds: immediate,passive
- #276 LANDSCAPE GARDENER | dataset kinds: passive
- #278 AMAZON HOUSE | dataset kinds: immediate
- #279 UNDERWATER TUNNEL | dataset kinds: endgame,immediate,passive
- #280 RECONSTRUCTION | dataset kinds: immediate,passive
- #281 ARCADE | dataset kinds: endgame,income
- #282 PROMOTION TEAM | dataset kinds: immediate,passive

### Sponsor cards with no explicit effect data in the dataset

- #223 SCIENCE INSTITUTE

## Final Scoring Cards

- F001_LargeAnimalZoo Large Animal Zoo | formula: large animals threshold bonus: 1/2/3/4 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F002_SmallAnimalZoo Small Animal Zoo | formula: small animals threshold bonus: 3/6/8/10 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F003_ResearchZoo Research Zoo | formula: science icons threshold bonus: 3/4/5/7 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F004_ArchitecturalZoo Architectural Zoo | formula: sum of 4 booleans: all water adj, all rock adj, all border adj, map fully covered | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F005_ConservationZoo Conservation Zoo | formula: supported conservation project actions: 2/3/4/5 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F006_NaturalistsZoo Naturalists Zoo | formula: empty fillable spaces: 6/12/18/24 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F007_FavoriteZoo Favorite Zoo | formula: reputation: 6/9/12/15 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F008_SponsoredZoo Sponsored Zoo | formula: played sponsor cards: 3/5/7/9 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F009_DiverseSpeciesZoo Diverse Species Zoo | formula: compare category counts vs next player, max 4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F010_ClimbingPark Climbing Park | formula: rock icons: 1/3/5/6 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F011_AquaticPark Aquatic Park | formula: water icons: 2/4/6/7 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F012_DesignerZoo Designer Zoo | formula: distinct building shapes: 4/6/7/8 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F013_SpecializedHabitatZoo Specialized Habitat Zoo | formula: best unsupported continent project icon count: 3/4/5/6 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F014_SpecializedSpeciesZoo Specialized Species Zoo | formula: best unsupported species project icon count: 3/4/5/6 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F015_CateredPicnicAreas Catered Picnic Areas | formula: min(kiosks, pavilions): 2/3/4/5 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F016_AccessibleZoo Accessible Zoo | formula: count card conditions in zoo: 4/7/10/12 -> 1/2/3/4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card
- F017_InternationalZoo International Zoo | formula: compare continent counts vs next player, partner zoos included, max 4 CP | ref: main.py:5914 _final_scoring_conservation_bonus_for_card

## Conservation Projects

- P101 SPECIES DIVERSITY | status: implemented | requirement: - | rewards: unique animal category icons count ; 5/4/3 icons -> left/mid/right gives 5/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P102 HABITAT DIVERSITY | status: implemented | requirement: - | rewards: unique continent icons count ; 5/4/3 icons -> left/mid/right gives 5/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P103 AFRICA | status: implemented | requirement: - | rewards: Africa icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P104 AMERICAS | status: implemented | requirement: - | rewards: America icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P105 AUSTRALIA | status: implemented | requirement: - | rewards: Australia icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P106 ASIA | status: implemented | requirement: - | rewards: Asia icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P107 EUROPE | status: implemented | requirement: - | rewards: Europe icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P108 PRIMATES | status: implemented | requirement: - | rewards: Primate icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P109 REPTILES | status: implemented | requirement: - | rewards: Reptile icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P110 PREDATORS | status: implemented | requirement: - | rewards: Predator icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P111 HERBIVORES | status: implemented | requirement: - | rewards: Herbivore icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P112 BIRDS | status: implemented | requirement: - | rewards: Bird icons count ; default 5/4/2 icons -> 5/4/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P113 BAVARIAN FOREST NATIONAL PARK | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P114 YOSEMITE NATIONAL PARK | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P115 ANGTHONG NATIONAL PARK | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P116 SERENGETI NATIONAL PARK | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P117 BLUE MOUNTAINS NATIONAL PARK | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P118 SAVANNA | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P119 LOW MOUNTAIN RANGE | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P120 BAMBOO FOREST | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P121 SEA CAVE | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P122 JUNGLE | status: unimplemented | requirement: - | rewards: falls through default `_project_requirement_value` return 0 ; generic/default project reward path only | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P123 BIRD BREEDING PROGRAM | status: implemented | requirement: - | rewards: breeding program: bird + matching partner zoo ; 1 icon -> left/mid/right gives 2CP+2rep / 1CP+2rep / 2CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P124 PREDATOR BREEDING PROGRAM | status: implemented | requirement: - | rewards: breeding program: predator + matching partner zoo ; 1 icon -> left/mid/right gives 2CP+2rep / 1CP+2rep / 2CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P125 REPTILE BREEDING PROGRAM | status: implemented | requirement: - | rewards: breeding program: reptile + matching partner zoo ; 1 icon -> left/mid/right gives 2CP+2rep / 1CP+2rep / 2CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P126 HERBIVORE BREEDING PROGRAM | status: implemented | requirement: - | rewards: breeding program: herbivore + matching partner zoo ; 1 icon -> left/mid/right gives 2CP+2rep / 1CP+2rep / 2CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P127 PRIMATE BREEDING PROGRAM | status: implemented | requirement: - | rewards: breeding program: primate + matching partner zoo ; 1 icon -> left/mid/right gives 2CP+2rep / 1CP+2rep / 2CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P128 AQUATIC | status: implemented | requirement: - | rewards: water icons count ; 5/4/3 icons -> 4/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P129 GEOLOGICAL | status: implemented | requirement: - | rewards: rock icons count ; 5/4/3 icons -> 4/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P130 SMALL ANIMALS | status: implemented | requirement: - | rewards: small animals count ; 8/5/2 icons -> 4/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P131 LARGE ANIMALS | status: implemented | requirement: - | rewards: large animals count ; 4/3/2 icons -> 4/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option
- P132 RESEARCH | status: implemented | requirement: - | rewards: science icons count ; 5/4/2 icons -> 4/3/2 CP | refs: main.py:8677 _project_requirement_value | main.py:1602 _project_level_rewards | main.py:12060 _apply_association_selected_option

## Raw TSV

- `ark_nova_card_effect_audit.tsv`

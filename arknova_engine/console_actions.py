"""Interactive console prompts for action details."""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def format_animals_play_step_for_human(
    *,
    step: Dict[str, Any],
    player: Any,
    find_building_fn: Callable[[Any, Tuple[int, int], str, int], Optional[Any]],
    building_cells_text_fn: Callable[[Any], str],
) -> str:
    card_no = int(step.get("card_number", -1))
    card_label = f"#{card_no}" if card_no >= 0 else "#?"
    card_name = str(step.get("card_name") or "unknown")
    enclosure_index = int(step.get("enclosure_index", -1))
    enclosure_size = int(step.get("enclosure_size", 0))
    enclosure_type = str(step.get("enclosure_type") or "standard")
    spaces_used = int(step.get("spaces_used", 1))
    origin_raw = step.get("enclosure_origin")
    rotation = str(step.get("enclosure_rotation") or "")
    cells_text = ""
    if isinstance(origin_raw, list) and len(origin_raw) == 2:
        origin = (int(origin_raw[0]), int(origin_raw[1]))
        building = find_building_fn(player, origin, rotation, enclosure_size)
        if building is not None:
            cells_text = f" cells={building_cells_text_fn(building)}"
    enclosure_label = f"E{enclosure_index + 1}"
    if enclosure_type == "reptile_house":
        enclosure_label += f" reptile_house(size=5,use={spaces_used})"
    elif enclosure_type == "large_bird_aviary":
        enclosure_label += f" large_bird_aviary(size=5,use={spaces_used})"
    elif enclosure_type == "petting_zoo":
        enclosure_label += f" petting_zoo(size=3,use={spaces_used})"
    else:
        enclosure_label += f" size={enclosure_size}"
    discount = int(step.get("card_discount", 0))
    effective_cost = int(step.get("card_cost", 0))
    if discount > 0:
        return (
            f"{card_label} {card_name} -> {enclosure_label}{cells_text} "
            f"(discount={discount}, effective_cost={effective_cost})"
        )
    return f"{card_label} {card_name} -> {enclosure_label}{cells_text}"


def prompt_animals_action_details_for_human(
    *,
    state: Any,
    player: Any,
    strength: int,
    player_id: int,
    animals_play_limit_fn: Callable[[int, bool], int],
    list_legal_animals_options_fn: Callable[[Any, int, int], List[Dict[str, Any]]],
    format_animals_play_step_fn: Callable[[Dict[str, Any], Any], str],
) -> Dict[str, Any]:
    upgraded = bool(player.action_upgraded["animals"])
    play_limit = animals_play_limit_fn(strength, upgraded)
    options = list_legal_animals_options_fn(state, player_id, strength)

    print(
        "Animals action settings: strength={} upgraded={} play_limit={}".format(
            strength, upgraded, play_limit
        )
    )
    if upgraded and strength >= 5:
        print("Animals II at strength 5+: +1 reputation will be applied.")
    if not options:
        print("No legal animal plays available.")
        return {}

    print("Legend: E# is animal-host enclosure index.")
    print("Animals options:")
    for option in options:
        parts = [format_animals_play_step_fn(step, player) for step in option["plays"]]
        action_label = parts[0] if len(parts) == 1 else " ; then ".join(parts)
        print(
            "{}. {} | total_cost={} total_appeal={} total_rep={} total_cons={}".format(
                option["index"],
                action_label,
                option["total_cost"],
                option["total_appeal"],
                option["total_reputation"],
                option["total_conservation"],
            )
        )
    while True:
        raw_pick = input(f"Select animals option [1-{len(options)}]: ").strip()
        if raw_pick == "":
            print("Please enter a number.")
            continue
        if not raw_pick.isdigit():
            print("Please enter a number.")
            continue
        picked = int(raw_pick)
        if 1 <= picked <= len(options):
            return {"animals_sequence_index": picked - 1}
        print("Out of range, try again.")


def prompt_cards_action_details_for_human(
    *,
    state: Any,
    player: Any,
    strength: int,
    cards_table_values_fn: Callable[[int, bool], Tuple[int, int, bool]],
    reputation_display_limit_fn: Callable[[int], int],
    format_card_line_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    upgraded = bool(player.action_upgraded["cards"])
    draw_target, discard_target, snap_allowed = cards_table_values_fn(strength, upgraded)
    print(
        "Cards action settings: strength={} upgraded={} draw_target={} discard_target={} snap_allowed={}".format(
            strength, upgraded, draw_target, discard_target, snap_allowed
        )
    )
    use_snap = False
    if snap_allowed and state.zoo_display:
        raw_mode = input("Use snap? [y/N]: ").strip().lower()
        use_snap = raw_mode in {"y", "yes"}
    if use_snap:
        while True:
            raw_idx = input(f"Snap which display folder [1-{len(state.zoo_display)}]: ").strip()
            if not raw_idx.isdigit():
                print("Please enter a number.")
                continue
            idx = int(raw_idx)
            if 1 <= idx <= len(state.zoo_display):
                return {"snap_display_index": idx - 1}
            print("Out of range, try again.")

    details: Dict[str, Any] = {}
    from_display_indices: List[int] = []
    if upgraded and state.zoo_display:
        accessible = reputation_display_limit_fn(int(player.reputation))
        if accessible > 0:
            raw = input(
                f"Display draw indices (within range 1-{min(accessible, len(state.zoo_display))}, blank=none): "
            ).strip()
            if raw:
                parts = raw.split()
                if not all(part.isdigit() for part in parts):
                    raise ValueError("Display draw indices must be numbers.")
                values = [int(part) for part in parts]
                if len(values) != len(set(values)):
                    raise ValueError("Display draw indices must be unique.")
                for value in values:
                    if value < 1 or value > len(state.zoo_display):
                        raise ValueError("Display draw index out of range.")
                    if value > accessible:
                        raise ValueError("Display draw index is outside reputation range.")
                from_display_indices = [value - 1 for value in sorted(values)]
    if from_display_indices:
        details["from_display_indices"] = from_display_indices

    remaining = max(0, draw_target - len(from_display_indices))
    if not upgraded:
        from_deck_count = draw_target
        print(f"Non-upgraded cards: draw is fixed to {from_deck_count} from deck.")
    else:
        while True:
            raw_deck = input(f"Draw how many from deck? [0-{remaining}] (default {remaining}): ").strip()
            if raw_deck == "":
                from_deck_count = remaining
                break
            if not raw_deck.isdigit():
                print("Please enter a number.")
                continue
            from_deck_count = int(raw_deck)
            if 0 <= from_deck_count <= remaining:
                break
            print("Out of range, try again.")
    details["from_deck_count"] = from_deck_count

    if discard_target > 0:
        preview_display_cards = [state.zoo_display[idx] for idx in sorted(from_display_indices)]
        preview_deck_cards = state.zoo_deck[:from_deck_count]
        preview_hand = list(player.hand) + preview_display_cards + preview_deck_cards
        if len(preview_hand) < discard_target:
            raise ValueError("Not enough cards in post-draw hand to choose required discard.")
        print("Choose discard from post-draw hand:")
        for idx, card in enumerate(preview_hand, start=1):
            print(f"{idx}. {format_card_line_fn(card)}")
        while True:
            raw_discard = input(f"Enter exactly {discard_target} discard index(es): ").strip()
            parts = raw_discard.split()
            if len(parts) != discard_target or not all(part.isdigit() for part in parts):
                print("Invalid input, try again.")
                continue
            values = [int(part) for part in parts]
            if len(values) != len(set(values)):
                print("Indices must be unique.")
                continue
            if any(value < 1 or value > len(preview_hand) for value in values):
                print("Index out of range.")
                continue
            details["discard_hand_indices"] = sorted(value - 1 for value in values)
            break
    return details


def prompt_build_action_details_for_human(
    *,
    state: Any,
    player: Any,
    strength: int,
    player_id: int,
    list_legal_build_options_fn: Callable[..., List[Dict[str, Any]]],
    building_type_enum: Any,
) -> Dict[str, Any]:
    upgraded = bool(player.action_upgraded["build"])
    remaining_size = strength
    built_types: Set[Any] = set()
    selections: List[Dict[str, Any]] = []
    bonus_targets: List[str] = []
    build_step = 1

    print(
        "Build action settings: strength={} upgraded={} max_total_size={}".format(
            strength, upgraded, strength
        )
    )
    while True:
        options = list_legal_build_options_fn(
            state=state,
            player_id=player_id,
            strength=remaining_size,
            already_built_types=built_types,
        )
        if not options:
            if build_step == 1:
                print("No legal building placements.")
            break
        by_type = Counter(opt["building_label"] for opt in options)
        print(f"Build step {build_step} | remaining_size={remaining_size}")
        print("Available building types:")
        for label, count in sorted(by_type.items()):
            print(f"- {label}: {count} options")
        print("Build options:")
        for opt in options:
            bonuses = ",".join(opt["placement_bonuses"]) if opt["placement_bonuses"] else "-"
            cells = ",".join(f"({x},{y})" for x, y in opt["cells"])
            print(
                f"{opt['index']}. {opt['building_label']} cells=[{cells}] "
                f"size={opt['size']} cost={opt['cost']} bonuses={bonuses}"
            )

        while True:
            prompt = f"Select build option [1-{len(options)}]"
            prompt += " (blank=finish)" if upgraded else " (blank=skip)"
            raw_pick = input(prompt + ": ").strip()
            if raw_pick == "":
                if not upgraded or selections:
                    return {"selections": selections, "bonus_action_to_slot_1_targets": bonus_targets}
                print("Build II requires at least one building if possible.")
                continue
            if not raw_pick.isdigit():
                print("Please enter a number.")
                continue
            idx = int(raw_pick)
            if 1 <= idx <= len(options):
                selected = options[idx - 1]
                break
            print("Out of range, try again.")

        selections.append({"building_type": selected["building_type"], "cells": selected["cells"]})
        remaining_size -= int(selected["size"])
        built_types.add(building_type_enum[selected["building_type"]])

        if "action_to_slot_1" in selected["placement_bonuses"]:
            print("Placement bonus requires moving one action card to slot 1.")
            for i, action_card in enumerate(player.action_order, start=1):
                print(f"{i}. {action_card}")
            while True:
                raw_card = input(f"Select action card [1-{len(player.action_order)}]: ").strip()
                if not raw_card.isdigit():
                    print("Please enter a number.")
                    continue
                action_idx = int(raw_card)
                if 1 <= action_idx <= len(player.action_order):
                    bonus_targets.append(player.action_order[action_idx - 1])
                    break
                print("Out of range, try again.")
        if not upgraded or remaining_size <= 0:
            break
        build_step += 1

    return {"selections": selections, "bonus_action_to_slot_1_targets": bonus_targets}


def prompt_association_action_details_for_human(
    *,
    state: Any,
    player: Any,
    strength: int,
    player_id: int,
    list_legal_association_options_fn: Callable[..., List[Dict[str, Any]]],
    partner_zoo_label_fn: Callable[[str], str],
    university_label_fn: Callable[[str], str],
    current_donation_cost_fn: Callable[[Any], int],
) -> Dict[str, Any]:
    upgraded = bool(player.action_upgraded["association"])
    options = list_legal_association_options_fn(state=state, player_id=player_id, strength=strength)
    partner_market = ", ".join(partner_zoo_label_fn(partner) for partner in sorted(state.available_partner_zoos)) or "-"
    university_market = ", ".join(university_label_fn(uni) for uni in sorted(state.available_universities)) or "-"
    print(
        "Association action settings: strength={} upgraded={} available_partner_zoos=[{}] "
        "available_universities=[{}]".format(strength, upgraded, partner_market, university_market)
    )
    if upgraded:
        print(f"Donation available: pay {current_donation_cost_fn(state)} money for +1 conservation.")
    if not options:
        print("No legal association tasks available.")
        return {}
    print("Association options:")
    for option in options:
        print(f"{option['index']}. {option['description']}")
    while True:
        raw_pick = input(f"Select association option [1-{len(options)}]: ").strip()
        if raw_pick == "":
            print("Please select one association option.")
            continue
        if not raw_pick.isdigit():
            print("Please enter a number.")
            continue
        picked = int(raw_pick)
        if 1 <= picked <= len(options):
            details: Dict[str, Any] = {"association_option_index": picked - 1}
            if upgraded:
                raw_donation = input("Make donation for +1 conservation? [y/N]: ").strip().lower()
                if raw_donation in {"y", "yes"}:
                    details["make_donation"] = True
            return details
        print("Out of range, try again.")


def prompt_sponsors_action_details_for_human(
    *,
    state: Any,
    player: Any,
    strength: int,
    sponsor_candidates_fn: Callable[[Any, Any, bool], List[Dict[str, Any]]],
    format_card_line_fn: Callable[[Any], str],
) -> Dict[str, Any]:
    upgraded = bool(player.action_upgraded["sponsors"])
    level_cap = strength + 1 if upgraded else strength
    candidates = sponsor_candidates_fn(state, player, upgraded)

    def _candidate_is_playable_under_cap(item: Dict[str, Any]) -> bool:
        return bool(item.get("playable_now")) and int(item.get("level", 0)) <= level_cap

    playable = [item for item in candidates if _candidate_is_playable_under_cap(item)]

    # If no sponsor card is playable under current strength cap, do not show
    # sponsor-mode/candidate prompts and directly resolve as break alternative.
    if not playable:
        return {"use_break_ability": True, "sponsor_selections": []}

    print(
        "Sponsors action settings: strength={} upgraded={} level_cap={}".format(
            strength, upgraded, level_cap
        )
    )
    print(
        "Choose mode: play sponsor card(s) or break alternative "
        "(break={} money{} and advance break by {}).".format(
            strength,
            "x2" if upgraded else "",
            strength,
        )
    )
    if candidates:
        print("Sponsor candidates:")
        for idx, cand in enumerate(candidates, start=1):
            card = cand["card"]
            source = str(cand["source"])
            folder = ""
            if source == "display":
                folder = f" folder={int(cand['source_index']) + 1}"
            if _candidate_is_playable_under_cap(cand):
                status = "OK"
            elif not cand.get("playable_now"):
                status = f"BLOCKED({cand.get('reason')})"
            else:
                status = f"BLOCKED(level_cap_{level_cap})"
            print(
                "{}. [{}{}] level={} pay={} {} | {}".format(
                    idx,
                    source,
                    folder,
                    int(cand["level"]),
                    int(cand["pay_cost"]),
                    status,
                    format_card_line_fn(card),
                )
            )
    else:
        print("No sponsor cards available to play.")

    while True:
        mode = input("Sponsors mode [p=play / b=break]: ").strip().lower()
        if mode in {"b", "break"}:
            return {"use_break_ability": True, "sponsor_selections": []}
        if mode not in {"p", "play"}:
            print("Please choose p or b.")
            continue

        if not playable:
            print("No playable sponsor cards now, choose break instead.")
            continue

        if not upgraded:
            while True:
                raw_pick = input(f"Select one sponsor option [1-{len(candidates)}]: ").strip()
                if not raw_pick.isdigit():
                    print("Please enter a number.")
                    continue
                picked = int(raw_pick)
                if not (1 <= picked <= len(candidates)):
                    print("Out of range, try again.")
                    continue
                chosen = candidates[picked - 1]
                if not _candidate_is_playable_under_cap(chosen):
                    print(f"That card is not playable now ({chosen.get('reason')}).")
                    continue
                details: Dict[str, Any] = {
                    "use_break_ability": False,
                    "sponsor_selections": [
                        {
                            "source": chosen["source"],
                            "source_index": int(chosen["source_index"]),
                            "card_instance_id": str(chosen["card_instance_id"]),
                        }
                    ],
                }
                if int(chosen["card"].number) == 227:
                    while True:
                        raw_mode = input("WAZA SPECIAL ASSIGNMENT mode [small/large]: ").strip().lower()
                        if raw_mode in {"small", "large"}:
                            details["sponsor_227_mode"] = raw_mode
                            break
                        print("Please input small or large.")
                return details

        while True:
            raw = input(
                f"Select sponsor option indices (space-separated, 1-{len(candidates)}): "
            ).strip()
            parts = raw.split()
            if not parts or not all(part.isdigit() for part in parts):
                print("Please enter one or more numbers.")
                continue
            picks = [int(part) for part in parts]
            if len(picks) != len(set(picks)):
                print("Indices must be unique.")
                continue
            if any(pick < 1 or pick > len(candidates) for pick in picks):
                print("Out of range, try again.")
                continue
            chosen_items = [candidates[pick - 1] for pick in picks]
            if not all(_candidate_is_playable_under_cap(item) for item in chosen_items):
                print("At least one selected card is not playable now.")
                continue
            total_level = sum(int(item["level"]) for item in chosen_items)
            if total_level > level_cap:
                print(f"Total sponsor level {total_level} exceeds cap {level_cap}.")
                continue
            details = {
                "use_break_ability": False,
                "sponsor_selections": [
                    {
                        "source": item["source"],
                        "source_index": int(item["source_index"]),
                        "card_instance_id": str(item["card_instance_id"]),
                    }
                    for item in chosen_items
                ],
            }
            if any(int(item["card"].number) == 227 for item in chosen_items):
                while True:
                    raw_mode = input("WAZA SPECIAL ASSIGNMENT mode [small/large]: ").strip().lower()
                    if raw_mode in {"small", "large"}:
                        details["sponsor_227_mode"] = raw_mode
                        break
                    print("Please input small or large.")
            return details

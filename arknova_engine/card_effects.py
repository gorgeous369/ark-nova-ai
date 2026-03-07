"""Card effect resolution/execution helpers for the lightweight main runner.

This module centralizes "what card effect should run" so that each card can be
mapped to a concrete effect code, while keeping main.py smaller.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ActionMover = Callable[[str], None]
BreakAdvancer = Callable[[int], bool]
DeckDrawer = Callable[[int], Sequence[Any]]
DiscardPusher = Callable[[Sequence[Any]], None]
MoneyGainer = Callable[[int], None]
DisplayTaker = Callable[[int, str, bool], int]
HandSeller = Callable[[int, int], int]
FreeSmallBuilder = Callable[[int], int]
MultiplierAdder = Callable[[str], bool]
TokenApplier = Callable[[int], int]
HypnosisExecutor = Callable[[int], str]
PilferingExecutor = Callable[[int], str]
DiscardScavenger = Callable[[int, int], Tuple[int, int]]
ExtraActionMarker = Callable[[str], None]
FinalScoringDrawer = Callable[[int, int], Tuple[int, int]]
BaseProjectTaker = Callable[[int], int]
XTokenGainer = Callable[[int], int]
IconCounter = Callable[[str], int]
PrimaryIconCounter = Callable[[], int]
FinalScoringAdapter = Callable[[int], Tuple[int, int]]
EnclosureRemover = Callable[[int], Tuple[int, int]]
AssociationWorkerReturner = Callable[[int], int]
DisplayMarker = Callable[[int], int]
LargeBirdAviaryPlacer = Callable[[int], int]
TradeExecutor = Callable[[int], int]
SharkAttackExecutor = Callable[[int], Tuple[int, int]]
SpecificProjectTaker = Callable[[str, int], int]
SymbiosisExecutor = Callable[[], List[str]]
CamouflageGrant = Callable[[int], None]


@dataclass(frozen=True)
class ResolvedCardEffect:
    code: str
    value: int = 0
    target: str = ""
    raw_title: str = ""
    raw_text: str = ""
    supported: bool = True


def _extract_int_from_title(prefix: str, title: str) -> Optional[int]:
    pattern = rf"^{re.escape(prefix)}\s+(\d+)$"
    match = re.match(pattern, title.strip(), flags=re.I)
    if not match:
        return None
    return int(match.group(1))


def _extract_int_from_text(pattern: str, text: str, default: int = 0) -> int:
    match = re.search(pattern, text, flags=re.I)
    if not match:
        return default
    return int(match.group(1))


def resolve_card_effect(card: Any) -> ResolvedCardEffect:
    card_type = str(getattr(card, "card_type", "")).strip().lower()
    title = str(getattr(card, "ability_title", "") or "").strip()
    text = str(getattr(card, "ability_text", "") or "").strip()

    if card_type not in {"animal", "sponsor"}:
        return ResolvedCardEffect(code="none", raw_title=title, raw_text=text)

    if card_type == "sponsor":
        # Sponsor effects are currently consumed by Sponsors action implementation.
        # Keep explicit code here for per-card effect mapping completeness.
        effects = getattr(card, "effects", ()) or ()
        if effects:
            return ResolvedCardEffect(code="sponsor_effects_from_dataset", raw_title=title, raw_text=text)
        return ResolvedCardEffect(code="none", raw_title=title, raw_text=text)

    if not title:
        return ResolvedCardEffect(code="none", raw_title=title, raw_text=text)

    sprint = _extract_int_from_title("Sprint", title)
    if sprint is not None:
        return ResolvedCardEffect(code="draw_from_deck", value=sprint, raw_title=title, raw_text=text)

    hunter = _extract_int_from_title("Hunter", title)
    if hunter is not None:
        return ResolvedCardEffect(code="hunter", value=hunter, raw_title=title, raw_text=text)

    snapping = _extract_int_from_title("Snapping", title)
    if snapping is not None:
        replenish_each = "replenish in between" in text.lower()
        target = "replenish_each" if replenish_each else "single_replenish"
        return ResolvedCardEffect(code="take_display_cards", value=snapping, target=target, raw_title=title, raw_text=text)

    perception = _extract_int_from_title("Perception", title)
    if perception is not None:
        keep_count = _extract_int_from_text(r"Add\s+(\d+)\s+to\s+your\s+hand", text, default=1)
        return ResolvedCardEffect(
            code="draw_keep_from_deck",
            value=perception,
            target=str(max(1, keep_count)),
            raw_title=title,
            raw_text=text,
        )

    scavenging = _extract_int_from_title("Scavenging", title)
    if scavenging is not None:
        keep_count = _extract_int_from_text(r"Add\s+(\d+)\s+to\s+your\s+hand", text, default=1)
        return ResolvedCardEffect(
            code="scavenge_from_discard",
            value=scavenging,
            target=str(max(1, keep_count)),
            raw_title=title,
            raw_text=text,
        )

    boost_match = re.match(r"^Boost:\s*(\w+)$", title, flags=re.I)
    if boost_match:
        target = boost_match.group(1).strip().lower()
        if target == "animal":
            target = "animals"
        if target == "sponsor":
            target = "sponsors"
        if target == "card":
            target = "cards"
        if target == "building":
            target = "build"
        if target in {"animals", "cards", "build", "association", "sponsors"}:
            return ResolvedCardEffect(code="move_action_to_slot_1", target=target, raw_title=title, raw_text=text)

    if title.lower() == "clever":
        return ResolvedCardEffect(code="move_any_action_to_slot_1", raw_title=title, raw_text=text)

    if title.lower() == "full-throated":
        return ResolvedCardEffect(code="hire_worker", value=1, raw_title=title, raw_text=text)

    if title.lower() in {"pack", "petting zoo animal"}:
        gain_match = re.search(r"Gain\s+(\d+)", text, flags=re.I)
        gain = int(gain_match.group(1)) if gain_match else 1
        return ResolvedCardEffect(code="gain_appeal", value=gain, raw_title=title, raw_text=text)

    glide = _extract_int_from_title("Glide", title)
    if glide is not None:
        return ResolvedCardEffect(code="advance_break", value=glide, raw_title=title, raw_text=text)

    jumping = _extract_int_from_title("Jumping", title)
    if jumping is not None:
        gain_money = _extract_int_from_text(r"Gain\s+(\d+)", text, default=0)
        return ResolvedCardEffect(
            code="jumping_break_and_money",
            value=jumping,
            target=str(max(0, gain_money)),
            raw_title=title,
            raw_text=text,
        )

    sun_bathing = _extract_int_from_title("Sun Bathing", title)
    if sun_bathing is not None:
        return ResolvedCardEffect(code="sell_hand_cards", value=sun_bathing, target="4", raw_title=title, raw_text=text)

    pouch = _extract_int_from_title("Pouch", title)
    if pouch is not None:
        return ResolvedCardEffect(
            code="discard_hand_for_money",
            value=pouch,
            target="2",
            raw_title=title,
            raw_text=text,
        )

    digging = _extract_int_from_title("Digging", title)
    if digging is not None:
        return ResolvedCardEffect(code="digging_cycle", value=digging, raw_title=title, raw_text=text)

    pilfering = _extract_int_from_title("Pilfering", title)
    if pilfering is not None:
        return ResolvedCardEffect(code="pilfering", value=pilfering, raw_title=title, raw_text=text)

    posturing = _extract_int_from_title("Posturing", title)
    if posturing is not None:
        return ResolvedCardEffect(code="free_kiosk_or_pavilion", value=posturing, raw_title=title, raw_text=text)

    venom = _extract_int_from_title("Venom", title)
    if venom is not None:
        return ResolvedCardEffect(code="venom_tokens", value=venom, raw_title=title, raw_text=text)

    if title.lower() == "constriction":
        return ResolvedCardEffect(code="constriction_tokens", value=1, raw_title=title, raw_text=text)

    if title.lower() == "sponsor magnet":
        return ResolvedCardEffect(code="take_display_sponsors", value=999, raw_title=title, raw_text=text)

    multiplier_match = re.match(r"^Multiplier:\s*(\w+)$", title, flags=re.I)
    if multiplier_match:
        target = multiplier_match.group(1).strip().lower()
        if target == "card":
            target = "cards"
        if target == "building":
            target = "build"
        if target in {"animals", "cards", "build", "association", "sponsors"}:
            return ResolvedCardEffect(code="multiplier_token", target=target, value=1, raw_title=title, raw_text=text)

    action_match = re.match(r"^Action:\s*(\w+)$", title, flags=re.I)
    if action_match:
        target = action_match.group(1).strip().lower()
        if target == "animal":
            target = "animals"
        if target == "sponsor":
            target = "sponsors"
        if target == "card":
            target = "cards"
        if target == "building":
            target = "build"
        if target in {"animals", "cards", "build", "association", "sponsors"}:
            return ResolvedCardEffect(code="extra_action_granted", target=target, value=1, raw_title=title, raw_text=text)

    iconic_match = re.match(r"^Iconic Animal:\s*([a-zA-Z]+)$", title, flags=re.I)
    if iconic_match:
        gain = _extract_int_from_text(r"Gain\s+(\d+)", text, default=1)
        return ResolvedCardEffect(code="gain_conservation", value=max(0, gain), raw_title=title, raw_text=text)

    if title.lower() == "resistance":
        return ResolvedCardEffect(code="draw_final_scoring_keep", value=2, target="1", raw_title=title, raw_text=text)

    if title.lower() == "assertion":
        return ResolvedCardEffect(code="take_unused_base_project", value=1, raw_title=title, raw_text=text)

    if title.lower() == "determination":
        return ResolvedCardEffect(code="extra_action_any", value=1, raw_title=title, raw_text=text)

    hypnosis = _extract_int_from_title("Hypnosis", title)
    if hypnosis is not None:
        return ResolvedCardEffect(code="hypnosis", value=hypnosis, raw_title=title, raw_text=text)

    if title.lower() == "inventive":
        return ResolvedCardEffect(code="gain_x_tokens", value=1, raw_title=title, raw_text=text)

    if title.lower() == "inventive: bear":
        return ResolvedCardEffect(code="gain_x_tokens_from_icon", target="Bear", raw_title=title, raw_text=text)

    if title.lower() == "inventive: primary":
        return ResolvedCardEffect(code="gain_x_tokens_from_primary", value=3, raw_title=title, raw_text=text)

    flock = _extract_int_from_title("Flock Animal", title)
    if flock is not None:
        return ResolvedCardEffect(code="flock_optional", value=flock, raw_title=title, raw_text=text)

    if title.lower() == "symbiosis":
        return ResolvedCardEffect(code="symbiosis_copy", value=1, raw_title=title, raw_text=text)

    if title.lower() == "camouflage":
        return ResolvedCardEffect(code="camouflage_grant", value=1, raw_title=title, raw_text=text)

    if title.lower() == "scuba dive x":
        return ResolvedCardEffect(code="scuba_dive_x", target="seaanimal", raw_title=title, raw_text=text)

    if title.lower() == "monkey gang":
        return ResolvedCardEffect(code="reveal_until_badge", target="Primate", raw_title=title, raw_text=text)

    adapt = _extract_int_from_title("Adapt", title)
    if adapt is not None:
        return ResolvedCardEffect(code="adapt_final_scoring", value=adapt, raw_title=title, raw_text=text)

    if title.lower() == "cut down":
        return ResolvedCardEffect(code="remove_empty_enclosure_refund", value=1, raw_title=title, raw_text=text)

    if title.lower() == "dominance":
        return ResolvedCardEffect(code="take_specific_base_project", target="primates", value=1, raw_title=title, raw_text=text)

    if title.lower() == "extra shift":
        return ResolvedCardEffect(code="return_association_worker", value=1, raw_title=title, raw_text=text)

    if title.lower() == "mark":
        return ResolvedCardEffect(code="mark_display_animal", value=1, raw_title=title, raw_text=text)

    if title.lower() == "peacocking":
        return ResolvedCardEffect(code="place_free_large_bird_aviary", value=1, raw_title=title, raw_text=text)

    if title.lower() == "sea animal magnet":
        return ResolvedCardEffect(code="take_display_by_badge", target="SeaAnimal", value=999, raw_title=title, raw_text=text)

    shark = _extract_int_from_title("Shark Attack", title)
    if shark is not None:
        return ResolvedCardEffect(code="shark_attack", value=shark, raw_title=title, raw_text=text)

    if title.lower() == "trade":
        return ResolvedCardEffect(code="trade_hand_with_display", value=1, raw_title=title, raw_text=text)

    return ResolvedCardEffect(
        code=f"unimplemented:{title}",
        raw_title=title,
        raw_text=text,
        supported=False,
    )


def apply_animal_effect(
    *,
    card: Any,
    move_action_to_slot_1: ActionMover,
    advance_break: BreakAdvancer,
    draw_from_deck: DeckDrawer,
    push_to_discard: DiscardPusher,
    choose_action_for_clever: Optional[Callable[[], str]] = None,
    increase_workers: Optional[Callable[[int], None]] = None,
    increase_appeal: Optional[Callable[[int], None]] = None,
    gain_money: Optional[MoneyGainer] = None,
    take_display_cards: Optional[DisplayTaker] = None,
    sell_hand_cards: Optional[HandSeller] = None,
    place_free_kiosk_or_pavilion: Optional[FreeSmallBuilder] = None,
    add_multiplier_token: Optional[MultiplierAdder] = None,
    apply_venom: Optional[TokenApplier] = None,
    apply_constriction: Optional[TokenApplier] = None,
    perform_hypnosis: Optional[HypnosisExecutor] = None,
    perform_pilfering: Optional[PilferingExecutor] = None,
    scavenge_from_discard: Optional[DiscardScavenger] = None,
    mark_extra_action: Optional[ExtraActionMarker] = None,
    increase_conservation: Optional[Callable[[int], None]] = None,
    draw_final_scoring_keep: Optional[FinalScoringDrawer] = None,
    take_unused_base_project: Optional[BaseProjectTaker] = None,
    gain_x_tokens: Optional[XTokenGainer] = None,
    count_icon: Optional[IconCounter] = None,
    count_primary_icons: Optional[PrimaryIconCounter] = None,
    adapt_final_scoring: Optional[FinalScoringAdapter] = None,
    remove_empty_enclosure_refund: Optional[EnclosureRemover] = None,
    return_association_worker: Optional[AssociationWorkerReturner] = None,
    mark_display_animal: Optional[DisplayMarker] = None,
    place_free_large_bird_aviary: Optional[LargeBirdAviaryPlacer] = None,
    trade_hand_with_display: Optional[TradeExecutor] = None,
    shark_attack: Optional[SharkAttackExecutor] = None,
    take_specific_base_project: Optional[SpecificProjectTaker] = None,
    symbiosis_copy: Optional[SymbiosisExecutor] = None,
    grant_camouflage_ignore: Optional[CamouflageGrant] = None,
) -> List[str]:
    effect = resolve_card_effect(card)
    messages: List[str] = []

    if effect.code == "none":
        return messages

    if effect.code == "draw_from_deck":
        drawn = list(draw_from_deck(max(0, effect.value)))
        messages.append(f"effect[{effect.code}] drew={len(drawn)}")
        return messages

    if effect.code == "hunter":
        reveal_count = max(0, effect.value)
        revealed = list(draw_from_deck(reveal_count))
        if not revealed:
            messages.append("effect[hunter] reveal=0")
            return messages
        picked = None
        for candidate in revealed:
            if str(getattr(candidate, "card_type", "")).lower() == "animal":
                picked = candidate
                break
        if picked is not None:
            # Keep picked in hand because draw_from_deck has already moved it there.
            leftovers = [card_obj for card_obj in revealed if card_obj is not picked]
            if leftovers:
                push_to_discard(leftovers)
            messages.append(f"effect[hunter] revealed={len(revealed)} kept=1 discarded={len(leftovers)}")
            return messages
        # No animal found -> all revealed should be discarded from hand by caller policy.
        push_to_discard(revealed)
        messages.append(f"effect[hunter] revealed={len(revealed)} kept=0 discarded={len(revealed)}")
        return messages

    if effect.code == "take_display_cards":
        if take_display_cards is None:
            messages.append("effect[take_display_cards] unsupported(no_callback)")
            return messages
        replenish_each = effect.target == "replenish_each"
        taken = take_display_cards(max(0, effect.value), "", replenish_each)
        messages.append(f"effect[take_display_cards] taken={taken}")
        return messages

    if effect.code == "draw_keep_from_deck":
        draw_count = max(0, effect.value)
        keep_count = max(1, int(effect.target or "1"))
        drawn = list(draw_from_deck(draw_count))
        if len(drawn) > keep_count:
            leftovers = drawn[keep_count:]
            push_to_discard(leftovers)
        kept = min(len(drawn), keep_count)
        discarded = max(0, len(drawn) - kept)
        messages.append(f"effect[draw_keep_from_deck] drew={len(drawn)} kept={kept} discarded={discarded}")
        return messages

    if effect.code == "scavenge_from_discard":
        if scavenge_from_discard is None:
            messages.append("effect[scavenge_from_discard] unsupported(no_callback)")
            return messages
        draw_count = max(0, effect.value)
        keep_count = max(1, int(effect.target or "1"))
        drew, kept = scavenge_from_discard(draw_count, keep_count)
        messages.append(f"effect[scavenge_from_discard] drew={drew} kept={kept} discarded={max(0, drew - kept)}")
        return messages

    if effect.code == "move_action_to_slot_1":
        if effect.target:
            move_action_to_slot_1(effect.target)
            messages.append(f"effect[{effect.code}] target={effect.target}")
        return messages

    if effect.code == "move_any_action_to_slot_1":
        chosen = choose_action_for_clever() if choose_action_for_clever else "cards"
        move_action_to_slot_1(chosen)
        messages.append(f"effect[{effect.code}] target={chosen}")
        return messages

    if effect.code == "hire_worker":
        if increase_workers is not None:
            increase_workers(max(0, effect.value))
            messages.append(f"effect[{effect.code}] workers=+{max(0, effect.value)}")
        return messages

    if effect.code == "gain_appeal":
        if increase_appeal is not None:
            increase_appeal(max(0, effect.value))
            messages.append(f"effect[{effect.code}] appeal=+{max(0, effect.value)}")
        return messages

    if effect.code == "advance_break":
        advance_break(max(0, effect.value))
        messages.append(f"effect[{effect.code}] break=+{max(0, effect.value)}")
        return messages

    if effect.code == "jumping_break_and_money":
        advance_break(max(0, effect.value))
        money = max(0, int(effect.target or "0"))
        if gain_money is not None and money > 0:
            gain_money(money)
        messages.append(f"effect[jumping_break_and_money] break=+{max(0, effect.value)} money=+{money}")
        return messages

    if effect.code == "sell_hand_cards":
        if sell_hand_cards is None:
            messages.append("effect[sell_hand_cards] unsupported(no_callback)")
            return messages
        sold = sell_hand_cards(max(0, effect.value), max(0, int(effect.target or "0")))
        messages.append(f"effect[sell_hand_cards] sold={sold}")
        return messages

    if effect.code == "discard_hand_for_money":
        if sell_hand_cards is None:
            messages.append("effect[discard_hand_for_money] unsupported(no_callback)")
            return messages
        sold = sell_hand_cards(max(0, effect.value), max(0, int(effect.target or "0")))
        messages.append(f"effect[discard_hand_for_money] discarded={sold}")
        return messages

    if effect.code == "digging_cycle":
        if take_display_cards is None or sell_hand_cards is None or draw_from_deck is None:
            messages.append("effect[digging_cycle] unsupported(no_callback)")
            return messages
        loops = max(0, effect.value)
        did = 0
        for _ in range(loops):
            took = take_display_cards(1, "", True)
            if took > 0:
                did += 1
                continue
            sold = sell_hand_cards(1, 0)
            if sold <= 0:
                break
            draw_from_deck(1)
            did += 1
        messages.append(f"effect[digging_cycle] loops={did}")
        return messages

    if effect.code == "pilfering":
        if perform_pilfering is None:
            messages.append("effect[pilfering] unsupported(no_callback)")
            return messages
        summary = perform_pilfering(max(0, effect.value))
        messages.append(f"effect[pilfering] {summary}".rstrip())
        return messages

    if effect.code == "free_kiosk_or_pavilion":
        if place_free_kiosk_or_pavilion is None:
            messages.append("effect[free_kiosk_or_pavilion] unsupported(no_callback)")
            return messages
        placed = place_free_kiosk_or_pavilion(max(0, effect.value))
        messages.append(f"effect[free_kiosk_or_pavilion] placed={placed}")
        return messages

    if effect.code == "venom_tokens":
        if apply_venom is None:
            messages.append("effect[venom_tokens] unsupported(no_callback)")
            return messages
        affected = apply_venom(max(0, effect.value))
        messages.append(f"effect[venom_tokens] affected_players={affected}")
        return messages

    if effect.code == "constriction_tokens":
        if apply_constriction is None:
            messages.append("effect[constriction_tokens] unsupported(no_callback)")
            return messages
        affected = apply_constriction(max(0, effect.value))
        messages.append(f"effect[constriction_tokens] affected_players={affected}")
        return messages

    if effect.code == "hypnosis":
        if perform_hypnosis is None:
            messages.append("effect[hypnosis] unsupported(no_callback)")
            return messages
        summary = perform_hypnosis(max(0, effect.value))
        messages.append(f"effect[hypnosis] {summary}".rstrip())
        return messages

    if effect.code == "take_display_sponsors":
        if take_display_cards is None:
            messages.append("effect[take_display_sponsors] unsupported(no_callback)")
            return messages
        taken = take_display_cards(max(0, effect.value), "sponsor", False)
        messages.append(f"effect[take_display_sponsors] taken={taken}")
        return messages

    if effect.code == "multiplier_token":
        if add_multiplier_token is None or not effect.target:
            messages.append("effect[multiplier_token] unsupported(no_callback)")
            return messages
        ok = add_multiplier_token(effect.target)
        messages.append(f"effect[multiplier_token] target={effect.target} applied={1 if ok else 0}")
        return messages

    if effect.code == "extra_action_granted":
        if mark_extra_action is None:
            messages.append("effect[extra_action_granted] unsupported(no_callback)")
            return messages
        mark_extra_action(effect.target or "unknown")
        messages.append(f"effect[extra_action_granted] target={effect.target}")
        return messages

    if effect.code == "gain_conservation":
        if increase_conservation is None:
            messages.append("effect[gain_conservation] unsupported(no_callback)")
            return messages
        increase_conservation(max(0, effect.value))
        messages.append(f"effect[gain_conservation] conservation=+{max(0, effect.value)}")
        return messages

    if effect.code == "draw_final_scoring_keep":
        if draw_final_scoring_keep is None:
            messages.append("effect[draw_final_scoring_keep] unsupported(no_callback)")
            return messages
        drew, kept = draw_final_scoring_keep(max(0, effect.value), max(1, int(effect.target or "1")))
        messages.append(f"effect[draw_final_scoring_keep] drew={drew} kept={kept} discarded={max(0, drew-kept)}")
        return messages

    if effect.code == "take_unused_base_project":
        if take_unused_base_project is None:
            messages.append("effect[take_unused_base_project] unsupported(no_callback)")
            return messages
        taken = take_unused_base_project(max(0, effect.value))
        messages.append(f"effect[take_unused_base_project] taken={taken}")
        return messages

    if effect.code == "extra_action_any":
        if mark_extra_action is None:
            messages.append("effect[extra_action_any] unsupported(no_callback)")
            return messages
        mark_extra_action("any")
        messages.append("effect[extra_action_any] granted=1")
        return messages

    if effect.code == "extra_action_strength":
        if mark_extra_action is None:
            messages.append("effect[extra_action_strength] unsupported(no_callback)")
            return messages
        mark_extra_action(f"strength:{max(0, effect.value)}")
        messages.append(f"effect[extra_action_strength] strength={max(0, effect.value)}")
        return messages

    if effect.code == "gain_x_tokens":
        if gain_x_tokens is None:
            messages.append("effect[gain_x_tokens] unsupported(no_callback)")
            return messages
        gained = gain_x_tokens(max(0, effect.value))
        messages.append(f"effect[gain_x_tokens] gained={gained}")
        return messages

    if effect.code == "gain_x_tokens_from_icon":
        if gain_x_tokens is None or count_icon is None:
            messages.append("effect[gain_x_tokens_from_icon] unsupported(no_callback)")
            return messages
        amount = max(0, count_icon(effect.target))
        gained = gain_x_tokens(amount)
        messages.append(f"effect[gain_x_tokens_from_icon] icon={effect.target} amount={amount} gained={gained}")
        return messages

    if effect.code == "gain_x_tokens_from_primary":
        if gain_x_tokens is None or count_primary_icons is None:
            messages.append("effect[gain_x_tokens_from_primary] unsupported(no_callback)")
            return messages
        amount = min(max(0, effect.value), max(0, count_primary_icons()))
        gained = gain_x_tokens(amount)
        messages.append(f"effect[gain_x_tokens_from_primary] amount={amount} gained={gained}")
        return messages

    if effect.code == "flock_optional":
        messages.append(f"effect[flock_optional] size={max(0, effect.value)}")
        return messages

    if effect.code == "symbiosis_copy":
        if symbiosis_copy is None:
            messages.append("effect[symbiosis_copy] unsupported(no_callback)")
            return messages
        nested = symbiosis_copy()
        messages.append(f"effect[symbiosis_copy] nested={len(nested)}")
        messages.extend(nested)
        return messages

    if effect.code == "camouflage_grant":
        if grant_camouflage_ignore is None:
            messages.append("effect[camouflage_grant] unsupported(no_callback)")
            return messages
        grant_camouflage_ignore(max(0, effect.value))
        messages.append(f"effect[camouflage_grant] granted={max(0, effect.value)}")
        return messages

    if effect.code == "scuba_dive_x":
        if count_icon is None:
            messages.append("effect[scuba_dive_x] unsupported(no_icon_counter)")
            return messages
        x = max(0, count_icon(effect.target))
        revealed = list(draw_from_deck(x))
        if not revealed:
            messages.append("effect[scuba_dive_x] reveal=0")
            return messages
        picked = None
        for candidate in revealed:
            if str(getattr(candidate, "card_type", "")).lower() == "sponsor":
                picked = candidate
                break
        leftovers = [card_obj for card_obj in revealed if card_obj is not picked]
        if leftovers:
            push_to_discard(leftovers)
        kept = 1 if picked is not None else 0
        messages.append(f"effect[scuba_dive_x] x={x} revealed={len(revealed)} kept={kept} discarded={len(leftovers)}")
        return messages

    if effect.code == "reveal_until_badge":
        max_reveal = 20
        revealed: List[Any] = []
        kept = 0
        target_badge = effect.target.strip().lower()
        for _ in range(max_reveal):
            draw = list(draw_from_deck(1))
            if not draw:
                break
            card_obj = draw[0]
            revealed.append(card_obj)
            badges = tuple(str(item).strip().lower() for item in getattr(card_obj, "badges", ()) or ())
            if target_badge in badges:
                kept = 1
                break
        if revealed:
            leftovers = revealed[:-1] if kept == 1 else list(revealed)
            if leftovers:
                push_to_discard(leftovers)
        messages.append(f"effect[reveal_until_badge] revealed={len(revealed)} kept={kept}")
        return messages

    if effect.code == "adapt_final_scoring":
        if adapt_final_scoring is None:
            messages.append("effect[adapt_final_scoring] unsupported(no_callback)")
            return messages
        drew, discarded = adapt_final_scoring(max(0, effect.value))
        messages.append(f"effect[adapt_final_scoring] drew={drew} discarded={discarded}")
        return messages

    if effect.code == "remove_empty_enclosure_refund":
        if remove_empty_enclosure_refund is None:
            messages.append("effect[remove_empty_enclosure_refund] unsupported(no_callback)")
            return messages
        removed, refunded = remove_empty_enclosure_refund(max(0, effect.value))
        messages.append(f"effect[remove_empty_enclosure_refund] removed={removed} refunded={refunded}")
        return messages

    if effect.code == "return_association_worker":
        if return_association_worker is None:
            messages.append("effect[return_association_worker] unsupported(no_callback)")
            return messages
        returned = return_association_worker(max(0, effect.value))
        messages.append(f"effect[return_association_worker] returned={returned}")
        return messages

    if effect.code == "mark_display_animal":
        if mark_display_animal is None:
            messages.append("effect[mark_display_animal] unsupported(no_callback)")
            return messages
        marked = mark_display_animal(max(0, effect.value))
        messages.append(f"effect[mark_display_animal] marked={marked}")
        return messages

    if effect.code == "place_free_large_bird_aviary":
        if place_free_large_bird_aviary is None:
            messages.append("effect[place_free_large_bird_aviary] unsupported(no_callback)")
            return messages
        placed = place_free_large_bird_aviary(max(0, effect.value))
        messages.append(f"effect[place_free_large_bird_aviary] placed={placed}")
        return messages

    if effect.code == "take_display_by_badge":
        if take_display_cards is None:
            messages.append("effect[take_display_by_badge] unsupported(no_callback)")
            return messages
        taken = take_display_cards(max(0, effect.value), f"badge:{effect.target}", False)
        messages.append(f"effect[take_display_by_badge] badge={effect.target} taken={taken}")
        return messages

    if effect.code == "shark_attack":
        if shark_attack is None:
            messages.append("effect[shark_attack] unsupported(no_callback)")
            return messages
        discarded, money = shark_attack(max(0, effect.value))
        messages.append(f"effect[shark_attack] discarded={discarded} money=+{money}")
        return messages

    if effect.code == "trade_hand_with_display":
        if trade_hand_with_display is None:
            messages.append("effect[trade_hand_with_display] unsupported(no_callback)")
            return messages
        traded = trade_hand_with_display(max(0, effect.value))
        messages.append(f"effect[trade_hand_with_display] traded={traded}")
        return messages

    if effect.code == "take_specific_base_project":
        if take_specific_base_project is None:
            messages.append("effect[take_specific_base_project] unsupported(no_callback)")
            return messages
        taken = take_specific_base_project(effect.target, max(0, effect.value))
        messages.append(f"effect[take_specific_base_project] target={effect.target} taken={taken}")
        return messages

    messages.append(f"effect[{effect.code}] unsupported title='{effect.raw_title}'")
    return messages


def build_effect_coverage(cards: Sequence[Any]) -> Dict[str, int]:
    total = 0
    supported = 0
    unsupported = 0
    no_effect = 0
    for card in cards:
        if str(getattr(card, "card_type", "")).lower() not in {"animal", "sponsor"}:
            continue
        total += 1
        resolved = resolve_card_effect(card)
        if resolved.code == "none":
            no_effect += 1
        elif resolved.supported:
            supported += 1
        else:
            unsupported += 1
    return {
        "total": total,
        "supported": supported,
        "unsupported": unsupported,
        "no_effect": no_effect,
    }

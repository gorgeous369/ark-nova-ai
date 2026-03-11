"""Card effect resolution/execution helpers for the lightweight main runner.

This module centralizes "what card effect should run" so that each card can be
mapped to a concrete effect code, while keeping main.py smaller.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


ActionMover = Callable[[str], None]
BoostActionExecutor = Callable[[str], str]
BreakTrackAdvancer = Callable[[int], bool]
DeckDrawer = Callable[[int], Sequence[Any]]
DiscardPusher = Callable[[Sequence[Any]], None]
MoneyGainer = Callable[[int], None]
DisplayTaker = Callable[[int, str, bool], int]
HandSeller = Callable[[int, int], int]
HandPoucher = Callable[[int, int], int]
FreeSmallBuilder = Callable[[int], int]
MultiplierAdder = Callable[[str], bool]
TokenApplier = Callable[[int], int]
HypnosisExecutor = Callable[[int], str]
PilferingExecutor = Callable[[int], str]
DiggingExecutor = Callable[[int], int]
DiscardScavenger = Callable[[int, int], Tuple[int, int]]
ExtraActionMarker = Callable[[str], None]
FinalScoringDrawer = Callable[[int, int], Tuple[int, int]]
DeckKeepChoiceExecutor = Callable[[int, int], Tuple[int, int]]
RevealKeepByTypeExecutor = Callable[[int, str], Tuple[int, int]]
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
GlideRewardExecutor = Callable[[int], str]


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
        return ResolvedCardEffect(code="sprint", value=sprint, raw_title=title, raw_text=text)

    hunter = _extract_int_from_title("Hunter", title)
    if hunter is not None:
        return ResolvedCardEffect(code="hunter", value=hunter, raw_title=title, raw_text=text)

    snapping = _extract_int_from_title("Snapping", title)
    if snapping is not None:
        replenish_each = "replenish in between" in text.lower()
        target = "replenish_each" if replenish_each else "single_replenish"
        return ResolvedCardEffect(code="snapping", value=snapping, target=target, raw_title=title, raw_text=text)

    perception = _extract_int_from_title("Perception", title)
    if perception is not None:
        keep_count = _extract_int_from_text(r"Add\s+(\d+)\s+to\s+your\s+hand", text, default=1)
        return ResolvedCardEffect(
            code="perception",
            value=perception,
            target=str(max(1, keep_count)),
            raw_title=title,
            raw_text=text,
        )

    scavenging = _extract_int_from_title("Scavenging", title)
    if scavenging is not None:
        keep_count = _extract_int_from_text(r"Add\s+(\d+)\s+to\s+your\s+hand", text, default=1)
        return ResolvedCardEffect(
            code="scavenging",
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
            return ResolvedCardEffect(code="boost", target=target, raw_title=title, raw_text=text)

    if title.lower() == "clever":
        return ResolvedCardEffect(code="clever", raw_title=title, raw_text=text)

    if title.lower() == "full-throated":
        return ResolvedCardEffect(code="full_throated", value=1, raw_title=title, raw_text=text)

    if title.lower() == "pack":
        gain_match = re.search(r"Gain\s+(\d+)", text, flags=re.I)
        gain = int(gain_match.group(1)) if gain_match else 1
        return ResolvedCardEffect(code="pack", value=gain, raw_title=title, raw_text=text)

    if title.lower() == "petting zoo animal":
        gain_match = re.search(r"Gain\s+(\d+)", text, flags=re.I)
        gain = int(gain_match.group(1)) if gain_match else 1
        return ResolvedCardEffect(code="petting_zoo_animal", value=gain, raw_title=title, raw_text=text)

    glide = _extract_int_from_title("Glide", title)
    if glide is not None:
        return ResolvedCardEffect(code="glide", value=glide, raw_title=title, raw_text=text)

    jumping = _extract_int_from_title("Jumping", title)
    if jumping is not None:
        gain_money = _extract_int_from_text(r"Gain\s+(\d+)", text, default=0)
        return ResolvedCardEffect(
            code="jumping",
            value=jumping,
            target=str(max(0, gain_money)),
            raw_title=title,
            raw_text=text,
        )

    sun_bathing = _extract_int_from_title("Sun Bathing", title)
    if sun_bathing is not None:
        return ResolvedCardEffect(code="sun_bathing", value=sun_bathing, target="4", raw_title=title, raw_text=text)

    pouch = _extract_int_from_title("Pouch", title)
    if pouch is not None:
        return ResolvedCardEffect(
            code="pouch",
            value=pouch,
            target="2",
            raw_title=title,
            raw_text=text,
        )

    digging = _extract_int_from_title("Digging", title)
    if digging is not None:
        return ResolvedCardEffect(code="digging", value=digging, raw_title=title, raw_text=text)

    pilfering = _extract_int_from_title("Pilfering", title)
    if pilfering is not None:
        return ResolvedCardEffect(code="pilfering", value=pilfering, raw_title=title, raw_text=text)

    posturing = _extract_int_from_title("Posturing", title)
    if posturing is not None:
        return ResolvedCardEffect(code="posturing", value=posturing, raw_title=title, raw_text=text)

    venom = _extract_int_from_title("Venom", title)
    if venom is not None:
        return ResolvedCardEffect(code="venom", value=venom, raw_title=title, raw_text=text)

    if title.lower() == "constriction":
        return ResolvedCardEffect(code="constriction", value=1, raw_title=title, raw_text=text)

    if title.lower() == "sponsor magnet":
        return ResolvedCardEffect(code="sponsor_magnet", value=999, raw_title=title, raw_text=text)

    multiplier_match = re.match(r"^Multiplier:\s*(\w+)$", title, flags=re.I)
    if multiplier_match:
        target = multiplier_match.group(1).strip().lower()
        if target == "card":
            target = "cards"
        if target == "building":
            target = "build"
        if target in {"animals", "cards", "build", "association", "sponsors"}:
            return ResolvedCardEffect(code="multiplier", target=target, value=1, raw_title=title, raw_text=text)

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
            return ResolvedCardEffect(code="action", target=target, value=1, raw_title=title, raw_text=text)

    iconic_match = re.match(r"^Iconic Animal:\s*([a-zA-Z]+)$", title, flags=re.I)
    if iconic_match:
        gain = _extract_int_from_text(r"Gain\s+(\d+)", text, default=1)
        return ResolvedCardEffect(code="iconic_animal", value=max(0, gain), raw_title=title, raw_text=text)

    if title.lower() == "resistance":
        return ResolvedCardEffect(code="resistance", value=2, target="1", raw_title=title, raw_text=text)

    if title.lower() == "assertion":
        return ResolvedCardEffect(code="assertion", value=1, raw_title=title, raw_text=text)

    if title.lower() == "determination":
        return ResolvedCardEffect(code="determination", value=1, raw_title=title, raw_text=text)

    hypnosis = _extract_int_from_title("Hypnosis", title)
    if hypnosis is not None:
        return ResolvedCardEffect(code="hypnosis", value=hypnosis, raw_title=title, raw_text=text)

    if title.lower() == "inventive":
        return ResolvedCardEffect(code="inventive", value=1, raw_title=title, raw_text=text)

    if title.lower() == "inventive: bear":
        return ResolvedCardEffect(code="inventive_bear", target="Bear", raw_title=title, raw_text=text)

    if title.lower() == "inventive: primary":
        return ResolvedCardEffect(code="inventive_primary", value=3, raw_title=title, raw_text=text)

    flock = _extract_int_from_title("Flock Animal", title)
    if flock is not None:
        return ResolvedCardEffect(code="flock_animal", value=flock, raw_title=title, raw_text=text)

    if title.lower() == "symbiosis":
        return ResolvedCardEffect(code="symbiosis", value=1, raw_title=title, raw_text=text)

    if title.lower() == "camouflage":
        return ResolvedCardEffect(code="camouflage", value=1, raw_title=title, raw_text=text)

    if title.lower() == "scuba dive x":
        return ResolvedCardEffect(code="scuba_dive_x", target="seaanimal", raw_title=title, raw_text=text)

    if title.lower() == "monkey gang":
        return ResolvedCardEffect(code="monkey_gang", target="Primate", raw_title=title, raw_text=text)

    adapt = _extract_int_from_title("Adapt", title)
    if adapt is not None:
        return ResolvedCardEffect(code="adapt", value=adapt, raw_title=title, raw_text=text)

    if title.lower() == "cut down":
        return ResolvedCardEffect(code="cut_down", value=1, raw_title=title, raw_text=text)

    if title.lower() == "dominance":
        return ResolvedCardEffect(code="dominance", target="primates", value=1, raw_title=title, raw_text=text)

    if title.lower() == "extra shift":
        return ResolvedCardEffect(code="extra_shift", value=1, raw_title=title, raw_text=text)

    if title.lower() == "mark":
        return ResolvedCardEffect(code="mark", value=1, raw_title=title, raw_text=text)

    if title.lower() == "peacocking":
        return ResolvedCardEffect(code="peacocking", value=1, raw_title=title, raw_text=text)

    if title.lower() == "sea animal magnet":
        return ResolvedCardEffect(code="sea_animal_magnet", target="SeaAnimal", value=999, raw_title=title, raw_text=text)

    shark = _extract_int_from_title("Shark Attack", title)
    if shark is not None:
        return ResolvedCardEffect(code="shark_attack", value=shark, raw_title=title, raw_text=text)

    if title.lower() == "trade":
        return ResolvedCardEffect(code="trade", value=1, raw_title=title, raw_text=text)

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
    advance_break: BreakTrackAdvancer,
    draw_from_deck: DeckDrawer,
    push_to_discard: DiscardPusher,
    clever: Optional[Callable[[], str]] = None,
    full_throated: Optional[Callable[[int], None]] = None,
    increase_appeal: Optional[Callable[[int], None]] = None,
    gain_money: Optional[MoneyGainer] = None,
    take_display_cards: Optional[DisplayTaker] = None,
    sun_bathing: Optional[HandSeller] = None,
    pouch: Optional[HandPoucher] = None,
    posturing: Optional[FreeSmallBuilder] = None,
    multiplier: Optional[MultiplierAdder] = None,
    venom: Optional[TokenApplier] = None,
    constriction: Optional[TokenApplier] = None,
    perform_hypnosis: Optional[HypnosisExecutor] = None,
    perform_pilfering: Optional[PilferingExecutor] = None,
    digging: Optional[DiggingExecutor] = None,
    scavenging: Optional[DiscardScavenger] = None,
    mark_extra_action: Optional[ExtraActionMarker] = None,
    iconic_animal: Optional[Callable[[int], None]] = None,
    resistance: Optional[FinalScoringDrawer] = None,
    perception: Optional[DeckKeepChoiceExecutor] = None,
    reveal_keep_by_card_type: Optional[RevealKeepByTypeExecutor] = None,
    assertion: Optional[BaseProjectTaker] = None,
    inventive: Optional[XTokenGainer] = None,
    count_icon: Optional[IconCounter] = None,
    count_primary_icons: Optional[PrimaryIconCounter] = None,
    adapt: Optional[FinalScoringAdapter] = None,
    cut_down: Optional[EnclosureRemover] = None,
    extra_shift: Optional[AssociationWorkerReturner] = None,
    mark: Optional[DisplayMarker] = None,
    peacocking: Optional[LargeBirdAviaryPlacer] = None,
    trade: Optional[TradeExecutor] = None,
    shark_attack: Optional[SharkAttackExecutor] = None,
    dominance: Optional[SpecificProjectTaker] = None,
    symbiosis: Optional[SymbiosisExecutor] = None,
    camouflage: Optional[CamouflageGrant] = None,
    boost: Optional[BoostActionExecutor] = None,
    glide: Optional[GlideRewardExecutor] = None,
) -> List[str]:
    effect = resolve_card_effect(card)
    messages: List[str] = []

    if effect.code == "none":
        return messages

    if effect.code == "sprint":
        drawn = list(draw_from_deck(max(0, effect.value)))
        messages.append(f"effect[{effect.code}] drew={len(drawn)}")
        return messages

    if effect.code == "hunter":
        if reveal_keep_by_card_type is None:
            raise ValueError("hunter requires reveal_keep_by_card_type callback.")
        drew, kept = reveal_keep_by_card_type(max(0, effect.value), "animal")
        messages.append(
            f"effect[hunter] revealed={drew} kept={kept} discarded={max(0, drew - kept)}"
        )
        return messages

    if effect.code == "snapping":
        if take_display_cards is None:
            messages.append("effect[snapping] unsupported(no_callback)")
            return messages
        replenish_each = effect.target == "replenish_each"
        taken = take_display_cards(max(0, effect.value), "", replenish_each)
        messages.append(f"effect[snapping] taken={taken}")
        return messages

    if effect.code == "perception":
        if perception is None:
            raise ValueError("perception requires perception callback.")
        drew, kept = perception(
            max(0, effect.value),
            max(1, int(effect.target or "1")),
        )
        messages.append(
            f"effect[perception] drew={drew} kept={kept} discarded={max(0, drew - kept)}"
        )
        return messages

    if effect.code == "scavenging":
        if scavenging is None:
            messages.append("effect[scavenging] unsupported(no_callback)")
            return messages
        draw_count = max(0, effect.value)
        keep_count = max(1, int(effect.target or "1"))
        drew, kept = scavenging(draw_count, keep_count)
        messages.append(f"effect[scavenging] drew={drew} kept={kept} discarded={max(0, drew - kept)}")
        return messages

    if effect.code == "boost":
        if not effect.target:
            raise ValueError("boost requires effect target.")
        if boost is None:
            raise ValueError("boost requires boost callback.")
        result = str(boost(effect.target)).strip()
        if result:
            messages.append(f"effect[{effect.code}] target={effect.target} {result}")
        else:
            messages.append(f"effect[{effect.code}] target={effect.target}")
        return messages

    if effect.code == "clever":
        if clever is None:
            raise ValueError("clever requires clever callback.")
        chosen = clever()
        move_action_to_slot_1(chosen)
        messages.append(f"effect[{effect.code}] target={chosen}")
        return messages

    if effect.code == "full_throated":
        if full_throated is None:
            raise ValueError("full_throated requires full_throated callback.")
        full_throated(max(0, effect.value))
        messages.append(f"effect[{effect.code}] workers=+{max(0, effect.value)}")
        return messages

    if effect.code in {"pack", "petting_zoo_animal"}:
        if increase_appeal is None:
            raise ValueError(f"{effect.code} requires increase_appeal callback.")
        increase_appeal(max(0, effect.value))
        messages.append(f"effect[{effect.code}] appeal=+{max(0, effect.value)}")
        return messages

    if effect.code == "glide":
        if glide is None:
            messages.append("effect[glide] unsupported(no_callback)")
            return messages
        summary = str(glide(max(0, effect.value))).strip()
        if summary:
            messages.append(f"effect[glide] {summary}")
        else:
            messages.append("effect[glide] resolved")
        return messages

    if effect.code == "jumping":
        money = max(0, int(effect.target or "0"))
        if gain_money is None:
            raise ValueError("jumping requires gain_money callback.")
        advance_break(max(0, effect.value))
        if money > 0:
            gain_money(money)
        messages.append(f"effect[jumping] break=+{max(0, effect.value)} money=+{money}")
        return messages

    if effect.code == "sun_bathing":
        if sun_bathing is None:
            messages.append("effect[sun_bathing] unsupported(no_callback)")
            return messages
        sold = sun_bathing(max(0, effect.value), max(0, int(effect.target or "0")))
        messages.append(f"effect[sun_bathing] sold={sold}")
        return messages

    if effect.code == "pouch":
        if pouch is None:
            messages.append("effect[pouch] unsupported(no_callback)")
            return messages
        pouched = pouch(max(0, effect.value), max(0, int(effect.target or "0")))
        messages.append(f"effect[pouch] pouched={pouched}")
        return messages

    if effect.code == "digging":
        if digging is None:
            raise ValueError("digging requires digging callback.")
        did = digging(max(0, effect.value))
        messages.append(f"effect[digging] loops={did}")
        return messages

    if effect.code == "pilfering":
        if perform_pilfering is None:
            messages.append("effect[pilfering] unsupported(no_callback)")
            return messages
        summary = perform_pilfering(max(0, effect.value))
        messages.append(f"effect[pilfering] {summary}".rstrip())
        return messages

    if effect.code == "posturing":
        if posturing is None:
            messages.append("effect[posturing] unsupported(no_callback)")
            return messages
        placed = posturing(max(0, effect.value))
        messages.append(f"effect[posturing] placed={placed}")
        return messages

    if effect.code == "venom":
        if venom is None:
            messages.append("effect[venom] unsupported(no_callback)")
            return messages
        affected = venom(max(0, effect.value))
        messages.append(f"effect[venom] affected_players={affected}")
        return messages

    if effect.code == "constriction":
        if constriction is None:
            messages.append("effect[constriction] unsupported(no_callback)")
            return messages
        affected = constriction(max(0, effect.value))
        messages.append(f"effect[constriction] affected_players={affected}")
        return messages

    if effect.code == "hypnosis":
        if perform_hypnosis is None:
            messages.append("effect[hypnosis] unsupported(no_callback)")
            return messages
        summary = perform_hypnosis(max(0, effect.value))
        messages.append(f"effect[hypnosis] {summary}".rstrip())
        return messages

    if effect.code == "sponsor_magnet":
        if take_display_cards is None:
            messages.append("effect[sponsor_magnet] unsupported(no_callback)")
            return messages
        taken = take_display_cards(max(0, effect.value), "sponsor", False)
        messages.append(f"effect[sponsor_magnet] taken={taken}")
        return messages

    if effect.code == "multiplier":
        if multiplier is None or not effect.target:
            messages.append("effect[multiplier] unsupported(no_callback)")
            return messages
        ok = multiplier(effect.target)
        messages.append(f"effect[multiplier] target={effect.target} applied={1 if ok else 0}")
        return messages

    if effect.code == "action":
        if mark_extra_action is None:
            messages.append("effect[action] unsupported(no_callback)")
            return messages
        mark_extra_action(effect.target or "unknown")
        messages.append(f"effect[action] target={effect.target}")
        return messages

    if effect.code == "iconic_animal":
        if iconic_animal is None:
            messages.append("effect[iconic_animal] unsupported(no_callback)")
            return messages
        iconic_animal(max(0, effect.value))
        messages.append(f"effect[iconic_animal] conservation=+{max(0, effect.value)}")
        return messages

    if effect.code == "resistance":
        if resistance is None:
            messages.append("effect[resistance] unsupported(no_callback)")
            return messages
        drew, kept = resistance(max(0, effect.value), max(1, int(effect.target or "1")))
        messages.append(f"effect[resistance] drew={drew} kept={kept} discarded={max(0, drew-kept)}")
        return messages

    if effect.code == "assertion":
        if assertion is None:
            messages.append("effect[assertion] unsupported(no_callback)")
            return messages
        taken = assertion(max(0, effect.value))
        messages.append(f"effect[assertion] taken={taken}")
        return messages

    if effect.code == "determination":
        if mark_extra_action is None:
            messages.append("effect[determination] unsupported(no_callback)")
            return messages
        mark_extra_action("any")
        messages.append("effect[determination] granted=1")
        return messages

    if effect.code == "extra_action_strength":
        if mark_extra_action is None:
            messages.append("effect[extra_action_strength] unsupported(no_callback)")
            return messages
        mark_extra_action(f"strength:{max(0, effect.value)}")
        messages.append(f"effect[extra_action_strength] strength={max(0, effect.value)}")
        return messages

    if effect.code == "inventive":
        if inventive is None:
            messages.append("effect[inventive] unsupported(no_callback)")
            return messages
        gained = inventive(max(0, effect.value))
        messages.append(f"effect[inventive] gained={gained}")
        return messages

    if effect.code == "inventive_bear":
        if inventive is None or count_icon is None:
            messages.append("effect[inventive_bear] unsupported(no_callback)")
            return messages
        amount = max(0, count_icon(effect.target))
        gained = inventive(amount)
        messages.append(f"effect[inventive_bear] icon={effect.target} amount={amount} gained={gained}")
        return messages

    if effect.code == "inventive_primary":
        if inventive is None or count_primary_icons is None:
            messages.append("effect[inventive_primary] unsupported(no_callback)")
            return messages
        amount = min(max(0, effect.value), max(0, count_primary_icons()))
        gained = inventive(amount)
        messages.append(f"effect[inventive_primary] amount={amount} gained={gained}")
        return messages

    if effect.code == "flock_animal":
        messages.append(f"effect[flock_animal] size={max(0, effect.value)}")
        return messages

    if effect.code == "symbiosis":
        if symbiosis is None:
            messages.append("effect[symbiosis] unsupported(no_callback)")
            return messages
        nested = symbiosis()
        messages.append(f"effect[symbiosis] nested={len(nested)}")
        messages.extend(nested)
        return messages

    if effect.code == "camouflage":
        if camouflage is None:
            messages.append("effect[camouflage] unsupported(no_callback)")
            return messages
        camouflage(max(0, effect.value))
        messages.append(f"effect[camouflage] granted={max(0, effect.value)}")
        return messages

    if effect.code == "scuba_dive_x":
        if reveal_keep_by_card_type is None:
            raise ValueError("scuba_dive_x requires reveal_keep_by_card_type callback.")
        if count_icon is None:
            raise ValueError("scuba_dive_x requires count_icon callback.")
        x = max(0, count_icon(effect.target))
        drew, kept = reveal_keep_by_card_type(x, "sponsor")
        messages.append(
            f"effect[scuba_dive_x] x={x} revealed={drew} kept={kept} discarded={max(0, drew - kept)}"
        )
        return messages

    if effect.code == "monkey_gang":
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
        messages.append(f"effect[monkey_gang] revealed={len(revealed)} kept={kept}")
        return messages

    if effect.code == "adapt":
        if adapt is None:
            messages.append("effect[adapt] unsupported(no_callback)")
            return messages
        drew, discarded = adapt(max(0, effect.value))
        messages.append(f"effect[adapt] drew={drew} discarded={discarded}")
        return messages

    if effect.code == "cut_down":
        if cut_down is None:
            messages.append("effect[cut_down] unsupported(no_callback)")
            return messages
        removed, refunded = cut_down(max(0, effect.value))
        messages.append(f"effect[cut_down] removed={removed} refunded={refunded}")
        return messages

    if effect.code == "extra_shift":
        if extra_shift is None:
            messages.append("effect[extra_shift] unsupported(no_callback)")
            return messages
        returned = extra_shift(max(0, effect.value))
        messages.append(f"effect[extra_shift] returned={returned}")
        return messages

    if effect.code == "mark":
        if mark is None:
            messages.append("effect[mark] unsupported(no_callback)")
            return messages
        marked = mark(max(0, effect.value))
        messages.append(f"effect[mark] marked={marked}")
        return messages

    if effect.code == "peacocking":
        if peacocking is None:
            messages.append("effect[peacocking] unsupported(no_callback)")
            return messages
        placed = peacocking(max(0, effect.value))
        messages.append(f"effect[peacocking] placed={placed}")
        return messages

    if effect.code == "sea_animal_magnet":
        if take_display_cards is None:
            messages.append("effect[sea_animal_magnet] unsupported(no_callback)")
            return messages
        taken = take_display_cards(max(0, effect.value), f"badge:{effect.target}", False)
        messages.append(f"effect[sea_animal_magnet] badge={effect.target} taken={taken}")
        return messages

    if effect.code == "shark_attack":
        if shark_attack is None:
            messages.append("effect[shark_attack] unsupported(no_callback)")
            return messages
        discarded, money = shark_attack(max(0, effect.value))
        messages.append(f"effect[shark_attack] discarded={discarded} money=+{money}")
        return messages

    if effect.code == "trade":
        if trade is None:
            messages.append("effect[trade] unsupported(no_callback)")
            return messages
        traded = trade(max(0, effect.value))
        messages.append(f"effect[trade] traded={traded}")
        return messages

    if effect.code == "dominance":
        if dominance is None:
            messages.append("effect[dominance] unsupported(no_callback)")
            return messages
        taken = dominance(effect.target, max(0, effect.value))
        messages.append(f"effect[dominance] target={effect.target} taken={taken}")
        return messages

    messages.append(f"effect[{effect.code}] unsupported title='{effect.raw_title}'")
    return messages


def build_effect_coverage(cards: Sequence[Any]) -> Dict[str, int]:
    total = 0
    mapped = 0
    unmapped = 0
    no_effect = 0
    for card in cards:
        if str(getattr(card, "card_type", "")).lower() not in {"animal", "sponsor"}:
            continue
        total += 1
        resolved = resolve_card_effect(card)
        if resolved.code == "none":
            no_effect += 1
        elif resolved.supported:
            mapped += 1
        else:
            unmapped += 1
    return {
        "total": total,
        "mapped": mapped,
        "unmapped": unmapped,
        "supported": mapped,
        "unsupported": unmapped,
        "no_effect": no_effect,
    }

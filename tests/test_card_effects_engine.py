from types import SimpleNamespace

import pytest

from arknova_engine.card_effects import apply_animal_effect, build_effect_coverage, resolve_card_effect


def _make_card(
    *,
    card_type: str = "animal",
    ability_title: str = "",
    ability_text: str = "",
    effects=(),
):
    return SimpleNamespace(
        card_type=card_type,
        ability_title=ability_title,
        ability_text=ability_text,
        effects=tuple(effects),
    )


def test_resolve_card_effect_for_sprint_and_hunter():
    sprint = _make_card(ability_title="Sprint 3", ability_text="Draw 3 card(s) from the deck.")
    hunter = _make_card(ability_title="Hunter 2")

    sprint_effect = resolve_card_effect(sprint)
    hunter_effect = resolve_card_effect(hunter)

    assert sprint_effect.code == "sprint"
    assert sprint_effect.value == 3
    assert hunter_effect.code == "hunter"
    assert hunter_effect.value == 2


def test_resolve_card_effect_for_snapping_and_sun_bathing_pouch_and_multiplier():
    snapping = _make_card(ability_title="Snapping 2", ability_text="2x: Gain any 1 card from the display. You may replenish in between.")
    sun_bathing = _make_card(ability_title="Sun Bathing 3", ability_text="You may sell up to 3 card(s) from your hand for 4")
    pouch = _make_card(ability_title="Pouch 1", ability_text="You may place 1 card(s) from your hand under this card to gain 2")
    multiplier = _make_card(ability_title="Multiplier: Sponsors", ability_text="Place 1")

    s = resolve_card_effect(snapping)
    b = resolve_card_effect(sun_bathing)
    p = resolve_card_effect(pouch)
    m = resolve_card_effect(multiplier)

    assert s.code == "snapping"
    assert s.value == 2
    assert s.target == "replenish_each"
    assert b.code == "sun_bathing"
    assert b.value == 3
    assert p.code == "pouch"
    assert p.value == 1
    assert p.target == "2"
    assert m.code == "multiplier"
    assert m.target == "sponsors"


def test_resolve_card_effect_for_inventive_and_resistance_and_assertion():
    inventive = _make_card(ability_title="Inventive", ability_text="Gain")
    inventive_primary = _make_card(ability_title="Inventive: Primary", ability_text="Gain 1/2/3")
    resistance = _make_card(
        ability_title="Resistance",
        ability_text="Draw 2 Final Scoring cards. Keep 1 and discard the other.",
    )
    assertion = _make_card(
        ability_title="Assertion",
        ability_text="You may add any 1 of the unused base conservation projects to your hand.",
    )

    i = resolve_card_effect(inventive)
    ip = resolve_card_effect(inventive_primary)
    r = resolve_card_effect(resistance)
    a = resolve_card_effect(assertion)

    assert i.code == "inventive"
    assert ip.code == "inventive_primary"
    assert r.code == "resistance"
    assert a.code == "assertion"


def test_resolve_card_effect_for_hypnosis_and_pilfering():
    hypnosis = _make_card(ability_title="Hypnosis 3")
    pilfering = _make_card(ability_title="Pilfering 2")

    h = resolve_card_effect(hypnosis)
    p = resolve_card_effect(pilfering)

    assert h.code == "hypnosis"
    assert h.value == 3
    assert p.code == "pilfering"
    assert p.value == 2


def test_apply_hunter_effect_keeps_first_animal_and_discards_rest():
    played_card = _make_card(ability_title="Hunter 3")
    seen = []

    messages = apply_animal_effect(
        card=played_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        reveal_keep_by_card_type=lambda draw_count, card_type: seen.append((draw_count, card_type)) or (3, 1),
    )

    assert seen == [(3, "animal")]
    assert any("effect[hunter] revealed=3 kept=1 discarded=2" == msg for msg in messages)


def test_apply_snapping_effect_takes_cards_from_display():
    played_card = _make_card(ability_title="Snapping 1", ability_text="Gain any 1 card from the display.")
    taken = []

    messages = apply_animal_effect(
        card=played_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        take_display_cards=lambda count, _card_type_filter, _replenish_each: taken.append(count) or count,
    )

    assert taken == [1]
    assert any("effect[snapping]" in msg for msg in messages)


def test_apply_boost_action_card_effect_uses_callback():
    played_card = _make_card(
        ability_title="Boost: Association",
        ability_text="After finishing this action, you may place your Association Action card",
    )
    boosted = []

    messages = apply_animal_effect(
        card=played_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        boost=lambda target: boosted.append(target) or "slot=5",
    )

    assert boosted == ["association"]
    assert any("effect[boost] target=association slot=5" == msg for msg in messages)


def test_apply_resistance_and_assertion_and_inventive_callbacks():
    card_resistance = _make_card(ability_title="Resistance")
    card_assertion = _make_card(ability_title="Assertion")
    card_inventive = _make_card(ability_title="Inventive")

    resistance = apply_animal_effect(
        card=card_resistance,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        resistance=lambda draw_n, keep_n: (draw_n, keep_n),
    )
    assertion = apply_animal_effect(
        card=card_assertion,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        assertion=lambda count: count,
    )
    inventive = apply_animal_effect(
        card=card_inventive,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        inventive=lambda n: n,
    )

    assert any("effect[resistance]" in msg for msg in resistance)
    assert any("effect[assertion]" in msg for msg in assertion)
    assert any("effect[inventive]" in msg for msg in inventive)


def test_apply_hypnosis_and_pilfering_callbacks():
    hypnosis_card = _make_card(ability_title="Hypnosis 3")
    pilfering_card = _make_card(ability_title="Pilfering 1")

    hypnosis = apply_animal_effect(
        card=hypnosis_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        perform_hypnosis=lambda value: f"target=P2 slot<={value}",
    )
    pilfering = apply_animal_effect(
        card=pilfering_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        perform_pilfering=lambda value: f"targets={value}",
    )

    assert any("effect[hypnosis] target=P2 slot<=3" == msg for msg in hypnosis)
    assert any("effect[pilfering] targets=1" == msg for msg in pilfering)


def test_apply_digging_uses_required_callback():
    digging_card = _make_card(ability_title="Digging 2")
    seen = []

    messages = apply_animal_effect(
        card=digging_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        digging=lambda value: seen.append(value) or value,
    )

    assert seen == [2]
    assert any("effect[digging] loops=2" == msg for msg in messages)


def test_apply_digging_requires_callback():
    digging_card = _make_card(ability_title="Digging 1")

    with pytest.raises(ValueError, match="digging requires digging callback"):
        apply_animal_effect(
            card=digging_card,
            move_action_to_slot_1=lambda _: None,
            advance_break=lambda _: False,
            draw_from_deck=lambda _: [],
            push_to_discard=lambda _: None,
        )


@pytest.mark.parametrize(
    ("card", "extra_kwargs", "error_pattern"),
    [
        (
            _make_card(ability_title="Hunter 1"),
            {},
            "hunter requires reveal_keep_by_card_type callback",
        ),
        (
            _make_card(ability_title="Perception 2", ability_text="Add 1 to your hand."),
            {},
            "perception requires perception callback",
        ),
        (
            _make_card(ability_title="Clever"),
            {},
            "clever requires clever callback",
        ),
        (
            _make_card(ability_title="Full-throated"),
            {},
            "full_throated requires full_throated callback",
        ),
        (
            _make_card(ability_title="Pack", ability_text="Gain 2"),
            {},
            "pack requires increase_appeal callback",
        ),
        (
            _make_card(ability_title="Jumping 2", ability_text="Gain 4"),
            {},
            "jumping requires gain_money callback",
        ),
        (
            _make_card(
                ability_title="Boost: Association",
                ability_text="After finishing this action, you may place your Association Action card",
            ),
            {},
            "boost requires boost callback",
        ),
        (
            _make_card(ability_title="Scuba Dive X"),
            {"count_icon": lambda _: 2},
            "scuba_dive_x requires reveal_keep_by_card_type callback",
        ),
        (
            _make_card(ability_title="Scuba Dive X"),
            {"reveal_keep_by_card_type": lambda draw_count, card_type: (draw_count, 1)},
            "scuba_dive_x requires count_icon callback",
        ),
    ],
)
def test_apply_effect_callbacks_are_required(card, extra_kwargs, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        apply_animal_effect(
            card=card,
            move_action_to_slot_1=lambda _: None,
            advance_break=lambda _: False,
            draw_from_deck=lambda _: [],
            push_to_discard=lambda _: None,
            **extra_kwargs,
        )


def test_build_effect_coverage_counts_supported_and_unsupported():
    cards = [
        _make_card(card_type="animal"),
        _make_card(card_type="animal", ability_title="Sprint 1"),
        _make_card(card_type="animal", ability_title="Unseen Ability"),
        _make_card(card_type="sponsor", effects=[("income", "Take 1 card.")]),
    ]

    coverage = build_effect_coverage(cards)

    assert coverage["total"] == 4
    assert coverage["mapped"] == 2
    assert coverage["unmapped"] == 1
    assert coverage["supported"] == 2
    assert coverage["unsupported"] == 1
    assert coverage["no_effect"] == 1

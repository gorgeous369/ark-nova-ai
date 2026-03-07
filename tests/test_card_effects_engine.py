from types import SimpleNamespace

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

    assert sprint_effect.code == "draw_from_deck"
    assert sprint_effect.value == 3
    assert hunter_effect.code == "hunter"
    assert hunter_effect.value == 2


def test_resolve_card_effect_for_snapping_and_sun_bathing_and_multiplier():
    snapping = _make_card(ability_title="Snapping 2", ability_text="2x: Gain any 1 card from the display. You may replenish in between.")
    sun_bathing = _make_card(ability_title="Sun Bathing 3", ability_text="You may sell up to 3 card(s) from your hand for 4")
    multiplier = _make_card(ability_title="Multiplier: Sponsors", ability_text="Place 1")

    s = resolve_card_effect(snapping)
    b = resolve_card_effect(sun_bathing)
    m = resolve_card_effect(multiplier)

    assert s.code == "take_display_cards"
    assert s.value == 2
    assert s.target == "replenish_each"
    assert b.code == "sell_hand_cards"
    assert b.value == 3
    assert m.code == "multiplier_token"
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

    assert i.code == "gain_x_tokens"
    assert ip.code == "gain_x_tokens_from_primary"
    assert r.code == "draw_final_scoring_keep"
    assert a.code == "take_unused_base_project"


def test_apply_hunter_effect_keeps_first_animal_and_discards_rest():
    played_card = _make_card(ability_title="Hunter 3")
    hand = []
    discard = []
    deck = [
        _make_card(card_type="sponsor"),
        _make_card(card_type="animal"),
        _make_card(card_type="animal"),
    ]

    def draw_from_deck(count: int):
        drawn = deck[:count]
        del deck[:count]
        hand.extend(drawn)
        return drawn

    def push_to_discard(cards):
        for card in cards:
            if card in hand:
                hand.remove(card)
            discard.append(card)

    messages = apply_animal_effect(
        card=played_card,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=draw_from_deck,
        push_to_discard=push_to_discard,
        choose_action_for_clever=lambda: "cards",
        increase_workers=lambda _: None,
        increase_appeal=lambda _: None,
    )

    assert len(hand) == 1
    assert hand[0].card_type == "animal"
    assert len(discard) == 2
    assert any("effect[hunter]" in msg for msg in messages)


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
    assert any("effect[take_display_cards]" in msg for msg in messages)


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
        draw_final_scoring_keep=lambda draw_n, keep_n: (draw_n, keep_n),
    )
    assertion = apply_animal_effect(
        card=card_assertion,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        take_unused_base_project=lambda count: count,
    )
    inventive = apply_animal_effect(
        card=card_inventive,
        move_action_to_slot_1=lambda _: None,
        advance_break=lambda _: False,
        draw_from_deck=lambda _: [],
        push_to_discard=lambda _: None,
        gain_x_tokens=lambda n: n,
    )

    assert any("effect[draw_final_scoring_keep]" in msg for msg in resistance)
    assert any("effect[take_unused_base_project]" in msg for msg in assertion)
    assert any("effect[gain_x_tokens]" in msg for msg in inventive)


def test_build_effect_coverage_counts_supported_and_unsupported():
    cards = [
        _make_card(card_type="animal"),
        _make_card(card_type="animal", ability_title="Sprint 1"),
        _make_card(card_type="animal", ability_title="Unseen Ability"),
        _make_card(card_type="sponsor", effects=[("income", "Take 1 card.")]),
    ]

    coverage = build_effect_coverage(cards)

    assert coverage["total"] == 4
    assert coverage["supported"] == 2
    assert coverage["unsupported"] == 1
    assert coverage["no_effect"] == 1

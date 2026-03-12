from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence

import main


DEFAULT_PLAYER_NAMES = ["P1", "P2"]


def make_state(seed: int, *, include_marine_world: bool = False) -> main.GameState:
    return main.setup_game(
        seed=seed,
        player_names=list(DEFAULT_PLAYER_NAMES),
        include_marine_world=include_marine_world,
    )


def set_action_strength(player: main.PlayerState, action_name: str, strength: int) -> None:
    slot_index = max(1, min(len(main.MAIN_ACTION_CARDS), int(strength))) - 1
    action_order = [name for name in main.MAIN_ACTION_CARDS if name != action_name]
    action_order.insert(slot_index, action_name)
    player.action_order = action_order


def make_card(
    name: str,
    *,
    number: int,
    instance_id: str,
    card_type: str = "animal",
    cost: int = 0,
    size: int = 1,
    appeal: int = 0,
    conservation: int = 0,
    reputation_gain: int = 0,
    badges: Sequence[str] = (),
    ability_title: str = "",
    ability_text: str = "",
    effects: Sequence[tuple[str, str]] = (),
) -> main.AnimalCard:
    return main.AnimalCard(
        name=name,
        cost=cost,
        size=size,
        appeal=appeal,
        conservation=conservation,
        reputation_gain=reputation_gain,
        card_type=card_type,
        badges=tuple(str(badge) for badge in badges),
        ability_title=ability_title,
        ability_text=ability_text,
        effects=tuple((str(code), str(text)) for code, text in effects),
        number=number,
        instance_id=instance_id,
    )


def configure_player(
    state: main.GameState,
    *,
    player_id: int = 0,
    current_player: bool = True,
    action_name: Optional[str] = None,
    strength: Optional[int] = None,
    action_order: Optional[Sequence[str]] = None,
    money: Optional[int] = None,
    zoo_card_numbers: Sequence[int] = (),
    zoo_cards: Sequence[main.AnimalCard] = (),
    hand: Optional[Sequence[main.AnimalCard]] = None,
    enclosure_sizes: Optional[Sequence[int]] = None,
    upgraded_actions: Optional[Sequence[str]] = None,
) -> main.PlayerState:
    player = state.players[player_id]
    if current_player:
        state.current_player = player_id
    if action_order is not None:
        player.action_order = list(action_order)
    if action_name is not None and strength is not None:
        set_action_strength(player, action_name, strength)
    if money is not None:
        player.money = int(money)
    if zoo_card_numbers:
        player.zoo_cards.extend(take_card_by_number(state, number) for number in zoo_card_numbers)
    if zoo_cards:
        player.zoo_cards.extend(list(zoo_cards))
    if hand is not None:
        player.hand = list(hand)
    if enclosure_sizes is not None:
        player.enclosures = [
            main.Enclosure(size=int(size), occupied=False)
            for size in enclosure_sizes
        ]
    if upgraded_actions is not None:
        for upgraded_action in upgraded_actions:
            player.action_upgraded[str(upgraded_action)] = True
    return player


def configure_standard_enclosures(
    player: main.PlayerState,
    *,
    sizes: Sequence[int],
    origins: Optional[Sequence[tuple[int, int]]] = None,
    rotations: Optional[Sequence[str]] = None,
    occupied: Optional[Sequence[bool]] = None,
) -> None:
    enclosure_origins = list(origins or [(index, 0) for index, _size in enumerate(sizes)])
    enclosure_rotations = list(rotations or ["ROT_0"] * len(sizes))
    enclosure_occupied = list(occupied or [False] * len(sizes))
    if not (
        len(enclosure_origins) == len(sizes)
        and len(enclosure_rotations) == len(sizes)
        and len(enclosure_occupied) == len(sizes)
    ):
        raise AssertionError("Standard enclosure configuration lengths must match sizes.")

    player.enclosures = []
    player.enclosure_objects = []
    for size, origin, rotation, is_occupied in zip(
        sizes,
        enclosure_origins,
        enclosure_rotations,
        enclosure_occupied,
        strict=True,
    ):
        player.enclosures.append(
            main.Enclosure(
                size=int(size),
                occupied=bool(is_occupied),
                origin=tuple(origin),
                rotation=str(rotation),
            )
        )
        player.enclosure_objects.append(
            main.EnclosureObject(
                size=int(size),
                enclosure_type=f"enclosure_{int(size)}",
                adjacent_rock=0,
                adjacent_water=0,
                animals_inside=0,
                origin=tuple(origin),
                rotation=str(rotation),
            )
        )


def make_basic_animals_play_state(
    seed: int = 610,
    *,
    player_id: int = 0,
    money: int = 25,
    reputation: int = 6,
    hand: Optional[Sequence[main.AnimalCard]] = (),
    enclosure_sizes: Sequence[int] = (2,),
    action_name: Optional[str] = None,
    strength: Optional[int] = None,
    action_order: Optional[Sequence[str]] = None,
) -> tuple[main.GameState, main.PlayerState]:
    state = make_state(seed)
    player = configure_player(
        state,
        player_id=player_id,
        money=money,
        hand=hand,
        enclosure_sizes=enclosure_sizes,
        action_name=action_name,
        strength=strength,
        action_order=action_order,
    )
    player.reputation = int(reputation)
    player.zoo_cards = []
    if enclosure_sizes:
        configure_standard_enclosures(player, sizes=enclosure_sizes)
    return state, player


def take_card_by_number(state: main.GameState, number: int) -> main.AnimalCard:
    for zone in (state.zoo_deck, state.zoo_display, state.zoo_discard):
        for idx, card in enumerate(zone):
            if card.number == number:
                return zone.pop(idx)
    for player in state.players:
        for idx, card in enumerate(player.hand):
            if card.number == number:
                return player.hand.pop(idx)
    raise AssertionError(f"Card #{number} not found in known zones")


def find_action(
    actions: Iterable[main.Action],
    *,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> main.Action:
    for action in actions:
        details = dict(action.details or {})
        if any(details.get(key) != value for key, value in expected_details.items()):
            continue
        if predicate is not None and not predicate(action):
            continue
        return action
    raise AssertionError(f"Action not found for details={expected_details}")


def has_action(
    actions: Iterable[main.Action],
    *,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> bool:
    try:
        find_action(actions, predicate=predicate, **expected_details)
    except AssertionError:
        return False
    return True


def materialize_first_action(state: main.GameState, player_id: int) -> main.Action:
    player = state.players[player_id]
    legal_actions = main.legal_actions(
        player,
        state=state,
        player_id=player_id,
    )
    if not legal_actions:
        raise AssertionError("No legal concrete actions available.")
    for action in legal_actions:
        if action.type == main.ActionType.PENDING_DECISION:
            return action
    for action in legal_actions:
        if action.type == main.ActionType.MAIN_ACTION and action.card_name == "build":
            return action
    for action in legal_actions:
        if (
            action.type == main.ActionType.MAIN_ACTION
            and action.card_name == "sponsors"
            and bool((action.details or {}).get("use_break_ability"))
        ):
            return action
    for action in legal_actions:
        if action.type == main.ActionType.X_TOKEN:
            return action
    return legal_actions[0]


def set_pending_decision(
    state: main.GameState,
    *,
    player_id: int,
    kind: str,
    payload: Optional[dict[str, object]] = None,
) -> None:
    state.pending_decision_kind = str(kind)
    state.pending_decision_player_id = int(player_id)
    state.pending_decision_payload = dict(payload or {})


def legal_actions_for_player(state: main.GameState, *, player_id: int) -> list[main.Action]:
    return list(
        main.legal_actions(
            state.players[player_id],
            state=state,
            player_id=player_id,
        )
    )


def pending_actions(state: main.GameState) -> list[main.Action]:
    if state.pending_decision_player_id is None:
        raise AssertionError("No pending decision player is set.")
    return legal_actions_for_player(state, player_id=int(state.pending_decision_player_id))


def find_main_action(
    actions: Iterable[main.Action],
    *,
    card_name: str,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> main.Action:
    return find_action(
        actions,
        predicate=lambda action: (
            action.type == main.ActionType.MAIN_ACTION
            and str(action.card_name or "") == str(card_name)
            and (predicate(action) if predicate is not None else True)
        ),
        **expected_details,
    )


def find_pending_action(
    state: main.GameState,
    *,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> main.Action:
    return find_action(
        pending_actions(state),
        predicate=predicate,
        **expected_details,
    )


def animals_sequence_index(
    state: main.GameState,
    *,
    player_id: int,
    strength: int,
    card_instance_ids: Sequence[str],
) -> int:
    target_card_ids = [str(card_instance_id) for card_instance_id in card_instance_ids]
    for option in main.list_legal_animals_options(
        state=state,
        player_id=player_id,
        strength=int(strength),
    ):
        option_card_ids = [
            str(play.get("card_instance_id") or "")
            for play in list(option.get("plays") or [])
        ]
        if option_card_ids == target_card_ids:
            return int(option["index"]) - 1
    raise AssertionError(f"Animals sequence not found for card ids={target_card_ids}")


def find_animals_sequence_action(
    state: main.GameState,
    *,
    player_id: int,
    strength: int,
    card_instance_ids: Sequence[str],
) -> main.Action:
    player = state.players[player_id]
    sequence_index = animals_sequence_index(
        state,
        player_id=player_id,
        strength=strength,
        card_instance_ids=card_instance_ids,
    )
    return find_main_action(
        main.legal_actions(player, state=state, player_id=player_id),
        card_name="animals",
        animals_sequence_index=sequence_index,
    )


def play_animals_sequence(
    state: main.GameState,
    *,
    player_id: int,
    strength: int,
    card_instance_ids: Sequence[str],
) -> main.Action:
    action = find_animals_sequence_action(
        state,
        player_id=player_id,
        strength=strength,
        card_instance_ids=card_instance_ids,
    )
    main.apply_action(state, action)
    return action


def apply_sponsors_from_hand(
    state: main.GameState,
    *,
    player_id: int,
    cards: Sequence[main.AnimalCard],
    use_break_ability: bool = False,
) -> None:
    player = state.players[player_id]
    selections = []
    for card in cards:
        source_index = next(
            idx
            for idx, hand_card in enumerate(player.hand)
            if str(hand_card.instance_id or "") == str(card.instance_id or "")
        )
        selections.append(
            {
                "source": "hand",
                "source_index": source_index,
                "card_instance_id": str(card.instance_id or ""),
            }
        )
    main.apply_action(
        state,
        main.Action(
            main.ActionType.MAIN_ACTION,
            card_name="sponsors",
            details={
                "use_break_ability": bool(use_break_ability),
                "sponsor_selections": selections,
            },
        ),
    )


def play_main_action(
    state: main.GameState,
    *,
    player_id: int,
    card_name: str,
    value: int = 0,
    details: Optional[dict[str, object]] = None,
) -> main.Action:
    action = main.Action(
        main.ActionType.MAIN_ACTION,
        value=int(value),
        card_name=str(card_name),
        details=dict(details or {}),
    )
    main.apply_action(state, action)
    return action


def play_legal_main_action(
    state: main.GameState,
    *,
    player_id: int,
    card_name: str,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> main.Action:
    action = find_main_action(
        legal_actions_for_player(state, player_id=player_id),
        card_name=card_name,
        predicate=predicate,
        **expected_details,
    )
    main.apply_action(state, action)
    return action


def play_pending_action(
    state: main.GameState,
    *,
    predicate: Optional[Callable[[main.Action], bool]] = None,
    **expected_details: object,
) -> main.Action:
    action = find_pending_action(
        state,
        predicate=predicate,
        **expected_details,
    )
    main.apply_action(state, action)
    return action

import json
from pathlib import Path

from arknova_engine.setup_data import load_combined_cards_payload


def test_cards_dataset_exists_and_has_expected_shape():
    assert Path("data/cards/cards.base.json").exists()
    assert Path("data/cards/cards.marine_world.json").exists()

    payload = load_combined_cards_payload()
    cards = payload["cards"]
    stats = payload["stats"]

    assert isinstance(cards, list)
    assert stats["total"] == len(cards)
    assert len(cards) >= 250

    card_ids = [card["data_id"] for card in cards]
    assert len(card_ids) == len(set(card_ids))

    by_type = stats["by_type"]
    assert by_type.get("animal", 0) > 0
    assert by_type.get("sponsor", 0) > 0
    assert by_type.get("conservation_project", 0) > 0
    assert by_type.get("final_scoring", 0) > 0

    known_ids = {card["data_id"] for card in cards}
    assert "A401_Cheetah" in known_ids
    assert "S201_ScienceLab" in known_ids
    assert "P101_SpeciesDiversity" in known_ids
    assert "F001_LargeAnimalZoo" in known_ids


def test_animal_prerequisites_are_parsed_as_required_icons():
    payload = load_combined_cards_payload()
    cards = payload["cards"]
    card_468 = next(card for card in cards if card.get("number") == 468)

    assert card_468["required_icons"] == {"science": 2}
    assert card_468["badges"] == ["Primate", "Americas"]


def test_manual_card_corrections_are_present_in_combined_and_split_datasets():
    combined_cards = load_combined_cards_payload()["cards"]
    base_cards = json.loads(Path("data/cards/cards.base.json").read_text(encoding="utf-8"))["cards"]
    marine_cards = json.loads(Path("data/cards/cards.marine_world.json").read_text(encoding="utf-8"))["cards"]

    combined_223 = next(card for card in combined_cards if card.get("number") == 223)
    base_223 = next(card for card in base_cards if card.get("number") == 223)
    combined_277 = next(card for card in combined_cards if card.get("number") == 277)
    marine_277 = next(card for card in marine_cards if card.get("number") == 277)

    assert combined_223["badges"] == ["Science", "Science"]
    assert base_223["badges"] == ["Science", "Science"]

    expected_277 = {
        "level": 6,
        "reputation": 3,
        "required_icons": {"science": 2, "seaanimal": 1},
        "badges": ["Science"],
    }
    for card in (combined_277, marine_277):
        for key, value in expected_277.items():
            assert card[key] == value


def test_sponsorship_cards_231_to_235_have_matching_icon_prerequisites_in_dataset():
    cards = json.loads(Path("data/cards/cards.base.json").read_text(encoding="utf-8"))["cards"]
    expected = {
        231: {"primate": 1},
        232: {"reptile": 1},
        233: {"bird": 1},
        234: {"predator": 1},
        235: {"herbivore": 1},
    }

    for number, required_icons in expected.items():
        card = next(card for card in cards if card.get("number") == number)
        assert card["required_icons"] == required_icons


def test_science_sponsors_203_204_206_207_208_209_210_214_215_216_218_220_221_227_228_229_236_241_242_245_246_249_251_253_255_and_256_include_their_effect_texts_and_constraints():
    cards = json.loads(Path("data/cards/cards.base.json").read_text(encoding="utf-8"))["cards"]
    card_203 = next(card for card in cards if card.get("number") == 203)
    card_204 = next(card for card in cards if card.get("number") == 204)
    card_206 = next(card for card in cards if card.get("number") == 206)
    card_207 = next(card for card in cards if card.get("number") == 207)
    card_208 = next(card for card in cards if card.get("number") == 208)
    card_209 = next(card for card in cards if card.get("number") == 209)
    card_210 = next(card for card in cards if card.get("number") == 210)
    card_214 = next(card for card in cards if card.get("number") == 214)
    card_215 = next(card for card in cards if card.get("number") == 215)
    card_216 = next(card for card in cards if card.get("number") == 216)
    card_218 = next(card for card in cards if card.get("number") == 218)
    card_220 = next(card for card in cards if card.get("number") == 220)
    card_221 = next(card for card in cards if card.get("number") == 221)
    card_227 = next(card for card in cards if card.get("number") == 227)
    card_228 = next(card for card in cards if card.get("number") == 228)
    card_229 = next(card for card in cards if card.get("number") == 229)
    card_236 = next(card for card in cards if card.get("number") == 236)
    card_241 = next(card for card in cards if card.get("number") == 241)
    card_242 = next(card for card in cards if card.get("number") == 242)
    card_245 = next(card for card in cards if card.get("number") == 245)
    card_246 = next(card for card in cards if card.get("number") == 246)
    card_249 = next(card for card in cards if card.get("number") == 249)
    card_251 = next(card for card in cards if card.get("number") == 251)
    card_253 = next(card for card in cards if card.get("number") == 253)
    card_255 = next(card for card in cards if card.get("number") == 255)
    card_256 = next(card for card in cards if card.get("number") == 256)

    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Money-2} / {Money-5} / {Money-10} if you have 1 / 2 / 3 universities."
        }
        for effect in card_203["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Money-2} for each research icon in your zoo."
        }
        for effect in card_204["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-2} for each conservation project you have supported so far."
        }
        for effect in card_206["effects"]
    )
    assert any(
        effect == {
            "kind": "income",
            "text": "Income: Gain {ConservationPoint-1} ."
        }
        for effect in card_206["effects"]
    )
    assert card_207["required_icons"] == {"sponsorsii": 1}
    assert card_207["max_appeal"] == 25
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {ConservationPoint-1} each for 2 different continent and/or animal category icons. For each {ConservationPoint-1} that you gain this way, all other players gain {Money-2} ."
        }
        for effect in card_207["effects"]
    )
    assert any(
        effect == {
            "kind": "passive",
            "text": "Each time research icons are played into any zoo, gain {Money-2} for each such icon."
        }
        for effect in card_208["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-1} for each research icon in your zoo."
        }
        for effect in card_208["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain 1 {XToken} -token."
        }
        for effect in card_209["effects"]
    )
    assert any(
        effect == {
            "kind": "income",
            "text": "Income: Gain 1 {XToken} -token."
        }
        for effect in card_209["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if you have at least 3 universities."
        }
        for effect in card_209["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-1} for each America icon in your zoo."
        }
        for effect in card_210["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if you have at least 5 kiosks."
        }
        for effect in card_210["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-1} for each Africa icon in your zoo."
        }
        for effect in card_214["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {Appeal-1} for each X-token you have."
        }
        for effect in card_214["effects"]
    )
    assert any(
        effect == {
            "kind": "passive",
            "text": "When playing this card, place 2 player tokens on it. When supporting a **base** conservation project, you may discard exactly 1 token as any icon."
        }
        for effect in card_215["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if you have at least 9 reputation."
        }
        for effect in card_216["effects"]
    )
    assert any(
        effect == {
            "kind": "passive",
            "text": "When playing this card, place 2 player tokens on it. When supporting a **base** conservation project, you may discard exactly 1 token as any icon."
        }
        for effect in card_218["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Money-3} ."
        }
        for effect in card_220["effects"]
    )
    assert any(
        effect == {
            "kind": "income",
            "text": "Income: Gain {Money-3} ."
        }
        for effect in card_220["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if you have at least 9 reputation."
        }
        for effect in card_220["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if all border spaces in your zoo are covered."
        }
        for effect in card_221["effects"]
    )
    assert card_227["required_icons"] == {"reputation": 6}
    assert card_228["required_icons"] == {"reputation": 3}
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Money-2} for each small animal in your zoo."
        }
        for effect in card_228["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-1} for each small animal in your zoo."
        }
        for effect in card_229["effects"]
    )
    assert card_236["level"] == 4
    assert any(
        effect == {
            "kind": "passive",
            "text": "Each time a primate icon is played into **any** zoo, gain {Money-3} ."
        }
        for effect in card_236["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-1} for each water icon in your zoo."
        }
        for effect in card_241["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if all water spaces are connected."
        }
        for effect in card_241["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Gain {Appeal-3} for each pair of rock icons in your zoo."
        }
        for effect in card_242["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if all rock spaces are connected."
        }
        for effect in card_242["effects"]
    )
    assert card_245["required_icons"] == {"reputation": 3}
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if you have at least 6 water icons."
        }
        for effect in card_245["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} if you have at least 6 rock icons."
        }
        for effect in card_246["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Place this 3-space building in a straight line."
        }
        for effect in card_249["effects"]
    )
    assert card_251["badges"] == ["Predator", "Bear"]
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Place next to at least 1 water space."
        }
        for effect in card_251["effects"]
    )
    assert any(
        effect == {
            "kind": "endgame",
            "text": "Gain {ConservationPoint-1} / {ConservationPoint-2} for 3 / 6 bear icons."
        }
        for effect in card_251["effects"]
    )
    assert card_253["badges"] == ["Herbivore"]
    assert any(
        effect == {
            "kind": "passive",
            "text": "Place 3 player tokens on this card. Each time you play a herbivore icon into your zoo, you may remove exactly 1 token and play 1 sponsors card from your hand by paying its level."
        }
        for effect in card_253["effects"]
    )
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Place this 4-space building."
        }
        for effect in card_253["effects"]
    )
    assert card_255["appeal"] == 4
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Place next to at least 1 rock space."
        }
        for effect in card_255["effects"]
    )
    assert card_256["appeal"] == 4
    assert any(
        effect == {
            "kind": "immediate",
            "text": "Place next to at least 1 water space."
        }
        for effect in card_256["effects"]
    )

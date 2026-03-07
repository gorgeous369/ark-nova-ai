import json
from pathlib import Path


def test_cards_dataset_exists_and_has_expected_shape():
    dataset_path = Path("data/cards/cards.json")
    assert dataset_path.exists(), "Run tools/fetch_cards.py to generate data/cards/cards.json"

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
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
    dataset_path = Path("data/cards/cards.json")
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    cards = payload["cards"]
    card_468 = next(card for card in cards if card.get("number") == 468)

    assert card_468["required_icons"] == {"science": 2}
    assert card_468["badges"] == ["Primate", "Americas"]

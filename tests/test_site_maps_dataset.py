import json
from pathlib import Path


def test_maps_dataset_exists_and_has_expected_shape():
    dataset_path = Path("data/maps/maps.json")
    assert dataset_path.exists(), "Run tools/fetch_maps.py to generate data/maps/maps.json"

    payload = json.loads(dataset_path.read_text(encoding="utf-8"))
    maps = payload["maps"]
    stats = payload["stats"]

    assert isinstance(maps, list)
    assert stats["total"] == len(maps)
    assert len(maps) >= 25

    map_ids = [item["id"] for item in maps]
    assert len(map_ids) == len(set(map_ids))

    by_source = stats["by_card_source"]
    assert by_source.get("Base", 0) > 0
    assert by_source.get("Alternative", 0) > 0
    assert by_source.get("Promo", 0) > 0
    assert by_source.get("Beginner", 0) > 0

    known_ids = {item["id"] for item in maps}
    assert "m0" in known_ids
    assert "mA" in known_ids
    assert "m1" in known_ids
    assert "m1a" in known_ids
    assert "m14" in known_ids
    assert "mt1" in known_ids

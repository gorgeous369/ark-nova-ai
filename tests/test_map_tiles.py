from arknova_engine.map_tiles import create_map_tile_template, validate_map_tile_payload


def test_create_map_tile_template_has_full_grid():
    payload = create_map_tile_template(
        map_id="m1a",
        map_name="Observation Tower",
        image_name="plan1a",
        image_path="data/maps/images/plan1a.jpg",
        map_effects=["effect line 1"],
    )
    assert payload["map_id"] == "m1a"
    assert payload["map_name"] == "Observation Tower"
    assert len(payload["tiles"]) == 58
    assert payload["tiles"][0]["terrain"] == "plain"
    assert payload["tiles"][0]["build2_required"] is False


def test_validate_map_tile_payload_success_and_failure():
    payload = create_map_tile_template(
        map_id="m1a",
        map_name="Observation Tower",
        image_name="plan1a",
        image_path="data/maps/images/plan1a.jpg",
        map_effects=[],
    )
    assert validate_map_tile_payload(payload) == []

    broken = dict(payload)
    broken_tiles = list(payload["tiles"])
    broken_tiles.pop()
    broken["tiles"] = broken_tiles
    issues = validate_map_tile_payload(broken)
    assert issues
    assert any("tile count is" in issue for issue in issues)

    broken2 = create_map_tile_template(
        map_id="m1a",
        map_name="Observation Tower",
        image_name="plan1a",
        image_path="data/maps/images/plan1a.jpg",
        map_effects=[],
    )
    broken2["tiles"][0]["terrain"] = "lava"
    broken2["tiles"][0]["build2_required"] = "yes"
    broken2["tiles"][0]["placement_bonus"] = "unknown_bonus"
    issues2 = validate_map_tile_payload(broken2)
    assert any("terrain" in issue for issue in issues2)
    assert any("build2_required must be boolean" in issue for issue in issues2)
    assert any("placement_bonus" in issue for issue in issues2)

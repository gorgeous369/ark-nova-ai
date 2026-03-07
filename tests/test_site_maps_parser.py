from arknova_engine.site_maps import (
    parse_map_entries_from_chunk,
    parse_maps_translations_from_page,
    resolve_maps_translation,
)


def test_parse_map_entries_from_chunk():
    chunk = """
    ,7120:function(e,t,a){"use strict";a.d(t,{i:function(){return n}});var r=a(4391);let n=[
    {id:"m1",name:"maps.name1",image:"plan1",cardSource:r.t.BASE,description:["maps.desc1-1","maps.desc1-2"]},
    {id:"m1a",name:"maps.name1",image:"plan1a",cardSource:r.t.ALTERNATIVE,description:["maps.desc1-1"]},
    {id:"m9",name:"maps.name9",image:"plan9",cardSource:r.t.PROMO,description:["maps.desc9-1","maps.desc9-2"]},
    {id:"m0",name:"maps.name0",image:"plan0",cardSource:r.t.BEGINNER,description:[]}
    ]}
    """
    maps = parse_map_entries_from_chunk(chunk)
    assert [item["id"] for item in maps] == ["m1", "m1a", "m9", "m0"]

    assert maps[0]["card_source"] == "Base"
    assert maps[1]["card_source"] == "Alternative"
    assert maps[2]["card_source"] == "Promo"
    assert maps[3]["card_source"] == "Beginner"

    assert maps[0]["image_url"].endswith("/img/maps/plan1.jpg")
    assert maps[1]["description_keys"] == ["maps.desc1-1"]
    assert maps[3]["description_keys"] == []


def test_parse_maps_translations_from_page():
    page_html = """
    <html><body>
      <script id="__NEXT_DATA__" type="application/json">
        {"props":{"pageProps":{"_nextI18Next":{"initialI18nStore":{"en":{"common":{"maps":{
          "name1":"Observation Tower",
          "name0":"Map 0",
          "desc1-1":"Gain 2 Appeal",
          "descA-1":"Starter map bonus"
        }}}}}}}}
      </script>
    </body></html>
    """
    translations = parse_maps_translations_from_page(page_html)
    assert translations["name1"] == "Observation Tower"
    assert translations["name0"] == "Map 0"
    assert translations["desc1-1"] == "Gain 2 Appeal"
    assert translations["descA-1"] == "Starter map bonus"


def test_resolve_maps_translation_with_prefixed_key_and_alias():
    translations = {
        "name1": "Observation Tower",
        "desc1-2": "Fallback text",
        "desc3-2": "maps.desc1-2",
    }
    assert resolve_maps_translation(translations, "maps.name1") == "Observation Tower"
    assert resolve_maps_translation(translations, "maps.desc3-2") == "Fallback text"
    assert resolve_maps_translation(translations, "maps.unknown-key") == "maps.unknown-key"

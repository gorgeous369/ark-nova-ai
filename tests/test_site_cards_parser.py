from arknova_engine.site_cards import (
    _parse_sponsor_chunk_records,
    parse_best_card_from_page,
    parse_card_block,
    parse_first_card_from_page,
)


def test_parse_card_block_animal_fields():
    block = """
    <div id="card-A401_Cheetah" data-id="A401_Cheetah" class="ark-card zoo-card animal-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-enclosure-regular">5</div>
          <div class="arknova-icon icon-money original-cost">17</div>
          <div class="zoo-card-prerequisites">
            <div class="zoo-card-badge"><div class="badge-icon" data-type="Science"></div></div>
            <div class="zoo-card-badge"><div class="badge-icon" data-type="Science"></div></div>
          </div>
        </div>
        <div class="ark-card-top-right">
          <div class="zoo-card-badge"><div class="badge-icon" data-type="Predator"></div></div>
          <div class="zoo-card-badge"><div class="badge-icon" data-type="Africa"></div></div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number">401</div>
          <div class="ark-card-title-wrapper">
            <div class="ark-card-title pt-1">CHEETAH</div>
            <div class="ark-card-subtitle">Acinonyx jubatus</div>
          </div>
        </div>
        <div class="zoo-card-bonuses">
          <div class="zoo-card-bonus reputation">1</div>
          <div class="zoo-card-bonus conservation">2</div>
          <div class="zoo-card-bonus appeal">6</div>
        </div>
        <h6 class="animal-ability-title">Sprint 3</h6>
        <div class="animal-ability-desc">Draw<!-- --> 3 card(s) from the deck.</div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="A401_Cheetah",
        card_class="ark-card zoo-card animal-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/401",
    )
    assert card["type"] == "animal"
    assert card["number"] == 401
    assert card["title"] == "CHEETAH"
    assert card["subtitle"] == "Acinonyx jubatus"
    assert card["cost"] == 17
    assert card["enclosure_size"] == 5
    assert card["required_icons"] == {"science": 2}
    assert card["reputation"] == 1
    assert card["conservation"] == 2
    assert card["appeal"] == 6
    assert card["badges"] == ["Predator", "Africa"]
    assert card["ability_title"] == "Sprint 3"
    assert card["ability_text"] == "Draw 3 card(s) from the deck."


def test_parse_card_block_animal_rock_adjacency_requirement():
    block = """
    <div id="card-A451_TestRock" data-id="A451_TestRock" class="ark-card zoo-card animal-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-enclosure-regular-rock-rock">4</div>
          <div class="arknova-icon icon-money original-cost">11</div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number">451</div>
          <div class="ark-card-title-wrapper">
            <div class="ark-card-title pt-1">TEST ROCK</div>
          </div>
        </div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="A451_TestRock",
        card_class="ark-card zoo-card animal-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/451",
    )
    assert card["type"] == "animal"
    assert card["enclosure_size"] == 4
    assert card["required_rock_adjacency"] == 2
    assert "required_water_adjacency" not in card


def test_parse_card_block_animal_mixed_adjacency_requirement():
    block = """
    <div id="card-A452_TestMixed" data-id="A452_TestMixed" class="ark-card zoo-card animal-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-enclosure-regular-rock-water">3</div>
          <div class="arknova-icon icon-money original-cost">9</div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number">452</div>
          <div class="ark-card-title-wrapper">
            <div class="ark-card-title pt-1">TEST MIXED</div>
          </div>
        </div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="A452_TestMixed",
        card_class="ark-card zoo-card animal-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/452",
    )
    assert card["type"] == "animal"
    assert card["enclosure_size"] == 3
    assert card["required_rock_adjacency"] == 1
    assert card["required_water_adjacency"] == 1


def test_parse_card_block_animal_special_enclosure_sizes():
    block = """
    <div id="card-A484_EuropeanPondTurtle" data-id="A484_EuropeanPondTurtle" class="ark-card zoo-card animal-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-enclosure-regular-water">2</div>
          <div class="arknova-icon icon-enclosure-special-reptile-house">1</div>
          <div class="arknova-icon icon-enclosure-special-large-bird-aviary">1</div>
          <div class="arknova-icon icon-money original-cost">8</div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number">484</div>
          <div class="ark-card-title-wrapper">
            <div class="ark-card-title pt-1">EUROPEAN POND TURTLE</div>
          </div>
        </div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="A484_EuropeanPondTurtle",
        card_class="ark-card zoo-card animal-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/484",
    )
    assert card["type"] == "animal"
    assert card["enclosure_size"] == 2
    assert card["required_water_adjacency"] == 1
    assert card["reptile_house_size"] == 1
    assert card["large_bird_aviary_size"] == 1


def test_parse_card_block_animal_special_enclosure_size_defaults_to_zero_when_icon_has_no_number():
    block = """
    <div id="card-A485_CommonEuropeanAdder" data-id="A485_CommonEuropeanAdder" class="ark-card zoo-card animal-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-enclosure-regular">1</div>
          <div class="arknova-icon icon-enclosure-special-reptile-house"></div>
          <div class="arknova-icon icon-money original-cost">10</div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number">485</div>
          <div class="ark-card-title-wrapper">
            <div class="ark-card-title pt-1">COMMON EUROPEAN ADDER</div>
          </div>
        </div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="A485_CommonEuropeanAdder",
        card_class="ark-card zoo-card animal-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/485",
    )
    assert card["reptile_house_size"] == 0


def test_parse_card_block_sponsor_effects():
    block = """
    <div id="card-S201_ScienceLab" data-id="S201_ScienceLab" class="ark-card zoo-card sponsor-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-level">1</div>
          <div class="arknova-icon icon-money original-cost">8</div>
          <div class="zoo-card-prerequisites">
            <div class="zoo-card-badge"><div class="badge-icon" data-type="Science"></div></div>
          </div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number sf-hidden">201</div>
          <div class="ark-card-title-wrapper"><div class="ark-card-title">SCIENCE LAB</div></div>
        </div>
        <div class="zoo-card-bonuses">
          <div class="zoo-card-bonus reputation">1</div>
          <div class="zoo-card-bonus appeal">2</div>
        </div>
        <h6 class="sponsor-ability-title">Lab Partner</h6>
        <div class="sponsor-ability-desc">Gain 1 card.</div>
        <div class="ark-card-bottom text-start">
          <ul class="sponsor-effects-list">
            <li class="effect-income">Take 1 card from the deck or in reputation range.</li>
          </ul>
          <ul class="sponsor-effects-list">
            <li class="effect-endgame">Gain 1 / 2 for 3 / 6 research icons.</li>
          </ul>
        </div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="S201_ScienceLab",
        card_class="ark-card zoo-card sponsor-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/201",
    )
    assert card["type"] == "sponsor"
    assert card["number"] == 201
    assert card["title"] == "SCIENCE LAB"
    assert card["level"] == 1
    assert card["cost"] == 8
    assert card["required_icons"] == {"science": 1}
    assert card["appeal"] == 2
    assert card["reputation"] == 1
    assert card["ability_title"] == "Lab Partner"
    assert card["ability_text"] == "Gain 1 card."
    assert card["effects"] == [
        {"kind": "income", "text": "Take 1 card from the deck or in reputation range."},
        {"kind": "endgame", "text": "Gain 1 / 2 for 3 / 6 research icons."},
    ]


def test_parse_first_card_from_page_extracts_balanced_div_block():
    page_html = """
    <html><body>
      <div id="card-P101_SpeciesDiversity" data-id="P101_SpeciesDiversity" class="ark-card zoo-card project-card tooltipable">
        <div class="ark-card-wrapper">
          <div class="ark-card-middle">
            <div class="ark-card-number sf-hidden">101</div>
            <div class="ark-card-title-wrapper"><div class="ark-card-title">SPECIES DIVERSITY</div></div>
          </div>
          <div class="ark-card-bottom">
            <div class="project-card-description sf-hidden">Requires different animal category icons in your zoo.</div>
          </div>
        </div>
      </div>
      <div>tail</div>
    </body></html>
    """
    card = parse_first_card_from_page(page_html, source_url="https://example.test/cards")
    assert card is not None
    assert card["data_id"] == "P101_SpeciesDiversity"
    assert card["type"] == "conservation_project"
    assert card["number"] == 101
    assert card["title"] == "SPECIES DIVERSITY"
    assert card["description"] == "Requires different animal category icons in your zoo."


def test_parse_best_card_from_page_prefers_matching_id_and_richer_fields():
    page_html = """
    <html><body>
      <div id="card-S206_MedicalBreakthrough" data-id="S206_MedicalBreakthrough" class="ark-card zoo-card sponsor-card tooltipable">
        <div class="ark-card-wrapper">
          <div class="ark-card-middle">
            <div class="ark-card-number">206</div>
            <div class="ark-card-title-wrapper"><div class="ark-card-title">MEDICAL BREAKTHROUGH</div></div>
          </div>
        </div>
      </div>
      <div id="card-S206_MedicalBreakthrough" data-id="S206_MedicalBreakthrough" class="ark-card zoo-card sponsor-card tooltipable">
        <div class="ark-card-wrapper">
          <div class="ark-card-top-left">
            <div class="arknova-icon icon-level">5</div>
            <div class="arknova-icon icon-money">20</div>
          </div>
          <div class="ark-card-middle">
            <div class="ark-card-number">206</div>
            <div class="ark-card-title-wrapper"><div class="ark-card-title">MEDICAL BREAKTHROUGH</div></div>
          </div>
          <div class="ark-card-bottom text-start">
            <ul class="sponsor-effects-list">
              <li class="effect-income">Gain 1 conservation point.</li>
            </ul>
          </div>
        </div>
      </div>
    </body></html>
    """

    card = parse_best_card_from_page(
        page_html,
        source_url="https://example.test/card/206",
        expected_card_id=206,
    )
    assert card is not None
    assert card["number"] == 206
    assert card["level"] == 5
    assert card["cost"] == 20
    assert card["effects"] == [{"kind": "income", "text": "Gain 1 conservation point."}]


def test_parse_card_block_sponsor_prerequisites_from_arknova_icons():
    block = """
    <div id="card-S206_MedicalBreakthrough" data-id="S206_MedicalBreakthrough" class="ark-card zoo-card sponsor-card tooltipable">
      <div class="ark-card-wrapper">
        <div class="ark-card-top-left">
          <div class="arknova-icon icon-level">5</div>
          <div class="arknova-icon icon-money">20</div>
          <div class="zoo-card-prerequisites">
            <div class="arknova-icon icon-science">4</div>
            <div class="arknova-icon icon-sponsors">5</div>
          </div>
        </div>
        <div class="ark-card-middle">
          <div class="ark-card-number">206</div>
          <div class="ark-card-title-wrapper"><div class="ark-card-title">MEDICAL BREAKTHROUGH</div></div>
        </div>
      </div>
    </div>
    """
    card = parse_card_block(
        data_id="S206_MedicalBreakthrough",
        card_class="ark-card zoo-card sponsor-card tooltipable",
        block_html=block,
        source_url="https://example.test/card/206",
    )
    assert card["type"] == "sponsor"
    assert card["level"] == 5
    assert card["cost"] == 20
    assert card["required_icons"] == {"science": 4, "sponsors": 5}


def test_parse_sponsor_chunk_records_extracts_requirements_and_effect_texts():
    chunk = """
    {id:"206",name:"MEDICAL BREAKTHROUGH",strength:5,rock:0,water:0,
    requirements:[c.Qq.Science,c.Qq.Science,c.Qq.Science,c.Qq.Science],
    tags:[c.Qq.Science],
    effects:[{effectType:o.A.INCOME,effectDesc:"sponsors.s206_desc1",fontSize:"lg"}],
    reputation:0,appeal:0,conservationPoint:0,source:r.t.BASE}
    """
    texts = {"s206_desc1": "Income: Gain {ConservationPoint-1} ."}
    records = _parse_sponsor_chunk_records(chunk, texts)
    assert 206 in records
    card = records[206]
    assert card["number"] == 206
    assert card["title"] == "MEDICAL BREAKTHROUGH"
    assert card["level"] == 5
    assert card["required_icons"] == {"science": 4}
    assert card["badges"] == ["Science"]
    assert card["effects"] == [{"kind": "income", "text": "Income: Gain {ConservationPoint-1} ."}]

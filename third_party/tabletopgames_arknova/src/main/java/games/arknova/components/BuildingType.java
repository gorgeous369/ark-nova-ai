package games.arknova.components;

import java.util.Arrays;

/** An enum of all possible `BuildingType` with the corresponding layout. */
public enum BuildingType {
  SIZE_1(
      BuildingSubType.ENCLOSURE_BASIC,
      new HexTile[] {new HexTile(0, 0)},
      "size_1_full.png",
      "size_1_empty.png"),
  SIZE_2(
      BuildingSubType.ENCLOSURE_BASIC,
      new HexTile[] {new HexTile(0, 0), new HexTile(0, -1)},
      "size_2_full.png",
      "size_2_empty.png"),
  SIZE_3(
      BuildingSubType.ENCLOSURE_BASIC,
      new HexTile[] {new HexTile(0, 0), new HexTile(0, -1), new HexTile(1, -1)},
      "size_3_full.png",
      "size_3_empty.png"),
  SIZE_4(
      BuildingSubType.ENCLOSURE_BASIC,
      new HexTile[] {new HexTile(0, 0), new HexTile(0, -1), new HexTile(1, -2), new HexTile(1, -1)},
      "size_4_full.png",
      "size_4_empty.png"),
  SIZE_5(
      BuildingSubType.ENCLOSURE_BASIC,
      new HexTile[] {
        new HexTile(0, 0),
        new HexTile(0, -1),
        new HexTile(1, -2),
        new HexTile(1, -1),
        new HexTile(0, -2)
      },
      "size_5_full.png",
      "size_5_empty.png"),

  PETTING_ZOO(
      BuildingSubType.ENCLOSURE_SPECIAL,
      new HexTile[] {new HexTile(0, 0), new HexTile(0, -1), new HexTile(1, -2)},
      "petting_zoo.png",
      3),

  REPTILE_HOUSE(
      BuildingSubType.ENCLOSURE_SPECIAL,
      new HexTile[] {
        new HexTile(0, 0),
        new HexTile(0, -1),
        new HexTile(1, -1),
        new HexTile(2, -2),
        new HexTile(2, -1)
      },
      "reptile_house.png",
      5),

  LARGE_BIRD_AVIARY(
      BuildingSubType.ENCLOSURE_SPECIAL,
      new HexTile[] {
        new HexTile(0, 0),
        new HexTile(0, -1),
        new HexTile(1, -2),
        new HexTile(1, -1),
        new HexTile(2, -1)
      },
      "large_bird_aviary.png",
      5),

  PAVILION(BuildingSubType.PAVILION, new HexTile[] {new HexTile(0, 0)}, "pavilion.png"),

  KIOSK(BuildingSubType.KIOSK, new HexTile[] {new HexTile(0, 0)}, "kiosk.png");

  BuildingSubType subType;
  HexTile[] layout;
  boolean isSpecial;

  // Frontal image of the building (full enclosure)
  String frontImagePath;

  // Back image of the building (empty enclosure)
  String backImagePath;
  int maxCapacity;

  BuildingType(
      BuildingSubType subType, HexTile[] layout, String frontImagePath, String backImagePath) {
    if (subType != BuildingSubType.ENCLOSURE_BASIC) {
      throw new IllegalArgumentException("This constructor is reserved for basic enclosures.");
    }
    this.subType = subType;
    this.layout = layout;
    this.frontImagePath = frontImagePath;
    this.backImagePath = backImagePath;
    this.maxCapacity = 1;
  }

  BuildingType(BuildingSubType subType, HexTile[] layout, String frontImagePath, int maxCapacity) {
    if (subType != BuildingSubType.ENCLOSURE_SPECIAL) {
      throw new IllegalArgumentException("This constructor is reserved for special enclosures.");
    }
    this.subType = subType;
    this.layout = layout;
    this.frontImagePath = frontImagePath;
    this.backImagePath = "";
    this.maxCapacity = maxCapacity;
  }

  BuildingType(BuildingSubType subType, HexTile[] layout, String frontImagePath) {
    if (!Arrays.asList(BuildingSubType.KIOSK, BuildingSubType.PAVILION, BuildingSubType.UNIQUE)
        .contains(subType)) {
      throw new IllegalArgumentException(
          "This constructor is reserved for non-habitable enclosures.");
    }
    this.subType = subType;
    this.layout = layout;
    this.frontImagePath = frontImagePath;
    this.backImagePath = "";
    this.maxCapacity = -1;
  }

  public HexTile[] getLayout() {
    return layout;
  }

  public boolean isSpecial() {
    return isSpecial;
  }

  public String getBackImagePath() {
    return "data/arknova/buildings/" + backImagePath;
  }

  public String getFrontImagePath() {
    return "data/arknova/buildings/" + frontImagePath;
  }

  public enum BuildingSubType {
    ENCLOSURE_BASIC,
    ENCLOSURE_SPECIAL,
    UNIQUE,
    KIOSK,
    PAVILION,
    SPONSOR_BUILDING,
  }
}

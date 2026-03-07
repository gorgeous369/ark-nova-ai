package games.arknova.components;

import static org.testng.Assert.*;

import java.util.*;
import java.util.stream.Collectors;
import org.testng.annotations.Test;

public class ArkNovaMapTest {

  @Test
  public void testGetBorderTilesWithoutIgnoringTerrain() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);

    HashSet<HexTile> calculatedBorderTiles = map.getBorderTiles(false);

    HexTile[] groundTruthBorderTiles =
        new HexTile[] {
          new HexTile(1, -1),
          new HexTile(2, -1),
          new HexTile(3, -2),
          new HexTile(5, -3),
          new HexTile(6, -3),
          new HexTile(0, 2),
          new HexTile(0, 3),
          new HexTile(0, 4),
          new HexTile(0, 5),
          new HexTile(8, -1),
          new HexTile(8, 0),
          new HexTile(1, 5),
          new HexTile(2, 4),
          new HexTile(3, 4),
          new HexTile(4, 3),
          new HexTile(5, 3),
          new HexTile(6, 2),
          new HexTile(7, 2),
        };

    assert (groundTruthBorderTiles.length == calculatedBorderTiles.size());

    for (HexTile groundTruthTile : groundTruthBorderTiles) {
      assert (calculatedBorderTiles.contains(groundTruthTile));
    }

    assertFalse(calculatedBorderTiles.contains(new HexTile(0, 0)));
  }

  @Test
  public void testGetBorderTilesIgnoringTerrain() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);

    HashSet<HexTile> calculatedBorderTiles = map.getBorderTiles(true);

    HexTile[] groundTruthBorderTiles =
        new HexTile[] {
          new HexTile(0, 0),
          new HexTile(0, 1),
          new HexTile(8, 1),
          new HexTile(8, -2),
          new HexTile(8, -3),
          new HexTile(8, -4),
          new HexTile(7, -4),
          new HexTile(4, -2),
          new HexTile(1, -1),
          new HexTile(2, -1),
          new HexTile(3, -2),
          new HexTile(5, -3),
          new HexTile(6, -3),
          new HexTile(0, 2),
          new HexTile(0, 3),
          new HexTile(0, 4),
          new HexTile(0, 5),
          new HexTile(8, -1),
          new HexTile(8, 0),
          new HexTile(1, 5),
          new HexTile(2, 4),
          new HexTile(3, 4),
          new HexTile(4, 3),
          new HexTile(5, 3),
          new HexTile(6, 2),
          new HexTile(7, 2),
        };

    assert (groundTruthBorderTiles.length == calculatedBorderTiles.size());

    for (HexTile groundTruthTile : groundTruthBorderTiles) {
      assert (calculatedBorderTiles.contains(groundTruthTile));
    }
  }

  @Test
  public void testGetLegalBuildingsPlacementsPettingZooOnlyOnce() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);
    map.addBuilding(
        new Building(BuildingType.PETTING_ZOO, new HexTile(6, 1), Building.Rotation.ROT_180));

    ArrayList<Building> legalBuildingsPlacements =
        map.getLegalBuildingsPlacements(false, false, 100, new HashSet<>());

    assert (legalBuildingsPlacements.stream()
        .filter(building -> building.getType() == BuildingType.PETTING_ZOO)
        .collect(Collectors.toSet())
        .isEmpty());
  }

  @Test
  public void testGetLegalBuildingsPlacementAviaryOnlyOnce() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);
    map.addBuilding(
        new Building(BuildingType.LARGE_BIRD_AVIARY, new HexTile(6, 1), Building.Rotation.ROT_180));

    ArrayList<Building> legalBuildingsPlacements =
        map.getLegalBuildingsPlacements(true, true, 100, new HashSet<>());

    assert (legalBuildingsPlacements.stream()
        .filter(building -> building.getType() == BuildingType.LARGE_BIRD_AVIARY)
        .collect(Collectors.toSet())
        .isEmpty());
  }

  @Test
  public void testGetLegalBuildingsPlacementsSize1AlreadyPlaced() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);

    HashSet<BuildingType> alreadyPlacedBuildings =
        new HashSet<>(Arrays.asList(BuildingType.SIZE_1));

    ArrayList<Building> legalBuildingsPlacements =
        map.getLegalBuildingsPlacements(false, false, 100, alreadyPlacedBuildings);

    assert (legalBuildingsPlacements.stream()
        .filter(building -> building.getType() == BuildingType.SIZE_1)
        .collect(Collectors.toSet())
        .isEmpty());
  }

  @Test
  public void testGetLegalBuildingsPlacementKioskTooClose() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);
    map.addBuilding(new Building(BuildingType.KIOSK, new HexTile(7, 2), Building.Rotation.ROT_0));

    // We can't add a kiosk to a single kiosk on board
    assert (map.getLegalBuildingsPlacements(true, true, 100, new HashSet<>()).stream()
        .filter(building -> building.getType() == BuildingType.KIOSK)
        .collect(Collectors.toSet())
        .isEmpty());

    // We add another pavilion to the kiosk -> but we are still too close for kiosk
    map.addBuilding(
        new Building(BuildingType.PAVILION, new HexTile(6, 2), Building.Rotation.ROT_0));
    assert (map.getLegalBuildingsPlacements(true, true, 100, new HashSet<>()).stream()
        .filter(building -> building.getType() == BuildingType.KIOSK)
        .collect(Collectors.toSet())
        .isEmpty());

    // After we add another kiosk, we can add a single kiosk on (4, 3)
    map.addBuilding(
        new Building(BuildingType.PAVILION, new HexTile(5, 2), Building.Rotation.ROT_0));

    List<HexTile> kioskLocations =
        map.getLegalBuildingsPlacements(true, false, 100, new HashSet<>()).stream()
            .filter(building -> building.getType() == BuildingType.KIOSK)
            .map(Building::getOriginHex)
            .collect(Collectors.toList());

    assertEquals(kioskLocations.size(), 1);
    assert (kioskLocations.contains(new HexTile(4, 3)));

    // If we have a Diversity researcher, we have 3 possible choices
    kioskLocations =
        map.getLegalBuildingsPlacements(true, true, 100, new HashSet<>()).stream()
            .filter(building -> building.getType() == BuildingType.KIOSK)
            .map(Building::getOriginHex)
            .collect(Collectors.toList());

    assertEquals(kioskLocations.size(), 3);
    assert (kioskLocations.containsAll(
        Arrays.asList(new HexTile(4, 3), new HexTile(4, 2), new HexTile(5, 1))));
  }

  @Test
  public void testGetLegalBuildingsPlacements() {
    // TODO: check whether everything is correctly generated
  }

  @Test
  public void testCanBuildOnHex() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);

    // HexTile (0, 0) is a rock tile and cannot be occupied
    HexTile hexTile = new HexTile(0, 0);

    assertFalse(map.canBuildOnHex(hexTile, new HashSet<>(), false, false));

    // But we can built on it with the Diversity Researcher
    assert (map.canBuildOnHex(hexTile, new HashSet<>(), false, true));

    // However if we add a building on it, it cannot be occupied anymore
    map.addBuilding(new Building(BuildingType.SIZE_1, hexTile, Building.Rotation.ROT_0));
    assertFalse(map.canBuildOnHex(hexTile, map.getCoveredHexes(), false, true));

    // HexTile (3, 4) requires an upgraded build 2 card
    HexTile hexTileBuild2 = new HexTile(3, 4);
    assertFalse(map.canBuildOnHex(hexTileBuild2, map.getCoveredHexes(), false, true));
    assert (map.canBuildOnHex(hexTileBuild2, map.getCoveredHexes(), true, true));

    // However if we add a building on it, it cannot be occupied anymore
    map.addBuilding(new Building(BuildingType.SIZE_1, hexTileBuild2, Building.Rotation.ROT_0));
    assertFalse(map.canBuildOnHex(hexTileBuild2, map.getCoveredHexes(), true, true));
  }

  @Test
  public void testGetPossibleStartingBuildingHexesWithNoBuildings() {
    ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);

    HashSet<HexTile> possibleStartingBuildingHexes =
        map.getPossibleStartingBuildingHexes(false, false, new HashSet<>());

    // It should return only border tiles
    assert (possibleStartingBuildingHexes.containsAll(map.getBorderTiles(false)));
  }
}

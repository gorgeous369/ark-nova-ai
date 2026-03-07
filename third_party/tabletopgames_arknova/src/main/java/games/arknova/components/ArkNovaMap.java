package games.arknova.components;

import core.CoreConstants;
import core.components.Component;
import games.arknova.ArkNovaConstants;
import games.arknova.actions.Bonus;
import games.arknova.actions.GainBonus;
import java.awt.*;
import java.util.*;
import java.util.AbstractMap.SimpleEntry;
import java.util.stream.Collectors;
import utilities.Vector2D;

/**
 * An abstraction for Ark Nova maps, containing the grid, helper methods for placing buildings,
 * calculating border spaces and other map related things.
 */
public class ArkNovaMap extends Component {
  static final int WIDTH = 8;
  static final int HEIGHT = 5;
  // Mapping between axial coords and the underlying hex
  protected HashMap<HexTile, HexTile> grid;

  protected HashMap<HexTile, Building> buildings;

  // TODO: change this to array list, as a tile can have more than one terrain (build + bonus)
  protected HashMap<HexTile, Terrain> terrain;
  protected MapData mapData;

  public ArkNovaMap(MapData mapData) {
    super(CoreConstants.ComponentType.BOARD, "Map");
    this.mapData = mapData;

    grid = new HashMap<>();
    buildings = new HashMap<>();
    terrain = mapData.getTerrain();
    for (int q = 0; q <= WIDTH; q++) {
      int qOffset = (int) Math.floor((q + 1) / 2.0);
      for (int r = -qOffset; r <= HEIGHT - qOffset; r++) {
        grid.put(new HexTile(q, r), new HexTile(q, r));
      }

      if (q % 2 == 1) {
        int newR = HEIGHT - qOffset + 1;
        grid.put(new HexTile(q, newR), new HexTile(q, newR));
      }
    }
  }

  public HexTile pixelToHex(Point pixel, int hexSize) {
    double q = (2. / 3 * pixel.x) / hexSize;
    double r = (-1. / 3 * pixel.x + Math.sqrt(3) / 3 * pixel.y) / hexSize;

    HexTile roundedHex = axialHexRound(q, r);
    return grid.getOrDefault(roundedHex, null);
  }

  public void addBuilding(Building building) {
    this.buildings.put(building.getOriginHex(), building);
  }

  public HexTile axialHexRound(double q, double r) {
    double s = -q - r;
    int qi = (int) (Math.round(q));
    int ri = (int) (Math.round(r));
    int si = (int) (Math.round(s));

    double q_diff = Math.abs(qi - q);
    double r_diff = Math.abs(ri - r);
    double s_diff = Math.abs(si - s);

    if (q_diff > r_diff && q_diff > s_diff) {
      qi = -ri - si;
    } else if (r_diff > s_diff) {
      ri = -qi - si;
    } else {
      si = -qi - ri;
    }
    return new HexTile(qi, ri);
  }

  public HashMap<HexTile, HexTile> getGrid() {
    return grid;
  }

  public HashMap<HexTile, Building> getBuildings() {
    return buildings;
  }

  public MapData getMapData() {
    return mapData;
  }

  @Override
  public ArkNovaMap copy() {
    ArkNovaMap copy = new ArkNovaMap(this.mapData);

    for (HexTile tile : this.buildings.keySet()) {
      copy.buildings.put(tile, this.buildings.get(tile));
    }
    return copy;
  }

  public HashSet<HexTile> getCoveredHexes() {
    HashSet<HexTile> hexes = new HashSet<>();

    for (HexTile tile : buildings.keySet()) {
      hexes.addAll(buildings.get(tile).getLayout());
    }

    return hexes;
  }

  /**
   * Return border hex tiles.
   *
   * @param ignoreTerrain Whether to ignore ROCK, WATER icons (e.g. Diversity Researcher)
   * @return Border hex tiles set.
   */
  public HashSet<HexTile> getBorderTiles(boolean ignoreTerrain) {
    HashSet<HexTile> tiles = new HashSet<>();
    for (HexTile tile : grid.keySet()) {
      Terrain tileTerrain = terrain.get(tile);
      if (!ignoreTerrain && (tileTerrain == Terrain.WATER || tileTerrain == Terrain.ROCK)) {
        continue;
      }
      Vector2D doubled = tile.getDoubledCoordinates();

      if (doubled.getX() == 0
          || doubled.getX() == WIDTH
          || doubled.getY() == -1
          || doubled.getY() == 0
          || doubled.getY() == 2 * HEIGHT
          || doubled.getY() == 2 * HEIGHT + 1) {
        tiles.add(tile);
      }
    }

    return tiles;
  }

  /**
   * Get all legal building placements.
   *
   * @param isBuildUpgraded whether Build action is upgraded (flipped)
   * @param hasDiversityResearcher has the player played Diversity Researcher (allows to build
   *     anywhere)
   * @return all possible building placements with all possible rotations
   */
  public ArrayList<Building> getLegalBuildingsPlacements(
      boolean isBuildUpgraded,
      boolean hasDiversityResearcher,
      int maxBuildingSize,
      HashSet<BuildingType> alreadyBuiltBuildings) {
    ArrayList<Building> placements = new ArrayList<>();

    HashSet<HexTile> coveredHexes = getCoveredHexes();

    // Find all possible hexes from where we can build buildings
    // If there is not building on the map yet, we have to start from the border tiles
    HashSet<HexTile> possibleStartingHexes =
        getPossibleStartingBuildingHexes(isBuildUpgraded, hasDiversityResearcher, coveredHexes);

    Set<BuildingType> existingSpecialBuildings =
        buildings.values().stream()
            .map(Building::getType)
            .filter(
                buildingType ->
                    buildingType.subType == BuildingType.BuildingSubType.ENCLOSURE_SPECIAL)
            .collect(Collectors.toSet());

    // Define which buildings can be built -> special enclosures can only be built once
    HashSet<BuildingType> buildingTypes =
        Arrays.stream(BuildingType.values())
            .filter(
                buildingType ->
                    buildingType.getLayout().length <= maxBuildingSize
                        && !alreadyBuiltBuildings.contains(buildingType)
                        && buildingType.subType != BuildingType.BuildingSubType.SPONSOR_BUILDING
                        && !(buildingType.subType == BuildingType.BuildingSubType.ENCLOSURE_SPECIAL
                            && existingSpecialBuildings.contains(buildingType)))
            .collect(Collectors.toCollection(HashSet::new));

    // Find existing kiosk for distance calculation
    Set<Building> existingKiosks =
        buildings.values().stream()
            .filter(building -> building.getType() == BuildingType.KIOSK)
            .collect(Collectors.toSet());

    // We cannot build aviary and reptile house if build is not upgraded
    if (!isBuildUpgraded && maxBuildingSize >= BuildingType.LARGE_BIRD_AVIARY.getLayout().length) {
      buildingTypes.remove(BuildingType.LARGE_BIRD_AVIARY);
      buildingTypes.remove(BuildingType.REPTILE_HOUSE);
    }

    // For every hex try to place the building on the hex with all possible rotations
    for (HexTile startingHex : possibleStartingHexes) {
      for (BuildingType buildingType : buildingTypes) {
        if (buildingType == BuildingType.KIOSK) {
          long numOfTooCloseKiosks =
              existingKiosks.stream()
                  .map(kiosk -> kiosk.getOriginHex().distance(startingHex))
                  .filter(distance -> distance < ArkNovaConstants.MINIMUM_KIOSK_DISTANCE)
                  .count();

          if (numOfTooCloseKiosks > 0) {
            continue;
          }
        }

        // TODO: optimize
        for (Building.Rotation rotation : Building.Rotation.values()) {
          Building building = new Building(buildingType, startingHex, rotation);

          boolean legalPlacement = true;
          for (HexTile buildingTile : building.getLayout()) {
            legalPlacement &=
                canBuildOnHex(buildingTile, coveredHexes, isBuildUpgraded, hasDiversityResearcher);
          }

          if (legalPlacement) {
            placements.add(building);
          }

          // No need to check all rotations for size 1
          if (building.getLayout().size() == 1) {
            break;
          }
        }
      }
    }

    return placements;
  }

  /**
   * Get all HexTile from where we can start a building, e.g. if a buildings exist, the hexes have
   * to be adjacent, if nothing is built yet, we have to start from the edge (border tiles).
   *
   * @param isBuildUpgraded is build upgraded (flipped)
   * @param hasDiversityResearcher has player played Diversity researcher (allows to build anywhere)
   * @param coveredHexes which hexes are already covered by buildings
   * @return a set of HexTile from where we can start a building
   */
  public HashSet<HexTile> getPossibleStartingBuildingHexes(
      boolean isBuildUpgraded, boolean hasDiversityResearcher, HashSet<HexTile> coveredHexes) {
    HashSet<HexTile> possibleStartingHexes = new HashSet<>();
    if (coveredHexes.isEmpty()) {
      possibleStartingHexes.addAll(getBorderTiles(hasDiversityResearcher));
    } else {

      for (HexTile coveredHex : coveredHexes) {
        for (HexTile neighbor : coveredHex.getNeighbors()) {
          if (canBuildOnHex(neighbor, coveredHexes, isBuildUpgraded, hasDiversityResearcher)) {
            possibleStartingHexes.add(neighbor);
          }
        }
      }
    }
    return possibleStartingHexes;
  }

  public boolean canBuildOnHex(
      HexTile tile,
      HashSet<HexTile> coveredHexes,
      boolean buildUpgraded,
      boolean hasDiversityResearcher) {
    if (grid.containsKey(tile) && !coveredHexes.contains(tile)) {
      Terrain terrain = mapData.getTerrain().get(tile);

      // Does the hex require build 2 upgraded?
      if (terrain == Terrain.BUILD_2_REQUIRED && !buildUpgraded) {
        return false;
      }

      // Otherwise check terrain is water/rock (ignore for diversity researcher
      return hasDiversityResearcher || (terrain != Terrain.ROCK && terrain != Terrain.WATER);
    }

    return false;
  }

  public enum MapData {
    Map7(
        "Ice Cream parlors",
        "data/arknova/Map7.png",
        new Object() {
          HashMap<HexTile, Terrain> evaluate() {
            HashMap<HexTile, Terrain> map = new HashMap<>();
            for (SimpleEntry<HexTile, Terrain> hexTileTerrainSimpleEntry :
                Arrays.asList(
                    new SimpleEntry<>(new HexTile(0, 0), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(0, 1), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(1, 3), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(2, 3), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(3, 0), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(4, -1), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(4, -2), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(4, 2), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(5, 1), Terrain.ROCK),
                    new SimpleEntry<>(new HexTile(7, -4), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(8, -4), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(8, -3), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(8, -2), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(7, -2), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(7, 1), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(8, 1), Terrain.WATER),
                    new SimpleEntry<>(new HexTile(3, 2), Terrain.BUILD_2_REQUIRED),
                    new SimpleEntry<>(new HexTile(3, 3), Terrain.BUILD_2_REQUIRED),
                    new SimpleEntry<>(new HexTile(3, 4), Terrain.BUILD_2_REQUIRED))) {
              map.put(hexTileTerrainSimpleEntry.getKey(), hexTileTerrainSimpleEntry.getValue());
            }
            return map;
          }
        }.evaluate(),
        new Object() {
          HashMap<HexTile, Bonus> evaluate() {
            HashMap<HexTile, Bonus> map = new HashMap<>();
            for (SimpleEntry<HexTile, GainBonus> bonusHexTile :
                Arrays.asList(
                    new SimpleEntry<>(
                        new HexTile(0, 2),
                        new GainBonus(ArkNovaConstants.Resource.REPUTATION, true, 1)))) {

              map.put(bonusHexTile.getKey(), bonusHexTile.getValue());
            }
            return map;
          }
        }.evaluate());

    String fullName;

    String mapImagePath;

    HashMap<HexTile, Terrain> terrain;
    HashMap<HexTile, Bonus> placementBonuses;

    MapData(
        String fullName,
        String mapImagePath,
        HashMap<HexTile, Terrain> terrain,
        HashMap<HexTile, Bonus> placementBonuses) {
      this.fullName = fullName;
      this.mapImagePath = mapImagePath;
      this.terrain = terrain;
      this.placementBonuses = placementBonuses;
    }

    public HashMap<HexTile, Terrain> getTerrain() {
      return terrain;
    }

    public HashMap<HexTile, Bonus> getPlacementBonuses() {
      return placementBonuses;
    }

    public String getMapImagePath() {
      return mapImagePath;
    }
  }

  public enum Terrain {
    ROCK,
    WATER,
    BUILD_2_REQUIRED,
    BONUS_REP_PLUS_ONE(new Bonus(), true);

    boolean isPlacementBonus = false;
    Bonus bonus;

    Terrain() {}

    Terrain(Bonus bonus, boolean isPlacementBonus) {
      this.bonus = bonus;
      this.isPlacementBonus = isPlacementBonus;
    }
  }
}

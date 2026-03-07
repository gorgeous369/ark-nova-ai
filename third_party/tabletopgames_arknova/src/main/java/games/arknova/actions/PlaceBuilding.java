package games.arknova.actions;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.interfaces.IExtendedSequence;
import games.arknova.ArkNovaConstants;
import games.arknova.ArkNovaGameState;
import games.arknova.components.Building;
import games.arknova.components.HexTile;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Objects;

public class PlaceBuilding extends ArkNovaAction implements IExtendedSequence {

  protected Building building;

  protected List<ArkNovaAction> placementBonuses;

  protected boolean buildingPlaced;

  public PlaceBuilding(int playerId, Building building, boolean isFree) {
    super(playerId, 0, isFree);
    this.building = building;
    this.buildingPlaced = false;

    this.placementBonuses = new ArrayList<>();
  }

  @Override
  public List<AbstractAction> _computeAvailableActions(AbstractGameState state) {
    List<AbstractAction> actions = new ArrayList<>();

    if (!buildingPlaced) {
      actions.add(this.copy());
    }

    actions.addAll(placementBonuses);

    return actions;
  }

  @Override
  public boolean _execute(ArkNovaGameState gs) {
    gs.setActionInProgress(this);
    // Check for any placement bonuses and execute them
    // Otherwise just place the building
    gs.getMaps()[playerId].addBuilding(building);
    buildingPlaced = true;

    if (!isFree) {
      gs.incMoney(
          playerId, -ArkNovaConstants.MONEY_PER_ONE_BUILDING_HEX * building.getLayout().size());
    }

    // Find all covered tiles and create an action
    HashMap<HexTile, Bonus> mapPlacementBonuses =
        gs.getMaps()[playerId].getMapData().getPlacementBonuses();
    for (HexTile coveredTile : building.getLayout()) {
      if (mapPlacementBonuses.containsKey(coveredTile)) {
        placementBonuses.add(getActionFromBonus(mapPlacementBonuses.get(coveredTile)));
      }
    }

    return true;
  }

  @Override
  public void _afterAction(AbstractGameState state, AbstractAction action) {
    ArkNovaAction anAction = (ArkNovaAction) action;
    placementBonuses.remove(anAction);
  }

  public boolean executionComplete(AbstractGameState state) {
    return buildingPlaced && placementBonuses.isEmpty();
  }

  public ArkNovaAction getActionFromBonus(Bonus bonus) {
    if (bonus instanceof GainBonus) {
      return new GainResourceAction(playerId, ((GainBonus) bonus).amount, true, (GainBonus) bonus);
    }

    return null;
  }

  @Override
  public int getCurrentPlayer(AbstractGameState state) {
    return playerId;
  }

  @Override
  public PlaceBuilding copy() {
    PlaceBuilding copy = new PlaceBuilding(playerId, building, isFree);

    copy.buildingPlaced = buildingPlaced;
    copy.placementBonuses.addAll(placementBonuses);
    return copy;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    PlaceBuilding that = (PlaceBuilding) o;
    return buildingPlaced == that.buildingPlaced
        && Objects.equals(building, that.building)
        && Objects.equals(placementBonuses, that.placementBonuses);
  }

  @Override
  public int hashCode() {
    return Objects.hash(super.hashCode(), building, placementBonuses, buildingPlaced);
  }

  @Override
  public String toString() {
    return String.format("Place Building: %s", this.building);
  }
}

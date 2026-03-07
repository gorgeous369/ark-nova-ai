package games.arknova.actions;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.interfaces.IExtendedSequence;
import games.arknova.ArkNovaConstants;
import games.arknova.ArkNovaGameState;
import games.arknova.components.Building;
import games.arknova.components.BuildingType;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;

/**
 * The extended actions framework supports 2 use-cases:
 *
 * <ol>
 *   <li>A sequence of decisions required to complete an action (e.g. play a card in a game area -
 *       which card? - which area?). This avoids very large action spaces in favour of more
 *       decisions throughout the game (alternative: all unit actions with parameters supplied at
 *       initialization, all combinations of parameters computed beforehand).
 *   <li>A sequence of actions triggered by specific decisions (e.g. play a card which forces
 *       another player to discard a card - other player: which card to discard?)
 * </ol>
 *
 * <p>Extended actions should implement the {@link IExtendedSequence} interface and appropriate
 * methods, as detailed below.
 *
 * <p>They should also extend the {@link AbstractAction} class, or any other core actions. As such,
 * all guidelines in {@link ArkNovaAction} apply here as well.
 */
public class ArkNovaBuildAction extends ArkNovaAction implements IExtendedSequence {

  boolean executed = false;

  boolean isBuildUpgraded;

  boolean hasUsedEngineerThisRound = false;

  HashSet<BuildingType> alreadyBuiltBuildings;

  public ArkNovaBuildAction(int playerId, int strength, boolean isFree, boolean isBuildUpgraded) {
    super(playerId, strength, isFree);
    this.isBuildUpgraded = isBuildUpgraded;

    alreadyBuiltBuildings = new HashSet<>();
  }

  @Override
  public List<AbstractAction> _computeAvailableActions(AbstractGameState state) {
    ArkNovaGameState gs = (ArkNovaGameState) state;
    List<AbstractAction> actions = new ArrayList<>();

    boolean hasDiversityResearcher = gs.hasPlayedDiversityResearcher(playerId);
    boolean hasPlayedEngineer = gs.hasPlayedEngineer(playerId);

    // TODO: take into account Engineer and filter the buildings
    int maxBuildingSize = Math.min(strength, gs.getMaxBuildableBuildingSize(playerId));
    ArrayList<Building> legalBuildingsPlacements =
        gs.getMaps()[playerId].getLegalBuildingsPlacements(
            isBuildUpgraded, hasDiversityResearcher, maxBuildingSize, alreadyBuiltBuildings);

    //    System.out.format(
    //        "[%s] Legal buildings placements: %s \n",
    //        state.getCurrentPlayer(), legalBuildingsPlacements.size());
    for (Building building : legalBuildingsPlacements) {
      actions.add(new PlaceBuilding(playerId, building, isFree));
    }

    if (!actions.isEmpty() && !alreadyBuiltBuildings.isEmpty()) {
      actions.add(new PassAction(playerId));
    }

    return actions;
  }

  @Override
  public int getCurrentPlayer(AbstractGameState state) {
    return playerId;
  }

  @Override
  public void _afterAction(AbstractGameState state, AbstractAction action) {
    ArkNovaGameState gs = (ArkNovaGameState) state;

    if (action instanceof PlaceBuilding) {
      PlaceBuilding placeAction = (PlaceBuilding) action;
      alreadyBuiltBuildings.add(placeAction.building.getType());

      // TODO: calculate if engineer was used this round

      // Using engineer does not change the strength of the action
      if (!hasUsedEngineerThisRound) {
        strength -= placeAction.building.getLayout().size();
      } else {

      }
    }

    boolean hasPlayedEngineer = gs.hasPlayedEngineer(playerId);
    if (strength == 0
        || action instanceof PassAction
        || !isBuildUpgraded && (!hasPlayedEngineer || hasUsedEngineerThisRound)) {

      gs.setMainActionIndexTo(gs.getCurrentPlayer(), ArkNovaConstants.MainAction.BUILD, 0);
      executed = true;
    }
  }

  @Override
  public boolean executionComplete(AbstractGameState state) {
    return executed;
  }

  @Override
  public boolean execute(AbstractGameState gs) {
    gs.setActionInProgress(this);
    return true;
  }

  @Override
  public ArkNovaBuildAction copy() {
    ArkNovaBuildAction copy = new ArkNovaBuildAction(playerId, strength, isFree, isBuildUpgraded);

    copy.alreadyBuiltBuildings.addAll(this.alreadyBuiltBuildings);

    return copy;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    if (!super.equals(o)) return false;
    ArkNovaBuildAction that = (ArkNovaBuildAction) o;
    return executed == that.executed
        && isBuildUpgraded == that.isBuildUpgraded
        && hasUsedEngineerThisRound == that.hasUsedEngineerThisRound
        && Objects.equals(alreadyBuiltBuildings, that.alreadyBuiltBuildings);
  }

  @Override
  public int hashCode() {
    return Objects.hash(
        super.hashCode(),
        executed,
        isBuildUpgraded,
        hasUsedEngineerThisRound,
        alreadyBuiltBuildings);
  }

  @Override
  public String toString() {
    return String.format("[%s] ArkNovaBuildAction(%s)", isBuildUpgraded ? "UPG" : "BASE", strength);
  }

  /**
   * @param gameState - game state provided for context.
   * @return A more descriptive alternative to the toString action, after access to the game state
   *     to e.g. retrieve components for which only the ID is stored on the action object, and
   *     include the name of those components. Optional.
   */
  @Override
  public String getString(AbstractGameState gameState) {
    return toString();
  }
}

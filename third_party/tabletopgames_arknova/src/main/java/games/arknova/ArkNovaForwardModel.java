package games.arknova;

import core.AbstractGameState;
import core.StandardForwardModel;
import core.actions.AbstractAction;
import core.components.Counter;
import games.arknova.actions.ArkNovaAction;
import games.arknova.actions.ArkNovaBuildAction;
import games.arknova.components.ArkNovaMap;
import java.util.*;
import java.util.stream.Collectors;

/**
 * The forward model contains all the game rules and logic. It is mainly responsible for declaring
 * rules for:
 *
 * <ol>
 *   <li>Game setup
 *   <li>Actions available to players in a given game state
 *   <li>Game events or rules applied after a player's action
 *   <li>Game end
 * </ol>
 */
public class ArkNovaForwardModel extends StandardForwardModel {

  /**
   * Initializes all variables in the given game state. Performs initial game setup according to
   * game rules, e.g.:
   *
   * <ul>
   *   <li>Sets up decks of cards and shuffles them
   *   <li>Gives player cards
   *   <li>Places tokens on boards
   *   <li>...
   * </ul>
   *
   * @param firstState - the state to be modified to the initial game state.
   */
  @Override
  protected void _setup(AbstractGameState firstState) {
    // TODO: perform initialization of variables and game setup

    ArkNovaGameState gs = (ArkNovaGameState) firstState;

    int nPlayers = gs.getNPlayers();

    gs.maps = new ArkNovaMap[nPlayers];
    gs.playerIcons = new HashMap[nPlayers];
    gs.playerResources = new HashMap[nPlayers];
    gs.mainActionOrder = new ArrayList[nPlayers];
    gs.mainActionLevel = new HashMap[nPlayers];
    for (int i = 0; i < nPlayers; i++) {
      gs.maps[i] = new ArkNovaMap(ArkNovaMap.MapData.Map7);
      gs.mainActionOrder[i] = new ArrayList<>();
      gs.mainActionLevel[i] = new HashMap<>();

      gs.playerIcons[i] = new HashMap<>();
      gs.playerResources[i] = new HashMap<>();

      // Set actions
      gs.mainActionOrder[i].add(ArkNovaConstants.MainAction.ANIMALS);
      List<ArkNovaConstants.MainAction> shuffledActionsWithoutAnimals =
          Arrays.stream(ArkNovaConstants.MainAction.values())
              .filter(mainAction -> mainAction != ArkNovaConstants.MainAction.ANIMALS)
              .collect(Collectors.toList());

      Collections.shuffle(shuffledActionsWithoutAnimals);
      gs.mainActionOrder[i].addAll(shuffledActionsWithoutAnimals);

      for (ArkNovaConstants.MainAction action : ArkNovaConstants.MainAction.values()) {
        gs.mainActionLevel[i].put(action, ArkNovaConstants.MainActionLevel.UPGRADED);
      }

      // Set all resource to the initial value
      gs.playerResources[i].put(
          ArkNovaConstants.Resource.MONEY,
          new Counter(ArkNovaConstants.STARTING_MONEY, 0, Integer.MAX_VALUE, "MoneyCounter"));
      gs.playerResources[i].put(
          ArkNovaConstants.Resource.APPEAL,
          new Counter(i, 0, ArkNovaConstants.MAXIMUM_APPEAL, "AppealCounter"));

      gs.playerResources[i].put(
          ArkNovaConstants.Resource.CONSERVATION_POINTS,
          new Counter(
              0, 0, ArkNovaConstants.MAXIMUM_CONSERVATION_POINTS, "ConservationPointsCounter"));

      gs.playerResources[i].put(
          ArkNovaConstants.Resource.REPUTATION,
          new Counter(1, 1, ArkNovaConstants.MAXIMUM_REPUTATION, "ReputationCounter"));
      gs.playerResources[i].put(
          ArkNovaConstants.Resource.X_TOKEN,
          new Counter(0, 0, ArkNovaConstants.MAXIMUM_X_TOKEN, "XTokenCounter"));
      gs.playerResources[i].put(
          ArkNovaConstants.Resource.WORKER,
          new Counter(1, 1, ArkNovaConstants.MAXIMUM_WORKERS, "WorkerCounter"));

      // Set all icons to 0
      for (ArkNovaConstants.Icon icon : ArkNovaConstants.Icon.values()) {
        gs.playerIcons[i].put(icon, new Counter(icon.name() + "IconCounter"));
      }
    }
    gs.breakCounter =
        new Counter(0, 0, ArkNovaConstants.MAXIMUM_BREAK[nPlayers - 2], "BreakCounter");
  }

  /**
   * Calculates the list of currently available actions, possibly depending on the game phase.
   *
   * @return - List of AbstractAction objects.
   */
  @Override
  protected List<AbstractAction> _computeAvailableActions(AbstractGameState state) {
    ArkNovaGameState gs = (ArkNovaGameState) state;

    List<AbstractAction> actions = new ArrayList<>();

    int playerId = gs.getCurrentPlayer();

    for (int actionStrength = 0;
        actionStrength < gs.getMainActionOrder()[playerId].size();
        actionStrength++) {
      ArkNovaConstants.MainAction mainAction =
          gs.getMainActionOrder()[playerId].get(actionStrength);

      if (mainAction == ArkNovaConstants.MainAction.BUILD) {
        boolean isBuildUpgraded =
            gs.getMainActionLevel()[playerId].get(ArkNovaConstants.MainAction.BUILD)
                == ArkNovaConstants.MainActionLevel.UPGRADED;
        ArkNovaAction buildAction =
            new ArkNovaBuildAction(playerId, actionStrength + 1, false, isBuildUpgraded);

        actions.add(buildAction);
      }
    }

    return actions;
  }

  @Override
  protected void _afterAction(AbstractGameState currentState, AbstractAction actionTaken) {
    ArkNovaGameState gs = (ArkNovaGameState) currentState;

    if (currentState.isActionInProgress()) {
      return;
    }

    endPlayerTurn(currentState);
  }
}

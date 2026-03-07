package games.arknova;

import core.AbstractGameState;
import core.AbstractParameters;
import core.components.Component;
import core.components.Counter;
import games.GameType;
import games.arknova.components.ArkNovaMap;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

/**
 * The game state encapsulates all game information. It is a data-only class, with game
 * functionality present in the Forward Model or actions modifying the state of the game.
 *
 * <p>Most variables held here should be {@link Component} subclasses as much as possible.
 *
 * <p>No initialisation or game logic should be included here (not in the constructor either). This
 * is all handled externally.
 *
 * <p>Computation may be included in functions here for ease of access, but only if this is querying
 * the game state information. Functions on the game state should never <b>change</b> the state of
 * the game.
 */
public class ArkNovaGameState extends AbstractGameState {

  protected ArkNovaMap[] maps;

  protected ArrayList<ArkNovaConstants.MainAction>[] mainActionOrder;
  protected HashMap<ArkNovaConstants.MainAction, ArkNovaConstants.MainActionLevel>[]
      mainActionLevel;
  protected HashMap<ArkNovaConstants.Icon, Counter>[] playerIcons;
  protected HashMap<ArkNovaConstants.Resource, Counter>[] playerResources;
  protected Counter breakCounter;

  /**
   * @param gameParameters - game parameters.
   * @param nPlayers - number of players in the game
   */
  public ArkNovaGameState(AbstractParameters gameParameters, int nPlayers) {
    super(gameParameters, nPlayers);
  }

  public ArrayList<ArkNovaConstants.MainAction>[] getMainActionOrder() {
    return mainActionOrder;
  }

  public HashMap<ArkNovaConstants.MainAction, ArkNovaConstants.MainActionLevel>[]
      getMainActionLevel() {
    return mainActionLevel;
  }

  /**
   * @return the enum value corresponding to this game, declared in {@link GameType}.
   */
  @Override
  protected GameType _getGameType() {
    return GameType.ArkNova;
  }

  public Counter getBreakCounter() {
    return breakCounter;
  }

  public HashMap<ArkNovaConstants.Icon, Counter>[] getPlayerIcons() {
    return playerIcons;
  }

  public HashMap<ArkNovaConstants.Resource, Counter>[] getPlayerResources() {
    return playerResources;
  }

  public Counter getAppeal(int playerId) {
    return getPlayerResources()[playerId].get(ArkNovaConstants.Resource.APPEAL);
  }

  public Counter getConservationPoints(int playerId) {
    return getPlayerResources()[playerId].get(ArkNovaConstants.Resource.CONSERVATION_POINTS);
  }

  public Counter getReputation(int playerId) {
    return getPlayerResources()[playerId].get(ArkNovaConstants.Resource.REPUTATION);
  }

  public Counter getMoney(int playerId) {
    return getPlayerResources()[playerId].get(ArkNovaConstants.Resource.MONEY);
  }

  public Counter getXTokens(int playerId) {
    return getPlayerResources()[playerId].get(ArkNovaConstants.Resource.X_TOKEN);
  }

  public Counter getWorkers(int playerId) {
    return getPlayerResources()[playerId].get(ArkNovaConstants.Resource.WORKER);
  }

  public Counter getIcon(int playerId, ArkNovaConstants.Icon icon) {
    return getPlayerIcons()[playerId].get(icon);
  }

  /** Get max building size that can be built given the current money. */
  public int getMaxBuildableBuildingSize(int playerId) {
    return getMoney(playerId).getValue() / ArkNovaConstants.MONEY_PER_ONE_BUILDING_HEX;
  }

  // Counter
  public void incMoney(int playerId, int amount) {
    Counter money = getMoney(playerId);
    if (amount < 0 && money.getValue() < amount) {
      throw new RuntimeException("Not enough money to deduce!");
    }
    getMoney(playerId).increment(amount);
  }

  public void incReputation(int playerId, int amount) {
    getReputation(playerId).increment(amount);
  }

  public void setMainActionIndexTo(
      int playerId, ArkNovaConstants.MainAction mainAction, int index) {
    mainActionOrder[playerId].remove(mainAction);
    mainActionOrder[playerId].add(index, mainAction);
  }

  // TODO: fix when adding cards
  public boolean hasPlayedEngineer(int playerId) {
    return false;
  }

  public boolean hasPlayedDiversityResearcher(int playerId) {
    return false;
  }

  /**
   * Returns all Components used in the game and referred to by componentId from actions or rules.
   * This method is called after initialising the game state, so all components will be initialised
   * already.
   *
   * @return - List of Components in the game.
   */
  @Override
  protected List<Component> _getAllComponents() {
    // TODO: add all components to the list
    return new ArrayList<>();
  }

  /**
   * Create a deep copy of the game state containing only those components the given player can
   * observe.
   *
   * <p>If the playerID is NOT -1 and If any components are not visible to the given player (e.g.
   * cards in the hands of other players or a face-down deck), then these components should instead
   * be randomized (in the previous examples, the cards in other players' hands would be combined
   * with the face-down deck, shuffled together, and then new cards drawn for the other players).
   *
   * <p>If the playerID passed is -1, then full observability is assumed and the state should be
   * faithfully deep-copied.
   *
   * <p>Make sure the return type matches the class type, and is not AbstractGameState.
   *
   * @param playerId - player observing this game state.
   */
  @Override
  protected ArkNovaGameState _copy(int playerId) {
    ArkNovaGameState copy = new ArkNovaGameState(gameParameters, getNPlayers());
    copy.maps = new ArkNovaMap[getNPlayers()];
    copy.playerIcons = new HashMap[nPlayers];
    copy.playerResources = new HashMap[nPlayers];
    copy.mainActionOrder = new ArrayList[nPlayers];
    copy.mainActionLevel = new HashMap[nPlayers];

    for (int i = 0; i < getNPlayers(); i++) {
      copy.maps[i] = this.maps[i].copy();

      copy.mainActionOrder[i] = new ArrayList<>();
      copy.mainActionLevel[i] = new HashMap<>();

      copy.playerIcons[i] = new HashMap<>();
      copy.playerResources[i] = new HashMap<>();

      for (ArkNovaConstants.MainAction mainAction : getMainActionOrder()[i]) {
        copy.mainActionOrder[i].add(mainAction);
      }

      for (ArkNovaConstants.MainAction mainAction : getMainActionLevel()[i].keySet()) {
        copy.mainActionLevel[i].put(mainAction, getMainActionLevel()[i].get(mainAction));
      }

      for (ArkNovaConstants.Icon playerIcon : getPlayerIcons()[i].keySet()) {
        copy.playerIcons[i].put(playerIcon, getPlayerIcons()[i].get(playerIcon));
      }

      for (ArkNovaConstants.Resource playerResource : getPlayerResources()[i].keySet()) {
        copy.playerResources[i].put(playerResource, getPlayerResources()[i].get(playerResource));
      }
    }
    // TODO: deep copy all variables to the new game state.
    return copy;
  }

  /**
   * @param playerId - player observing the state.
   * @return a score for the given player approximating how well they are doing (e.g. how close they
   *     are to winning the game); a value between 0 and 1 is preferred, where 0 means the game was
   *     lost, and 1 means the game was won.
   */
  @Override
  protected double _getHeuristicScore(int playerId) {
    if (isNotTerminal()) {
      // TODO calculate an approximate value
      return 0;
    } else {
      // The game finished, we can instead return the actual result of the game for the given
      // player.
      return getPlayerResults()[playerId].value;
    }
  }

  /**
   * @param playerId - player observing the state.
   * @return the true score for the player, according to the game rules. May be 0 if there is no
   *     score in the game.
   */
  @Override
  public double getGameScore(int playerId) {
    // The score is combination of appeal going from left-to-right and
    // score from conservation points going from right-to-left
    int cp = getConservationPoints(playerId).getValue();

    // The conversation points go reverse from the max appeal, the first 10 CPs span over 2 score
    // slots, minimal score is -14
    int conversationScore =
        (ArkNovaConstants.TRIGGER_END_GAME_SCORE - ArkNovaConstants.MAXIMUM_APPEAL - 1)
            + Math.min(10, cp) * 2;

    // The conservation points above 10 are worth 3 score slots
    if (cp > 10) {
      conversationScore += (cp - 10) * 3;
    }

    int appeal = getAppeal(playerId).getValue();
    return conversationScore + appeal;
  }

  public ArkNovaMap[] getMaps() {
    return maps;
  }

  public ArkNovaMap getCurrentPlayerMap() {
    return getMaps()[getCurrentPlayer()];
  }

  @Override
  protected boolean _equals(Object o) {
    // TODO: compare all variables in the state
    return o instanceof ArkNovaGameState;
  }

  @Override
  public int hashCode() {
    // TODO: include the hash code of all variables
    return super.hashCode();
  }

  // TODO: Consider the methods below for possible implementation
  // TODO: These all have default implementations in AbstractGameState, so are not required to be
  // implemented here.
  // TODO: If the game has 'teams' that win/lose together, then implement the next two nethods.
  /**
   * Returns the number of teams in the game. The default is to have one team per player. If the
   * game does not have 'teams' that win/lose together, then ignore these two methods.
   */
  // public int getNTeams();
  /** Returns the team number the specified player is on. */
  // public int getTeam(int player);

  // TODO: If your game has multiple special tiebreak options, then implement the next two methods.
  // TODO: The default is to tie-break on the game score (if this is the case, ignore these)
  // public double getTiebreak(int playerId, int tier);
  // public int getTiebreakLevels();

  // TODO: If your game does not have a score of any type, and is an 'insta-win' type game which
  // ends
  // TODO: as soon as a player achieves a winning condition, and has some bespoke method for
  // determining 1st, 2nd, 3rd etc.
  // TODO: Then you *may* want to implement:.
  // public int getOrdinalPosition(int playerId);

  // TODO: Review the methods below...these are all supported by the default implementation in
  // AbstractGameState
  // TODO: So you do not (and generally should not) implement your own versions - take advantage of
  // the framework!
  // public Random getRnd() returns a Random number generator for the game. This will be derived
  // from the seed
  // in game parameters, and will be updated correctly on a reset

  // Ths following provide access to the id of the current player; the first player in the Round (if
  // that is relevant to a game)
  // and the current Turn and Round numbers.
  // public int getCurrentPlayer()
  // public int getFirstPlayer()
  // public int getRoundCounter()
  // public int getTurnCounter()
  // also make sure you check out the standard endPlayerTurn() and endRound() methods in
  // StandardForwardModel

  // This method can be used to log a game event (e.g. for something game-specific that you want to
  // include in the metrics)
  // public void logEvent(IGameEvent...)
}

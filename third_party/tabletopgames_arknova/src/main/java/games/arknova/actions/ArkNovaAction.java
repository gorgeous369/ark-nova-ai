package games.arknova.actions;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.components.Component;
import games.arknova.ArkNovaGameState;
import java.util.Objects;

/**
 * Actions are unit things players can do in the game (e.g. play a card, move a pawn, roll dice,
 * attack etc.).
 *
 * <p>Actions in the game can (and should, if applicable) extend one of the other existing actions,
 * in package {@link core.actions}. Or, a game may simply reuse one of the existing core actions.
 *
 * <p>Actions may have parameters, so as not to duplicate actions for the same type of
 * functionality, e.g. playing card of different types (see {@link games.sushigo.actions.ChooseCard}
 * action from SushiGo as an example). Include these parameters in the class constructor.
 *
 * <p>They need to extend at a minimum the {@link AbstractAction} super class and implement the
 * {@link AbstractAction#execute(AbstractGameState)} method. This is where the main functionality of
 * the action should be inserted, which modifies the given game state appropriately (e.g. if the
 * action is to play a card, then the card will be moved from the player's hand to the discard pile,
 * and the card's effect will be applied).
 *
 * <p>They also need to include {@link Object#equals(Object)} and {@link Object#hashCode()} methods.
 *
 * <p>They <b>MUST NOT</b> keep references to game components. Instead, store the {@link
 * Component#getComponentID()} in variables for any components that must be referenced in the
 * action. Then, in the execute() function, use the {@link AbstractGameState#getComponentById(int)}
 * function to retrieve the actual reference to the component, given your componentID.
 */
public class ArkNovaAction extends AbstractAction {

  protected int playerId;

  protected boolean isFree;

  protected int strength;

  public ArkNovaAction(int playerId, int strength, boolean isFree) {
    this.playerId = playerId;
    this.strength = strength;
    this.isFree = isFree;
  }

  public ArkNovaAction(int playerId, int strength) {
    this(playerId, strength, false);
  }

  public ArkNovaAction(int playerId) {
    this(playerId, 0);
  }

  /**
   * Executes this action, applying its effect to the given game state. Can access any component IDs
   * stored through the {@link AbstractGameState#getComponentById(int)} method.
   *
   * @param gs - game state which should be modified by this action.
   * @return - true if successfully executed, false otherwise.
   */
  @Override
  public boolean execute(AbstractGameState gs) {
    ArkNovaGameState state = (ArkNovaGameState) gs;
    // TODO: Some functionality applied which changes the given game state.
    //        try {
    //            Thread.sleep(100000);
    //        } catch (InterruptedException e) {
    //            throw new RuntimeException(e);
    //        }

    this._execute(state);
    return true;
  }

  public boolean _execute(ArkNovaGameState gameState) {
    return true;
  }

  /**
   * @return Make sure to return an exact <b>deep</b> copy of the object, including all of its
   *     variables. Make sure the return type is this class (e.g. GTAction) and NOT the super class
   *     AbstractAction.
   *     <p>If all variables in this class are final or effectively final (which they should be),
   *     then you can just return <code>`this`</code>.
   */
  @Override
  public ArkNovaAction copy() {
    // TODO: copy non-final variables appropriately
    return this;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    ArkNovaAction that = (ArkNovaAction) o;
    return playerId == that.playerId && isFree == that.isFree && strength == that.strength;
  }

  @Override
  public int hashCode() {
    return Objects.hash(playerId, isFree, strength);
  }

  @Override
  public String toString() {
    // TODO: Replace with appropriate string, including any action parameters
    return getClass().getSimpleName();
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

  /**
   * This next one is optional.
   *
   * <p>May optionally be implemented if Actions are not fully visible The only impact this has is
   * in the GUI, to avoid this giving too much information to the human player.
   *
   * <p>An example is in Resistance or Sushi Go, in which all cards are technically revealed
   * simultaneously, but the game engine asks for the moves sequentially. In this case, the action
   * should be able to output something like "Player N plays card", without saying what the card is.
   *
   * @param gameState - game state to be used to generate the string.
   * @param playerId - player to whom the action should be represented.
   * @return
   */
  // @Override
  // public String getString(AbstractGameState gameState, int playerId);
}

package games.arknova.actions;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.interfaces.IExtendedSequence;
import java.util.ArrayList;
import java.util.List;

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
public class ArkNovaExtendedSequenceAction extends ArkNovaAction implements IExtendedSequence {

  public ArkNovaExtendedSequenceAction(int playerId, int strength, boolean isFree) {
    super(playerId, strength, isFree);
  }

  /**
   * Forward Model delegates to this from {@link
   * core.StandardForwardModel#computeAvailableActions(AbstractGameState)} if this Extended Sequence
   * is currently active.
   *
   * @param state The current game state
   * @return the list of possible actions for the {@link AbstractGameState#getCurrentPlayer()}.
   *     These may be instances of this same class, with more choices between different values for a
   *     not-yet filled in parameter.
   */
  @Override
  public List<AbstractAction> _computeAvailableActions(AbstractGameState state) {
    // TODO populate this list with available actions
    return new ArrayList<>();
  }

  /**
   * TurnOrder delegates to this from {@link
   * core.turnorders.TurnOrder#getCurrentPlayer(AbstractGameState)} if this Extended Sequence is
   * currently active.
   *
   * @param state The current game state
   * @return The player ID whose move it is.
   */
  @Override
  public int getCurrentPlayer(AbstractGameState state) {
    return playerId;
  }

  /**
   * This is called by ForwardModel whenever an action is about to be taken. It enables the
   * IExtendedSequence to maintain local state in whichever way is most suitable.
   *
   * <p>After this call, the state of IExtendedSequence should be correct ahead of the next decision
   * to be made. In some cases, there is no need to implement anything in this method - if for
   * example you can tell if all actions are complete from the state directly, then that can be
   * implemented purely in {@link #executionComplete(AbstractGameState)}
   *
   * @param state The current game state
   * @param action The action about to be taken (so the game state has not yet been updated with it)
   */
  @Override
  public void _afterAction(AbstractGameState state, AbstractAction action) {
    // TODO: Process the action that was taken.
  }

  /**
   * @param state The current game state
   * @return True if this extended sequence has now completed and there is nothing left to do.
   */
  @Override
  public boolean executionComplete(AbstractGameState state) {
    // TODO is execution of this sequence of actions complete?
    return true;
  }

  /**
   * Executes this action, applying its effect to the given game state. Can access any component IDs
   * stored through the {@link AbstractGameState#getComponentById(int)} method.
   *
   * <p>In extended sequences, this function makes a call to the {@link
   * AbstractGameState#setActionInProgress(IExtendedSequence)} method with the argument <code>`this`
   * </code> to indicate that this action has multiple steps and is now in progress. This call could
   * be wrapped in an <code>`if`</code> statement if sometimes the action simply executes an effect
   * in one step, or all parameters have values associated.
   *
   * @param gs - game state which should be modified by this action.
   * @return - true if successfully executed, false otherwise.
   */
  @Override
  public boolean execute(AbstractGameState gs) {
    // TODO: Some functionality applied which changes the given game state.
    gs.setActionInProgress(this);
    super.execute(gs);
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
  public ArkNovaExtendedSequenceAction copy() {
    // TODO: copy non-final variables appropriately
    return this;
  }

  @Override
  public String toString() {
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
}

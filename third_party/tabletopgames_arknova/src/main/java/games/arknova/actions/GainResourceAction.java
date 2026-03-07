package games.arknova.actions;

import core.AbstractGameState;
import core.actions.AbstractAction;
import core.interfaces.IExtendedSequence;
import games.arknova.ArkNovaConstants;
import games.arknova.ArkNovaGameState;
import java.util.ArrayList;
import java.util.List;

public class GainResourceAction extends ArkNovaAction implements IExtendedSequence {

  protected boolean resourceIncreased;
  GainBonus bonus;
  List<ArkNovaAction> spawnedActions;

  public GainResourceAction(int playerId, int strength, boolean isFree, GainBonus bonus) {
    super(playerId, strength, isFree);
    this.bonus = bonus;

    this.resourceIncreased = false;

    spawnedActions = new ArrayList<>();
  }

  @Override
  public List<AbstractAction> _computeAvailableActions(AbstractGameState state) {
    List<AbstractAction> actions = new ArrayList<>();
    actions.add(this);

    return actions;
  }

  @Override
  public int getCurrentPlayer(AbstractGameState state) {
    return playerId;
  }

  @Override
  public void _afterAction(AbstractGameState state, AbstractAction action) {}

  @Override
  public boolean _execute(ArkNovaGameState gs) {
    gs.setActionInProgress(this);
    if (bonus.resource == ArkNovaConstants.Resource.REPUTATION) {
      gs.incReputation(playerId, bonus.amount);

      resourceIncreased = true;
    }

    return true;
  }

  @Override
  public boolean executionComplete(AbstractGameState state) {
    return resourceIncreased && spawnedActions.isEmpty();
  }

  @Override
  public GainResourceAction copy() {
    return new GainResourceAction(playerId, strength, isFree, bonus);
  }

  @Override
  public String toString() {
    return String.format("GainResourceAction (%s: %d)", bonus.resource.name(), bonus.amount);
  }
}

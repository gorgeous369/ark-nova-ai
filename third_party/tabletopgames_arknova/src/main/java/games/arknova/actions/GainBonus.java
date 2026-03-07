package games.arknova.actions;

import games.arknova.ArkNovaConstants;

public class GainBonus extends Bonus {

  ArkNovaConstants.Resource resource;

  boolean isAutomatic;

  int amount;

  public GainBonus(ArkNovaConstants.Resource resource, boolean isAutomatic, int amount) {
    this.resource = resource;
    this.isAutomatic = isAutomatic;
    this.amount = amount;
  }
}

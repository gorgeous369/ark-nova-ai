package games.arknova;

public class ArkNovaConstants {

  public static final int MINIMUM_KIOSK_DISTANCE = 3;
  public static final int STARTING_MONEY = 25;
  public static final int MONEY_PER_ONE_BUILDING_HEX = 2;

  public static final int MAXIMUM_APPEAL = 113;
  public static final int MAXIMUM_CONSERVATION_POINTS = 41;
  public static final int MAXIMUM_REPUTATION = 15;
  public static final int MAXIMUM_X_TOKEN = 5;
  public static final int MAXIMUM_WORKERS = 4;

  public static final int[] MAXIMUM_BREAK = new int[] {9, 12, 15};

  // Game ends after the score reach 100 (when CP marker and appeal marker cross each other)
  public static final int TRIGGER_END_GAME_SCORE = 100;

  public enum Resource {
    MONEY,
    APPEAL,
    CONSERVATION_POINTS,
    REPUTATION,
    X_TOKEN,
    WORKER,
  }

  public enum Icon {
    AFRICA,
    EUROPE,
    ASIA,
    AMERICA,
    AUSTRALIA,
    BIRD,
    PREDATOR,
    HERBIVORE,
    REPTILE,
    PRIMATE,
    BEAR,
    PETTING_ZOO_ANIMAL,
    SCIENCE,
    ROCK, // not really a tag per-se, but they follow a similar logic
    WATER; // not really a tag per-se, but they follow a similar logic

    public String getImagePath() {
      return String.format("data/arknova/icons/%s.png", this.name().toLowerCase());
    }
  }

  public enum MainAction {
    ANIMALS,
    SPONSOR,
    ASSOCIATION,
    CARDS,
    BUILD
  }

  public enum MainActionLevel {
    BASE,
    UPGRADED
  }
}

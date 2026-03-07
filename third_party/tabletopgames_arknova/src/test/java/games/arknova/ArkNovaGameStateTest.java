package games.arknova;

import static org.testng.Assert.assertEquals;

import core.AbstractPlayer;
import core.Game;
import games.GameType;
import games.arknova.actions.PlaceBuilding;
import games.arknova.components.Building;
import games.arknova.components.BuildingType;
import games.arknova.components.HexTile;
import java.util.Arrays;
import java.util.List;
import org.testng.annotations.Test;
import players.simple.RandomPlayer;

public class ArkNovaGameStateTest {

  List<AbstractPlayer> players =
      Arrays.asList(new RandomPlayer(), new RandomPlayer(), new RandomPlayer());

  ArkNovaForwardModel fm = new ArkNovaForwardModel();

  public Game getNewGame() {
    return new Game(
        GameType.ArkNova,
        players,
        new ArkNovaForwardModel(),
        new ArkNovaGameState(new ArkNovaParameters(), players.size()));
  }

  @Test
  public void testSetup() {
    Game game = getNewGame();
    // TODO: add other checks
    ArkNovaGameState gs = (ArkNovaGameState) game.getGameState();

    assertEquals(gs.getBreakCounter().getMaximum(), 12);

    for (AbstractPlayer player : players) {
      assertEquals(gs.getMoney(player.getPlayerID()).getValue(), 25);
      assertEquals(gs.getConservationPoints(player.getPlayerID()).getValue(), 0);
      assertEquals(gs.getReputation(player.getPlayerID()).getValue(), 1);
      assertEquals(gs.getXTokens(player.getPlayerID()).getValue(), 0);
      assertEquals(gs.getWorkers(player.getPlayerID()).getValue(), 1);
    }

    assertEquals(gs.getAppeal(0).getValue(), 0);
    assertEquals(gs.getAppeal(1).getValue(), 1);
    assertEquals(gs.getAppeal(2).getValue(), 2);
  }

  @Test
  public void testGetGameScore() {
    Game game = getNewGame();
    ArkNovaGameState gs = (ArkNovaGameState) game.getGameState();

    assertEquals(gs.getGameScore(0), -14);
    assertEquals(gs.getGameScore(1), -13);
    assertEquals(gs.getGameScore(2), -12);

    // Random value check
    gs.getConservationPoints(0).setValue(9);
    gs.getAppeal(0).setValue(62);

    assertEquals(gs.getGameScore(0), 66);

    // Edge case check
    gs.getConservationPoints(0).setValue(41);
    gs.getAppeal(0).setValue(113);

    assertEquals(gs.getGameScore(0), 212);
  }

  @Test
  public void testPlaceBuildingAction() {
    Game game = getNewGame();
    ArkNovaGameState gs = (ArkNovaGameState) game.getGameState();

    int playerId = 0;
    Building toBePlaced =
        new Building(BuildingType.SIZE_2, new HexTile(0, 0), Building.Rotation.ROT_0);

    fm.next(gs, new PlaceBuilding(playerId, toBePlaced, false));

    // Should have less money and building set
    assertEquals(gs.getMoney(playerId).getValue(), 21);
    assertEquals(gs.getMaps()[playerId].getBuildings().size(), 1);
    assertEquals(gs.getMaps()[playerId].getBuildings().get(new HexTile(0, 0)), toBePlaced);

    // Take a free place building action, shouldn't cost any money
    fm.next(
        gs,
        new PlaceBuilding(
            playerId,
            new Building(BuildingType.SIZE_2, new HexTile(3, 0), Building.Rotation.ROT_0),
            true));

    assertEquals(gs.getMoney(playerId).getValue(), 21);
    assertEquals(gs.getMaps()[playerId].getBuildings().size(), 2);
  }
}

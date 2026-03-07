package games.arknova.components;

import static org.testng.Assert.*;

import org.testng.annotations.Test;

public class HexTileTest {

  @Test
  public void testDistance() {
    assertEquals(new HexTile(5, 1).distance(new HexTile(5, 1)), 0);
    assertEquals(new HexTile(1, -1).distance(new HexTile(8, -1)), 7);
    assertEquals(new HexTile(7, 2).distance(new HexTile(4, 3)), 3);
  }
}

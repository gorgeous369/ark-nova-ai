package games.arknova;

import games.arknova.components.ArkNovaMap;
import games.arknova.components.Building;
import games.arknova.components.BuildingType;
import games.arknova.components.HexTile;
import org.junit.Test;

import java.awt.*;

import static org.junit.Assert.*;

public class TestHexTiles {

    @Test
    public void hexCoordinateEquality() {
        HexTile a = new HexTile(45, 23);
        HexTile b = new HexTile(45, 23);

        assertEquals(a, b);
        assertNotSame(a, b);
    }

    @Test
    public void hexToPixel() {
        int hexSize = 50;
        Point selectedPoint = new Point(217, 311);

        ArkNovaMap map = new ArkNovaMap(ArkNovaMap.MapData.Map7);
        HexTile mappedHex = map.pixelToHex(selectedPoint, hexSize);

        assertEquals(mappedHex, new HexTile(3, 2));
    }

    @Test
    public void buildingCreation() {
        Building building = new Building(BuildingType.PETTING_ZOO, new HexTile(2, 1), Building.Rotation.ROT_120);

        assertArrayEquals(building.getLayout().toArray(),
                          new HexTile[]{new HexTile(2, 1), new HexTile(3, 1), new HexTile(3, 2)}
        );
        assertEquals(building.getEmptySpaces(), 3);
    }


    @Test
    public void decreaseEmptySpaces() {
        Building building = new Building(BuildingType.PETTING_ZOO, new HexTile(2, 1), Building.Rotation.ROT_120);

        assertEquals(building.getEmptySpaces(), 3);
        building.decreaseEmptySpaces(2);
        assertEquals(building.getEmptySpaces(), 1);

        assertThrows(IllegalArgumentException.class, () -> building.decreaseEmptySpaces(2));

    }
}

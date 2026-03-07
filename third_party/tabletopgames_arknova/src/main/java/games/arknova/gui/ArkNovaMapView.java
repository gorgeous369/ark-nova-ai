package games.arknova.gui;

import static games.terraformingmars.gui.Utils.drawImage;

import games.arknova.ArkNovaGameState;
import games.arknova.components.ArkNovaMap;
import games.arknova.components.Building;
import games.arknova.components.HexTile;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.geom.AffineTransform;
import javax.swing.*;
import utilities.ImageIO;

public class ArkNovaMapView extends JComponent {

  final double SCALE = 1.0;
  Point HEX_GRID_OFFSET = new Point(270, 130);
  Point MAP_SIZE = new Point(1500, 1500);
  int HEX_TILE_SIZE = 50;
  ArkNovaGameState gs;
  Image mapImage;

  HexTile selectedHex;
  ArkNovaGUIManager gui;

  public ArkNovaMapView(ArkNovaGUIManager gui, ArkNovaGameState gs) {
    this.gs = gs;
    this.gui = gui;

    ArkNovaMap map = gs.getCurrentPlayerMap();

    // Dynamic variables based on the scale of the GUI
    // TODO: check if even needed
    HEX_TILE_SIZE = (int) (SCALE * HEX_TILE_SIZE);
    HEX_GRID_OFFSET =
        new Point((int) (HEX_GRID_OFFSET.x * SCALE), (int) (HEX_GRID_OFFSET.y * SCALE));
    MAP_SIZE = new Point((int) (MAP_SIZE.x * SCALE), (int) (MAP_SIZE.y * SCALE));

    mapImage = ImageIO.GetInstance().getImage(map.getMapData().getMapImagePath());

    addMouseListener(
        new MouseAdapter() {
          @Override
          public void mouseClicked(MouseEvent e) {
            if (e.getButton() == MouseEvent.BUTTON1) {
              Point pixel = e.getPoint();
              pixel.translate(-HEX_GRID_OFFSET.x, -HEX_GRID_OFFSET.y);
              selectedHex = map.pixelToHex(pixel, HEX_TILE_SIZE);
            }
          }
        });
  }

  public void drawBuilding(Graphics2D g, Building building) {
    Image img = ImageIO.GetInstance().getImage(building.getImage());
    HexTile hexTile = building.getOriginHex();
    int x = (int) (HEX_GRID_OFFSET.x + (HEX_TILE_SIZE + 1) * (3. / 2 * hexTile.q));
    int y =
        (int)
            (HEX_GRID_OFFSET.y
                + (HEX_TILE_SIZE) * (Math.sqrt(3) / 2 * hexTile.q + Math.sqrt(3) * hexTile.r));

    int size = HEX_TILE_SIZE;
    int w = img.getWidth(null);
    int h = img.getHeight(null);
    double scale;
    if (w > h) scale = size * 1.0 / w;
    else scale = size * 1.0 / h;

    scale = SCALE;

    AffineTransform tr = new AffineTransform();
    y = y - h + (int) (Math.sqrt(3) * HEX_TILE_SIZE / 2);
    x -= HEX_TILE_SIZE;

    //    tr.translate(x - (int) gui.xModel.getValue(), y - (int) gui.yModel.getValue());
    tr.translate(x, y);

    tr.scale(scale, scale);

    tr.rotate(
        //        Math.toRadians((int) gui.model.getValue()),
        Math.toRadians(building.getRotation().getAngle()),
        HEX_TILE_SIZE,
        h - (Math.sqrt(3) * (HEX_TILE_SIZE + 1) / 2.0));

    g.drawImage(img, tr, null);

    g.setColor(Color.RED);
    g.drawLine(x, y, x + 2, y + 2);
  }

  public Dimension getPreferredSize() {
    return new Dimension(MAP_SIZE.x, MAP_SIZE.y / 2);
  }

  @Override
  protected void paintComponent(Graphics gBase) {
    Graphics2D g = (Graphics2D) gBase.create();
    drawImage(g, mapImage, 0, 0, MAP_SIZE.y);

    ArkNovaMap map = gs.getMaps()[gui.getCurrentlyObservedPlayer()];

    map.getBuildings().values().forEach((building) -> drawBuilding(g, building));

    if (ArkNovaGUIManager.DEBUG) {
      for (HexTile hexTile : map.getGrid().values()) {
        double center_x = (HEX_TILE_SIZE + 1) * (3. / 2 * hexTile.q);
        double center_y = HEX_TILE_SIZE * (Math.sqrt(3) / 2 * hexTile.q + Math.sqrt(3) * hexTile.r);

        Polygon h = getPolygon(center_x, center_y);

        if (map.getMapData().getTerrain().containsKey(hexTile)) {
          ArkNovaMap.Terrain terrain = map.getMapData().getTerrain().get(hexTile);

          switch (terrain) {
            case ROCK:
              g.setColor(new Color(114, 42, 5, 80));
              g.fillPolygon(h);
              break;
            case WATER:
              g.setColor(new Color(6, 63, 191, 90));
              g.fillPolygon(h);
              break;
            case BUILD_2_REQUIRED:
              g.setColor(new Color(247, 0, 235, 90));
              g.fillPolygon(h);
              break;
            default:
              g.setColor(new Color(221, 197, 42, 150));
              g.fillPolygon(h);
              break;
          }
        }

        if (hexTile.equals(selectedHex)) {
          g.setColor(new Color(0, 255, 0, 50));
          g.fillPolygon(h);
        }

        g.setColor(Color.BLACK);
        g.drawPolygon(h);

        g.setStroke(new BasicStroke(5));
        g.setFont(new Font("Arial", Font.BOLD, 15));
        g.drawString(
            hexTile.q + ", " + hexTile.r,
            (int) center_x + HEX_GRID_OFFSET.x,
            (int) center_y + HEX_GRID_OFFSET.y);
      }
    }

    super.paintComponent(g);
  }

  private Polygon getPolygon(double center_x, double center_y) {
    Polygon h = new Polygon();
    for (int i = 0; i < 6; i++) {
      double angle = 2.0 * Math.PI * i / 6;
      double offset_x = HEX_TILE_SIZE * Math.cos(angle);
      double offset_y = HEX_TILE_SIZE * Math.sin(angle);

      int hex_x = (int) (HEX_GRID_OFFSET.x + center_x + offset_x);
      int hex_y = (int) (HEX_GRID_OFFSET.y + center_y + offset_y);

      h.addPoint(hex_x, hex_y);
    }
    return h;
  }
}

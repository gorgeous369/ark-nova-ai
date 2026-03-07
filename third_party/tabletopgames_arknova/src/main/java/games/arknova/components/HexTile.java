package games.arknova.components;

import java.util.ArrayList;
import java.util.Objects;
import utilities.Vector2D;

/**
 * Axial hex tile grid implementation based on the excellent article
 * (https://www.redblobgames.com/grids/hexagons/)
 */
public class HexTile {

  public static HexTile[] NEIGHBORS =
      new HexTile[] {
        new HexTile(1, 0),
        new HexTile(1, -1),
        new HexTile(0, -1),
        new HexTile(-1, 0),
        new HexTile(-1, 1),
        new HexTile(0, 1)
      };

  public int q;

  public int r;

  public HexTile(int q, int r) {
    this.q = q;
    this.r = r;
  }

  public int getS() {
    return -q - r;
  }

  public HexTile add(HexTile other) {
    return new HexTile(this.q + other.q, this.r + other.r);
  }

  public HexTile subtract(HexTile other) {
    return new HexTile(this.q - other.q, this.r - other.r);
  }

  /**
   * Calculates distance (how many hops is needed to reach other hex tile).
   *
   * @param other Hex tile to calculate distance for.
   * @return integer distance between the hex tiles.
   */
  public int distance(HexTile other) {
    return (Math.abs(this.q - other.q)
            + Math.abs(this.q + this.r - other.q - other.r)
            + Math.abs(this.r - other.r))
        / 2;
  }

  public HexTile rotateLeft() {
    return new HexTile(-getS(), -q);
  }

  public HexTile rotateRight() {
    return new HexTile(-r, -getS());
  }

  /**
   * Returns doubled coordinates (https://www.redblobgames.com/grids/hexagons/#conversions-doubled)
   * of the axial hex.
   *
   * @return Vector of column and row of doubled coordinates.
   */
  public Vector2D getDoubledCoordinates() {
    int col = q;
    int row = 2 * r + q;

    return new Vector2D(col, row);
  }

  public ArrayList<HexTile> getNeighbors() {
    ArrayList<HexTile> neighbors = new ArrayList<>();

    for (HexTile neighbor : NEIGHBORS) {
      neighbors.add(this.add(neighbor));
    }

    return neighbors;
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    HexTile hexTile = (HexTile) o;
    return q == hexTile.q && r == hexTile.r;
  }

  @Override
  public int hashCode() {
    return Objects.hash(q, r);
  }

  @Override
  public String toString() {
    return "HexTile{" + "q=" + q + ", r=" + r + '}';
  }
}

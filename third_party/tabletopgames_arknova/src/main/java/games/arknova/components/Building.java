package games.arknova.components;

import java.util.ArrayList;
import java.util.Objects;

public class Building {
  protected BuildingType type;

  // Location of the starting hex, together with rotation uniquely determines the building's
  // location
  protected HexTile originHex;
  protected Rotation rotation;
  protected ArrayList<HexTile> layout;
  protected int emptySpaces;

  public Building(BuildingType type, HexTile originHex, Rotation rotation) {
    this.type = type;
    this.originHex = originHex;
    this.rotation = Rotation.ROT_0;
    this.emptySpaces = type.maxCapacity;

    this.layout = new ArrayList<>();
    for (HexTile tile : type.getLayout()) {
      this.layout.add(originHex.add(tile));
    }

    this.applyRotation(rotation);
  }

  public ArrayList<HexTile> getLayout() {
    return layout;
  }

  public BuildingType getType() {
    return type;
  }

  public HexTile getOriginHex() {
    return originHex;
  }

  public Rotation getRotation() {
    return rotation;
  }

  public int getEmptySpaces() {
    return emptySpaces;
  }

  /**
   * Applies rotation to the current building until it hits specified `rotation`.
   *
   * @param newRotation Angle to rotate to.
   */
  public void applyRotation(Rotation newRotation) {
    double diffAngle = newRotation.getAngle() - this.rotation.getAngle();

    for (double angle = 0; angle < Math.abs(diffAngle); angle += 60) {
      if (diffAngle < 0) {
        this.rotateLeft();
      } else {
        this.rotateRight();
      }
    }
    this.rotation = newRotation;
  }

  public void rotateLeft() {
    ArrayList<HexTile> newLayout = new ArrayList<>();
    for (HexTile tile : layout) {
      newLayout.add(tile.subtract(originHex).rotateLeft().add(originHex));
    }

    this.layout = newLayout;
  }

  public void rotateRight() {
    ArrayList<HexTile> newLayout = new ArrayList<>();
    for (HexTile tile : layout) {
      newLayout.add(tile.subtract(originHex).rotateRight().add(originHex));
    }

    this.layout = newLayout;
  }

  public void decreaseEmptySpaces(int emptySpaces) {
    if (this.emptySpaces - emptySpaces < 0) {
      throw new IllegalArgumentException("Cannot decrease empty space below 0.");
    }
    this.emptySpaces -= emptySpaces;
  }

  public String getImage() {
    if (this.emptySpaces != 0 && type.subType == BuildingType.BuildingSubType.ENCLOSURE_BASIC) {
      return type.getBackImagePath();
    } else {
      return type.getFrontImagePath();
    }
  }

  @Override
  public String toString() {
    return "Building{"
        + "type="
        + type
        + ", originHex="
        + originHex
        + ", rotation="
        + rotation
        + ", layout="
        + layout
        + ", emptySpaces="
        + emptySpaces
        + '}';
  }

  @Override
  public boolean equals(Object o) {
    if (this == o) return true;
    if (o == null || getClass() != o.getClass()) return false;
    Building building = (Building) o;
    return emptySpaces == building.emptySpaces
        && type == building.type
        && Objects.equals(originHex, building.originHex)
        && rotation == building.rotation
        && Objects.equals(layout, building.layout);
  }

  @Override
  public int hashCode() {
    return Objects.hash(type, originHex, rotation, layout, emptySpaces);
  }

  public enum Rotation {
    ROT_0(0),
    ROT_60(60),
    ROT_120(120),
    ROT_180(180),
    ROT_240(240),
    ROT_300(360);

    private final double angle;

    Rotation(double angle) {
      this.angle = angle;
    }

    public double getAngle() {
      return angle;
    }
  }
}

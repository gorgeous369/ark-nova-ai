package games.arknova.gui;

import core.AbstractGameState;
import core.AbstractPlayer;
import core.Game;
import games.arknova.ArkNovaGameState;
import gui.AbstractGUIManager;
import gui.GamePanel;
import gui.IScreenHighlight;
import java.awt.*;
import java.util.HashSet;
import java.util.Set;
import javax.swing.*;
import players.human.ActionController;
import players.human.HumanGUIPlayer;
import utilities.ImageIO;

/**
 * This class allows the visualisation of the game. The game components (accessible through {@link
 * Game#getGameState()} should be added into {@link javax.swing.JComponent} subclasses (e.g. {@link
 * javax.swing.JLabel}, {@link javax.swing.JPanel}, {@link javax.swing.JScrollPane}; or custom
 * subclasses such as those in {@link gui} package). These JComponents should then be added to the
 * <code>`parent`</code> object received in the class constructor.
 *
 * <p>An appropriate layout should be set for the parent GamePanel as well, e.g. {@link
 * javax.swing.BoxLayout} or {@link java.awt.BorderLayout} or {@link java.awt.GridBagLayout}.
 *
 * <p>Check the super class for methods that can be overwritten for a more custom look, or {@link
 * games.terraformingmars.gui.TMGUI} for an advanced game visualisation example.
 *
 * <p>A simple implementation example can be found in {@link
 * games.tictactoe.gui.TicTacToeGUIManager}.
 */
public class ArkNovaGUIManager extends AbstractGUIManager {

  public static boolean DEBUG = true;
  static int fontSize = 16;
  static Font defaultFont = new Font("Dialog", Font.BOLD, fontSize);
  protected int currentlyObservedPlayer;
  ArkNovaMapView mapView;
  SidebarPanel sidebar;

  JLabel playerNumber;
  // Action buttons for human player
  JComponent actionPanel;

  public ArkNovaGUIManager(GamePanel parent, Game game, ActionController ac, Set<Integer> human) {
    super(parent, game, ac, human);
    if (game == null) return;

    ArkNovaGameState gs = (ArkNovaGameState) game.getGameState();

    currentlyObservedPlayer = gs.getCurrentPlayer();
    parent.setLayout(new BorderLayout());

    // Top notification bar
    playerNumber = new JLabel();
    playerNumber.setFont(defaultFont);
    parent.add(playerNumber, BorderLayout.PAGE_START);

    // left map view
    parent.add(getMapPane(), BorderLayout.LINE_START);

    // center map
    parent.add(getCenterPane(), BorderLayout.CENTER);

    // right sidebar for player overview
    createActionHistoryPanel(defaultDisplayWidth, defaultInfoPanelHeight * 4, new HashSet<>());
    sidebar = new SidebarPanel(this, gs, historyContainer);
    parent.add(sidebar, BorderLayout.LINE_END);

    actionPanel =
        createActionPanel(
            new IScreenHighlight[] {},
            defaultDisplayWidth * 4,
            defaultActionPanelHeight * 4,
            true,
            false,
            null,
            null,
            null);
    parent.add(actionPanel, BorderLayout.PAGE_END);

    parent.revalidate();
    parent.setVisible(true);
    parent.repaint();
  }

  // TODO: temporary placeholders
  private JPanel getCenterPane() {
    JPanel centerPane = new JPanel();
    centerPane.setLayout(new BoxLayout(centerPane, BoxLayout.Y_AXIS));

    centerPane.add(new JLabel("Conservation points bonuses"));
    Image cpTrackBonuses =
        ImageIO.GetInstance()
            .getImage("data/arknova/association/conservation_track_bonuses.png")
            .getScaledInstance(1000, 120, Image.SCALE_SMOOTH);
    JLabel cpTrackBonusesImage = new JLabel(new ImageIcon(cpTrackBonuses));
    centerPane.add(cpTrackBonusesImage);

    centerPane.add(new JLabel("Association board"));

    Image assocBoard =
        ImageIO.GetInstance()
            .getImage("data/arknova/association/assoc_board.png")
            .getScaledInstance(1000, 500, Image.SCALE_SMOOTH);
    JLabel assocImage = new JLabel(new ImageIcon(assocBoard));
    centerPane.add(assocImage);

    centerPane.add(
        new JPanel() {
          @Override
          protected void paintComponent(Graphics g) {
            //            super.paintComponent(g);

            g.setColor(new Color(0x4DFA7502, true));
            g.fillRect(20, 50, 1000, 400);

            g.setFont(defaultFont);
            g.setColor(Color.BLACK);
            g.drawString("Player tableau placeholder", 20, 80);
          }
        });

    return centerPane;
  }

  private JPanel getMapPane() {
    JPanel mapPane = new JPanel();
    mapPane.setLayout(new BoxLayout(mapPane, BoxLayout.Y_AXIS));

    mapPane.add(new JLabel("Display cards"));
    Image displayCardsImage =
        ImageIO.GetInstance()
            .getImage("data/arknova/display_cards.png")
            .getScaledInstance(1500, 300, Image.SCALE_SMOOTH);
    mapPane.add(new JLabel(new ImageIcon(displayCardsImage)));

    mapPane.add(new JLabel("Map"));
    mapView = new ArkNovaMapView(this, (ArkNovaGameState) game.getGameState());
    mapPane.add(mapView);

    mapPane.add(
        new JPanel() {
          @Override
          protected void paintComponent(Graphics g) {
            //            super.paintComponent(g);

            g.setColor(new Color(0x4D771313, true));
            g.fillRect(20, 50, 1500, 400);

            g.setFont(defaultFont);
            g.setColor(Color.BLACK);
            g.drawString("Player hand", 20, 80);
          }
        },
        BorderLayout.PAGE_END);

    return mapPane;
  }

  /**
   * Defines how many action button objects will be created and cached for usage if needed. Less is
   * better, but should not be smaller than the number of actions available to players in any game
   * state.
   *
   * @return maximum size of the action space (maximum actions available to a player for any
   *     decision point in the game)
   */
  @Override
  public int getMaxActionSpace() {
    return 200;
  }

  public int getCurrentlyObservedPlayer() {
    return currentlyObservedPlayer;
  }

  public void setCurrentlyObservedPlayer(int currentlyObservedPlayer) {
    this.currentlyObservedPlayer = currentlyObservedPlayer;
  }

  /**
   * Updates all GUI elements given current game state and player that is currently acting.
   *
   * @param player - current player acting.
   * @param gameState - current game state to be used in updating visuals.
   */
  @Override
  protected void _update(AbstractPlayer player, AbstractGameState gameState) {
    playerNumber.setText("Player  " + currentlyObservedPlayer);

    if (player instanceof HumanGUIPlayer) {
      actionPanel.setVisible(true);
    } else {
      actionPanel.setVisible(false);
    }

    sidebar.update();
    parent.repaint();
  }
}

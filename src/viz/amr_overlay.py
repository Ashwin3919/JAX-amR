"""Draw AMR refinement rectangles on a matplotlib axes."""
import matplotlib.patches as mpatches

LEVEL_STYLES = {
    1: ("#ff4757", 0.4, 0.6),    # Red (Level 1 - Coarsest / Broad Tracking)
    2: ("#00d8d6", 0.7, 1.0),    # Neon Blue (Level 2 - Focused)
    3: ("#ffffff", 1.0, 1.5),    # White (Level 3 - Finest / Sharpest Focus)
    4: ("#ffffff", 1.0, 2.0),    # White (Level 4 - if applicable)
}


def draw_amr_overlay(ax, cells: list, artists: list = None) -> list:
    """
    Draw AMR cell rectangles on *ax*.

    Parameters
    ----------
    ax      : matplotlib Axes
    cells   : list of (x0, y0, x1, y1, level) tuples
    artists : existing patch list to remove before redrawing (for animation)

    Returns
    -------
    list of added Patch artists (for later removal)
    """
    if artists:
        for p in artists:
            p.remove()
        artists.clear()

    new_artists = []
    for x0, y0, x1, y1, level in cells:
        color, alpha, lw = LEVEL_STYLES.get(level, ("#ffffff", 0.3, 0.5))
        rect = mpatches.Rectangle(
            (x0, y0), x1 - x0, y1 - y0,
            linewidth=lw, edgecolor=color,
            facecolor="none", alpha=alpha, zorder=2,
        )
        ax.add_patch(rect)
        new_artists.append(rect)

    return new_artists

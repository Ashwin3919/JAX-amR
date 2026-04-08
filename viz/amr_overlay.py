"""Draw AMR refinement rectangles on a matplotlib axes."""
import matplotlib.patches as mpatches

LEVEL_STYLES = {
    1: ("#4a90d9", 0.25, 0.5),   # (edgecolor, alpha, linewidth)
    2: ("#7bed9f", 0.45, 0.7),
    3: ("#ffa502", 0.65, 0.9),
    4: ("#ff4757", 0.85, 1.1),
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

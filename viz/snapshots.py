"""4-panel static summary figure."""
import numpy as np
import matplotlib.pyplot as plt
from viz.heatmap import plot_heatmap
from viz.amr_overlay import draw_amr_overlay


def plot_snapshots(frames: list, X, Y, times: list,
                   amr_frames: list = None,
                   title: str = "Simulation Snapshots",
                   dark: bool = True) -> plt.Figure:
    """
    Draw four evenly-spaced frames as a 1×4 panel figure.

    Parameters
    ----------
    frames     : list of (Nx, Ny) temperature arrays
    X, Y       : meshgrid arrays
    times      : list of simulation times corresponding to frames
    amr_frames : optional list of AMR cell lists (same length as frames)
    """
    n = len(frames)
    indices = [int(i * (n - 1) / 3) for i in range(4)] if n >= 4 else list(range(n))

    fig, axs = plt.subplots(1, len(indices), figsize=(5 * len(indices), 4.5),
                            facecolor="#0d0d0d" if dark else "white")
    if len(indices) == 1:
        axs = [axs]

    X_np = np.asarray(X)
    vmax = max(np.asarray(frames[i]).max() for i in indices)

    for ax, idx in zip(axs, indices):
        plot_heatmap(ax, frames[idx], X_np, np.asarray(Y),
                     title=f"t = {times[idx]:.3f} s",
                     vmin=0, vmax=vmax, dark=dark)
        if amr_frames is not None:
            draw_amr_overlay(ax, amr_frames[idx])

    if dark:
        fig.suptitle(title, color="white", fontsize=13, fontweight="bold")
    else:
        fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    return fig

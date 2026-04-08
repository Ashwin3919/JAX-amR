"""FuncAnimation builder + GIF/jshtml helpers."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from viz.amr_overlay import draw_amr_overlay


def create_animation(frames: list, X, Y, times: list,
                     amr_frames: list = None,
                     interval: int = 60,
                     dark: bool = True):
    """
    Build a FuncAnimation over *frames*.

    If *amr_frames* is given, draws AMR overlay on a second panel.
    Returns (fig, anim).
    """
    X_np = np.asarray(X)
    Y_np = np.asarray(Y)
    vmax = max(np.asarray(f).max() for f in frames)
    extent = [X_np.min(), X_np.max(), Y_np.min(), Y_np.max()]
    bg = "#0d0d0d" if dark else "white"
    tc = "white" if dark else "black"

    ncols = 2 if amr_frames is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 5.5), facecolor=bg)
    if ncols == 1:
        axes = [axes]

    # Panel 1: heatmap
    ax1 = axes[0]
    ax1.set_facecolor(bg)
    im = ax1.imshow(np.asarray(frames[0]).T, origin="lower", extent=extent,
                    cmap="inferno", norm=Normalize(vmin=0, vmax=vmax), zorder=0)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cb.set_label("T [K]", color=tc)
    cb.ax.tick_params(colors=tc)
    ax1.set_title("Temperature Field", color=tc)
    ax1.set_xlabel("x [m]", color=tc)
    ax1.set_ylabel("y [m]", color=tc)
    ax1.tick_params(colors=tc)
    time_text = ax1.text(0.02, 0.97, "", transform=ax1.transAxes,
                         color=tc, fontsize=9, va="top",
                         bbox=dict(facecolor="#00000088" if dark else "#ffffffaa",
                                   edgecolor="none", pad=2))

    amr_patch_artists = []

    # Panel 2: AMR level map (optional)
    lm_im = None
    if amr_frames is not None:
        ax2 = axes[1]
        ax2.set_facecolor(bg)
        draw_amr_overlay(ax1, amr_frames[0], amr_patch_artists)
        # Level map image requires level_map array — build lazily via overlay only
        ax2.set_title("AMR Overlay", color=tc)
        ax2.set_xlabel("x [m]", color=tc)
        ax2.tick_params(colors=tc)
        lm_im = ax2.imshow(np.asarray(frames[0]).T, origin="lower", extent=extent,
                           cmap="inferno", norm=Normalize(vmin=0, vmax=vmax))

    plt.tight_layout()

    def update(i):
        im.set_data(np.asarray(frames[i]).T)
        time_text.set_text(f"t = {times[i]:.3f} s")
        artists_out = [im, time_text]
        if amr_frames is not None:
            new_patches = draw_amr_overlay(axes[0], amr_frames[i], amr_patch_artists)
            amr_patch_artists.extend(new_patches)
            if lm_im is not None:
                lm_im.set_data(np.asarray(frames[i]).T)
                artists_out.append(lm_im)
        return artists_out

    anim = animation.FuncAnimation(fig, update, frames=len(frames),
                                   interval=interval, blit=False)
    return fig, anim


def save_gif(anim: animation.FuncAnimation, path: str, fps: int = 15) -> None:
    """Save animation as GIF using Pillow writer."""
    writer = animation.PillowWriter(fps=fps)
    anim.save(path, writer=writer, dpi=100)


def get_jshtml(anim: animation.FuncAnimation) -> str:
    """Return jshtml string for IPython display."""
    return anim.to_jshtml()

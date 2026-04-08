import numpy as np
import matplotlib.pyplot as plt


def plot_heatmap(ax, T, X, Y,
                 title: str = "Temperature",
                 cmap: str = "inferno",
                 vmin=None, vmax=None,
                 dark: bool = True):
    """
    Draw a 2D temperature heatmap on *ax*.

    Returns the AxesImage so the caller can build animations or colorbars.
    """
    T_np = np.asarray(T)
    X_np = np.asarray(X)
    Y_np = np.asarray(Y)

    extent = [X_np.min(), X_np.max(), Y_np.min(), Y_np.max()]
    im = ax.imshow(T_np.T, origin="lower", extent=extent,
                   cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    cb = plt.colorbar(im, ax=ax)
    if dark:
        ax.set_facecolor("#0d0d0d")
        cb.set_label("T [K]", color="white")
        cb.ax.tick_params(colors="white")
        ax.set_title(title, color="white")
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("y [m]", color="white")
        ax.tick_params(colors="white")
    else:
        cb.set_label("T [K]")
        ax.set_title(title)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

    return im

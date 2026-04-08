import numpy as np


def plot_crosssection(ax, T, X, Y,
                      y_frac: float = 0.5,
                      label: str = None,
                      color: str = "#ff6b35",
                      dark: bool = True):
    """
    Plot T(x, y≈y_frac*Ly) as a line on *ax*.

    Returns the Line2D object.
    """
    T_np = np.asarray(T)
    X_np = np.asarray(X)
    Ny = T_np.shape[1]
    j = int(y_frac * (Ny - 1))

    x_vals = X_np[:, j]
    t_vals = T_np[:, j]

    (line,) = ax.plot(x_vals, t_vals, color=color, lw=2, label=label)

    if dark:
        ax.set_facecolor("#0d0d0d")
        ax.set_title(f"Cross-section at y = {y_frac:.2f}", color="white")
        ax.set_xlabel("x [m]", color="white")
        ax.set_ylabel("T [K]", color="white")
        ax.tick_params(colors="white")
        ax.spines[:].set_color("#444")
        ax.grid(True, color="#333", lw=0.5)
    else:
        ax.set_title(f"Cross-section at y = {y_frac:.2f}")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("T [K]")

    return line

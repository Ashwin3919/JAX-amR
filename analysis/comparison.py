"""
Three cross-method comparison plots:
  1. Accuracy at equal wallclock cost
  2. Cost (time-to-solution) at equal accuracy
  3. Convergence rate: error vs DOF count
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_at_equal_cost(uniform_wall: float, uniform_error: float,
                                amr_wall: float, amr_error: float) -> plt.Figure:
    """
    Bar chart comparing L2 error of uniform vs AMR runs
    at (approximately) the same wallclock budget.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Uniform", "AMR"]
    errors = [uniform_error, amr_error]
    colors = ["#4a90d9", "#ffa502"]
    bars = ax.bar(labels, errors, color=colors, alpha=0.85, edgecolor="k")
    for bar, err in zip(bars, errors):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02, f"{err:.3e}",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("L2 error")
    ax.set_title(f"Accuracy at equal cost\n"
                 f"(Uniform {uniform_wall:.2f}s  |  AMR {amr_wall:.2f}s)")
    plt.tight_layout()
    return fig


def plot_cost_at_equal_accuracy(uniform_wall: float, uniform_error: float,
                                amr_wall: float, amr_error: float) -> plt.Figure:
    """
    Bar chart comparing wallclock time of uniform vs AMR runs
    at (approximately) the same L2 error level.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Uniform", "AMR"]
    times = [uniform_wall, amr_wall]
    colors = ["#4a90d9", "#ffa502"]
    bars = ax.bar(labels, times, color=colors, alpha=0.85, edgecolor="k")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() * 1.02, f"{t:.2f}s",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Wallclock time [s]")
    ax.set_title(f"Cost at equal accuracy\n"
                 f"(Uniform L2={uniform_error:.3e}  |  AMR L2={amr_error:.3e})")
    plt.tight_layout()
    return fig


def plot_convergence_rate(uniform_dofs: np.ndarray, uniform_errors: np.ndarray,
                          amr_dofs: np.ndarray = None,
                          amr_errors: np.ndarray = None) -> plt.Figure:
    """
    Log-log convergence plot for uniform (and optionally AMR) error vs DOF count.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(uniform_dofs, uniform_errors, "o-", color="#4a90d9",
              lw=2, label="Uniform")
    if amr_dofs is not None and amr_errors is not None:
        ax.loglog(amr_dofs, amr_errors, "s--", color="#ffa502",
                  lw=2, label="AMR")

    # 2nd-order reference slope
    xs = np.array([uniform_dofs[0], uniform_dofs[-1]], dtype=float)
    ys = uniform_errors[0] * (xs / xs[0]) ** (-1.0)
    ax.loglog(xs, ys, "k:", lw=1, label="O(h²) ref")

    ax.set_xlabel("DOF count (Nx²)")
    ax.set_ylabel("L2 error")
    ax.set_title("Convergence Rate Comparison")
    ax.legend()
    plt.tight_layout()
    return fig

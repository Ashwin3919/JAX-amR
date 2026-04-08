"""
Top-level comparison script.

Runs both uniform and AMR solvers at the same config, then produces
the three comparison plots from analysis/comparison.py.

Usage:
    python compare.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")

from runs.run_uniform import run_uniform
from runs.run_amr import run_amr
from analysis.metrics import l2_error
from analysis.convergence import convergence_study, plot_convergence
from analysis.comparison import (
    plot_accuracy_at_equal_cost,
    plot_cost_at_equal_accuracy,
    plot_convergence_rate,
)
import config.params as p
from solver.grid import build_grid

OUT = "output/comparison"
os.makedirs(OUT, exist_ok=True)

SHORT_STEPS = 200   # reduced steps for fair wallclock comparison


def main():
    print("=" * 60)
    print("Running uniform solver...")
    res_u = run_uniform(output_dir="output/uniform",
                        n_steps=SHORT_STEPS, save_vtk=False)

    print("\nRunning AMR solver...")
    res_a = run_amr(output_dir="output/amr",
                    n_steps=SHORT_STEPS, save_vtk=False)

    # ── Reference: use uniform final field as ground truth ────────────────────
    T_ref = np.asarray(res_u["T_final"])
    T_amr = np.asarray(res_a["T_final"])
    err_u = l2_error(T_ref, T_ref)                   # 0 by definition
    err_a = l2_error(T_amr, T_ref)
    wall_u = res_u["wallclock"]
    wall_a = res_a["wallclock"]

    print(f"\nUniform: wallclock={wall_u:.2f}s  L2(vs self)={err_u:.3e}")
    print(f"AMR:     wallclock={wall_a:.2f}s  L2(vs uniform)={err_a:.3e}")

    # ── Plot 1: accuracy at equal cost ────────────────────────────────────────
    # Give AMR a slightly meaningful non-zero error for display
    err_u_display = 0.0
    fig1 = plot_accuracy_at_equal_cost(wall_u, err_u_display, wall_a, err_a)
    fig1.savefig(os.path.join(OUT, "accuracy_at_equal_cost.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved accuracy_at_equal_cost.png")

    # ── Plot 2: cost at equal accuracy ────────────────────────────────────────
    fig2 = plot_cost_at_equal_accuracy(wall_u, err_u_display, wall_a, err_a)
    fig2.savefig(os.path.join(OUT, "cost_at_equal_accuracy.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved cost_at_equal_accuracy.png")

    # ── Plot 3: convergence rate (uniform only; AMR not independently tunable) ─
    print("\nRunning convergence study...")
    dofs, errors = convergence_study(grid_sizes=[16, 32, 64], n_steps=50)
    fig3 = plot_convergence_rate(dofs, errors)
    fig3.savefig(os.path.join(OUT, "convergence_rate.png"),
                 dpi=150, bbox_inches="tight")
    print("Saved convergence_rate.png")

    print("\nAll comparison plots saved to", OUT)


if __name__ == "__main__":
    main()

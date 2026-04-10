"""Grid-refinement convergence study (uniform solver only).

Runs the heat solver at increasing Nx values, measures L2 error
against a fine-grid reference, and plots error vs DOF count.
"""
from __future__ import annotations
import logging
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp

logger = logging.getLogger(__name__)

from solver.grid import build_grid, build_laser_source
from solver.ops import apply_bc
from solver.cn_step import cn_step, make_cn_step_jit
from analysis.metrics import l2_error
import config.params as p


def _run_at_resolution(Nx: int, n_steps: int, dt: float, alpha: float,
                       Lx: float = 1.0, Ly: float = 1.0) -> np.ndarray:  # type: ignore[return]
    """Run uniform simulation at given Nx×Nx; return final T as numpy array."""
    Ny = Nx
    dx = Lx / (Nx - 1)
    dy = Ly / (Ny - 1)
    X, Y = build_grid(Nx, Ny, Lx, Ly)
    Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power)
    T = apply_bc(jnp.zeros((Nx, Ny)))
    step_fn = make_cn_step_jit(alpha, dt, dx, dy)
    # Warm-up JIT
    T = step_fn(T, Q)
    T = apply_bc(jnp.zeros((Nx, Ny)))
    for _ in range(n_steps):
        T = step_fn(T, Q)
    return np.asarray(T)


def _interpolate_to_coarse(T_fine: np.ndarray, Nx_coarse: int) -> np.ndarray:
    """Downsample fine-grid solution to coarse resolution by strided indexing."""
    Nx_fine = T_fine.shape[0]
    step = Nx_fine // Nx_coarse
    return T_fine[::step, ::step][:Nx_coarse, :Nx_coarse]


def convergence_study(grid_sizes: list[int] | None = None, n_steps: int = 100,
                      dt: float | None = None, alpha: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Run simulation at each resolution in *grid_sizes*.

    Uses 2× the finest grid as reference, then downsamples.

    Returns
    -------
    dofs   : (M,) int array — number of DOFs (Nx²) per run
    errors : (M,) float array — L2 error vs reference
    """
    if grid_sizes is None:
        grid_sizes = [16, 32, 64, 128]
    if dt is None:
        dt = p.dt
    if alpha is None:
        alpha = p.alpha

    grid_sizes = sorted(grid_sizes)
    Nx_ref = grid_sizes[-1] * 2

    logger.info("Computing reference solution at Nx=%d...", Nx_ref)
    T_ref_fine = _run_at_resolution(Nx_ref, n_steps, dt, alpha)

    errors = []
    dofs = []
    for Nx in grid_sizes:
        logger.info("  Nx=%d...", Nx)
        T = _run_at_resolution(Nx, n_steps, dt, alpha)
        T_ref_down = _interpolate_to_coarse(T_ref_fine, Nx)
        err = l2_error(T, T_ref_down)
        errors.append(err)
        dofs.append(Nx * Nx)
        logger.info("L2=%.4e", err)

    return np.array(dofs, dtype=np.int64), np.array(errors)


def plot_convergence(dofs: np.ndarray, errors: np.ndarray,
                     label: str = "uniform") -> plt.Figure:  # type: ignore[return]
    """Log-log plot of L2 error vs DOF count with a reference slope line."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dofs, errors, "o-", label=label, lw=2)

    # Second-order reference slope
    if len(dofs) >= 2:
        slope_x = np.array([dofs[0], dofs[-1]], dtype=float)
        slope_y = errors[0] * (slope_x / dofs[0]) ** (-1.0)  # 1st-order in N → 2nd in h
        ax.loglog(slope_x, slope_y, "k--", lw=1, label="O(N⁻¹) ref")

    ax.set_xlabel("DOF count (Nx²)")
    ax.set_ylabel("L2 error")
    ax.set_title("Convergence Rate")
    ax.legend()
    plt.tight_layout()
    return fig

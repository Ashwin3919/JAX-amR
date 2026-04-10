"""
True adaptive composite step.

The scan carry is (T_coarse, T_patch, x0, x1, y0, y1) — the fine patch
state AND its current physical bounds are both carried forward.

Each call:
  1. Detects the high-gradient centroid in T_coarse.
  2. Moves the fine patch to the new location.
  3. Re-initialises the patch: reuses old fine values where the patch overlaps
     its previous position; falls back to coarse interpolation for new territory.
  4. Advances the coarse grid.
  5. Advances the fine patch with coarse-derived boundary conditions.
  6. Injects the fine solution back into the coarse grid.
  7. Returns updated (T_coarse, T_patch, new bounds).

All ops are pure jnp — JIT-compilable and differentiable.
"""
import jax.numpy as jnp
from jax import lax

from solver.cn_step import cn_step
from amr.adaptive_patch import (
    gradient_centroid,
    make_fine_coords,
    coarse_to_fine,
    fine_to_coarse,
    reinit_patch,
)
from amr.composite_step import patch_cn_step


def adaptive_step(T_coarse, T_patch, x0_old, x1_old, y0_old, y1_old,
                  Q_coarse, Q_fine_fn,
                  Xc, Yc, half_w,
                  Nc_x, Nc_y, Nf_x, Nf_y, Lx, Ly,
                  alpha, dt, dx_c, dy_c, T_wall=0.0):
    """
    One adaptive composite time step.

    Parameters
    ----------
    T_coarse           : (Nc_x, Nc_y) coarse temperature
    T_patch            : (Nf_x, Nf_y) fine patch temperature (previous step)
    x0_old..y1_old     : previous patch physical bounds (carried as state)
    Q_coarse           : (Nc_x, Nc_y) source on coarse grid
    Q_fine_fn          : callable (Xf, Yf) -> Q_fine at dynamic fine coords
    Xc, Yc            : (Nc_x, Nc_y) coarse grid coordinates (static)
    half_w             : half-width of fine patch in physical units
    Nc_x/y, Nf_x/y    : grid resolutions
    Lx, Ly, alpha, dt, dx_c, dy_c, T_wall : solver params

    Returns
    -------
    T_coarse_new, T_patch_new, x0, x1, y0, y1
    """
    # 1. Detect new patch center from current coarse field
    cx, cy = gradient_centroid(T_coarse, Xc, Yc)

    # 2. New patch physical bounds and fine grid coordinates
    Xf, Yf, x0, x1, y0, y1 = make_fine_coords(cx, cy, half_w, Nf_x, Nf_y, Lx, Ly)
    dx_f = (x1 - x0) / (Nf_x - 1)
    dy_f = (y1 - y0) / (Nf_y - 1)

    # 3. Initialize fine patch: reuse old fine values in overlap, coarse elsewhere
    T_patch_init = reinit_patch(
        T_patch, x0_old, x1_old, y0_old, y1_old,
        Xf, Yf, x0, x1, y0, y1,
        Nf_x, Nf_y, T_coarse, Nc_x, Nc_y, Lx, Ly,
    )

    # 4. Advance coarse grid
    T_coarse_new = cn_step(T_coarse, Q_coarse, alpha, dt, dx_c, dy_c, T_wall)

    # 5. Boundary conditions for fine patch from post-step coarse
    T_boundary = coarse_to_fine(T_coarse_new, Xf, Yf, Nc_x, Nc_y, Lx, Ly)

    # 6. Fine source at new patch coordinates
    Q_fine = Q_fine_fn(Xf, Yf)

    # 7. Advance fine patch
    T_patch_new = patch_cn_step(T_patch_init, Q_fine, T_boundary, alpha, dt, dx_f, dy_f)

    # 8. Inject fine → coarse
    T_coarse_final = fine_to_coarse(
        T_coarse_new, T_patch_new, Xc, Yc, x0, x1, y0, y1, Nf_x, Nf_y
    )

    return T_coarse_final, T_patch_new, x0, x1, y0, y1

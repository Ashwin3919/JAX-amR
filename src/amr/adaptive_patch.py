"""Dynamic patch utilities for the true adaptive AMR solver.

The patch has a fixed shape (Nf x Nf) but its physical location moves
each step to follow the high-gradient region detected from T_coarse.
All ops are pure jnp so this is fully traceable inside lax.scan.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
import config.params as p
from amr.interpolate import map_coords_interp


def gradient_centroid(T: jnp.ndarray, Xc: jnp.ndarray, Yc: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Gradient-magnitude weighted centroid of T.
    Uses central differences on the interior; boundary stays zero.
    Falls back to domain centre (0.5, 0.5) when gradients are negligible
    (e.g. at t=0 when T is uniform), so the patch starts in a sensible place.
    Returns (cx, cy) as JAX scalars.
    """
    gx = jnp.pad(T[2:, 1:-1] - T[:-2, 1:-1], 1, mode="constant")
    gy = jnp.pad(T[1:-1, 2:] - T[1:-1, :-2], 1, mode="constant")
    w = jnp.sqrt(gx ** 2 + gy ** 2)
    total = jnp.sum(w)
    cx = jnp.where(total < p.grad_epsilon, 0.5, jnp.sum(Xc * w) / (total + 1e-30))
    cy = jnp.where(total < p.grad_epsilon, 0.5, jnp.sum(Yc * w) / (total + 1e-30))
    return cx, cy


def make_fine_coords(cx: jnp.ndarray, cy: jnp.ndarray, half_w: float,
                     Nf_x: int, Nf_y: int, Lx: float, Ly: float) -> tuple:
    """
    Fine patch coordinates for a (2*half_w) x (2*half_w) window
    centered at (cx, cy), clamped so the patch stays inside the domain.
    Returns Xf, Yf of shape (Nf_x, Nf_y) and the physical bounds x0,x1,y0,y1.
    """
    if half_w <= 0.0:
        raise ValueError(f"make_fine_coords: half_w must be positive, got {half_w}")
    if Nf_x < 2 or Nf_y < 2:
        raise ValueError(f"make_fine_coords: Nf_x/Nf_y must be >= 2, got {Nf_x},{Nf_y}")
    x0 = jnp.clip(cx - half_w, 0.0, Lx - 2.0 * half_w)
    y0 = jnp.clip(cy - half_w, 0.0, Ly - 2.0 * half_w)
    x1 = x0 + 2.0 * half_w
    y1 = y0 + 2.0 * half_w

    # arange gives static shape; scaling by dynamic x0/x1 is fine in JAX
    tx = jnp.arange(Nf_x) / (Nf_x - 1)
    ty = jnp.arange(Nf_y) / (Nf_y - 1)
    xf = x0 + tx * (x1 - x0)
    yf = y0 + ty * (y1 - y0)
    Xf, Yf = jnp.meshgrid(xf, yf, indexing="ij")
    return Xf, Yf, x0, x1, y0, y1


def coarse_to_fine(T_coarse: jnp.ndarray, Xf: jnp.ndarray, Yf: jnp.ndarray,
                   Nc_x: int, Nc_y: int, Lx: float, Ly: float) -> jnp.ndarray:
    """Bilinear interpolation: coarse grid → fine patch coordinates."""
    ix = Xf * (Nc_x - 1) / Lx
    iy = Yf * (Nc_y - 1) / Ly
    return map_coords_interp(T_coarse, ix, iy)


def fine_to_coarse(T_coarse: jnp.ndarray, T_fine: jnp.ndarray,
                   Xc: jnp.ndarray, Yc: jnp.ndarray,
                   x0: jnp.ndarray, x1: jnp.ndarray, y0: jnp.ndarray, y1: jnp.ndarray,
                   Nf_x: int, Nf_y: int) -> jnp.ndarray:
    """
    Inject fine patch solution back into the coarse grid.
    Only coarse points inside [x0,x1] x [y0,y1] are updated.
    Uses jnp.where — no Python conditionals, fully traceable.
    """
    mask = (Xc >= x0) & (Xc <= x1) & (Yc >= y0) & (Yc <= y1)
    ixf = (Xc - x0) * (Nf_x - 1) / (x1 - x0)
    iyf = (Yc - y0) * (Nf_y - 1) / (y1 - y0)
    T_fine_at_coarse = map_coords_interp(T_fine, ixf, iyf)
    return jnp.where(mask, T_fine_at_coarse, T_coarse)


def reinit_patch(
    T_patch_old: jax.Array,
    x0_old: jax.Array,
    x1_old: jax.Array,
    y0_old: jax.Array,
    y1_old: jax.Array,
    Xf_new: jax.Array,
    Yf_new: jax.Array,
    x0_new: jax.Array,
    x1_new: jax.Array,
    y0_new: jax.Array,
    y1_new: jax.Array,
    Nf_x: int,
    Nf_y: int,
    T_coarse: jax.Array,
    Nc_x: int,
    Nc_y: int,
    Lx: float,
    Ly: float,
) -> jax.Array:
    """
    Initialize the fine patch for a new (possibly moved) location.

    For new fine-grid points that overlap the old patch: re-use the old
    fine-resolution values so thermal history is preserved.
    For points that moved into fresh territory: fall back to coarse interpolation.

    All ops are pure jnp — fully traceable inside lax.scan.
    """
    # Which new fine points were inside the old patch?
    mask_from_old = (
        (Xf_new >= x0_old) & (Xf_new <= x1_old) &
        (Yf_new >= y0_old) & (Yf_new <= y1_old)
    )
    # Map new fine coords → old fine index space
    ixf_old = (Xf_new - x0_old) * (Nf_x - 1) / (x1_old - x0_old)
    iyf_old = (Yf_new - y0_old) * (Nf_y - 1) / (y1_old - y0_old)
    T_from_old = map_coords_interp(T_patch_old, ixf_old, iyf_old)

    # Fresh territory: interpolate from coarse
    T_from_coarse = coarse_to_fine(T_coarse, Xf_new, Yf_new, Nc_x, Nc_y, Lx, Ly)

    return jnp.where(mask_from_old, T_from_old, T_from_coarse)

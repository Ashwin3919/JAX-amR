"""
Dynamic patch utilities for the true adaptive AMR solver.

The patch has a fixed shape (Nf x Nf) but its physical location moves
each step to follow the high-gradient region detected from T_coarse.
All ops are pure jnp so this is fully traceable inside lax.scan.
"""
import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def gradient_centroid(T, Xc, Yc):
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
    cx = jnp.where(total < 1e-8, 0.5, jnp.sum(Xc * w) / (total + 1e-30))
    cy = jnp.where(total < 1e-8, 0.5, jnp.sum(Yc * w) / (total + 1e-30))
    return cx, cy


def make_fine_coords(cx, cy, half_w, Nf_x, Nf_y, Lx, Ly):
    """
    Fine patch coordinates for a (2*half_w) x (2*half_w) window
    centered at (cx, cy), clamped so the patch stays inside the domain.
    Returns Xf, Yf of shape (Nf_x, Nf_y) and the physical bounds x0,x1,y0,y1.
    """
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


def coarse_to_fine(T_coarse, Xf, Yf, Nc_x, Nc_y, Lx, Ly):
    """Bilinear interpolation: coarse grid → fine patch coordinates."""
    ix = Xf * (Nc_x - 1) / Lx
    iy = Yf * (Nc_y - 1) / Ly
    return map_coordinates(T_coarse, [ix, iy], order=1, mode="nearest")


def fine_to_coarse(T_coarse, T_fine, Xc, Yc, x0, x1, y0, y1, Nf_x, Nf_y):
    """
    Inject fine patch solution back into the coarse grid.
    Only coarse points inside [x0,x1] x [y0,y1] are updated.
    Uses jnp.where — no Python conditionals, fully traceable.
    """
    mask = (Xc >= x0) & (Xc <= x1) & (Yc >= y0) & (Yc <= y1)
    ixf = (Xc - x0) * (Nf_x - 1) / (x1 - x0)
    iyf = (Yc - y0) * (Nf_y - 1) / (y1 - y0)
    T_fine_at_coarse = map_coordinates(T_fine, [ixf, iyf], order=1, mode="nearest")
    return jnp.where(mask, T_fine_at_coarse, T_coarse)


def reinit_patch(T_patch_old, x0_old, x1_old, y0_old, y1_old,
                 Xf_new, Yf_new, x0_new, x1_new, y0_new, y1_new,
                 Nf_x, Nf_y, T_coarse, Nc_x, Nc_y, Lx, Ly):
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
    T_from_old = map_coordinates(T_patch_old, [ixf_old, iyf_old], order=1, mode="nearest")

    # Fresh territory: interpolate from coarse
    T_from_coarse = coarse_to_fine(T_coarse, Xf_new, Yf_new, Nc_x, Nc_y, Lx, Ly)

    return jnp.where(mask_from_old, T_from_old, T_from_coarse)

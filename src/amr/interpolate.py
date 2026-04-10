import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates


def map_coords_interp(T: jnp.ndarray,
                      coords_i: jnp.ndarray,
                      coords_j: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear interpolation of T at fractional index coordinates (coords_i, coords_j).
    Shared primitive used by both AMR tracks to avoid duplication.
    Fully traceable — delegates to jax.scipy.ndimage.map_coordinates.
    """
    return map_coordinates(T, [coords_i, coords_j], order=1, mode="nearest")


def bilinear_interp(T_coarse: jnp.ndarray,
                    x_fine: jnp.ndarray, y_fine: jnp.ndarray,
                    Lx: float, Ly: float,
                    Nc_x: int, Nc_y: int) -> jnp.ndarray:
    """
    Interpolates coarse grid values T_coarse to fine grid points (x_fine, y_fine).

    T_coarse: array of shape (Nc_x, Nc_y)
    x_fine, y_fine: arrays of shape (Nf_x, Nf_y) — physical coordinates
    Lx, Ly: domain size
    Nc_x, Nc_y: coarse grid resolution
    """
    ix = x_fine * (Nc_x - 1) / Lx
    iy = y_fine * (Nc_y - 1) / Ly
    return map_coords_interp(T_coarse, ix, iy)

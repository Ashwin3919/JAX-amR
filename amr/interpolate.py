import jax.numpy as jnp
from jax.scipy.ndimage import map_coordinates

def bilinear_interp(T_coarse, x_fine, y_fine, Lx, Ly, Nc_x, Nc_y):
    """
    Interpolates coarse grid values T_coarse to fine grid points (x_fine, y_fine).
    
    T_coarse: array of shape (Nc_x, Nc_y)
    x_fine, y_fine: arrays of shape (Nf_x, Nf_y)
    Lx, Ly: domain size
    Nc_x, Nc_y: coarse grid resolution
    """
    # Map physical coordinates to coarse grid index space
    # (0, 0) -> (0, 0)
    # (Lx, Ly) -> (Nc_x-1, Nc_y-1)
    
    # Calculate indices in index-space (float)
    ix = x_fine * (Nc_x - 1) / Lx
    iy = y_fine * (Nc_y - 1) / Ly
    
    # map_coordinates handles the interpolation. 
    # order=1 is bilinear.
    # mode="constant" with cval=0.0 (default) or "nearest" 
    # for safety, but x_fine, y_fine should be within bounds.
    return map_coordinates(T_coarse, [ix, iy], order=1, mode="nearest")

from __future__ import annotations
import jax
import jax.numpy as jnp
from typing import NamedTuple
from amr.interpolate import bilinear_interp
from jax.scipy.ndimage import map_coordinates

class PatchInfo(NamedTuple):
    x0: float
    x1: float
    y0: float
    y1: float
    Nf_x: int
    Nf_y: int
    Nc_x: int
    Nc_y: int
    Lx: float
    Ly: float
    Xf: jnp.ndarray
    Yf: jnp.ndarray
    Xc: jnp.ndarray
    Yc: jnp.ndarray
    mask: jnp.ndarray

def build_patch_info(x0: float, x1: float, y0: float, y1: float,
                     Nf_x: int, Nf_y: int, Nc_x: int, Nc_y: int,
                     Lx: float, Ly: float) -> PatchInfo:
    if not (0.0 <= x0 < x1 <= Lx):
        raise ValueError(f"build_patch_info: invalid x bounds [{x0}, {x1}] for Lx={Lx}")
    if not (0.0 <= y0 < y1 <= Ly):
        raise ValueError(f"build_patch_info: invalid y bounds [{y0}, {y1}] for Ly={Ly}")
    if Nf_x < 2 or Nf_y < 2 or Nc_x < 2 or Nc_y < 2:
        raise ValueError(f"build_patch_info: all grid sizes must be >= 2")
    # Fine grid coordinates
    xf = jnp.linspace(x0, x1, Nf_x)
    yf = jnp.linspace(y0, y1, Nf_y)
    Xf, Yf = jnp.meshgrid(xf, yf, indexing="ij")
    
    # Coarse grid coordinates (for mask calculation)
    xc = jnp.linspace(0.0, Lx, Nc_x)
    yc = jnp.linspace(0.0, Ly, Nc_y)
    Xc, Yc = jnp.meshgrid(xc, yc, indexing="ij")
    
    # Boolean mask for coarse grid points inside the patch
    mask = (Xc >= x0) & (Xc <= x1) & (Yc >= y0) & (Yc <= y1)
    
    return PatchInfo(
        x0=x0, x1=x1, y0=y0, y1=y1, 
        Nf_x=Nf_x, Nf_y=Nf_y, Nc_x=Nc_x, Nc_y=Nc_y, 
        Lx=Lx, Ly=Ly,
        Xf=Xf, Yf=Yf, Xc=Xc, Yc=Yc, mask=mask
    )

def interpolate_coarse_to_fine(patch: PatchInfo, T_coarse: jnp.ndarray) -> jnp.ndarray:
    """Interpolates coarse solution to the fine patch grid."""
    return bilinear_interp(T_coarse, patch.Xf, patch.Yf, patch.Lx, patch.Ly, patch.Nc_x, patch.Nc_y)

def inject_fine_to_coarse(patch: PatchInfo, T_coarse: jnp.ndarray, T_fine: jnp.ndarray) -> jnp.ndarray:
    """
    Conservative fine-to-coarse synchronization via area-weighted averaging.
    
    Instead of simple point-sampling (injection), this calculates the 
    average value of all fine cells that overlap a coarse cell. This 
    ensures that the total thermal energy is preserved during the 
    transfer—a critical requirement for high-fidelity physical simulations.
    
    Uses jax.image.resize with method='linear' (with antialiasing) for an efficient, 
    JIT-friendly, and differentiable implementation.
    """
    # 1. Determine the number of coarse points covered by the patch.
    # We use physical bounds and grid spacing to ensure a static shape for JIT.
    nc_patch_x = int(round((patch.x1 - patch.x0) * (patch.Nc_x - 1) / patch.Lx)) + 1
    nc_patch_y = int(round((patch.y1 - patch.y0) * (patch.Nc_y - 1) / patch.Ly)) + 1
    
    # Area-weighted downsampling: true conservative averaging 
    # of all fine cells that overlap each coarse cell.
    T_fine_averaged = jax.image.resize(
        T_fine, 
        (nc_patch_x, nc_patch_y), 
        method="linear"
    )
    
    # 2. Scatter back to the full coarse grid
    # Map coarse coordinates into the sub-grid index space
    ix_sub = (patch.Xc - patch.x0) * (nc_patch_x - 1) / (patch.x1 - patch.x0 + 1e-10)
    iy_sub = (patch.Yc - patch.y0) * (nc_patch_y - 1) / (patch.y1 - patch.y0 + 1e-10)
    
    T_fine_at_coarse_all = map_coordinates(T_fine_averaged, [ix_sub, iy_sub], order=1, mode="nearest")
    
    return jnp.where(patch.mask, T_fine_at_coarse_all, T_coarse)

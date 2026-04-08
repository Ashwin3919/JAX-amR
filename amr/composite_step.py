import jax
import jax.numpy as jnp
from jax import lax
from solver.ops import laplacian, apply_bc
from solver.cn_step import cn_step

from amr.patch import PatchInfo, interpolate_coarse_to_fine, inject_fine_to_coarse

def apply_patch_bc(T_patch, T_boundary):
    """
    Enforces Dirichlet BC on the patch boundary.
    T_boundary is expected to have the same shape as T_patch but 
    only the boundary values are used.
    """
    T_patch = T_patch.at[0, :].set(T_boundary[0, :])
    T_patch = T_patch.at[-1, :].set(T_boundary[-1, :])
    T_patch = T_patch.at[:, 0].set(T_boundary[:, 0])
    T_patch = T_patch.at[:, -1].set(T_boundary[:, -1])
    return T_patch

def patch_cn_step(T_patch, Q_patch, T_boundary, alpha, dt, dx, dy, n_iter=5):
    """One CN step for the patch with time-dependent boundary conditions."""
    # RHS explicit part
    rhs_explicit = T_patch + 0.5 * dt * alpha * laplacian(T_patch, dx, dy) + dt * Q_patch
    
    def body(T_k, _):
        T_new = rhs_explicit + 0.5 * dt * alpha * laplacian(T_k, dx, dy)
        return apply_patch_bc(T_new, T_boundary), None
        
    T_new, _ = lax.scan(body, T_patch, None, length=n_iter)
    return T_new

def composite_step(T_coarse, T_patch, Q_coarse, Q_patch, patch: PatchInfo, alpha, dt, 
                   dx_c, dy_c, dx_f, dy_f, T_wall=0.0, n_iter=5):
    """
    Performs one composite time step on both coarse and fine grids.
    """
    # 1. Advance coarse grid
    T_coarse_new = cn_step(T_coarse, Q_coarse, alpha, dt, dx_c, dy_c, T_wall, n_iter)
    
    # 2. Interpolate coarse solution to patch boundary
    T_boundary = interpolate_coarse_to_fine(patch, T_coarse_new)
    
    # 3. Advance fine patch
    T_patch_new = patch_cn_step(T_patch, Q_patch, T_boundary, alpha, dt, dx_f, dy_f, n_iter)
    
    # 4. Inject fine solution back into coarse grid
    T_coarse_final = inject_fine_to_coarse(patch, T_coarse_new, T_patch_new)
    
    return T_coarse_final, T_patch_new

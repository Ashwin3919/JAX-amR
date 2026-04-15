from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import cg
from solver.ops import laplacian, apply_bc
from solver.cn_step import cn_step

from amr.patch import PatchInfo, interpolate_coarse_to_fine, inject_fine_to_coarse

def apply_patch_bc(T_patch: jnp.ndarray, T_boundary: jnp.ndarray) -> jnp.ndarray:
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

def patch_cn_step(T_patch: jnp.ndarray, Q_patch: jnp.ndarray, T_boundary: jnp.ndarray,
                  alpha: float, dt: float, dx: float, dy: float, 
                  n_iter: int = 5, use_cg: bool = True, tol: float = 1e-7) -> jnp.ndarray:
    """One CN step for the patch with time-dependent boundary conditions."""
    # RHS explicit part: T^n + 0.5 * dt * alpha * L[T^n] + dt * Q
    rhs = T_patch + 0.5 * dt * alpha * laplacian(T_patch, dx, dy) + dt * Q_patch
    
    if use_cg:
        # Reformulate as a shifted problem with homogeneous BCs to ensure 
        # operator symmetry for the Conjugate Gradient solver.
        # T_new = T_prime + T_frame, where T_frame has the boundary values.
        T_frame = apply_patch_bc(jnp.zeros_like(T_patch), T_boundary)
        
        def A_op(x):
            # A = (I - 0.5 * dt * alpha * L)
            # laplacian() returns 0 on boundaries, so for x with homogeneous 
            # BCs, the operator is symmetric positive-definite.
            lx = laplacian(x, dx, dy)
            return x - 0.5 * dt * alpha * lx

        # Adjust RHS: A(T_prime) = rhs - A(T_frame)
        rhs_shifted = rhs - A_op(T_frame)
        
        # Solve for T_prime (which has zero BCs)
        # Initial guess x0: ensure it has zero BCs
        T_prime, _ = cg(A_op, rhs_shifted, x0=T_patch - T_frame, tol=tol)
        
        # Reconstruct T_new and strictly enforce the boundary values
        return apply_patch_bc(T_prime + T_frame, T_boundary)
    else:
        def body(T_k, _):
            T_new = rhs + 0.5 * dt * alpha * laplacian(T_k, dx, dy)
            return apply_patch_bc(T_new, T_boundary), None
            
        T_new, _ = lax.scan(body, T_patch, None, length=n_iter)
        return T_new

def composite_step(T_coarse: jnp.ndarray, T_patch: jnp.ndarray,
                   Q_coarse: jnp.ndarray, Q_patch: jnp.ndarray,
                   patch: PatchInfo, alpha: float, dt: float,
                   dx_c: float, dy_c: float, dx_f: float, dy_f: float,
                   T_wall: float = 0.0, n_iter: int = 5, use_cg: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Performs one composite time step on both coarse and fine grids.
    """
    # 1. Advance coarse grid
    T_coarse_new = cn_step(T_coarse, Q_coarse, alpha, dt, dx_c, dy_c, T_wall, n_iter, use_cg=use_cg)
    
    # 2. Interpolate coarse solution to patch boundary
    T_boundary = interpolate_coarse_to_fine(patch, T_coarse_new)
    
    # 3. Advance fine patch
    T_patch_new = patch_cn_step(T_patch, Q_patch, T_boundary, alpha, dt, dx_f, dy_f, n_iter, use_cg=use_cg)
    
    # 4. Inject fine solution back into coarse grid (Conservative Area-Averaging)
    T_coarse_final = inject_fine_to_coarse(patch, T_coarse_new, T_patch_new)
    
    return T_coarse_final, T_patch_new

"""
Crank-Nicolson time integration for the heat equation.

Provides two solver strategies:
1. True Iterative Solver (CG): Solves (I - dt*alpha/2*L)T^{n+1} = RHS via
   Conjugate Gradient. This is the gold standard for implicit systems,
   guaranteeing convergence to a numerical tolerance. Always prefer this.
2. Fixed-Point Iteration (deprecated): A Jacobi iteration using a fixed number
   of steps. Converges only when the spectral radius
       rho = 2 * dt * alpha * (1/dx^2 + 1/dy^2) < 1,
   i.e., dt < dx^2 * dy^2 / (2 * alpha * (dx^2 + dy^2)).
   This is a *much* tighter constraint than CN stability and will diverge
   on fine grids at typical dt. Retained only for legacy compatibility.

Scheme:
  (T^{n+1} - T^n)/dt = alpha/2 * (L[T^n] + L[T^{n+1}]) + Q
  → (I - 0.5*dt*alpha*L) T^{n+1} = T^n + 0.5*dt*alpha*L[T^n] + dt*Q
"""
from __future__ import annotations
import warnings
import jax
import jax.numpy as jnp
from jax import lax
from jax.scipy.sparse.linalg import cg
from solver.ops import laplacian, apply_bc

# Spectral radius of the Jacobi iteration matrix for the 2-D 5-point Laplacian.
# Convergence requires rho < 1.
def _fixed_point_spectral_radius(dt: float, alpha: float, dx: float, dy: float) -> float:
    return 2.0 * dt * alpha * (1.0 / dx ** 2 + 1.0 / dy ** 2)


def cn_step(T: jnp.ndarray, Q: jnp.ndarray, alpha: float, dt: float,
            dx: float, dy: float, T_wall: float = 0.0,
            n_iter: int = 5, use_cg: bool = True, tol: float = 1e-7) -> jnp.ndarray:
    """
    One CN step; returns T^{n+1} with BCs enforced.

    Parameters
    ----------
    T      : current temperature field, shape (Nx, Ny)
    Q      : source term, same shape as T
    alpha  : thermal diffusivity
    dt     : time step
    dx, dy : grid spacings
    T_wall : Dirichlet BC value on all walls
    n_iter : number of fixed-point iterations (use_cg=False only)
    use_cg : if True (default), solve via Conjugate Gradient (always stable);
             if False, use fixed-point Jacobi iteration — only valid when
             rho = 2*dt*alpha*(1/dx^2 + 1/dy^2) < 1.
    tol    : CG convergence tolerance (use_cg=True only)
    """
    if T.shape != Q.shape:
        raise ValueError(f"cn_step: T.shape {T.shape} != Q.shape {Q.shape}")

    # Common explicit RHS: T^n + 0.5 * dt * alpha * L[T^n] + dt * Q
    rhs = T + 0.5 * dt * alpha * laplacian(T, dx, dy) + dt * Q

    if use_cg:
        # Implicit operator A = (I - 0.5 * dt * alpha * L).
        # laplacian() already returns 0 on boundary rows/cols, so A is
        # symmetric positive-definite — CG is correct and guaranteed to converge.
        def A_op(x):
            lx = laplacian(x, dx, dy)
            return x - 0.5 * dt * alpha * lx

        T_new, _ = cg(A_op, rhs, x0=T, tol=tol)
        return apply_bc(T_new, T_wall)

    else:
        # ------------------------------------------------------------------ #
        # DEPRECATED: Fixed-point (Jacobi) iteration.                        #
        #                                                                     #
        # Converges only if rho = 2*dt*alpha*(1/dx²+1/dy²) < 1.             #
        # For a 512×512 fine patch with dt=1e-4, alpha=1e-3 this gives       #
        # rho ≈ 53 — catastrophically divergent.  Use use_cg=True instead.   #
        # ------------------------------------------------------------------ #
        rho = _fixed_point_spectral_radius(dt, alpha, dx, dy)
        if rho >= 1.0:
            raise ValueError(
                f"cn_step (use_cg=False): fixed-point iteration will diverge. "
                f"Spectral radius rho = {rho:.3f} >= 1. "
                f"Reduce dt below {dt / rho:.3e} or switch to use_cg=True."
            )
        if rho > 0.5:
            warnings.warn(
                f"cn_step (use_cg=False): spectral radius rho = {rho:.3f} is "
                f"close to 1. Convergence in {n_iter} iterations is not "
                f"guaranteed. Consider use_cg=True.",
                stacklevel=2,
            )

        def body(T_k, _):
            T_new = rhs + 0.5 * dt * alpha * laplacian(T_k, dx, dy)
            return apply_bc(T_new, T_wall), None

        T_new, _ = lax.scan(body, T, None, length=n_iter)
        return T_new


def make_cn_step_jit(alpha: float, dt: float, dx: float, dy: float,
                     T_wall: float = 0.0, n_iter: int = 5, use_cg: bool = True):
    """
    Return a JIT-compiled step function with fixed hyperparameters closed over.
    Signature: step_fn(T, Q) → T_new
    """
    @jax.jit
    def _step(T, Q):
        return cn_step(T, Q, alpha, dt, dx, dy, T_wall, n_iter, use_cg)
    return _step

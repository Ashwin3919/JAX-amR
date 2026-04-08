"""
Crank-Nicolson step via fixed-point iteration (5 iterations).

Scheme:
  (T^{n+1} - T^n)/dt = alpha/2 * (L[T^n] + L[T^{n+1}]) + Q
  → T^{n+1}_{k+1} = T^n + dt*alpha/2*(L[T^n] + L[T^{n+1}_k]) + dt*Q

5 fixed-point iterations (matches v1/v2 approach; unconditionally stable CN).
"""
import jax
import jax.numpy as jnp
from jax import lax
from solver.ops import laplacian, apply_bc


def cn_step(T, Q, alpha: float, dt: float, dx: float, dy: float,
            T_wall: float = 0.0, n_iter: int = 5):
    """One CN step; returns T^{n+1} with BCs enforced."""
    rhs_explicit = T + 0.5 * dt * alpha * laplacian(T, dx, dy) + dt * Q

    def body(T_k, _):
        T_new = rhs_explicit + 0.5 * dt * alpha * laplacian(T_k, dx, dy)
        return apply_bc(T_new, T_wall), None

    T_new, _ = lax.scan(body, T, None, length=n_iter)
    return T_new


def make_cn_step_jit(alpha: float, dt: float, dx: float, dy: float,
                     T_wall: float = 0.0, n_iter: int = 5):
    """
    Return a JIT-compiled step function with fixed hyperparameters closed over.
    Signature: step_fn(T, Q) → T_new
    """
    @jax.jit
    def _step(T, Q):
        return cn_step(T, Q, alpha, dt, dx, dy, T_wall, n_iter)
    return _step

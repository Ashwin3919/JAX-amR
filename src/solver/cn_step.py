"""
Crank-Nicolson step via fixed-point iteration (5 iterations).

Scheme:
  (T^{n+1} - T^n)/dt = alpha/2 * (L[T^n] + L[T^{n+1}]) + Q
  → T^{n+1}_{k+1} = T^n + dt*alpha/2*(L[T^n] + L[T^{n+1}_k]) + dt*Q

5 fixed-point iterations (matches v1/v2 approach; unconditionally stable CN).
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from jax import lax
from solver.ops import laplacian, apply_bc


def cn_step(T: jnp.ndarray, Q: jnp.ndarray, alpha: float, dt: float,
            dx: float, dy: float, T_wall: float = 0.0, n_iter: int = 5) -> jnp.ndarray:
    """One CN step; returns T^{n+1} with BCs enforced."""
    if T.shape != Q.shape:
        raise ValueError(f"cn_step: T.shape {T.shape} != Q.shape {Q.shape}")
    if alpha <= 0.0:
        raise ValueError(f"cn_step: alpha must be positive, got {alpha}")
    if dt <= 0.0:
        raise ValueError(f"cn_step: dt must be positive, got {dt}")
    if dx <= 0.0 or dy <= 0.0:
        raise ValueError(f"cn_step: grid spacing must be positive, got dx={dx}, dy={dy}")
    if n_iter < 1:
        raise ValueError(f"cn_step: n_iter must be >= 1, got {n_iter}")
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

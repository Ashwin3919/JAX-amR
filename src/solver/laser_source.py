"""Application-specific laser source builder.

Separated from solver/grid.py to keep the PDE-agnostic grid utilities
free of config dependencies.
"""
import jax.numpy as jnp
import config.params as p


def build_laser_source(X, Y, cx: float, cy: float,
                       sigma: float, power: float, t: float = 0.0):
    """
    Gaussian laser source Q(x,y,t).
    Respects config.params.LASER_MODE (stationary or circular).
    """
    if p.LASER_MODE == "circular":
        omega = p.laser_omega
        R = p.laser_radius
        target_cx = 0.5 + R * jnp.cos(omega * t)
        target_cy = 0.5 + R * jnp.sin(omega * t)
    else:
        target_cx = cx
        target_cy = cy

    return power * jnp.exp(
        -((X - target_cx) ** 2 + (Y - target_cy) ** 2) / (2.0 * sigma ** 2)
    )

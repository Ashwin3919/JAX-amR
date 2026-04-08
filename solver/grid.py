import jax.numpy as jnp


def build_grid(Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0):
    """Return (X, Y) meshgrid arrays of shape (Nx, Ny), indexing='ij'."""
    x = jnp.linspace(0.0, Lx, Nx)
    y = jnp.linspace(0.0, Ly, Ny)
    return jnp.meshgrid(x, y, indexing="ij")


def build_laser_source(X, Y, cx: float, cy: float, sigma: float, power: float, t: float = 0.0):
    """
    Gaussian laser source Q(x,y,t) = power * exp(-r²/2σ²).
    The center (cx, cy) can be functions of time t.
    """
    # Moving laser: circular path
    # R = 0.2, Period = 0.1s
    omega = 2.0 * jnp.pi / 0.1
    R = 0.2
    target_cx = 0.5 + R * jnp.cos(omega * t)
    target_cy = 0.5 + R * jnp.sin(omega * t)
    
    return power * jnp.exp(-((X - target_cx) ** 2 + (Y - target_cy) ** 2) / (2.0 * sigma ** 2))

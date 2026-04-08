import jax.numpy as jnp


def build_grid(Nx: int, Ny: int, Lx: float = 1.0, Ly: float = 1.0):
    """Return (X, Y) meshgrid arrays of shape (Nx, Ny), indexing='ij'."""
    x = jnp.linspace(0.0, Lx, Nx)
    y = jnp.linspace(0.0, Ly, Ny)
    return jnp.meshgrid(x, y, indexing="ij")


def build_laser_source(X, Y, cx: float, cy: float, sigma: float, power: float):
    """Gaussian laser source Q(x,y) = power * exp(-r²/2σ²)."""
    return power * jnp.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / (2.0 * sigma ** 2))

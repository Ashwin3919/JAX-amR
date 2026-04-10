import jax.numpy as jnp


def compute_gradient_magnitude(T: jnp.ndarray, dx: float, dy: float) -> jnp.ndarray:
    """
    Compute |∇T| on a JAX array using second-order central differences.

    Parameters
    ----------
    T   : (Nx, Ny) float array
    dx  : grid spacing in x
    dy  : grid spacing in y

    Returns
    -------
    (Nx, Ny) float array of gradient magnitudes
    """
    T = jnp.asarray(T)
    gx_interior = (T[2:, 1:-1] - T[:-2, 1:-1]) / (2.0 * dx)
    gy_interior = (T[1:-1, 2:] - T[1:-1, :-2]) / (2.0 * dy)

    gx = jnp.pad(gx_interior, ((1, 1), (1, 1)), mode="edge")
    gy = jnp.pad(gy_interior, ((1, 1), (1, 1)), mode="edge")

    return jnp.sqrt(gx ** 2 + gy ** 2)

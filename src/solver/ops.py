import jax.numpy as jnp


def laplacian(T, dx: float, dy: float):
    """
    5-point FD Laplacian on interior; boundary rows/cols return 0.

    This is the *slice* form (no roll), so:
      lap[0,:] = lap[-1,:] = lap[:,0] = lap[:,-1] = 0
    which correctly encodes Dirichlet ghost-cell contributions as 0
    when boundary values are enforced by apply_bc().
    """
    d2x = (T[2:, 1:-1] - 2.0 * T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx ** 2
    d2y = (T[1:-1, 2:] - 2.0 * T[1:-1, 1:-1] + T[1:-1, :-2]) / dy ** 2
    interior = d2x + d2y
    return jnp.pad(interior, 1, mode="constant", constant_values=0.0)


def apply_bc(T, T_wall: float = 0.0):
    """Enforce Dirichlet BC on all four walls."""
    T = T.at[0,  :].set(T_wall)
    T = T.at[-1, :].set(T_wall)
    T = T.at[:,  0].set(T_wall)
    T = T.at[:, -1].set(T_wall)
    return T

import numpy as np


def compute_gradient_magnitude(T: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute |∇T| on a numpy array using second-order central differences.

    Parameters
    ----------
    T   : (Nx, Ny) float array
    dx  : grid spacing in x
    dy  : grid spacing in y

    Returns
    -------
    (Nx, Ny) float array of gradient magnitudes
    """
    T = np.asarray(T)
    gx = np.gradient(T, dx, axis=0)
    gy = np.gradient(T, dy, axis=1)
    return np.sqrt(gx ** 2 + gy ** 2)

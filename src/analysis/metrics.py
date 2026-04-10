"""L2 error computation and wall-clock timer."""
import time
import numpy as np


def l2_error(T: np.ndarray, T_ref: np.ndarray) -> float:
    """Root-mean-square L2 error between T and a reference field."""
    T = np.asarray(T, dtype=np.float64)
    T_ref = np.asarray(T_ref, dtype=np.float64)
    return float(np.sqrt(np.mean((T - T_ref) ** 2)))


class Timer:
    """Context-manager wall-clock timer. Elapsed time in seconds."""

    def __init__(self):
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start

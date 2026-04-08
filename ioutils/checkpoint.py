"""Save / load numpy temperature frames to .npz checkpoints."""
import numpy as np


def save_checkpoint(path: str, T: np.ndarray, step: int, t: float) -> None:
    np.savez(path, T=np.asarray(T), step=np.int64(step), t=np.float64(t))


def load_checkpoint(path: str):
    """Returns (T, step, t)."""
    data = np.load(path)
    return data["T"], int(data["step"]), float(data["t"])

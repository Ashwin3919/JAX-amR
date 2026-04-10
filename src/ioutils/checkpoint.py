"""Checkpoint save/load utilities."""
from __future__ import annotations
import numpy as np


def save_checkpoint(path: str, T: np.ndarray, step: int, t: float) -> None:
    """Save simulation state to a .npz checkpoint file."""
    T = np.asarray(T)
    if T.ndim != 2:
        raise ValueError(f"save_checkpoint: expected 2-D array, got shape {T.shape}")
    try:
        np.savez(path, T=T, step=np.int64(step), t=np.float64(t))
    except OSError as exc:
        raise RuntimeError(f"save_checkpoint: failed to write '{path}': {exc}") from exc


def load_checkpoint(path: str) -> tuple[np.ndarray, int, float]:
    """Load simulation state from a .npz checkpoint file."""
    try:
        data = np.load(path)
    except FileNotFoundError:
        raise FileNotFoundError(f"load_checkpoint: checkpoint not found: '{path}'")
    except OSError as exc:
        raise RuntimeError(f"load_checkpoint: failed to read '{path}': {exc}") from exc
    for key in ("T", "step", "t"):
        if key not in data:
            raise KeyError(f"load_checkpoint: key '{key}' missing in '{path}'")
    return data["T"], int(data["step"]), float(data["t"])

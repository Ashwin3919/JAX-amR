"""
Level-assignment logic for AMR cells.

Level 1 = base (coarsest)
Level 2 = first refinement
Level 3 = finest refinement

Thresholds list is ordered low-to-high:
  tiers = [thresh_for_level2, thresh_for_level3]
"""
import numpy as np


def assign_level(grad_val: float, tiers: list, max_level: int = 3) -> int:
    """Assign a single AMR level given one gradient magnitude value."""
    level = 1
    for l, thresh in enumerate(tiers, start=2):
        if grad_val >= thresh:
            level = l
    return min(level, max_level)


def assign_levels_array(grad_mag: np.ndarray, tiers: list,
                        max_level: int = 3) -> np.ndarray:
    """
    Vectorised level assignment for a whole array.

    Parameters
    ----------
    grad_mag  : (M, N) float array of gradient magnitudes
    tiers     : ascending list of threshold values, e.g. [2.0, 4.0, 16.0]
                tier[k] is the threshold to reach level k+2
    max_level : maximum allowed level (clamps output)

    Returns
    -------
    (M, N) int32 array with values in [1, max_level]
    """
    levels = np.ones(grad_mag.shape, dtype=np.int32)
    for k, thresh in enumerate(tiers):
        levels = np.where(grad_mag >= thresh, k + 2, levels)
    return np.clip(levels, 1, max_level)

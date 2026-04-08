"""
Build AMR macro-cell structures from a temperature field.

The domain is divided into MACRO×MACRO coarse cells.
Each macro-cell is assigned a refinement level based on max |∇T| inside it,
then sub-divided into 2^level × 2^level sub-cells for VTK/visualisation.
"""
import numpy as np
from amr.gradient import compute_gradient_magnitude
from amr.thresholds import assign_levels_array


def build_level_map(T: np.ndarray, dx: float, dy: float,
                    macro: int, tiers: list, max_level: int = 3) -> np.ndarray:
    """
    Return a (macro, macro) int array of refinement levels.

    Parameters
    ----------
    T         : (Nx, Ny) temperature array
    dx, dy    : fine-grid spacing
    macro     : number of macro-cells per axis
    tiers     : threshold list for assign_levels_array
    max_level : max AMR level
    """
    T = np.asarray(T)
    Nx, Ny = T.shape
    grad_mag = compute_gradient_magnitude(T, dx, dy)
    pts_x = Nx // macro
    pts_y = Ny // macro

    cell_grad_max = np.zeros((macro, macro), dtype=np.float32)
    for ci in range(macro):
        for cj in range(macro):
            block = grad_mag[ci * pts_x:(ci + 1) * pts_x,
                             cj * pts_y:(cj + 1) * pts_y]
            cell_grad_max[ci, cj] = block.max()

    return assign_levels_array(cell_grad_max, tiers, max_level)


def build_amr_cells(T: np.ndarray, dx: float, dy: float,
                    Lx: float, Ly: float,
                    macro: int, tiers: list, max_level: int = 3):
    """
    Build list of AMR sub-cell tuples (x0, y0, x1, y1, level).

    Each macro-cell at level L is sub-divided into 2^L × 2^L rectangles.
    Returns (cells, level_map) where:
      cells     : list of (x0, y0, x1, y1, level) tuples
      level_map : (macro, macro) int array
    """
    level_map = build_level_map(T, dx, dy, macro, tiers, max_level)
    cell_w = Lx / macro
    cell_h = Ly / macro
    cells = []

    for ci in range(macro):
        for cj in range(macro):
            level = int(level_map[ci, cj])
            n_sub = 2 ** (level - 1) if level > 1 else 1
            sub_w = cell_w / n_sub
            sub_h = cell_h / n_sub
            x0_base = ci * cell_w
            y0_base = cj * cell_h
            for si in range(n_sub):
                for sj in range(n_sub):
                    x0 = x0_base + si * sub_w
                    y0 = y0_base + sj * sub_h
                    cells.append((x0, y0, x0 + sub_w, y0 + sub_h, level))

    return cells, level_map

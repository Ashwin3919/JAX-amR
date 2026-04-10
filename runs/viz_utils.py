"""Shared grid-overlay utilities for run_* drivers."""
from __future__ import annotations


def coarse_cells(n: int, Lx: float = 1.0, Ly: float = 1.0) -> list[tuple]:
    """Return n×n equal cells covering the full domain (level 1)."""
    w, h = Lx / n, Ly / n
    return [(i * w, j * h, (i + 1) * w, (j + 1) * h, 1)
            for i in range(n) for j in range(n)]


def bounds_to_cells(
    x0: float, x1: float, y0: float, y1: float,
    n_coarse: int = 8, n_fine: int = 16,
    Lx: float = 1.0, Ly: float = 1.0,
) -> list[tuple]:
    """Coarse background grid + fine patch cells inside [x0,x1]x[y0,y1]."""
    x0, x1, y0, y1 = float(x0), float(x1), float(y0), float(y1)
    cells = coarse_cells(n_coarse, Lx, Ly)
    fw, fh = (x1 - x0) / n_fine, (y1 - y0) / n_fine
    for i in range(n_fine):
        for j in range(n_fine):
            cells.append((x0 + i * fw, y0 + j * fh,
                          x0 + (i + 1) * fw, y0 + (j + 1) * fh, 3))
    return cells

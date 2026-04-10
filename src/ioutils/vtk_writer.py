"""
Legacy VTK writers for the heat-equation solver.
Generates .vtk files (ASCII) instead of XML .vts/.vtu files.
"""
from __future__ import annotations
import os
import numpy as np


def write_legacy_vtk(path: str, X: np.ndarray, Y: np.ndarray, T: np.ndarray, title: str = "HeatMap") -> None:
    """
    Writes a legacy VTK file (STRUCTURED_GRID) containing coordinates and scalar data.
    """
    T = np.asarray(T)
    X = np.asarray(X)
    Y = np.asarray(Y)
    if not (T.shape == X.shape == Y.shape):
        raise ValueError(
            f"write_legacy_vtk: shape mismatch — T={T.shape}, X={X.shape}, Y={Y.shape}"
        )
    Nx, Ny = T.shape
    n_points = Nx * Ny

    try:
        with open(path, "w") as f:
            # Header
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"{title}\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_GRID\n")
            f.write(f"DIMENSIONS {Nx} {Ny} 1\n")

            # Points (Coordinates)
            f.write(f"POINTS {n_points} float\n")
            # VTK expects points in (x, y, z) order, flat array
            # meshgrid(indexing='ij') means we loop i then j
            for j in range(Ny):
                for i in range(Nx):
                    f.write(f"{X[i, j]:.6f} {Y[i, j]:.6f} 0.0\n")

            # Point Data (Temperature)
            f.write(f"POINT_DATA {n_points}\n")
            f.write("SCALARS Temperature float 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(Ny):
                for i in range(Nx):
                    f.write(f"{T[i, j]:.6f}\n")
    except OSError as exc:
        raise RuntimeError(f"write_legacy_vtk: failed to write '{path}': {exc}") from exc


def write_amr_legacy_vtk(path: str, cells: list, title: str = "AMR") -> None:
    """
    Writes a legacy VTK file (UNSTRUCTURED_GRID) for AMR-overlay cells.
    cells : list of (x0, y0, x1, y1, level)
    """
    n_cells = len(cells)
    n_points = 4 * n_cells

    try:
        with open(path, "w") as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write(f"{title}\n")
            f.write("ASCII\n")
            f.write("DATASET UNSTRUCTURED_GRID\n")

            # Points
            f.write(f"POINTS {n_points} float\n")
            for x0, y0, x1, y1, _ in cells:
                f.write(f"{x0:.6f} {y0:.6f} 0.0\n")
                f.write(f"{x1:.6f} {y0:.6f} 0.0\n")
                f.write(f"{x1:.6f} {y1:.6f} 0.0\n")
                f.write(f"{x0:.6f} {y1:.6f} 0.0\n")

            # Cells
            # Each cell has 4 points. Total size = n_cells * (1 count + 4 indices)
            f.write(f"CELLS {n_cells} {5 * n_cells}\n")
            for i in range(n_cells):
                b = 4 * i
                f.write(f"4 {b} {b+1} {b+2} {b+3}\n")

            f.write(f"CELL_TYPES {n_cells}\n")
            for _ in range(n_cells):
                f.write("9\n")  # VTK_QUAD

            # Cell Data (AMRLevel)
            f.write(f"CELL_DATA {n_cells}\n")
            f.write("SCALARS AMRLevel int 1\n")
            f.write("LOOKUP_TABLE default\n")
            for *_, level in cells:
                f.write(f"{level}\n")
    except OSError as exc:
        raise RuntimeError(f"write_amr_legacy_vtk: failed to write '{path}': {exc}") from exc


def write_pvd(path: str, entries: list) -> None:
    """
    Write a ParaView Data (.pvd) collection file.
    ParaView can handle legacy .vtk inside a .pvd.
    """
    try:
        with open(path, "w") as f:
            f.write('<?xml version="1.0"?>\n')
            f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
            f.write("  <Collection>\n")
            for t_val, fpath in entries:
                fname = os.path.basename(fpath)
                f.write(f'    <DataSet timestep="{t_val:.6f}" group="" part="0" file="{fname}"/>\n')
            f.write("  </Collection>\n")
            f.write("</VTKFile>\n")
    except OSError as exc:
        raise RuntimeError(f"write_pvd: failed to write '{path}': {exc}") from exc

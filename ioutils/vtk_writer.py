"""
VTK writers for the heat-equation solver.

Uniform grid:  StructuredGrid (.vts)  — mesh written once, scalars per step
AMR grid:      UnstructuredGrid (.vtu) — mesh + scalars per step (topology changes)
Collection:    .pvd file so ParaView sees a time series
"""
import os
import numpy as np


# ── Uniform: mesh ─────────────────────────────────────────────────────────────

def write_mesh_vtk(path: str, X: np.ndarray, Y: np.ndarray) -> None:
    """Write StructuredGrid VTK containing only X/Y coordinates (written once)."""
    Nx, Ny = X.shape
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {Nx-1} 0 {Ny-1} 0 0">\n')
        f.write(f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 0">\n')
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for i in range(Nx):
            for j in range(Ny):
                f.write(f"          {X[i, j]:.6f} {Y[i, j]:.6f} 0.0\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")
        f.write("    </Piece>\n")
        f.write("  </StructuredGrid>\n")
        f.write("</VTKFile>\n")


# ── Uniform: scalar per timestep ─────────────────────────────────────────────

def write_scalar_vtk(path: str, T: np.ndarray, t: float) -> None:
    """Write StructuredGrid VTK with Temperature PointData for one timestep."""
    T = np.asarray(T)
    Nx, Ny = T.shape
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="StructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write(f'  <StructuredGrid WholeExtent="0 {Nx-1} 0 {Ny-1} 0 0">\n')
        f.write(f'    <Piece Extent="0 {Nx-1} 0 {Ny-1} 0 0">\n')
        f.write('      <PointData Scalars="Temperature">\n')
        f.write('        <DataArray type="Float32" Name="Temperature" format="ascii">\n')
        for i in range(Nx):
            for j in range(Ny):
                f.write(f"          {T[i, j]:.6f}\n")
        f.write("        </DataArray>\n")
        f.write("      </PointData>\n")
        f.write("    </Piece>\n")
        f.write("  </StructuredGrid>\n")
        f.write("</VTKFile>\n")


# ── AMR: unstructured mesh + scalar per timestep ──────────────────────────────

def write_amr_vtk(path: str, cells: list, t: float) -> None:
    """
    Write UnstructuredGrid VTK for AMR cells.

    cells : list of (x0, y0, x1, y1, level) tuples
    Each cell becomes a VTK_QUAD (type 9) with 4 corner points.
    CellData: AMRLevel (int), Temperature (float, cell-centre average — optional).
    """
    n_cells = len(cells)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <UnstructuredGrid>\n")
        f.write(f'    <Piece NumberOfPoints="{4 * n_cells}" NumberOfCells="{n_cells}">\n')

        # Points
        f.write("      <Points>\n")
        f.write('        <DataArray type="Float32" NumberOfComponents="3" format="ascii">\n')
        for x0, y0, x1, y1, _ in cells:
            f.write(f"          {x0:.6f} {y0:.6f} 0.0\n")
            f.write(f"          {x1:.6f} {y0:.6f} 0.0\n")
            f.write(f"          {x1:.6f} {y1:.6f} 0.0\n")
            f.write(f"          {x0:.6f} {y1:.6f} 0.0\n")
        f.write("        </DataArray>\n")
        f.write("      </Points>\n")

        # Cells
        f.write("      <Cells>\n")
        f.write('        <DataArray type="Int32" Name="connectivity" format="ascii">\n')
        for i in range(n_cells):
            b = 4 * i
            f.write(f"          {b} {b+1} {b+2} {b+3}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="Int32" Name="offsets" format="ascii">\n')
        for i in range(1, n_cells + 1):
            f.write(f"          {4 * i}\n")
        f.write("        </DataArray>\n")
        f.write('        <DataArray type="UInt8" Name="types" format="ascii">\n')
        for _ in range(n_cells):
            f.write("          9\n")   # VTK_QUAD
        f.write("        </DataArray>\n")
        f.write("      </Cells>\n")

        # CellData
        f.write("      <CellData>\n")
        f.write('        <DataArray type="Int32" Name="AMRLevel" format="ascii">\n')
        for *_, level in cells:
            f.write(f"          {level}\n")
        f.write("        </DataArray>\n")
        f.write("      </CellData>\n")

        f.write("    </Piece>\n")
        f.write("  </UnstructuredGrid>\n")
        f.write("</VTKFile>\n")


# ── PVD collection ────────────────────────────────────────────────────────────

def write_pvd(path: str, entries: list) -> None:
    """
    Write a ParaView Data (.pvd) collection file.

    entries : list of (time: float, filepath: str)
    """
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n')
        f.write("  <Collection>\n")
        for t_val, fpath in entries:
            fname = os.path.basename(fpath)
            f.write(f'    <DataSet timestep="{t_val:.6f}" group="" part="0" file="{fname}"/>\n')
        f.write("  </Collection>\n")
        f.write("</VTKFile>\n")

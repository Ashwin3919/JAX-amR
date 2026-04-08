"""
Test suite — 6 tests covering solver correctness, AMR logic, and VTK output.

Run with:
    python tests.py
or:
    python -m pytest tests.py -v
"""
import os
import sys
import tempfile
import numpy as np
import jax
import jax.numpy as jnp

# ── Ensure repo root on path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from solver.grid import build_grid, build_laser_source
from solver.ops import laplacian, apply_bc
from solver.cn_step import cn_step
from amr.thresholds import assign_levels_array
from amr.cells import build_level_map
from ioutils.vtk_writer import write_mesh_vtk


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_grid(N=32):
    Lx = Ly = 1.0
    dx = Lx / (N - 1)
    dy = Ly / (N - 1)
    X, Y = build_grid(N, N, Lx, Ly)
    return X, Y, dx, dy


# ── Test 1: Laplacian known solution ─────────────────────────────────────────

def test_laplacian_known():
    """
    Apply laplacian() to T = sin(πx)sin(πy).
    Exact value: ∇²T = -2π²·T.
    FD error is O(dx²); check max interior error < 1% of |exact|.
    """
    N = 64
    X, Y, dx, dy = _make_grid(N)
    T = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    exact = -2.0 * jnp.pi ** 2 * T

    lap = laplacian(T, dx, dy)

    # Compare only interior (boundary rows are correctly 0 in our stencil)
    lap_int   = np.asarray(lap[1:-1, 1:-1])
    exact_int = np.asarray(exact[1:-1, 1:-1])

    rel_err = np.max(np.abs(lap_int - exact_int)) / np.max(np.abs(exact_int))
    assert rel_err < 0.01, (
        f"Laplacian relative error {rel_err:.3e} exceeds 1% tolerance"
    )
    print(f"  test_laplacian_known PASSED  (max rel err = {rel_err:.2e})")


# ── Test 2: apply_bc sets all walls ──────────────────────────────────────────

def test_apply_bc():
    """After apply_bc(), all four wall edges must equal T_wall."""
    N = 16
    T_wall = 42.0
    T = jnp.ones((N, N)) * 100.0
    T_bc = apply_bc(T, T_wall=T_wall)
    T_np = np.asarray(T_bc)

    assert np.all(T_np[0,  :] == T_wall), "Top wall failed"
    assert np.all(T_np[-1, :] == T_wall), "Bottom wall failed"
    assert np.all(T_np[:,  0] == T_wall), "Left wall failed"
    assert np.all(T_np[:, -1] == T_wall), "Right wall failed"
    print("  test_apply_bc PASSED")


# ── Test 3: CN step with no source leaves uniform field unchanged ─────────────

def test_cn_step_no_source():
    """
    With Q=0 and T uniform = T_wall = 0, one CN step must leave T unchanged.
    """
    N = 16
    X, Y, dx, dy = _make_grid(N)
    T = jnp.zeros((N, N))
    Q = jnp.zeros((N, N))

    T_new = cn_step(T, Q, alpha=1e-3, dt=1e-3, dx=dx, dy=dy)
    diff = float(jnp.max(jnp.abs(T_new - T)))
    assert diff < 1e-10, f"CN step changed uniform-zero field by {diff:.2e}"
    print(f"  test_cn_step_no_source PASSED  (max diff = {diff:.2e})")


# ── Test 4: laser source drives peak temperature upward ───────────────────────

def test_cn_energy_growth():
    """
    With a laser source active, peak temperature must strictly increase
    for the first 10 steps (source power >> diffusion loss).
    """
    N = 32
    X, Y, dx, dy = _make_grid(N)
    Q = build_laser_source(X, Y, cx=0.5, cy=0.5, sigma=0.05, power=500.0)
    T = apply_bc(jnp.zeros((N, N)))

    peaks = []
    for _ in range(10):
        T = cn_step(T, Q, alpha=1e-3, dt=1e-3, dx=dx, dy=dy)
        peaks.append(float(jnp.max(T)))

    for i in range(9):
        assert peaks[i + 1] > peaks[i], (
            f"Peak T did not increase at step {i+1}: "
            f"{peaks[i]:.4f} → {peaks[i+1]:.4f}"
        )
    print(f"  test_cn_energy_growth PASSED  (peak after 10 steps = {peaks[-1]:.4f})")


# ── Test 5: AMR level assignment correctness ──────────────────────────────────

def test_amr_level_assignment():
    """
    Construct a synthetic gradient field with known magnitudes.
    Verify build_level_map (via assign_levels_array) returns the correct level
    for each region.

    Tiers: [2.0, 4.0]  →  level 1 if < 2.0, level 2 if >= 2.0, level 3 if >= 4.0
    """
    tiers = [2.0, 4.0]
    grad = np.array([
        [0.5,  2.5, 5.0],
        [0.5,  2.5, 5.0],
        [0.5,  2.5, 5.0],
    ], dtype=np.float32)

    levels = assign_levels_array(grad, tiers=tiers, max_level=3)
    expected = np.array([
        [1, 2, 3],
        [1, 2, 3],
        [1, 2, 3],
    ], dtype=np.int32)

    np.testing.assert_array_equal(
        levels, expected,
        err_msg=f"Level map mismatch:\n{levels}\nvs expected:\n{expected}"
    )
    print("  test_amr_level_assignment PASSED")


# ── Test 6: VTK mesh file has required XML tags ───────────────────────────────

def test_vtk_mesh_xml():
    """
    Call write_mesh_vtk on a 4×4 grid, verify the output file exists
    and contains the expected XML tags.
    """
    N = 4
    x = np.linspace(0, 1, N)
    y = np.linspace(0, 1, N)
    X, Y = np.meshgrid(x, y, indexing="ij")

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test_mesh.vts")
        write_mesh_vtk(path, X, Y)

        assert os.path.exists(path), "VTK file was not created"

        with open(path) as f:
            content = f.read()

        required_tags = ["VTKFile", "StructuredGrid", "Points"]
        for tag in required_tags:
            assert tag in content, f"Expected XML tag <{tag}> not found in VTK output"

    print("  test_vtk_mesh_xml PASSED")


# ── Runner ────────────────────────────────────────────────────────────────────

TESTS = [
    test_laplacian_known,
    test_apply_bc,
    test_cn_step_no_source,
    test_cn_energy_growth,
    test_amr_level_assignment,
    test_vtk_mesh_xml,
]


if __name__ == "__main__":
    print(f"\nRunning {len(TESTS)} tests...\n")
    passed = 0
    failed = 0
    for test in TESTS:
        try:
            test()
            passed += 1
        except Exception as exc:
            print(f"  {test.__name__} FAILED: {exc}")
            failed += 1

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(TESTS)} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        sys.exit(1)
    else:
        print(" — all OK")

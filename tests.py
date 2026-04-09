"""
Physics tests — run with:
    PYTHONPATH=. python tests.py
"""
import sys
import numpy as np
import jax.numpy as jnp

from solver.grid import build_grid, build_laser_source
from solver.ops import laplacian, apply_bc
from solver.cn_step import cn_step
import config.params as p


def _grid(N=32):
    X, Y = build_grid(N, N, p.Lx, p.Ly)
    dx = p.Lx / (N - 1)
    dy = p.Ly / (N - 1)
    return X, Y, dx, dy


# 1. Laplacian on sin(πx)sin(πy) — exact: -2π²T, FD error < 1%
def test_laplacian():
    X, Y, dx, dy = _grid(64)
    T = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    exact = -2.0 * jnp.pi**2 * T
    lap = laplacian(T, dx, dy)
    interior = np.s_[1:-1, 1:-1]
    rel = float(np.max(np.abs(np.asarray(lap[interior]) - np.asarray(exact[interior])))
                / np.max(np.abs(np.asarray(exact[interior]))))
    assert rel < 0.01, f"Laplacian rel error {rel:.2e} > 1%"
    print(f"  laplacian           PASS  (rel err = {rel:.2e})")


# 2. apply_bc sets all four walls to T_wall
def test_bc():
    T = apply_bc(jnp.ones((16, 16)) * 99.0, T_wall=0.0)
    T = np.asarray(T)
    assert T[0, :].max() == 0 and T[-1, :].max() == 0
    assert T[:, 0].max() == 0 and T[:, -1].max() == 0
    print("  apply_bc            PASS")


# 3. Zero source + zero IC → field stays zero after one CN step
def test_cn_zero():
    _, _, dx, dy = _grid(16)
    T = jnp.zeros((16, 16))
    T_new = cn_step(T, T, alpha=p.alpha, dt=p.dt, dx=dx, dy=dy)
    diff = float(jnp.max(jnp.abs(T_new - T)))
    assert diff < 1e-10, f"CN changed zero field by {diff:.2e}"
    print(f"  cn_step zero        PASS  (max diff = {diff:.2e})")


# 4. Active laser → peak temperature strictly increases for 20 steps
def test_energy_growth():
    X, Y, dx, dy = _grid(32)
    Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power)
    T = apply_bc(jnp.zeros((32, 32)))
    prev = float(jnp.max(T))
    for _ in range(20):
        T = cn_step(T, Q, alpha=p.alpha, dt=p.dt, dx=dx, dy=dy)
        curr = float(jnp.max(T))
        assert curr > prev, f"Peak T dropped: {prev:.4f} → {curr:.4f}"
        prev = curr
    print(f"  energy growth       PASS  (peak after 20 steps = {prev:.4f} K)")


# 5. Dirichlet walls never heat up (laser centred away from walls)
def test_walls_cold():
    X, Y, dx, dy = _grid(32)
    Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power)
    T = apply_bc(jnp.zeros((32, 32)))
    for _ in range(50):
        T = cn_step(T, Q, alpha=p.alpha, dt=p.dt, dx=dx, dy=dy)
    T_np = np.asarray(T)
    wall_max = max(T_np[0, :].max(), T_np[-1, :].max(),
                   T_np[:, 0].max(), T_np[:, -1].max())
    assert wall_max == 0.0, f"Wall temperature is {wall_max:.4f} K (should be 0)"
    print(f"  walls cold          PASS  (wall max = {wall_max:.4f} K)")


TESTS = [test_laplacian, test_bc, test_cn_zero, test_energy_growth, test_walls_cold]

if __name__ == "__main__":
    print(f"\nRunning {len(TESTS)} physics tests...\n")
    passed = failed = 0
    for t in TESTS:
        try:
            t(); passed += 1
        except Exception as e:
            print(f"  {t.__name__} FAIL: {e}"); failed += 1
    print(f"\n{'='*40}")
    print(f"{passed}/{len(TESTS)} passed" + (f", {failed} FAILED" if failed else " — all OK"))
    sys.exit(1 if failed else 0)

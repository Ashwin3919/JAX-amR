"""
compare.py — run all three solvers and print a side-by-side comparison.

Usage:
    python runs/compare.py
"""
import sys, os
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np

from run_uniform import run_uniform
from run_amr import run_amr
from run_composite_amr import run_simulation as run_fixed
import config.params as p

N_STEPS = p.n_steps   # use the same n_steps for all three (default 5000)

print("=" * 60)
print("JAX-amR  benchmark comparison")
print(f"n_steps={N_STEPS}  dt={p.dt}  alpha={p.alpha}")
print("=" * 60)

# ── 1. Uniform ────────────────────────────────────────────────────────────────
print("\n[1/3] Uniform 1024×1024 ...")
res_u = run_uniform(Nx=1024, Ny=1024, n_steps=N_STEPS, save_vtk=False)
T_u   = np.asarray(res_u["T_final"])
peak_u   = float(T_u.max())
wall_u   = res_u["wallclock"]
dof_u    = 1024 * 1024

# ── 2. AMR Dynamic ────────────────────────────────────────────────────────────
print("\n[2/3] AMR Dynamic (128×128 coarse + 512×512 moving patch) ...")
res_a = run_amr(Nc=128, Nf=512, n_steps=N_STEPS, save_vtk=False)
T_a   = np.asarray(res_a["T_final"])
peak_a   = float(T_a.max())
wall_a   = res_a["wallclock"]
dof_a    = 128 * 128 + 512 * 512

# ── 3. AMR Fixed ──────────────────────────────────────────────────────────────
print("\n[3/3] AMR Fixed (128×128 coarse + 512×512 at [0.25,0.75]²) ...")
res_f = run_fixed(Nc_x=128, Nc_y=128, Nf_x=512, Nf_y=512,
                  patch_bounds=(0.25, 0.75, 0.25, 0.75),
                  n_steps=N_STEPS, save_vtk=False)
Tc_f, Tp_f = res_f["T_final"]
peak_f   = float(np.asarray(Tp_f).max())
wall_f   = res_f["wallclock"]
dof_f    = 128 * 128 + 512 * 512

# ── Results ───────────────────────────────────────────────────────────────────
err_a = abs(peak_a - peak_u) / peak_u * 100
err_f = abs(peak_f - peak_u) / peak_u * 100

print("\n" + "=" * 60)
print(f"{'Model':<20} {'DOF':>10} {'Wallclock':>12} {'Peak T (K)':>12} {'Error':>8}")
print("-" * 60)
print(f"{'Uniform 1024x1024':<20} {dof_u:>10,} {wall_u:>11.2f}s {peak_u:>12.4f} {'—':>8}")
print(f"{'AMR Dynamic':<20} {dof_a:>10,} {wall_a:>11.2f}s {peak_a:>12.4f} {err_a:>7.2f}%")
print(f"{'AMR Fixed':<20} {dof_f:>10,} {wall_f:>11.2f}s {peak_f:>12.4f} {err_f:>7.4f}%")
print("=" * 60)
print(f"\nSpeedup  AMR Dynamic : {wall_u / wall_a:.1f}x")
print(f"Speedup  AMR Fixed   : {wall_u / wall_f:.1f}x")
print(f"DOF ratio (AMR / Uniform): {dof_a / dof_u:.3f}x ({dof_u // dof_a:.1f}x fewer)")

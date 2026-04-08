"""
compare.py  —  Uniform-grid vs AMR-overlay performance study
Author: Ashwin Shirke

Three analyses
──────────────
1. Convergence      : L2 error vs grid resolution (Nx = 16 → 128, ref = Nx 256)
                      Shows how spatial accuracy scales with cost (DOF count).

2. Timing breakdown : per-step solver cost vs solver + AMR-cell-build overhead
                      at Nx = 128.  Averaged over multiple warm trials so JIT
                      compile time does not pollute the measurement.

3. Efficiency frontier : (wallclock, L2 error) for Nx = 16, 32, 64, 128.
                         The classic computational-scientist plot — lower-left
                         is better.  Both uniform and AMR curves are shown so
                         you can see the overhead AMR adds at each resolution.

Output
──────
  output/comparison/
    1_convergence.png        — log-log L2 error vs DOF count
    2_timing_breakdown.png   — grouped bar: solver / AMR overhead per step
    3_efficiency_frontier.png — accuracy vs wallclock scatter
    summary.txt              — plain-text result table

Usage
─────
    python compare.py
"""

import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import jax
import jax.numpy as jnp

import config.params as p
from solver.grid import build_grid, build_laser_source
from solver.ops import apply_bc
from solver.cn_step import make_cn_step_jit
from amr.cells import build_amr_cells
from analysis.metrics import l2_error, Timer
from analysis.comparison import plot_convergence_rate
from runs.run_composite_amr import run_simulation as run_composite_sim

# ── JIT compiled version of composite sim ─────────────────────────────────────
# Marking grid sizes and n_steps as static
run_composite_sim_jit = jax.jit(run_composite_sim, static_argnums=(0,1,2,3,4,6))

# COMPOSITE_CONFIGS: list of (Nc, Nf)
COMP_CONFIGS = [(128, 256), (256, 512)]

# ── Output directory ──────────────────────────────────────────────────────────
OUT = "output/comparison"
os.makedirs(OUT, exist_ok=True)

# ── Study parameters (keep fast; increase for publication quality) ────────────
CONV_GRID_SIZES   = [128, 256, 512, 1024] # resolutions for convergence study
CONV_N_STEPS      = 50                 # Reduced steps for ultra-high-res speed
TIMING_NX         = 128                # resolution for timing breakdown
TIMING_WARMUP     = 5                  # JIT warm-up steps (discarded)
TIMING_TRIALS     = 3                  # averaged timing trials
TIMING_STEPS      = 50                 # steps per trial
FRONTIER_N_STEPS  = 50                 # steps for efficiency-frontier runs
Nx_REF            = 2048               # Extreme Reference resolution



# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _build_problem(Nx: int):
    """Return (X, Y, Q, T0, step_fn) for a given resolution."""
    Ny  = Nx
    dx  = p.Lx / (Nx - 1)
    dy  = p.Ly / (Ny - 1)
    X, Y = build_grid(Nx, Ny, p.Lx, p.Ly)
    Q    = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power)
    T0   = apply_bc(jnp.zeros((Nx, Ny)))
    step_fn = make_cn_step_jit(p.alpha, p.dt, dx, dy)
    return X, Y, Q, T0, step_fn, dx, dy


def _run(step_fn, T0, Q, n_steps: int) -> tuple:
    """Warm-up JIT then run n_steps; return (T_final_np, wallclock_s)."""
    # warm-up — not timed
    _ = step_fn(T0, Q)
    T = T0
    with Timer() as tm:
        for _ in range(n_steps):
            T = step_fn(T, Q)
        T.block_until_ready()   # flush async JAX dispatch
    return np.asarray(T), tm.elapsed


def _downsample(T_fine: np.ndarray, Nx_coarse: int) -> np.ndarray:
    """Stride-downsample a fine-grid array to a coarser resolution."""
    s = T_fine.shape[0] // Nx_coarse
    return T_fine[::s, ::s][:Nx_coarse, :Nx_coarse]


def _save(fig: plt.Figure, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 1 — Convergence: L2 error vs grid resolution
# ═════════════════════════════════════════════════════════════════════════════

def run_convergence() -> tuple:
    """
    Run the solver at each resolution in CONV_GRID_SIZES.
    Reference is Nx = Nx_REF.
    """
    Nx_ref = Nx_REF
    print(f"\n[1/3] Convergence study — reference at Nx={Nx_ref}")

    _, _, Q_ref, T0_ref, step_ref, *_ = _build_problem(Nx_ref)
    T_ref, _ = _run(step_ref, T0_ref, Q_ref, CONV_N_STEPS)
    print(f"      Reference computed  (peak T = {T_ref.max():.3f} K)")

    dofs, errors, walltimes = [], [], []
    for Nx in CONV_GRID_SIZES:
        _, _, Q, T0, step_fn, *_ = _build_problem(Nx)
        T, wall = _run(step_fn, T0, Q, CONV_N_STEPS)
        T_ref_down = _downsample(T_ref, Nx)
        err = l2_error(T, T_ref_down)
        dofs.append(Nx * Nx)
        errors.append(err)
        walltimes.append(wall)
        print(f"      Nx={Nx:4d}  DOF={Nx*Nx:7d}  L2={err:.4e}  t={wall:.3f}s")

    return (CONV_GRID_SIZES,
            np.array(dofs, dtype=np.int64),
            np.array(errors),
            np.array(walltimes))


def plot_convergence(grid_sizes, dofs, errors):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.loglog(dofs, errors, "o-", color="#4a90d9", lw=2, ms=7, label="Uniform CN")

    # Fit slope in log-log space
    log_d = np.log10(dofs.astype(float))
    log_e = np.log10(errors)
    slope, intercept = np.polyfit(log_d, log_e, 1)

    # Reference lines
    xs = np.array([dofs[0], dofs[-1]], dtype=float)
    ax.loglog(xs, errors[0] * (xs / xs[0]) ** (-1.0), "k--",
              lw=1.2, label="O(N⁻¹) slope")
    ax.loglog(xs, errors[0] * (xs / xs[0]) ** (-2.0), "k:",
              lw=1.2, label="O(N⁻²) slope")

    # Annotate Nx values
    for Nx, d, e in zip(grid_sizes, dofs, errors):
        ax.annotate(f"Nx={Nx}", (d, e), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#333")

    ax.set_xlabel("DOF count  (Nx²)")
    ax.set_ylabel("L2 error  (vs Nx=256 reference)")
    ax.set_title(f"Convergence — fitted slope = {slope:.2f}")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 2 — Timing breakdown: solver vs AMR overhead
# ═════════════════════════════════════════════════════════════════════════════

def run_timing() -> dict:
    """
    At Nx = TIMING_NX, measure:
      - solver_only  : time per step for pure cn_step
      - amr_overhead : extra time per step for build_amr_cells

    JIT warm-up steps are discarded. TIMING_TRIALS runs are averaged.
    """
    Nx = TIMING_NX
    print(f"\n[2/3] Timing breakdown at Nx={Nx}  "
          f"({TIMING_TRIALS} trials × {TIMING_STEPS} steps)")

    X, Y, Q, T0, step_fn, dx, dy = _build_problem(Nx)
    tiers = p.REFINE_TIERS

    # ── JIT warm-up (discarded) ───────────────────────────────────────────
    T = T0
    for _ in range(TIMING_WARMUP):
        T = step_fn(T, Q)
    T.block_until_ready()

    # ── Solver-only trials ────────────────────────────────────────────────
    solver_times = []
    for trial in range(TIMING_TRIALS):
        T = T0
        t0 = time.perf_counter()
        for _ in range(TIMING_STEPS):
            T = step_fn(T, Q)
        T.block_until_ready()
        solver_times.append(time.perf_counter() - t0)

    solver_per_step = np.mean(solver_times) / TIMING_STEPS
    solver_std      = np.std(solver_times)  / TIMING_STEPS

    # ── Solver + AMR-cell-build trials ───────────────────────────────────
    amr_times = []
    for trial in range(TIMING_TRIALS):
        T = T0
        t0 = time.perf_counter()
        for _ in range(TIMING_STEPS):
            T = step_fn(T, Q)
            T.block_until_ready()
            _ = build_amr_cells(np.asarray(T), dx, dy,
                                 p.Lx, p.Ly, p.MACRO, tiers, p.MAX_LEVEL)
        amr_times.append(time.perf_counter() - t0)

    total_per_step  = np.mean(amr_times) / TIMING_STEPS
    amr_overhead    = total_per_step - solver_per_step
    overhead_pct    = 100.0 * amr_overhead / solver_per_step

    print(f"      Solver only   : {solver_per_step*1e3:.2f} ms/step "
          f"(±{solver_std*1e3:.2f})")
    print(f"      + AMR cells   : {amr_overhead*1e3:.2f} ms/step overhead "
          f"({overhead_pct:.1f}%)")

    return dict(
        solver_ms   = solver_per_step * 1e3,
        solver_std  = solver_std      * 1e3,
        amr_ms      = amr_overhead    * 1e3,
        overhead_pct= overhead_pct,
    )


def plot_timing(timing: dict):
    solver_ms  = timing["solver_ms"]
    amr_ms     = timing["amr_ms"]
    solver_std = timing["solver_std"]
    pct        = timing["overhead_pct"]

    fig, ax = plt.subplots(figsize=(5, 4))
    x = [0]
    bar_s = ax.bar(x, [solver_ms], width=0.4, color="#4a90d9",
                   label="Solver  (cn_step)", alpha=0.9, edgecolor="k")
    bar_a = ax.bar(x, [amr_ms], width=0.4, bottom=[solver_ms],
                   color="#ffa502", label=f"AMR overhead  (+{pct:.1f}%)",
                   alpha=0.9, edgecolor="k")

    ax.errorbar(x, [solver_ms / 2], yerr=[solver_std], fmt="none",
                color="white", capsize=5, lw=1.5)

    ax.set_xticks([0])
    ax.set_xticklabels([f"Nx = {TIMING_NX}"])
    ax.set_ylabel("Time per step  [ms]")
    ax.set_title(f"Per-step timing breakdown\n"
                 f"({TIMING_TRIALS} trials × {TIMING_STEPS} steps, post JIT warm-up)")
    ax.legend(fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(axis="y", which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Analysis 3 — Efficiency frontier: accuracy vs wallclock
# ═════════════════════════════════════════════════════════════════════════════

def run_frontier() -> dict:
    """
    For each Nx in CONV_GRID_SIZES run FRONTIER_N_STEPS and record:
      - uniform wallclock
      - uniform L2 error (vs reference)
      - AMR wallclock  (solver + cell build every step)
      - AMR L2 error   (same T field, so same accuracy — shows pure overhead)

    Returns dict with keys: grid_sizes, uniform_wall, uniform_err,
                                        amr_wall,     amr_err
    """
    Nx_ref = Nx_REF
    print(f"\n[3/3] Efficiency frontier — {FRONTIER_N_STEPS} steps, "
          f"ref Nx={Nx_ref}")

    _, _, Q_ref, T0_ref, step_ref, *_ = _build_problem(Nx_ref)
    T_ref, _ = _run(step_ref, T0_ref, Q_ref, FRONTIER_N_STEPS)

    uniform_wall, uniform_err = [], []
    amr_wall,     amr_err     = [], []
    comp_wall,    comp_err    = [], []
    comp_dofs                 = []

    for Nx in CONV_GRID_SIZES:
        X, Y, Q, T0, step_fn, dx, dy = _build_problem(Nx)
        tiers = p.REFINE_TIERS
        T_ref_d = _downsample(T_ref, Nx)

        # Warm-up
        _ = step_fn(T0, Q)

        # Uniform
        T_u, wall_u = _run(step_fn, T0, Q, FRONTIER_N_STEPS)
        
        # 1. Slice T_u to the patch bounds
        ix0_u = int(p.patch_x0 * (Nx - 1))
        ix1_u = int(p.patch_x1 * (Nx - 1))
        iy0_u = int(p.patch_y0 * (Nx - 1))
        iy1_u = int(p.patch_y1 * (Nx - 1))
        T_u_patch = T_u[ix0_u:ix1_u+1, iy0_u:iy1_u+1]
        
        # 2. Slice T_ref to the patch bounds
        ix0_r = int(p.patch_x0 * (Nx_ref - 1))
        ix1_r = int(p.patch_x1 * (Nx_ref - 1))
        iy0_r = int(p.patch_y0 * (Nx_ref - 1))
        iy1_r = int(p.patch_y1 * (Nx_ref - 1))
        T_r_patch = T_ref[ix0_r:ix1_r+1, iy0_r:iy1_r+1]
        
        # 3. Downsample T_r_patch to current Nx-patch resolution
        # Use current patch size
        Nu_f_x = ix1_u - ix0_u + 1
        Nu_f_y = iy1_u - iy0_u + 1
        
        from jax.scipy.ndimage import map_coordinates
        ix_f = jnp.linspace(0, T_r_patch.shape[0]-1, Nu_f_x)
        iy_f = jnp.linspace(0, T_r_patch.shape[1]-1, Nu_f_y)
        IXf, IYf = jnp.meshgrid(ix_f, iy_f, indexing="ij")
        T_r_patch_d = map_coordinates(jnp.array(T_r_patch), [IXf, IYf], order=1)
        
        err_u = l2_error(np.asarray(T_u_patch), np.asarray(T_r_patch_d))
        
        uniform_wall.append(wall_u)
        uniform_err.append(err_u)

        # AMR (solver + cell build each step — measures real overhead)
        T = T0
        t0 = time.perf_counter()
        for _ in range(FRONTIER_N_STEPS):
            T = step_fn(T, Q)
            T.block_until_ready()
            build_amr_cells(np.asarray(T), dx, dy,
                            p.Lx, p.Ly, p.MACRO, tiers, p.MAX_LEVEL)
        wall_a = time.perf_counter() - t0
        err_a = l2_error(np.asarray(T), T_ref_d)   # same T → same error
        amr_wall.append(wall_a)
        amr_err.append(err_a)

        print(f"      Nx={Nx:4d}  "
              f"uniform ({wall_u:.3f}s, L2={err_u:.3e})  "
              f"amr-overlay ({wall_a:.3f}s)")

    # Composite AMR runs
    for (Nc, Nf) in COMP_CONFIGS:
        # Warm-up
        _ = run_composite_sim_jit(Nc_x=Nc, Nc_y=Nc, Nf_x=Nf, Nf_y=Nf, n_steps=FRONTIER_N_STEPS)

        t0 = time.perf_counter()
        Tc, Tp = run_composite_sim_jit(Nc_x=Nc, Nc_y=Nc, Nf_x=Nf, Nf_y=Nf, n_steps=FRONTIER_N_STEPS)
        Tc.block_until_ready()
        wall_c = time.perf_counter() - t0
        # Error calculation: We compare the FINE PATCH against the reference.
        # Reference is Nx_REF (e.g. 512). 
        # We need to extract the same region from the reference and downsample to Nf.
        
        # 1. Slice reference to the patch bounds
        # Ref indices for x0, x1:
        ix0 = int(p.patch_x0 * (Nx_ref - 1))
        ix1 = int(p.patch_x1 * (Nx_ref - 1))
        iy0 = int(p.patch_y0 * (Nx_ref - 1))
        iy1 = int(p.patch_y1 * (Nx_ref - 1))
        T_ref_patch = T_ref[ix0:ix1+1, iy0:iy1+1]
        
        # 2. Downsample ref_patch to Nf
        # Stride s:
        s_x = T_ref_patch.shape[0] / Nf
        s_y = T_ref_patch.shape[1] / Nf
        
        # Simplest way: just use map_coordinates for high-quality downsampling
        from jax.scipy.ndimage import map_coordinates
        ix_f = jnp.linspace(0, T_ref_patch.shape[0]-1, Nf)
        iy_f = jnp.linspace(0, T_ref_patch.shape[1]-1, Nf)
        IXf, IYf = jnp.meshgrid(ix_f, iy_f, indexing="ij")
        T_ref_patch_d = map_coordinates(jnp.array(T_ref_patch), [IXf, IYf], order=1)
        
        err_c = l2_error(np.asarray(Tp), np.asarray(T_ref_patch_d))
        
        comp_wall.append(wall_c)
        comp_err.append(err_c)
        print(f"      Composite (Nc={Nc}, Nf={Nf})  ({wall_c:.3f}s, Patch-L2={err_c:.3e})")

    return dict(
        grid_sizes   = CONV_GRID_SIZES,
        uniform_wall = np.array(uniform_wall),
        uniform_err  = np.array(uniform_err),
        amr_wall     = np.array(amr_wall),
        amr_err      = np.array(amr_err),
        comp_wall    = np.array(comp_wall),
        comp_err     = np.array(comp_err),
        comp_configs = COMP_CONFIGS,
    )


def plot_frontier(frontier: dict):
    uw = frontier["uniform_wall"]
    ue = frontier["uniform_err"]
    aw = frontier["amr_wall"]
    ae = frontier["amr_err"]
    cw = frontier["comp_wall"]
    ce = frontier["comp_err"]
    gs = frontier["grid_sizes"]
    cs = frontier["comp_configs"]

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogy(uw, ue, "o-", color="#4a90d9", lw=2, ms=8, label="Uniform")
    ax.semilogy(aw, ae, "s--", color="#ffa502", lw=2, ms=8, label="Uniform + AMR-overlay")
    ax.semilogy(cw, ce, "D-", color="#2ed573", lw=2, ms=8, label="Composite JIT-AMR")

    # Annotate resolution
    for Nx, w, e in zip(gs, uw, ue):
        ax.annotate(f"Nx={Nx}", (w, e), textcoords="offset points",
                    xytext=(-6, 8), fontsize=8, color="#4a90d9", ha="right")
    
    for (Nc, Nf), w, e in zip(cs, cw, ce):
        ax.annotate(f"Nc={Nc},Nf={Nf}", (w, e), textcoords="offset points",
                    xytext=(6, 8), fontsize=8, color="#2ed573", ha="left")

    ax.set_xlabel("Wallclock time  [s]")
    ax.set_ylabel("L2 error  (vs Nx=256 reference)")
    ax.set_title(f"Efficiency frontier  ({FRONTIER_N_STEPS} steps)\n"
                 "Lower-left = better (More accuracy for less time)")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", ls=":", alpha=0.4)
    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# Summary table
# ═════════════════════════════════════════════════════════════════════════════

def write_summary(conv, timing, frontier):
    grid_sizes, dofs, conv_errors, conv_wall = conv
    path = os.path.join(OUT, "summary.txt")
    lines = [
        "JAX-AMR  Performance Summary",
        "Author : Ashwin Shirke",
        "=" * 60,
        "",
        f"Convergence study  (n_steps={CONV_N_STEPS}, ref Nx={Nx_REF})",
        f"{'Nx':>6}  {'DOF':>8}  {'L2 error':>12}  {'wallclock':>10}",
        "-" * 46,
    ]
    for Nx, d, e, w in zip(grid_sizes, dofs, conv_errors, conv_wall):
        lines.append(f"{Nx:6d}  {d:8d}  {e:12.4e}  {w:9.3f}s")

    lines += [
        "",
        f"Timing breakdown  (Nx={TIMING_NX}, {TIMING_TRIALS} trials × {TIMING_STEPS} steps)",
        f"  Solver only     : {timing['solver_ms']:.3f} ms/step  "
        f"(±{timing['solver_std']:.3f})",
        f"  AMR overhead    : {timing['amr_ms']:.3f} ms/step  "
        f"({timing['overhead_pct']:.1f}% of solver cost)",
        "",
        f"Efficiency frontier  (n_steps={FRONTIER_N_STEPS}, ref Nx={Nx_REF})",
        f"{'Grid':>12}  {'wall':>10}  {'L2 error':>12}",
        "-" * 40,
    ]
    for Nx, uw, ue in zip(frontier["grid_sizes"], frontier["uniform_wall"], frontier["uniform_err"]):
        lines.append(f"{'Uniform':>5} {Nx:6d}  {uw:9.3f}s  {ue:12.4e}")
    
    for (Nc, Nf), cw, ce in zip(frontier["comp_configs"], frontier["comp_wall"], frontier["comp_err"]):
        lines.append(f"{'Comp':>5} {Nc:d},{Nf:d}  {cw:9.3f}s  {ce:12.4e}")
    
    lines += ["", "=" * 60]

    text = "\n".join(lines)
    with open(path, "w") as f:
        f.write(text)
    print(f"\n{text}")
    print(f"\n  Saved → {path}")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("JAX-AMR  comparison study")
    print("Author : Ashwin Shirke")
    print(f"JAX devices: {jax.devices()}")
    print("=" * 60)

    # 1. Convergence
    conv    = run_convergence()
    fig1    = plot_convergence(*conv[:3])
    _save(fig1, "1_convergence.png")

    # 2. Timing breakdown
    timing  = run_timing()
    fig2    = plot_timing(timing)
    _save(fig2, "2_timing_breakdown.png")

    # 3. Efficiency frontier
    frontier = run_frontier()
    fig3     = plot_frontier(frontier)
    _save(fig3, "3_efficiency_frontier.png")

    # Summary
    write_summary(conv, timing, frontier)

    print(f"\nDone — all outputs in {OUT}/")


if __name__ == "__main__":
    main()

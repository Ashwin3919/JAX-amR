"""
Composite JIT-AMR driver (Model 3).

High-performance two-level solver (coarse + fine patch).
Fully JIT-compilable and differentiable.

Usage:
    PYTHONPATH=. python runs/run_composite_amr.py               # normal output
    PYTHONPATH=. python runs/run_composite_amr.py --plot-grid   # + fixed-patch overlay
"""
import sys, os

# Precision Toggle: Use True for float64 (high accuracy), False for float32 (fast)
USE_64BIT = True
os.environ["JAX_ENABLE_X64"] = "1" if USE_64BIT else "0"

from typing import Any, Dict, List, Optional, Tuple

_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "src"))
os.environ.setdefault("JAX_PLATFORMS", "") 
import argparse
import logging
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import config.params as p

logger = logging.getLogger(__name__)
from solver.grid import build_grid
from solver.laser_source import build_laser_source
from amr.patch import build_patch_info, PatchInfo
from amr.composite_step import composite_step

from ioutils.vtk_writer import write_legacy_vtk, write_pvd
from ioutils.checkpoint import save_checkpoint
from analysis.metrics import Timer
from viz.snapshots import plot_snapshots
from viz.animate import create_animation, save_gif
from viz_utils import coarse_cells, bounds_to_cells


def run_simulation(
        Nc_x: Optional[int] = None,
        Nc_y: Optional[int] = None,
        Nf_x: Optional[int] = None,
        Nf_y: Optional[int] = None,
        patch_bounds: Optional[Tuple[float, float, float, float]] = None,
        laser_power: Optional[float] = None,
        n_steps: Optional[int] = None,
        output_dir: str = "output/amr_fixed",
        save_vtk: bool = True,
) -> Dict[str, Any]:
    """
    Run the composite fixed-patch AMR simulation.

    Parameters
    ----------
    Nc_x, Nc_y   : coarse grid resolution (defaults to config.params)
    Nf_x, Nf_y   : fine patch resolution  (defaults to config.params)
    patch_bounds : (px0, px1, py0, py1) physical bounds of the fine patch
    laser_power  : laser power density W/m²  (defaults to config.params)
    n_steps      : number of time steps      (defaults to config.params)
    output_dir   : directory for VTK / checkpoint output
    save_vtk     : whether to write VTK and PVD files

    Returns
    -------
    Dict with keys:
      T_final      : (T_coarse, T_patch) arrays at the final step
      frames       : list of coarse np.ndarray snapshots (one per chunk)
      times        : list of float times corresponding to each frame
      patch_bounds : (px0, px1, py0, py1) used for the fine patch
      wallclock    : total wall-clock time in seconds
    """
    os.makedirs(output_dir, exist_ok=True)
    Nc_x = p.Nc_x if Nc_x is None else Nc_x
    Nc_y = p.Nc_y if Nc_y is None else Nc_y
    Nf_x = p.Nf_x if Nf_x is None else Nf_x
    Nf_y = p.Nf_y if Nf_y is None else Nf_y
    if patch_bounds is None:
        px0, px1, py0, py1 = p.patch_x0, p.patch_x1, p.patch_y0, p.patch_y1
    else:
        px0, px1, py0, py1 = patch_bounds
    laser_power = p.laser_power if laser_power is None else laser_power
    n_steps = p.n_steps if n_steps is None else n_steps

    # 1. Initialize Grids
    Xc, Yc = build_grid(Nc_x, Nc_y, p.Lx, p.Ly)
    dx_c = p.Lx / (Nc_x - 1)
    dy_c = p.Ly / (Nc_y - 1)

    patch = build_patch_info(px0, px1, py0, py1,
                             Nf_x, Nf_y, Nc_x, Nc_y, p.Lx, p.Ly)
    dx_f = (px1 - px0) / (Nf_x - 1)
    dy_f = (py1 - py0) / (Nf_y - 1)

    # 2. Initial state
    Tc = jnp.full((Nc_x, Nc_y), p.T_init)
    Tp = jnp.full((Nf_x, Nf_y), p.T_init)

    # 3. Output containers
    pvd_coarse = []
    pvd_patch = []
    frames = [np.asarray(Tc)]
    times = [0.0]

    chunk_size = p.save_every
    n_chunks = n_steps // chunk_size
    if n_steps % chunk_size != 0:
        import warnings
        warnings.warn(
            f"n_steps={n_steps} is not divisible by save_every={chunk_size}. "
            f"The last {n_steps % chunk_size} steps will be skipped. "
            "Set n_steps to a multiple of save_every to avoid this.",
            stacklevel=2,
        )

    @jax.jit
    def run_chunk(state, t_start):
        def body(carry, step_idx):
            Tc_k, Tp_k = carry
            t_curr = t_start + step_idx * p.dt
            Qc_k = build_laser_source(Xc, Yc, p.laser_cx, p.laser_cy,
                                      p.laser_sigma, laser_power, t_curr)
            Qf_k = build_laser_source(patch.Xf, patch.Yf, p.laser_cx, p.laser_cy,
                                      p.laser_sigma, laser_power, t_curr)
            return composite_step(Tc_k, Tp_k, Qc_k, Qf_k, patch, p.alpha, p.dt,
                                  dx_c, dy_c, dx_f, dy_f, p.T_wall), None

        final_s, _ = lax.scan(body, state, jnp.arange(chunk_size))
        return final_s

    with Timer() as timer:
        for i in range(n_chunks):
            t_chunk_start = i * chunk_size * p.dt
            Tc, Tp = run_chunk((Tc, Tp), t_chunk_start)
            step = (i + 1) * chunk_size
            t = step * p.dt

            frames.append(np.asarray(Tc))
            times.append(t)

            if save_vtk and p.vtk_every > 0 and step % p.vtk_every == 0:
                c_path = os.path.join(output_dir, f"coarse_t{step:05d}.vtk")
                p_path = os.path.join(output_dir, f"patch_t{step:05d}.vtk")
                write_legacy_vtk(c_path, np.asarray(Xc), np.asarray(Yc),
                                 np.asarray(Tc), title=f"Coarse_t{step}")
                write_legacy_vtk(p_path, np.asarray(patch.Xf), np.asarray(patch.Yf),
                                 np.asarray(Tp), title=f"Patch_t{step}")
                pvd_coarse.append((t, c_path))
                pvd_patch.append((t, p_path))

            if p.checkpoint_every > 0 and step % p.checkpoint_every == 0:
                save_checkpoint(os.path.join(output_dir, f"ckpt_{step:05d}.npz"),
                                np.asarray(Tc), step, t)

    if save_vtk and pvd_coarse:
        write_pvd(os.path.join(output_dir, "amr_coarse.pvd"), pvd_coarse)
        write_pvd(os.path.join(output_dir, "amr_patch.pvd"), pvd_patch)

    logger.info("[amr] %d steps | %.2fs | peak T = %.4f K",
                n_steps, timer.elapsed, np.asarray(Tp).max())

    return dict(T_final=(Tc, Tp), frames=frames, times=times,
                patch_bounds=(px0, px1, py0, py1), wallclock=timer.elapsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Composite JIT-AMR solver")
    parser.add_argument("--plot-grid", action="store_true",
                        help="Also generate fixed-patch overlay animation and snapshots")
    args = parser.parse_args()

    n_steps = p.n_steps
    Nc, Nf = 128, 512
    # Patch covers [0.25, 0.75]² — fully contains the laser orbit (R=0.2, centre=0.5)
    # dx_f = 0.5/511 ≈ 1/1024, matching the uniform reference resolution
    patch = (0.25, 0.75, 0.25, 0.75)
    logger.info("Starting AMR FIXED simulation (coarse=%dx%d dx≈1/128, fixed fine patch=%dx%d dx≈1/1024 at [%s,%s]²)...",
                Nc, Nc, Nf, Nf, patch[0], patch[1])

    res = run_simulation(Nc_x=Nc, Nc_y=Nc, Nf_x=Nf, Nf_y=Nf,
                         patch_bounds=patch, n_steps=n_steps)
    Tc_final, Tp_final = res["T_final"]
    frames = res["frames"]
    times = res["times"]
    px0, px1, py0, py1 = res["patch_bounds"]

    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)

    # --- always: standard snapshots + animation ---
    fig = plot_snapshots(frames, Xc, Yc, times,
                         title=f"AMR Fixed — pre-placed patch (coarse {Nc}x{Nc}, fine {Nf}x{Nf})")
    fig.savefig("output/amr_fixed/snapshots.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    logger.info("Saved output/amr_fixed/snapshots.png")

    fig2, anim = create_animation(frames, Xc, Yc, times)
    save_gif(anim, "output/amr_fixed/animation.gif")
    logger.info("Saved output/amr_fixed/animation.gif")

    jnp.save("output/amr_fixed/composite_coarse.npy", Tc_final)
    jnp.save("output/amr_fixed/composite_patch.npy", Tp_final)
    logger.info("Saved results to output/amr_fixed/")

    # --- --plot-grid: fixed patch rectangle on every frame ---
    if args.plot_grid:
        patch_cell = bounds_to_cells(px0, px1, py0, py1)
        amr_frames = [patch_cell] * len(frames)

        fig3 = plot_snapshots(frames, Xc, Yc, times,
                              amr_frames=amr_frames,
                              title=f"AMR Fixed — patch region (coarse {Nc}x{Nc}, fine {Nf}x{Nf})")
        fig3.savefig("output/amr_fixed/snapshots_grid.png", dpi=150,
                     bbox_inches="tight", facecolor="#0d0d0d")
        logger.info("Saved output/amr_fixed/snapshots_grid.png")

        fig4, anim2 = create_animation(frames, Xc, Yc, times, amr_frames=amr_frames)
        save_gif(anim2, "output/amr_fixed/animation_grid.gif")
        logger.info("Saved output/amr_fixed/animation_grid.gif")

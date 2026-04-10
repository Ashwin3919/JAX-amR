"""
Uniform-grid driver (Model 1).

Usage:
    PYTHONPATH=. python runs/run_uniform.py               # normal output
    PYTHONPATH=. python runs/run_uniform.py --plot-grid   # + gradient-cell overlay
"""
import sys, os
_root = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(_root, "src"))
os.environ.setdefault("JAX_PLATFORMS", "")  # suppress "no TPU" warnings
import argparse
import logging
import numpy as np
import jax.numpy as jnp

import config.params as p

logger = logging.getLogger(__name__)
from solver.grid import build_grid
from solver.laser_source import build_laser_source
from solver.ops import apply_bc
from solver.cn_step import make_cn_step_jit
from ioutils.vtk_writer import write_legacy_vtk, write_pvd
from ioutils.checkpoint import save_checkpoint
from viz.snapshots import plot_snapshots
from viz.animate import create_animation, save_gif
from analysis.metrics import Timer


def run_uniform(Nx: int = None, Ny: int = None,
                output_dir: str = "output/uniform",
                n_steps: int = None,
                save_vtk: bool = True) -> dict:
    """
    Run the uniform-grid heat solver and write VTK + checkpoint output.

    Returns a dict with keys: T_final, frames, times, wallclock.
    """
    os.makedirs(output_dir, exist_ok=True)
    if n_steps is None:
        n_steps = p.n_steps
    Nx = Nx or p.Nx
    Ny = Ny or p.Ny
    dx = p.Lx / (Nx - 1)
    dy = p.Ly / (Ny - 1)

    X, Y = build_grid(Nx, Ny, p.Lx, p.Ly)
    T = apply_bc(jnp.zeros((Nx, Ny)))

    step_fn = make_cn_step_jit(p.alpha, p.dt, dx, dy)
    Q0 = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power, 0.0)
    _ = step_fn(T, Q0)

    frames = [np.asarray(T)]
    times  = [0.0]
    pvd_entries = []

    with Timer() as timer:
        for step in range(n_steps):
            t = step * p.dt
            Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power, t)
            T = step_fn(T, Q)
            t_next = (step + 1) * p.dt

            if (step + 1) % p.save_every == 0:
                frames.append(np.asarray(T))
                times.append(t_next)

            if save_vtk and p.vtk_every > 0 and (step + 1) % p.vtk_every == 0:
                vtk_path = os.path.join(output_dir, f"temp_t{step+1:05d}.vtk")
                write_legacy_vtk(vtk_path, np.asarray(X), np.asarray(Y),
                                 np.asarray(T), title=f"Uniform_t{step+1}")
                pvd_entries.append((t_next, vtk_path))

            if p.checkpoint_every > 0 and (step + 1) % p.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f"ckpt_{step+1:05d}.npz"),
                    np.asarray(T), step + 1, t_next,
                )

    if save_vtk and pvd_entries:
        write_pvd(os.path.join(output_dir, "uniform.pvd"), pvd_entries)

    logger.info("[uniform] %d steps | %.2fs | peak T = %.4f K",
                n_steps, timer.elapsed, np.asarray(T).max())

    return dict(T_final=T, frames=frames, times=times, wallclock=timer.elapsed)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Uniform grid solver")
    parser.add_argument("--plot-grid", action="store_true",
                        help="Also generate grid-structure overlay animation")
    args = parser.parse_args()

    Nx = 1024
    logger.info("Starting ultra-high-resolution UNIFORM simulation (%dx%d)...", Nx, Nx)
    res = run_uniform(Nx=Nx, Ny=Nx, n_steps=p.n_steps)
    X, Y = build_grid(Nx, Nx, p.Lx, p.Ly)

    # --- always: standard snapshots + animation ---
    fig = plot_snapshots(res["frames"], X, Y, res["times"], title="Uniform Grid")
    fig.savefig("output/uniform/snapshots.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    logger.info("Saved output/uniform/snapshots.png")

    fig2, anim = create_animation(res["frames"], X, Y, res["times"])
    save_gif(anim, "output/uniform/animation.gif")
    logger.info("Saved output/uniform/animation.gif")

    # --- --plot-grid only: 16x16 equal white cells across the full domain ---
    # Shows that resolution is the same everywhere — no coarse/fine distinction.
    if args.plot_grid:
        n = 16
        w = p.Lx / n
        h = p.Ly / n
        uniform_cells = [(i*w, j*h, (i+1)*w, (j+1)*h, 3)
                         for i in range(n) for j in range(n)]
        amr_frames = [uniform_cells] * len(res["frames"])

        fig3 = plot_snapshots(res["frames"], X, Y, res["times"],
                              amr_frames=amr_frames,
                              title="Uniform Grid — same fine resolution everywhere")
        fig3.savefig("output/uniform/snapshots_grid.png", dpi=150,
                     bbox_inches="tight", facecolor="#0d0d0d")
        logger.info("Saved output/uniform/snapshots_grid.png")

        fig4, anim2 = create_animation(res["frames"], X, Y, res["times"],
                                       amr_frames=amr_frames)
        save_gif(anim2, "output/uniform/animation_grid.gif")
        logger.info("Saved output/uniform/animation_grid.gif")

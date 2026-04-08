"""
AMR driver (v2 path).

Full fine-grid solve in JAX; AMR grid built as numpy post-process overlay.

Usage:
    python -m runs.run_amr
"""
import os
import numpy as np
import jax.numpy as jnp

import config.params as p
from solver.grid import build_grid, build_laser_source
from solver.ops import apply_bc
from solver.cn_step import make_cn_step_jit
from amr.cells import build_amr_cells
from ioutils.vtk_writer import write_amr_vtk, write_pvd
from ioutils.checkpoint import save_checkpoint
from viz.snapshots import plot_snapshots
from viz.animate import create_animation, save_gif
from analysis.metrics import Timer


def run_amr(output_dir: str = "output/amr",
            n_steps: int = None,
            save_vtk: bool = True) -> dict:
    """
    Run the AMR-overlay heat solver and write VTK + checkpoint output.

    Returns a dict with keys: T_final, frames, times, amr_frames, wallclock.
    """
    os.makedirs(output_dir, exist_ok=True)
    if n_steps is None:
        n_steps = p.n_steps

    X, Y = build_grid(p.Nx, p.Ny, p.Lx, p.Ly)
    Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power)
    T = apply_bc(jnp.zeros((p.Nx, p.Ny)))

    step_fn = make_cn_step_jit(p.alpha, p.dt, p.dx, p.dy)
    _ = step_fn(T, Q)   # warm-up

    tiers = p.REFINE_TIERS
    frames     = [np.asarray(T)]
    times      = [0.0]
    amr_frames = []
    pvd_entries = []

    with Timer() as timer:
        for step in range(n_steps):
            T = step_fn(T, Q)
            t = (step + 1) * p.dt

            if (step + 1) % p.save_every == 0:
                T_np = np.asarray(T)
                frames.append(T_np)
                times.append(t)
                cells, _ = build_amr_cells(T_np, p.dx, p.dy, p.Lx, p.Ly,
                                           p.MACRO, tiers, p.MAX_LEVEL)
                amr_frames.append(cells)

            if save_vtk and p.vtk_every > 0 and (step + 1) % p.vtk_every == 0:
                T_np = np.asarray(T)
                cells, _ = build_amr_cells(T_np, p.dx, p.dy, p.Lx, p.Ly,
                                           p.MACRO, tiers, p.MAX_LEVEL)
                vtk_path = os.path.join(output_dir, f"amr_t{step+1:05d}.vtu")
                write_amr_vtk(vtk_path, cells, t)
                pvd_entries.append((t, vtk_path))

            if p.checkpoint_every > 0 and (step + 1) % p.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f"ckpt_{step+1:05d}.npz"),
                    np.asarray(T), step + 1, t,
                )

    if save_vtk and pvd_entries:
        write_pvd(os.path.join(output_dir, "amr.pvd"), pvd_entries)

    print(f"[amr] {n_steps} steps | {timer.elapsed:.2f}s | "
          f"peak T = {np.asarray(T).max():.2f} K")

    return dict(T_final=T, frames=frames, times=times,
                amr_frames=amr_frames, wallclock=timer.elapsed)


if __name__ == "__main__":
    res = run_amr()
    X, Y = build_grid(p.Nx, p.Ny, p.Lx, p.Ly)

    # AMR frames start at save_every — align with frames (skip frame[0] = T_init)
    amr_for_snap = res["amr_frames"] if res["amr_frames"] else None
    frames_for_snap = res["frames"][1:] if amr_for_snap else res["frames"]
    times_for_snap  = res["times"][1:]  if amr_for_snap else res["times"]

    fig = plot_snapshots(frames_for_snap, X, Y, times_for_snap,
                         amr_frames=amr_for_snap,
                         title="AMR Overlay")
    fig.savefig("output/amr/snapshots.png", dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    print("Saved snapshots.png")

    fig2, anim = create_animation(frames_for_snap, X, Y, times_for_snap,
                                  amr_frames=amr_for_snap)
    save_gif(anim, "output/amr/animation.gif")
    print("Saved animation.gif")

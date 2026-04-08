"""
Uniform-grid driver (Model 1).

Usage:
    PYTHONPATH=. python runs/run_uniform.py
"""
import os
import numpy as np
import jax.numpy as jnp

import config.params as p
from solver.grid import build_grid, build_laser_source
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
    # Warm-up JIT with t=0
    Q0 = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power, 0.0)
    _ = step_fn(T, Q0)

    frames = [np.asarray(T)]
    times  = [0.0]
    pvd_entries = []

    with Timer() as timer:
        for step in range(n_steps):
            t = step * p.dt
            # Recalculate source for moving laser
            Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power, t)
            
            T = step_fn(T, Q)
            t_next = (step + 1) * p.dt

            if (step + 1) % p.save_every == 0:
                T_np = np.asarray(T)
                frames.append(T_np)
                times.append(t_next)

            if save_vtk and p.vtk_every > 0 and (step + 1) % p.vtk_every == 0:
                vtk_path = os.path.join(output_dir, f"temp_t{step+1:05d}.vtk")
                write_legacy_vtk(vtk_path, np.asarray(X), np.asarray(Y), np.asarray(T), title=f"Uniform_t{step+1}")
                pvd_entries.append((t_next, vtk_path))

            if p.checkpoint_every > 0 and (step + 1) % p.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f"ckpt_{step+1:05d}.npz"),
                    np.asarray(T), step + 1, t_next,
                )

    if save_vtk and pvd_entries:
        write_pvd(os.path.join(output_dir, "uniform.pvd"), pvd_entries)

    print(f"[uniform] {n_steps} steps | {timer.elapsed:.2f}s | "
          f"peak T = {np.asarray(T).max():.4f} K")

    return dict(T_final=T, frames=frames, times=times, wallclock=timer.elapsed)


if __name__ == "__main__":
    Nx = 1024
    print(f"Starting ultra-high-resolution UNIFORM simulation ({Nx}x{Nx})...")
    res = run_uniform(Nx=Nx, Ny=Nx, n_steps=p.n_steps)
    X, Y = build_grid(Nx, Nx, p.Lx, p.Ly)

    fig = plot_snapshots(res["frames"], X, Y, res["times"], title="Uniform Grid")
    fig.savefig("output/uniform/snapshots.png", dpi=150, bbox_inches="tight",
                facecolor="#0d0d0d")
    print("Saved snapshots.png")

    fig2, anim = create_animation(res["frames"], X, Y, res["times"])
    save_gif(anim, "output/uniform/animation.gif")
    print("Saved animation.gif")

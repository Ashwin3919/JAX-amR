import jax
import jax.numpy as jnp
from jax import lax
import time
import os

import config.params as p
from solver.grid import build_grid, build_laser_source
from amr.patch import build_patch_info, PatchInfo
from amr.composite_step import composite_step

from ioutils.vtk_writer import write_legacy_vtk, write_pvd
from ioutils.checkpoint import save_checkpoint
from analysis.metrics import Timer

def run_simulation(Nc_x=None, Nc_y=None, Nf_x=None, Nf_y=None, 
                   patch_bounds=None, laser_power=None, n_steps=None,
                   return_frames=False, output_dir="output/amr", save_vtk=True):
    """
    Runs the composite grid simulation.
    Overrides config.params if arguments are provided.
    """
    os.makedirs(output_dir, exist_ok=True)
    Nc_x = Nc_x or p.Nc_x
    Nc_y = Nc_y or p.Nc_y
    Nf_x = Nf_x or p.Nf_x
    Nf_y = Nf_y or p.Nf_y
    if patch_bounds is None:
        px0, px1, py0, py1 = p.patch_x0, p.patch_x1, p.patch_y0, p.patch_y1
    else:
        px0, px1, py0, py1 = patch_bounds
    laser_power = laser_power or p.laser_power
    n_steps = n_steps or p.n_steps
        
    # 1. Initialize Grids
    Xc, Yc = build_grid(Nc_x, Nc_y, p.Lx, p.Ly)
    dx_c = p.Lx / (Nc_x - 1)
    dy_c = p.Ly / (Nc_y - 1)
    
    patch = build_patch_info(px0, px1, py0, py1, 
                             Nf_x, Nf_y, Nc_x, Nc_y, p.Lx, p.Ly)
    dx_f = (px1 - px0) / (Nf_x - 1)
    dy_f = (py1 - py0) / (Nf_y - 1)
    
    # 2. Pre-calculate Sources
    Qc = build_laser_source(Xc, Yc, p.laser_cx, p.laser_cy, p.laser_sigma, laser_power)
    Qf = build_laser_source(patch.Xf, patch.Yf, p.laser_cx, p.laser_cy, p.laser_sigma, laser_power)
    
    # 3. Initial state
    Tc = jnp.full((Nc_x, Nc_y), p.T_init)
    Tp = jnp.full((Nf_x, Nf_y), p.T_init)
    
    pvd_coarse = []
    pvd_patch = []
    frames = [np.asarray(Tc)]
    times = [0.0]

    chunk_size = p.save_every
    n_chunks = n_steps // chunk_size

    @jax.jit
    def run_chunk(state):
        def body(s, _):
            return composite_step(s[0], s[1], Qc, Qf, patch, p.alpha, p.dt, 
                                  dx_c, dy_c, dx_f, dy_f, p.T_wall), None
        final_s, _ = lax.scan(body, state, None, length=chunk_size)
        return final_s

    with Timer() as timer:
        for i in range(n_chunks):
            Tc, Tp = run_chunk((Tc, Tp))
            step = (i + 1) * chunk_size
            t = step * p.dt
            
            # 1. Frames for animation
            frames.append(np.asarray(Tc))
            times.append(t)
            
            # 2. VTK output
            if save_vtk and p.vtk_every > 0 and step % p.vtk_every == 0:
                c_path = os.path.join(output_dir, f"coarse_t{step:05d}.vtk")
                p_path = os.path.join(output_dir, f"patch_t{step:05d}.vtk")
                write_legacy_vtk(c_path, np.asarray(Xc), np.asarray(Yc), np.asarray(Tc), title=f"Coarse_t{step}")
                write_legacy_vtk(p_path, np.asarray(patch.Xf), np.asarray(patch.Yf), np.asarray(Tp), title=f"Patch_t{step}")
                pvd_coarse.append((t, c_path))
                pvd_patch.append((t, p_path))

            # 3. Checkpoints
            if p.checkpoint_every > 0 and step % p.checkpoint_every == 0:
                save_checkpoint(os.path.join(output_dir, f"ckpt_{step:05d}.npz"),
                                np.asarray(Tc), step, t)

    if save_vtk and pvd_coarse:
        write_pvd(os.path.join(output_dir, "amr_coarse.pvd"), pvd_coarse)
        write_pvd(os.path.join(output_dir, "amr_patch.pvd"), pvd_patch)

    print(f"[amr] {n_steps} steps | {timer.elapsed:.2f}s | "
          f"peak T = {np.asarray(Tp).max():.4f} K")

    return dict(T_final=(Tc, Tp), frames=frames, times=times, wallclock=timer.elapsed)

@jax.jit
def run_simulation_jit():
    return run_simulation()

import numpy as np
from viz.snapshots import plot_snapshots
from viz.animate import create_animation, save_gif

if __name__ == "__main__":
    n_steps = p.n_steps
    Nc, Nf = 256, 512
    print(f"Starting ultra-high-resolution COMPOSITE AMR simulation ({Nc}+{Nf})...")

    # Run simulation (this handles VTK/PVD internally now)
    res = run_simulation(
        Nc_x=Nc, Nc_y=Nc, Nf_x=Nf, Nf_y=Nf, n_steps=n_steps
    )
    Tc_final, Tp_final = res["T_final"]
    frames = res["frames"]
    times = res["times"]

    # 1. Snapshots
    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)
    fig = plot_snapshots(frames, Xc, Yc, times, title=f"Composite AMR (Nc={Nc}, Nf={Nf})")
    fig.savefig("output/amr/snapshots.png", dpi=150, bbox_inches="tight", facecolor="#0d0d0d")
    print("Saved output/amr/snapshots.png")
    
    # 2. Animation
    fig2, anim = create_animation(frames, Xc, Yc, times)
    save_gif(anim, "output/amr/animation.gif")
    print("Saved output/amr/animation.gif")

    # Save raw results
    jnp.save("output/amr/composite_coarse.npy", Tc_final)
    jnp.save("output/amr/composite_patch.npy", Tp_final)
    print("Saved results to output/amr/")


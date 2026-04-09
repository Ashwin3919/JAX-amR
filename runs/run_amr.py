"""
True Adaptive AMR driver (Model 2).

Solves on a coarse base grid everywhere. Each step, the gradient-weighted
centroid of T_coarse is computed and a fine patch of fixed shape (Nf x Nf)
is centered there. The patch follows the laser automatically — no pre-defined
location needed.

Usage:
    PYTHONPATH=. python runs/run_amr.py               # normal output
    PYTHONPATH=. python runs/run_amr.py --plot-grid   # + patch-tracking animation
"""
import os
import argparse
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import config.params as p
from solver.grid import build_grid, build_laser_source
from amr.adaptive_step import adaptive_step
from amr.adaptive_patch import make_fine_coords
from ioutils.vtk_writer import write_legacy_vtk, write_pvd
from ioutils.checkpoint import save_checkpoint
from analysis.metrics import Timer
from viz.snapshots import plot_snapshots
from viz.animate import create_animation, save_gif


def _coarse_cells(n, Lx=1.0, Ly=1.0):
    """n×n equal red cells covering the full domain — represents the coarse background grid."""
    w, h = Lx / n, Ly / n
    return [(i*w, j*h, (i+1)*w, (j+1)*h, 1)
            for i in range(n) for j in range(n)]


def _bounds_to_cells(x0, x1, y0, y1, n_coarse=8):
    """
    8×8 red coarse background + white fine-patch rectangle.
    Shows: red = coarse solve everywhere, white box = where fine solve is this frame.
    """
    return _coarse_cells(n_coarse) + [(float(x0), float(y0), float(x1), float(y1), 3)]


def run_amr(Nc: int = 128, Nf: int = 512, half_w: float = 0.25,
            output_dir: str = "output/amr",
            n_steps: int = None,
            save_vtk: bool = True) -> dict:
    """
    True adaptive AMR solver.

    Parameters
    ----------
    Nc       : coarse grid resolution (Nc x Nc, full domain)
    Nf       : fine patch resolution (Nf x Nf, fixed shape, dynamic location)
    half_w   : half-width of fine patch in physical units (patch = 2*half_w square)
    save_vtk : write VTK + PVD files for coarse and fine grids

    Returns
    -------
    dict with keys: T_final, frames, times, patch_bounds, wallclock
      patch_bounds : list of (x0, x1, y0, y1) tuples, one per saved frame
    """
    os.makedirs(output_dir, exist_ok=True)
    n_steps = n_steps or p.n_steps

    dx_c = p.Lx / (Nc - 1)
    dy_c = p.Ly / (Nc - 1)
    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)
    Xc_np = np.asarray(Xc)
    Yc_np = np.asarray(Yc)

    Tc = jnp.full((Nc, Nc), p.T_init)

    # Initial patch centred on domain
    _Xf0, _Yf0, x0_init, x1_init, y0_init, y1_init = make_fine_coords(
        jnp.array(0.5), jnp.array(0.5), half_w, Nf, Nf, p.Lx, p.Ly
    )
    Tp = jnp.full((Nf, Nf), p.T_init)

    chunk_size = p.save_every
    n_chunks = n_steps // chunk_size

    @jax.jit
    def run_chunk(state, t_start):
        Tc_k, Tp_k, x0_k, x1_k, y0_k, y1_k = state

        def body(carry, step_idx):
            Tc, Tp, x0, x1, y0, y1 = carry
            t = t_start + step_idx * p.dt

            Qc = build_laser_source(
                Xc, Yc, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power, t
            )

            def Q_fine_fn(Xf, Yf):
                return build_laser_source(
                    Xf, Yf, p.laser_cx, p.laser_cy, p.laser_sigma, p.laser_power, t
                )

            Tc_new, Tp_new, x0_new, x1_new, y0_new, y1_new = adaptive_step(
                Tc, Tp, x0, x1, y0, y1,
                Qc, Q_fine_fn,
                Xc, Yc, half_w,
                Nc, Nc, Nf, Nf, p.Lx, p.Ly,
                p.alpha, p.dt, dx_c, dy_c, p.T_wall,
            )
            return (Tc_new, Tp_new, x0_new, x1_new, y0_new, y1_new), None

        final_state, _ = lax.scan(body, (Tc_k, Tp_k, x0_k, x1_k, y0_k, y1_k),
                                  jnp.arange(chunk_size))
        return final_state

    frames = [np.asarray(Tc)]
    times = [0.0]
    patch_bounds = [(float(x0_init), float(x1_init), float(y0_init), float(y1_init))]
    state = (Tc, Tp, x0_init, x1_init, y0_init, y1_init)

    pvd_coarse = []
    pvd_patch = []

    with Timer() as timer:
        for i in range(n_chunks):
            t_start = jnp.array(i * chunk_size * p.dt)
            state = run_chunk(state, t_start)

            Tc = state[0]
            Tp = state[1]
            x0, x1, y0, y1 = state[2], state[3], state[4], state[5]
            step = (i + 1) * chunk_size
            t = float(step * p.dt)

            frames.append(np.asarray(Tc))
            times.append(t)
            patch_bounds.append((float(x0), float(x1), float(y0), float(y1)))

            if save_vtk and p.vtk_every > 0 and step % p.vtk_every == 0:
                # Coarse grid VTK (fixed coordinates)
                c_path = os.path.join(output_dir, f"coarse_t{step:05d}.vtk")
                write_legacy_vtk(c_path, Xc_np, Yc_np, np.asarray(Tc),
                                 title=f"AMR_Coarse_t{step}")
                pvd_coarse.append((t, c_path))

                # Fine patch VTK (dynamic coordinates — reconstruct from bounds)
                x0f, x1f = float(x0), float(x1)
                y0f, y1f = float(y0), float(y1)
                xf = np.linspace(x0f, x1f, Nf)
                yf = np.linspace(y0f, y1f, Nf)
                Xf_np, Yf_np = np.meshgrid(xf, yf, indexing="ij")
                p_path = os.path.join(output_dir, f"patch_t{step:05d}.vtk")
                write_legacy_vtk(p_path, Xf_np, Yf_np, np.asarray(Tp),
                                 title=f"AMR_Patch_t{step}")
                pvd_patch.append((t, p_path))

            if p.checkpoint_every > 0 and step % p.checkpoint_every == 0:
                save_checkpoint(
                    os.path.join(output_dir, f"ckpt_{step:05d}.npz"),
                    np.asarray(Tc), step, t,
                )

    if save_vtk and pvd_coarse:
        write_pvd(os.path.join(output_dir, "amr_coarse.pvd"), pvd_coarse)
        write_pvd(os.path.join(output_dir, "amr_patch.pvd"), pvd_patch)

    Tc_final = state[0]
    print(f"[amr-adaptive] {n_steps} steps | {timer.elapsed:.2f}s | "
          f"peak T = {np.asarray(Tc_final).max():.4f} K | "
          f"DOF = {Nc*Nc} + {Nf*Nf} = {Nc*Nc + Nf*Nf:,}")

    return dict(T_final=Tc_final, frames=frames, times=times,
                patch_bounds=patch_bounds, wallclock=timer.elapsed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Adaptive AMR solver")
    parser.add_argument("--plot-grid", action="store_true",
                        help="Also generate patch-tracking animation and snapshot grid")
    args = parser.parse_args()

    Nc, Nf = 128, 512
    print(f"Starting ADAPTIVE AMR simulation "
          f"(coarse={Nc}x{Nc} dx≈1/128, moving fine patch={Nf}x{Nf} dx≈1/1024)...")
    res = run_amr(Nc=Nc, Nf=Nf)

    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)

    # --- always: standard snapshots + animation ---
    fig = plot_snapshots(res["frames"], Xc, Yc, res["times"],
                         title=f"AMR — dynamic patch (coarse {Nc}x{Nc}, fine {Nf}x{Nf})")
    fig.savefig("output/amr/snapshots.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    print("Saved output/amr/snapshots.png")

    fig2, anim = create_animation(res["frames"], Xc, Yc, res["times"])
    save_gif(anim, "output/amr/animation.gif")
    print("Saved output/amr/animation.gif")

    # --- --plot-grid: patch-tracking overlay ---
    if args.plot_grid:
        amr_frames = [_bounds_to_cells(x0, x1, y0, y1)
                      for x0, x1, y0, y1 in res["patch_bounds"]]

        fig3 = plot_snapshots(res["frames"], Xc, Yc, res["times"],
                              amr_frames=amr_frames,
                              title=f"AMR — patch trajectory (coarse {Nc}x{Nc}, fine {Nf}x{Nf})")
        fig3.savefig("output/amr/snapshots_grid.png", dpi=150,
                     bbox_inches="tight", facecolor="#0d0d0d")
        print("Saved output/amr/snapshots_grid.png")

        fig4, anim2 = create_animation(res["frames"], Xc, Yc, res["times"],
                                       amr_frames=amr_frames)
        save_gif(anim2, "output/amr/animation_grid.gif")
        print("Saved output/amr/animation_grid.gif")

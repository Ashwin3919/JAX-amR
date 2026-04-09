"""
True Adaptive AMR driver (Model 2).

Solves on a coarse base grid everywhere. Each step, the gradient-weighted
centroid of T_coarse is computed and a fine patch of fixed shape (Nf x Nf)
is centered there. The patch follows the laser automatically — no pre-defined
location needed.

The fine patch carries its own thermal state between steps. When the patch
moves, old fine values are reused in the overlap region and fresh territory
is initialised from the coarse grid.

Usage:
    PYTHONPATH=. python runs/run_amr.py
"""
import os
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np

import config.params as p
from solver.grid import build_grid, build_laser_source
from amr.adaptive_step import adaptive_step
from amr.adaptive_patch import make_fine_coords
from analysis.metrics import Timer
from viz.snapshots import plot_snapshots
from viz.animate import create_animation, save_gif


def run_amr(Nc: int = 256, Nf: int = 512, half_w: float = 0.2,
            output_dir: str = "output/amr_overlay",
            n_steps: int = None) -> dict:
    """
    True adaptive AMR solver.

    Parameters
    ----------
    Nc     : coarse grid resolution (Nc x Nc, full domain)
    Nf     : fine patch resolution (Nf x Nf, fixed shape, dynamic location)
    half_w : half-width of fine patch in physical units (patch = 2*half_w square)
    """
    os.makedirs(output_dir, exist_ok=True)
    n_steps = n_steps or p.n_steps

    dx_c = p.Lx / (Nc - 1)
    dy_c = p.Ly / (Nc - 1)
    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)

    # Initial coarse field
    Tc = jnp.full((Nc, Nc), p.T_init)

    # Initial patch: centre on domain, full coarse interpolation
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
    state = (Tc, Tp, x0_init, x1_init, y0_init, y1_init)

    with Timer() as timer:
        for i in range(n_chunks):
            t_start = jnp.array(i * chunk_size * p.dt)
            state = run_chunk(state, t_start)
            Tc = state[0]
            frames.append(np.asarray(Tc))
            times.append(float((i + 1) * chunk_size * p.dt))

    Tc_final = state[0]
    print(f"[amr-adaptive] {n_steps} steps | {timer.elapsed:.2f}s | "
          f"peak T = {np.asarray(Tc_final).max():.4f} K | "
          f"DOF = {Nc*Nc} + {Nf*Nf} = {Nc*Nc + Nf*Nf:,}")

    return dict(T_final=Tc_final, frames=frames, times=times, wallclock=timer.elapsed)


if __name__ == "__main__":
    Nc, Nf = 256, 512
    print(f"Starting TRUE ADAPTIVE AMR simulation "
          f"(coarse={Nc}x{Nc}, moving fine patch={Nf}x{Nf})...")
    res = run_amr(Nc=Nc, Nf=Nf)

    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)
    fig = plot_snapshots(res["frames"], Xc, Yc, res["times"],
                         title=f"Adaptive AMR (Nc={Nc}, Nf={Nf}, tracking patch)")
    fig.savefig("output/amr_overlay/snapshots.png", dpi=150,
                bbox_inches="tight", facecolor="#0d0d0d")
    print("Saved output/amr_overlay/snapshots.png")

    fig2, anim = create_animation(res["frames"], Xc, Yc, res["times"])
    save_gif(anim, "output/amr_overlay/animation.gif")
    print("Saved output/amr_overlay/animation.gif")

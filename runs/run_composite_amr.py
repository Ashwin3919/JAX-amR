import jax
import jax.numpy as jnp
from jax import lax
import time

import config.params as p
from solver.grid import build_grid, build_laser_source
from amr.patch import build_patch_info, PatchInfo
from amr.composite_step import composite_step

def run_simulation(Nc_x=None, Nc_y=None, Nf_x=None, Nf_y=None, 
                   patch_bounds=None, laser_power=None, n_steps=None):
    """
    Runs the composite grid simulation.
    Overrides config.params if arguments are provided.
    """
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
    # Coarse grid (Nc_x, Nc_y)
    Xc, Yc = build_grid(Nc_x, Nc_y, p.Lx, p.Ly)
    dx_c = p.Lx / (Nc_x - 1)
    dy_c = p.Ly / (Nc_y - 1)
    
    # Fine patch info
    patch = build_patch_info(px0, px1, py0, py1, 
                             Nf_x, Nf_y, Nc_x, Nc_y, p.Lx, p.Ly)
    dx_f = (px1 - px0) / (Nf_x - 1)
    dy_f = (py1 - py0) / (Nf_y - 1)
    
    # 2. Pre-calculate Sources
    Qc = build_laser_source(Xc, Yc, p.laser_cx, p.laser_cy, p.laser_sigma, laser_power)
    Qf = build_laser_source(patch.Xf, patch.Yf, p.laser_cx, p.laser_cy, p.laser_sigma, laser_power)
    
    # 3. Initial state
    T_coarse_init = jnp.full((Nc_x, Nc_y), p.T_init)
    T_patch_init  = jnp.full((Nf_x, Nf_y), p.T_init)
    
    # 4. Time loop via lax.scan
    def body_fn(state, _):
        Tc, Tp = state
        Tc_next, Tp_next = composite_step(
            Tc, Tp, Qc, Qf, patch, p.alpha, p.dt, 
            dx_c, dy_c, dx_f, dy_f, p.T_wall
        )
        return (Tc_next, Tp_next), None

    state_final, _ = lax.scan(body_fn, (T_coarse_init, T_patch_init), None, length=n_steps)
    return state_final

@jax.jit
def run_simulation_jit():
    return run_simulation()

if __name__ == "__main__":
    print(f"Starting Composite AMR simulation with {p.n_steps} steps...")
    
    # First call (includes compilation)
    start = time.time()
    Tc_final, Tp_final = run_simulation_jit()
    Tc_final.block_until_ready()
    duration_comp = time.time() - start
    print(f"First run (with JIT): {duration_comp:.3f}s")
    
    # Second call (pure execution)
    start = time.time()
    Tc_final, Tp_final = run_simulation_jit()
    Tc_final.block_until_ready()
    duration_exec = time.time() - start
    print(f"Second run: {duration_exec:.3f}s")
    
    print(f"Max T (coarse): {Tc_final.max():.4f}")
    print(f"Max T (patch):  {Tp_final.max():.4f}")
    
    # Save results for comparison
    jnp.save("output/amr/composite_coarse.npy", Tc_final)
    jnp.save("output/amr/composite_patch.npy", Tp_final)
    print("Saved results to output/amr/")

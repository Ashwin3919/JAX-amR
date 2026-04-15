# JAX-amR : A Differentiable Adaptive Mesh Refinement Solver for PDEs

**Author:** Ashwin Shirke

![AMR Animation](docs/amr_animation.gif)

JAX-amR is a framework for differentiable adaptive mesh refinement (AMR) solver of PDEs, built entirely on JAX. The core idea: concentrate spatial resolution where gradients are large, keep the rest coarse, and do it in a way that remains fully JIT-compilable and end-to-end differentiable via `jax.grad`.

The reusable framework lives in `src/solver/`, `src/amr/`, `src/viz/`, and `src/ioutils/`. The `runs/` directory contains the example application: the 2D transient heat equation driven by a Gaussian laser on a circular orbit. Three solver strategies are benchmarked on this problem — a uniform reference, a dynamic AMR solver that tracks the gradient centroid each step, and a fixed-patch composite solver pre-placed over the laser orbit. All use Crank-Nicolson time integration with 5-point finite difference spatial discretization.

---

## Numerical Rigor & Differentiability

JAX-amR is designed for high-fidelity scientific computing, prioritizing numerical accuracy and end-to-end differentiability:

- **Iterative Solver (CG):** Uses `jax.scipy.sparse.linalg.cg` for Crank-Nicolson steps, ensuring implicit convergence to a $10^{-7}$ tolerance.
- **Conservative Synchronization:** Employs anti-aliased linear downsampling for fine-to-coarse transfers to preserve total thermal energy.
- **Scientific Rigor:** Global `float64` enabled (32-bit deprecated) for numerical stability and sub-millikelvin precision in deep `lax.scan` loops. This ensures absolute convergence for optimization tasks (hitting $10^{-10}$ loss).
- **Pure JIT:** 100% `jnp` implementation with static-shape moving patches to remain XLA-compliant and fully differentiable via `jax.grad`.

---

## Models & Performance

Benchmarks conducted on **Apple M2 CPU**, 5000 steps, dt=1e-4 s.

### 1. Legacy 32-bit Results (Float32)
*Fastest raw performance, but limited precision for complex optimizations.*

| Model | DOF | Wallclock | Peak T | Error |
| :--- | ---: | ---: | ---: | ---: |
| Uniform (1024×1024) | 1,048,576 | 147.02 s | 118.7011 K | — |
| AMR Dynamic | 278,528 | 12.80 s | 117.3004 K | 1.18% |
| AMR Fixed | 278,528 | 25.03 s | 118.7028 K | 0.0014% |

**Speedups (32-bit):** AMR Dynamic **11.5×**, AMR Fixed **5.9×**.

### 2. Current 64-bit Results (Float64)
*Mandatory for scientific rigor and gradient-based optimization. Higher compute cost per step.*

| Model | DOF | Wallclock | Peak T | Error |
| :--- | ---: | ---: | ---: | ---: |
| Uniform (1024×1024) | 1,048,576 | 49.26 s | 118.6983 K | — |
| AMR Dynamic | 278,528 | 54.00 s | 117.1260 K | 1.32% |
| AMR Fixed | 278,528 | 20.76 s | 118.7438 K | 0.0383% |

**Speedups (64-bit):** AMR Fixed **2.4×**.

> **Note on the transition:** The transition to 64-bit (Float64) increases raw compute time but is required to hit the high-precision convergence targets (e.g., $10^{-10}$ loss) seen in the `runs/Diffrential/` optimization scripts. The 32-bit mode is now considered deprecated for high-fidelity tasks.

Both AMR variants use **3.76× fewer degrees of freedom** than the uniform reference. The AMR Fixed variant achieves near-identical accuracy to the uniform reference because the fine patch carries uninterrupted thermal history from the first step. The AMR Dynamic variant tracks the laser with no prior knowledge of its path.


![AMR Snapshots](docs/amr_snapshots.png)
---

## Framework Structure

```
src/
  solver/          # PDE-agnostic numerics: grid builder, Crank-Nicolson step, BC enforcement
    laser_source.py      # Application-specific Gaussian laser source (example physics)
  amr/             # AMR primitives: coarse↔fine interpolation, gradient centroid, patch reinit
  viz/             # Visualization: snapshots, animations, mesh-overlay rendering
  ioutils/         # I/O: legacy VTK writer, PVD index, checkpoint save/load
  config/          # Physical and solver parameters (params.py)
runs/              # Example application (circular laser heat equation)
  run_uniform.py         # Model 1 — 1024×1024 uniform reference
  run_amr.py             # Model 2 — dynamic AMR, patch follows gradient centroid
  run_composite_amr.py   # Model 3 — fixed-patch AMR, pre-placed over known orbit
  compare.py             # Benchmark comparison of all three models
tests/             # Test suite (pytest)
docs/              # Technical report and visualizations
```

To adapt JAX-amR to a different PDE: replace `solver/grid.py`, `solver/cn_step.py`, and `solver/laser_source.py` with your own spatial operator, time integrator, and source term; the AMR layer in `amr/` is agnostic to the physics.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Requirements: `jax>=0.4.20`, `jaxlib>=0.4.20`, `numpy>=1.26`, `matplotlib>=3.8`, `imageio>=2.33`, `Pillow>=10.0`, `scipy>=1.11`. No CUDA required.

---

## Running

```bash
# Uniform reference
python runs/run_uniform.py

# AMR Dynamic
python runs/run_amr.py

# AMR Fixed
python runs/run_composite_amr.py
```

Output goes to `output/uniform/`, `output/amr/`, and `output/amr_fixed/`. Each directory contains `snapshots.png`, `animation.gif`, VTK files, and `.npz` checkpoints.

### Grid overlay animations

Pass `--plot-grid` to generate animations showing the mesh structure:

```bash
python runs/run_uniform.py --plot-grid       # 16x16 white cells, full domain
python runs/run_amr.py --plot-grid           # 8x8 red coarse + 16x16 white fine, moving
python runs/run_composite_amr.py --plot-grid # 8x8 red coarse + 16x16 white fine, fixed
```

---

## Differentiability

Every operation inside the JIT-compiled region is pure `jnp`: the 5-point Laplacian, Dirichlet BC enforcement, Gaussian laser source, bilinear interpolation, fine-to-coarse injection, gradient centroid detection, and the time loop via `lax.scan`. No Python conditionals. No NumPy calls. The computation from initial condition to final temperature is one continuous function that JAX can differentiate.

Check the `runs/Diffrential/` folder, where there are three optimization problems successfully solved because the code is entirely differentiable.

### Scientific Note: Static-Shape AMR
Traditional AMR utilizes dynamic tree-based tracking to spawn localized sub-grids. However, JAX's XLA compiler rigorously requires static array shapes for `jax.jit` compilation. To bridge this gap, this framework implements a differentiable foveated/moving-patch approach. It relies on continuous interpolations (`jnp.where` masking) of a statically-sized fine-grid patch that autonomously tracks gradient centroids over a coarse domain. This design concession allows the solver to mimic local refinement while remaining 100% compliant with static JIT compilation and end-to-end differentiation. *(See `docs/tech-report.md` for a full mathematical breakdown)*.

---

## ParaView

Open `output/amr/amr_coarse.pvd` and `output/amr/amr_patch.pvd` together to view the two-level grid. The patch VTK files store physical coordinates per frame, so the patch position animates correctly as it tracks the laser.

---

## License

MIT

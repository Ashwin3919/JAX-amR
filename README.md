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

Benchmarks conducted natively on an **Apple M2 CPU**, integrating 5000 Crank-Nicolson steps over simulated $t=0.5$s ($\Delta t=10^{-4}$ s).

### 1. Legacy 32-bit Results (Float32)
*Fast, low-precision benchmarking mode for rapid prototyping. (Run via `JAX_ENABLE_X64=0`)*

| Model | DOF | Wallclock | Peak T | Error |
| :--- | ---: | ---: | ---: | ---: |
| Uniform (1024×1024) | 1,048,576 | 36.44 s | 118.7001 K | — |
| AMR Dynamic | 278,528 | 45.86 s | 117.1248 K | 1.33% |
| AMR Fixed | 278,528 | 22.12 s | 118.7451 K | 0.0379% |

**Speedups (32-bit):** AMR Fixed **1.6×**. (Dynamic mode incurs compiler emulation overhead).

### 2. Current 64-bit Results (Float64)
*Default environment. Mandatory for scientific rigor, accurate inverse problems, and deep `lax.scan` optimizations.*

| Model | DOF | Wallclock | Peak T | Error |
| :--- | ---: | ---: | ---: | ---: |
| Uniform (1024×1024) | 1,048,576 | 50.73 s | 118.6983 K | — |
| AMR Dynamic | 278,528 | 56.77 s | 117.1260 K | 1.32% |
| AMR Fixed | 278,528 | 21.68 s | 118.7438 K | 0.0383% |

**Speedups (64-bit):** AMR Fixed **2.3×**. 

> **Scientific Note on XLA Overhead:** 
> Both AMR variants use **3.76× fewer degrees of freedom** (278,528 DOF) than the Uniform baseline. However, on highly optimized silicon like the Apple M2, brute-force spatial compute is aggressively bottlenecked out. 
> Notice that the **AMR Dynamic** solver is marginally slower than the Uniform grid (0.9× speedup). This is deliberate: maintaining static array shapes for `jax.jit` while dynamically relocating a fine patch requires differentiable masking (`jnp.where`) and continuous coarse-to-fine physical re-initialization. This JAX/XLA graph-level emulation overhead eclipses the raw ALUs saved by fewer DOFs. 
> The **AMR Fixed** solver strips out this tracking overhead, translating the 3.76× DOF reduction directly into a **2.3× absolute wallclock speedup**, while preserving uninterrupted thermal history from $t=0$ for a phenomenal 0.0383% error margin.



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

The default environment utilizes `JAX_ENABLE_X64=1` for maximum scientific fidelity.

```bash
# Uniform reference
python runs/run_uniform.py

# AMR Dynamic
python runs/run_amr.py

# AMR Fixed
python runs/run_composite_amr.py

# Automated Benchmarker Suite
python runs/compare.py
```

### Running in Legacy 32-bit (Float32) Mode

For faster rapid prototyping without strict high-precision convergence requirements:
```bash
JAX_ENABLE_X64=0 python runs/run_uniform.py
JAX_ENABLE_X64=0 python runs/run_amr.py
JAX_ENABLE_X64=0 python runs/run_composite_amr.py
JAX_ENABLE_X64=0 python runs/compare.py
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

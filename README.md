# JAX-AMR — 2D Heat Equation Solver with Adaptive Mesh Refinement

> **Author:** Ashwin Shirke

A modular JAX implementation of a 2D heat equation solver with a Gaussian laser source, Crank-Nicolson time stepping, and an Adaptive Mesh Refinement (AMR) overlay for visualisation and analysis.

---

## What this project does

| | |
|---|---|
| **Physics** | 2D heat equation: ∂T/∂t = α∇²T + Q(x,y) on a unit square with Dirichlet (cold-wall) BCs |
| **Source** | Stationary Gaussian laser Q = P · exp(−r²/2σ²) |
| **Solver** | Crank-Nicolson via 5-iteration fixed-point loop (unconditionally stable, JAX-JIT compiled) |
| **AMR** | Post-process overlay on the full fine-grid solution — macro-cells assigned refinement levels based on ‖∇T‖ tiers |
| **Output** | VTK files (`.vts` uniform, `.vtu` AMR) + `.pvd` collections for ParaView, `.npz` checkpoints, GIF animations, PNG snapshots |

---

## Repository layout

```
JAX-amR/
├── config/
│   └── params.py           # all constants (Nx, alpha, dt, laser_*, AMR thresholds, …)
├── solver/
│   ├── grid.py             # build_grid(), build_laser_source()
│   ├── ops.py              # laplacian(), apply_bc()
│   └── cn_step.py          # cn_step() + make_cn_step_jit()
├── amr/
│   ├── gradient.py         # compute_gradient_magnitude()
│   ├── cells.py            # build_amr_cells(), build_level_map()
│   └── thresholds.py       # assign_levels_array()
├── ioutils/
│   ├── vtk_writer.py       # write_mesh_vtk(), write_scalar_vtk(),
│   │                       # write_amr_vtk(), write_pvd()
│   └── checkpoint.py       # save_checkpoint(), load_checkpoint()
├── viz/
│   ├── heatmap.py          # plot_heatmap()
│   ├── crosssection.py     # plot_crosssection()
│   ├── amr_overlay.py      # draw_amr_overlay()
│   ├── snapshots.py        # plot_snapshots() — 4-panel static figure
│   └── animate.py          # create_animation(), save_gif(), get_jshtml()
├── analysis/
│   ├── metrics.py          # l2_error(), Timer
│   ├── convergence.py      # convergence_study(), plot_convergence()
│   └── comparison.py       # three cross-method comparison plots
├── runs/
│   ├── run_uniform.py      # uniform-grid driver (v1 path)
│   └── run_amr.py          # AMR-overlay driver  (v2 path)
├── compare.py              # top-level: run both → comparison plots
├── tests.py                # 6-test suite (no pytest required)
├── requirements.txt
└── v1.py / v2.py           # original monolithic notebooks (reference)
```

---

## Requirements

- Python 3.10 or 3.11
- A reasonably modern CPU (GPU optional — JAX auto-detects)

---

## Installation

### 1. Clone / download the repository

```bash
git clone <repo-url>
cd JAX-amR
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU (optional):** To use CUDA, replace the `jax` line above with the
> matching `jax[cuda12]` wheel from https://jax.readthedocs.io/en/latest/installation.html

---

## Quick start

### Run the test suite first

```bash
python tests.py
```

Expected output:

```
Running 6 tests...

  test_laplacian_known PASSED  (max rel err = 2.4e-04)
  test_apply_bc PASSED
  test_cn_step_no_source PASSED  (max diff = 0.00e+00)
  test_cn_energy_growth PASSED  (peak after 10 steps = ...)
  test_amr_level_assignment PASSED
  test_vtk_mesh_xml PASSED

========================================
Results: 6/6 passed — all OK
```

### Run the uniform solver (v1 path)

```bash
python -m runs.run_uniform
```

Writes output to `output/uniform/`:
- `mesh_t0000.vts` — mesh (written once)
- `temp_tNNNNN.vts` — temperature field every `vtk_every` steps
- `uniform.pvd` — ParaView collection
- `ckpt_NNNNN.npz` — NumPy checkpoints
- `snapshots.png`, `animation.gif`

### Run the AMR solver (v2 path)

```bash
python -m runs.run_amr
```

Writes output to `output/amr/`:
- `amr_tNNNNN.vtu` — AMR cell mesh + scalar every `vtk_every` steps
- `amr.pvd` — ParaView collection
- `snapshots.png`, `animation.gif`

### Run the full comparison

```bash
python compare.py
```

Writes three plots to `output/comparison/`:

| File | What it shows |
|------|--------------|
| `accuracy_at_equal_cost.png` | L2 error of uniform vs AMR at equal wallclock budget |
| `cost_at_equal_accuracy.png` | Time-to-solution at equal L2 error |
| `convergence_rate.png` | Error vs DOF count (grid-refinement study) |

---

## Configuration

All parameters live in **`config/params.py`** — edit this file to change anything:

```python
Nx, Ny        = 128, 128     # grid resolution
alpha         = 1e-3         # thermal diffusivity [m²/s]
dt            = 1e-3         # time step [s]
n_steps       = 600          # total solver steps
laser_power   = 500.0        # Gaussian laser strength [K/s]
laser_cx      = 0.5          # laser centre x
laser_cy      = 0.5          # laser centre y
laser_sigma   = 0.05         # laser width [m]
REFINE_THRESH = 2.0          # AMR level-1 gradient threshold
MAX_LEVEL     = 3            # maximum AMR refinement level
vtk_every     = 50           # VTK output interval (0 = off)
```

Both `run_uniform` and `run_amr` import from this single file, so comparisons are always apples-to-apples.

---

## Viewing VTK output in ParaView

1. Open ParaView (https://www.paraview.org/download/)
2. **File → Open** → select `output/uniform/uniform.pvd` (or `output/amr/amr.pvd`)
3. Click **Apply** in the Properties panel
4. Use the time scrubber to step through frames

---

## Module API reference

### `solver/cn_step.py`

```python
cn_step(T, Q, alpha, dt, dx, dy, T_wall=0.0, n_iter=5) -> jnp.ndarray
```

One Crank-Nicolson step. All arguments explicit — no global state.

```python
step_fn = make_cn_step_jit(alpha, dt, dx, dy)
T_new = step_fn(T, Q)   # JIT-compiled, reuse in loop
```

### `amr/cells.py`

```python
cells, level_map = build_amr_cells(T, dx, dy, Lx, Ly, macro, tiers, max_level)
# cells: list of (x0, y0, x1, y1, level) tuples
# level_map: (macro, macro) int array
```

### `ioutils/vtk_writer.py`

```python
write_mesh_vtk(path, X, Y)          # StructuredGrid, written once
write_scalar_vtk(path, T, t)        # StructuredGrid + Temperature PointData
write_amr_vtk(path, cells, t)       # UnstructuredGrid for AMR cells
write_pvd(path, [(t0, f0), ...])    # ParaView collection file
```

### `analysis/metrics.py`

```python
err = l2_error(T, T_ref)            # RMS L2 error
with Timer() as tm:
    ...
print(tm.elapsed)                   # wallclock seconds
```

---

## Design notes

- **AMR is post-process only** — the full fine-grid solve runs in JAX (static shapes), and the AMR grid is built in NumPy as a visualisation/analysis overlay. This keeps JAX's static-shape constraint satisfied.
- **`io/` → `ioutils/`** — the subdirectory is named `ioutils` rather than `io` to avoid shadowing Python's built-in `io` standard library module.
- **Single config** — `config/params.py` is the sole source of truth; both drivers import from it so grid resolution, solver settings, and AMR thresholds are always consistent between runs.

---

## License

MIT

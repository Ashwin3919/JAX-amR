# GEMINI.md - JAX-amR Project Context

## Project Overview
JAX-amR is a high-performance, differentiable framework for solving Partial Differential Equations (PDEs) using Adaptive Mesh Refinement (AMR), built entirely on **JAX**. It enables end-to-end differentiability (`jax.grad`) and full JIT-compilation (`jax.jit`) by employing a foveated/moving-patch AMR strategy that maintains static array shapes required by XLA.

The core application featured is a 2D transient heat equation solver driven by a Gaussian laser source.

### Main Technologies
- **JAX**: For JIT compilation, auto-differentiation, and GPU/CPU acceleration.
- **Python**: Primary development language.
- **NumPy & SciPy**: For numerical utilities (though core logic is `jnp`).
- **Matplotlib & Imageio**: For scientific visualization and animation.
- **Pytest**: For the test suite.

---

## Architecture & Framework Structure

### Core Framework (`src/`)
- **`amr/`**: Implementation of AMR primitives.
  - `adaptive_step.py`: The high-level adaptive time-step logic (moving patch).
  - `adaptive_patch.py`: Logic for gradient centroid detection and patch re-initialization.
  - `interpolate.py` & `gradient.py`: Numerical kernels for bilinear interpolation and gradient calculation.
  - `composite_step.py`: Fixed-patch composite grid logic.
- **`solver/`**: PDE-agnostic numerical core.
  - `ops.py`: Finite difference operators (5-point Laplacian, BC enforcement).
  - `cn_step.py`: Crank-Nicolson time integration.
  - `grid.py`: Coordinate and grid generation.
  - `laser_source.py`: Example physics implementation (Gaussian source).
- **`viz/`**: Specialized visualization for AMR (animations, snapshots, mesh overlays).
- **`ioutils/`**: I/O for VTK (ParaView compatible), PVD indices, and NPZ checkpoints.
- **`config/`**: Centralized parameter management in `params.py`.

### Applications & Benchmarks (`runs/`)
- `run_uniform.py`: Baseline uniform grid solver.
- `run_amr.py`: Dynamic AMR solver (patch follows the gradient centroid).
- `run_composite_amr.py`: Fixed-patch AMR solver (patch at a pre-defined location).
- `compare.py`: Automated benchmarking and comparison script.
- `Diffrential/`: Examples of solving optimization problems using the solver's differentiability.

---

## Building and Running

### Setup
Ensure you have Python 3.10+ installed.
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Key Commands
- **Run Dynamic AMR:** `python runs/run_amr.py`
- **Run Uniform Baseline:** `python runs/run_uniform.py`
- **Run Fixed-Patch AMR:** `python runs/run_composite_amr.py`
- **Run Benchmark:** `python runs/compare.py`
- **Enable Mesh Overlay Visualization:** Append `--plot-grid` to any run script.
- **Run Tests:** `pytest`

### Output
Results are stored in the `output/` directory (e.g., `output/amr/`), including:
- `animation.gif`: Time-evolution animation.
- `snapshots.png`: Selected time-step frames.
- `*.vtk` & `*.pvd`: 3D visualization files for ParaView.
- `ckpt_*.npz`: Checkpoint files.

---

## Development Conventions

### JAX & XLA Compatibility
- **Static Shapes:** All array shapes must be static at JIT-time. AMR is achieved by moving a statically-sized fine patch rather than dynamic tree-splitting.
- **Pure Functions:** Maintain functional purity for `jax.jit` and `jax.grad`. Avoid side effects and Python control flow inside JIT-decorated functions.
- **Vectorization:** Prefer `vmap` and `lax.scan` over Python loops for performance and differentiability.

### Numerical Standards
- **Dirichlet BCs:** Enforced explicitly via `apply_bc` in `src/solver/ops.py`.
- **Interpolation:** Bilinear interpolation is the standard for coarse-to-fine transfer.
- **Injection:** Fine-to-coarse data transfer (injection) is used for synchronization.

### Testing
- Tests are located in the `tests/` directory.
- Use `pytest` for execution.
- Key tests include validation of interpolation accuracy (`test_interpolate.py`) and end-to-end differentiability checks (`test_amr.py`).

### Performance Notes
- Cold XLA compilation for large grids (e.g., 1024x1024) can take ~2 minutes. Subsequent runs in the same process are extremely fast.
- AMR typically provides a 5x–12x speedup over uniform grids of equivalent effective resolution.

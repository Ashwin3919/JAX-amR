# JAX-AMR — 2D Heat Equation Solver with Adaptive Mesh Refinement

> **Author:** Ashwin Shirke

A modular JAX implementation of a 2D heat equation solver with a Gaussian laser source, Crank-Nicolson time stepping, and multiple AMR (Adaptive Mesh Refinement) strategies.

---

## 🚀 Two Types of AMR

This repository provides two distinct approaches to AMR:

### 1. Composite JIT-AMR (True Adaptive Solver)
*   **New & High-Performance:** Uses a two-level composite grid (coarse grid + fine patch).
*   **JAX-Native:** Fully JIT-compilable via `jax.lax.scan`.
*   **Differentiable:** Supports `jax.grad`, `jax.vmap`, and `jax.jacobian` through the entire simulation.
*   **Efficient:** Only solves high-resolution where needed (near the laser). 600 steps complete in **~0.03s** on CPU.
*   **Usage:** `python runs/run_composite_amr.py`

### 2. AMR-Overlay (Visualization Only)
*   **Analysis focused:** Solves a single uniform fine grid, then builds an AMR cell structure as a post-process overlay.
*   **ParaView Ready:** Generates `.vtu` unstructured grids showing refinement levels based on temperature gradients.
*   **Usage:** `python runs/run_amr.py`

---

## 🌡️ Physics & Numerical Methods

| Component | Detail |
|---|---|
| **Physics** | 2D heat equation: ∂T/∂t = α∇²T + Q(x,y) on a unit square |
| **Boundary** | Dirichlet (cold-wall) BCs (T=0) |
| **Source** | Stationary Gaussian laser Q = P · exp(−r²/2σ²) |
| **Solver** | Crank-Nicolson via 5-iteration fixed-point loop (unconditionally stable) |
| **Time-Stepping** | JAX `lax.scan` for zero-overhead loops inside JIT |

---

## 📂 Repository Layout

```
JAX-amR/
├── amr/
│   ├── patch.py            # Composite grid geometry & injection/interpolation
│   ├── composite_step.py   # Two-level coupled solver logic
│   ├── interpolate.py      # JAX-native bilinear interpolation
│   ├── cells.py            # (Overlay) build AMR cell structures
│   └── ...                 # (Overlay) gradient and thresholding
├── solver/
│   ├── grid.py             # meshgrid and laser source builders
│   ├── ops.py              # JAX Laplacian and BC operators
│   └── cn_step.py          # Crank-Nicolson core step
├── config/
│   └── params.py           # Central configuration (Nx, alpha, dt, etc.)
├── runs/
│   ├── run_composite_amr.py # Driver for the true JIT-AMR solver
│   ├── run_uniform.py       # Uniform grid driver
│   └── run_amr.py           # AMR-overlay driver (visualization only)
├── analysis/
│   ├── metrics.py          # L2 error and Timing utilities
│   └── comparison.py       # Plotting utilities
├── ioutils/
│   ├── vtk_writer.py       # VTK/PVD export for ParaView
│   └── checkpoint.py       # NumPy/JAX checkpointing
├── compare.py              # Performance/Accuracy benchmark suite
└── tests/
    └── test_amr.py         # Unit tests for the AMR system
```

---

## 🛠️ Installation

```bash
# Clone the repo
git clone <repo-url>
cd JAX-amR

# Setup environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 🏃 Usage

### 1. Run the Composite JIT-AMR (Recommended)
This is the most advanced part of the codebase. It runs a true two-level simulation that is fully differentiable.
```bash
PYTHONPATH=. python runs/run_composite_amr.py
```
*Outputs:* Saves `.npy` results to `output/amr/`.

### 2. Run the Performance Comparison
Compare the accuracy and wallclock time of Uniform vs. AMR-Overlay vs. Composite JIT-AMR.
```bash
PYTHONPATH=. python compare.py
```
*Outputs:* Three plots in `output/comparison/` and a `summary.txt`.

### 3. Run the AMR-Overlay (for ParaView)
If you want to visualize refinement levels as boxes in ParaView:
```bash
python runs/run_amr.py
```
*Outputs:* `.vtu` and `.pvd` files in `output/amr/`.

---

## 🧪 Testing

The codebase includes verification for interpolation, grid injection, and differentiability.
```bash
PYTHONPATH=. pytest tests/test_amr.py
```

---

## 🧠 Differentiability Example

Since the entire **Composite JIT-AMR** is pure JAX, you can optimize laser parameters:

```python
import jax
from runs.run_composite_amr import run_simulation

def loss(power):
    Tc_final, Tp_final = run_simulation(laser_power=power)
    return Tc_final.max()

# Compute gradient of peak temperature w.r.t. laser power
grad_power = jax.grad(loss)(500.0)
print(f"Sensitivity: {grad_power}")
```

---

## ⚙️ Configuration

Edit `config/params.py` to change simulation parameters:

*   `Nc_x, Nc_y`: Coarse grid resolution (e.g., 32x32)
*   `Nf_x, Nf_y`: Fine patch resolution (e.g., 64x64)
*   `patch_x0, patch_x1...`: Spatial bounds of the fine patch
*   `alpha, dt, n_steps`: Physical and numerical constants

---

## 📈 Performance & Accuracy Benchmark

We benchmarked the **Composite JIT-AMR** against a high-resolution **Uniform 512x512** reference (measured against a 1024x1024 ground truth) to verify that selective refinement delivers high precision at a fraction of the cost.

### Ultimate "512 Test" (100 Steps)

| Method | Effective Resolution | DOF Count | Accuracy (Patch L2) | Wallclock |
|---|---|---|---|---|
| **Uniform** | 512 x 512 | 262,144 | 1.240e-04 | 0.115s |
| **Composite AMR** | **64 (base) + 128 (patch)** | **20,480** | **1.110e-04** | **0.039s** |
| **Composite AMR** | **128 (base) + 256 (patch)**| **81,920** | **1.114e-04** | **0.103s** |

**Why it is "Faster and Better":**
1.  **3x Speedup:** The Composite AMR (64+128) is **3x faster** than the Uniform 512 grid (0.039s vs 0.115s).
2.  **Higher Precision:** It actually achieves **better local accuracy** in the laser zone ($1.11 \times 10^{-4}$ vs $1.24 \times 10^{-4}$) by focusing 100% of its fine-grid resources where the physics is happening.
3.  **Efficiency:** It matches or beats 512-uniform precision while using **12x fewer degrees of freedom** (20k vs 262k).

---

## 🛠️ How to Reproduce

### 1. Run the Full Benchmark Suite
To generate the convergence plots, timing breakdowns, and the efficiency frontier shown above:
```bash
PYTHONPATH=. python compare.py
```
This script will:
*   Compute a 1024x1024 reference solution.
*   Run uniform grids from 64 to 512.
*   Run composite AMR configurations.
*   Save results to `output/comparison/summary.txt` and generate PNG plots.

### 2. Run a Single High-Res AMR Simulation
To run the high-performance composite solver standalone:
```bash
PYTHONPATH=. python runs/run_composite_amr.py
```
This will run the simulation defined in `config/params.py` (default 128 base + 256 patch) and save the final temperature fields as `.npy` files for analysis.

### 3. Verify with Unit Tests
Ensure the JAX-native interpolation and differentiability are working correctly:
```bash
PYTHONPATH=. pytest tests/test_amr.py
```

---

## License
MIT

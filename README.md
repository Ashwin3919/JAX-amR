# JAX-AMR — Impulse Differentiable Solver

**Author:** Ashwin Shirke

![AMR Snapshots](amr_snapshots.png)


A JAX implementation of the 2D heat equation with adaptive mesh refinement. Solves a moving Gaussian laser source using three different solver strategies, benchmarked against each other. The composite solver is fully differentiable — you can run `jax.grad` through the entire time loop.

---

## What it does

Simulates laser-material thermal interaction on a $[0,1]^2$ domain using the transient heat equation. The laser moves on a circular orbit and the goal is to resolve the sharp thermal gradient at the focus without wasting compute everywhere else.

Three solver strategies are implemented:

**Model 1 — Uniform Grid**
A 1024×1024 single-resolution solver. Everything is resolved at the same spacing. Used as the reference.

**Model 2 — AMR-Overlay**
Same 1024×1024 solver, but after each save step the temperature field is analyzed with a gradient-based refinement detector that assigns cell levels (1–3) to a 16×16 macro-cell grid. The AMR is visualization and post-processing only — the PDE solve is unchanged.


![AMR Animation](amr_animation.gif) 


**Model 3 — Composite JIT-AMR**
A coupled two-level solver. A 256×256 coarse grid covers the full domain. A 512×512 fine patch sits over the laser zone $[0.3, 0.7]^2$. Both grids advance together each step: the coarse grid provides boundary conditions for the fine patch via bilinear interpolation, and the fine solution is injected back into the coarse grid. The entire 100-step chunk is compiled with `lax.scan` under `@jax.jit`.

---

## Benchmark (5000 steps, Apple M2 CPU)

| Model | Solve DOF | Adaptive solve? | Time | vs Uniform |
| :--- | ---: | :--- | ---: | :--- |
| Uniform | 1,048,576 | No | 147.02 s | — |
| AMR-Overlay | 1,048,576 | **No** (visualization only) | 22.98 s | 6.4× faster |
| Composite AMR | 327,680 | **Yes** | 31.60 s | 4.65× faster |

Peak temperature: Uniform 118.7011 K, Composite 118.6312 K (< 0.06% error).

AMR-Overlay solves on the same 1,048,576 DOF as uniform — no compute reduction. Its speed advantage is purely from writing compact AMR cell lists to VTK instead of dense 1M-point arrays. Composite AMR is the only model that genuinely reduces the number of points solved, with 3.2× fewer DOF and `lax.scan` batching 100 steps per XLA dispatch.

---

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## Running

```bash
# Model 1 — uniform reference
PYTHONPATH=. python runs/run_uniform.py

# Model 2 — AMR-overlay
PYTHONPATH=. python runs/run_amr.py

# Model 3 — composite JIT-AMR
PYTHONPATH=. python runs/run_composite_amr.py
```

Output goes to `output/uniform/`, `output/amr_overlay/`, `output/amr/`. Each directory contains `snapshots.png`, `animation.gif`, VTK files, and `.npz` checkpoints.

Laser mode (stationary or circular orbit) is toggled in `config/params.py`:

```python
LASER_MODE = "circular"   # or "stationary"
```

---

## Differentiability

The composite solver uses no Python conditionals or NumPy calls inside the JIT region. Every operation — the 5-point Laplacian, Dirichlet boundary enforcement, bilinear interpolation, fine-to-coarse injection, and the time loop itself via `lax.scan` — is pure `jnp`. This means:

```python
def peak_temperature(laser_power):
    res = run_simulation(laser_power=laser_power)
    return jnp.max(res["T_final"])

dT_dP = jax.grad(peak_temperature)(2500.0)
```

The gradient flows back through 5,000 time steps exactly, not via finite differences.

---

## ParaView

Open `output/amr/amr_coarse.pvd` and `output/amr/amr_patch.pvd` together to see the two-level grid overlaid.

---

## Tests

```bash
PYTHONPATH=. pytest tests/test_amr.py
```

---

## License

MIT

# Technical Report: JAX-AMR — Adaptive Mesh Refinement for 2D Thermal Simulation

**Project:** JAX-AMR
**Author:** Ashwin Shirke
**Date:** April 2026
**Hardware:** Apple M2 CPU

---

## 1. Problem Statement

Simulating high-power laser interactions with materials requires resolving sharp thermal gradients at the laser focus while tracking their evolution over time across the full domain. The naive approach — a uniform fine grid everywhere — is accurate but wastes compute on regions that are thermally flat. The research question is:

> Can we build a physically accurate, fast, and *differentiable* adaptive mesh refinement system using JAX, and how much do we gain versus brute-force uniform resolution?

We answer this by implementing and benchmarking three distinct solver architectures for the 2D heat equation with a moving Gaussian laser source.

---

## 2. Physical Model

### 2.1 Governing Equation

We solve the 2D transient heat equation:

$$\frac{\partial T}{\partial t} = \alpha \nabla^2 T + Q(x, y, t)$$

where:

| Symbol | Value | Description |
| :--- | :--- | :--- |
| $\alpha$ | $10^{-3}$ m²/s | Thermal diffusivity |
| $T$ | — | Temperature field (K) |
| $Q(x,y,t)$ | — | Gaussian laser source (W/m²) |
| $\Delta t$ | $10^{-4}$ s | Time step |
| $N_{steps}$ | 5,000 | Total steps (0.5 s simulation) |

**Domain:** $[0, 1] \times [0, 1]$ m with Dirichlet boundary conditions $T = 0$ K on all walls.

### 2.2 Laser Source

The laser is a moving Gaussian heat source:

$$Q(x, y, t) = P \cdot \exp\!\left(-\frac{(x - c_x(t))^2 + (y - c_y(t))^2}{2\sigma^2}\right)$$

It travels on a circular orbit to stress-test tracking:

$$c_x(t) = 0.5 + 0.2\cos(\omega t), \quad c_y(t) = 0.5 + 0.2\sin(\omega t), \quad \omega = 2\pi / 0.1$$

| Parameter | Value |
| :--- | :--- |
| Power $P$ | 2,500 W/m² |
| Beam width $\sigma$ | 0.05 m |
| Orbit radius $R$ | 0.2 m |
| Angular velocity $\omega$ | 20π rad/s |

### 2.3 Numerical Scheme: Crank-Nicolson

All three solvers use the **Crank-Nicolson (CN) scheme** — a second-order-accurate, unconditionally stable semi-implicit method:

$$\frac{T^{n+1} - T^n}{\Delta t} = \frac{\alpha}{2}\left(\nabla^2 T^n + \nabla^2 T^{n+1}\right) + Q^n$$

Because $T^{n+1}$ appears on both sides, we solve via **fixed-point iteration** (5 iterations):

```
T_k+1 = T^n + dt·α/2·(∇²T^n + ∇²T_k) + dt·Q^n
```

The spatial Laplacian uses the standard **5-point finite difference stencil**:

```python
# solver/ops.py
d2x = (T[2:, 1:-1] - 2*T[1:-1, 1:-1] + T[:-2, 1:-1]) / dx**2
d2y = (T[1:-1, 2:] - 2*T[1:-1, 1:-1] + T[1:-1, :-2]) / dy**2
interior = d2x + d2y
```

---

## 3. Why JAX

Standard PDE solvers in NumPy or C++ face three barriers when pushed to production use:

1. **Python loop overhead.** A 5,000-step simulation with a Python `for` loop means 5,000 Python bytecode dispatches, interpreter locks, and NumPy kernel launches — each carrying ~microseconds of overhead that compounds badly at scale.

2. **No hardware fusion.** NumPy executes operations one-at-a-time. `A + B * C` allocates two intermediate arrays. XLA (the compiler JAX is built on) fuses them into a single memory pass.

3. **No gradients.** Running sensitivity analysis — e.g. "how does peak temperature respond to laser power?" — requires either finite-difference perturbations (slow, inaccurate) or a hand-derived adjoint (correct but brittle). Neither is practical for iterative design.

JAX solves all three simultaneously:

| JAX Feature | What it does for us |
| :--- | :--- |
| `@jax.jit` | XLA-compiles a Python function into a single fused kernel; subsequent calls skip Python entirely |
| `jax.lax.scan` | Unrolls a loop into a single XLA operation, eliminating all Python overhead for time-stepping |
| `jax.grad` / `jax.jacobian` | Exact reverse-mode autodiff through any `jnp` computation, including our entire simulation |
| Static shapes | The same compiled kernel is reused for every time step since array shapes never change |

The single most impactful construct is `lax.scan`. Instead of:

```python
for step in range(n_steps):   # Python loop — slow
    T = step_fn(T, Q(t))
```

we write:

```python
def body(carry, step_idx):
    T, = carry
    return step_fn(T, Q(t)), None

T_final, _ = lax.scan(body, (T0,), jnp.arange(n_steps))
```

The entire 5,000-step loop compiles into **one** XLA operation and runs as a single kernel.

---

## 4. Three Solver Architectures

### Model 1 — Uniform Grid (`runs/run_uniform.py`)

**What it is:** The ground-truth reference. A single 1024×1024 grid covers the full domain at uniform resolution.

**How it works:**
1. Build grid $X, Y$ of shape $(1024, 1024)$.
2. JIT-compile `cn_step` with fixed $\alpha$, $\Delta t$, $\Delta x$, $\Delta y$.
3. Python `for` loop over 5,000 steps: compute $Q(t)$, call `step_fn(T, Q)`.
4. Write VTK snapshots and checkpoints at configured intervals.

**Code path:**
```
build_grid(1024, 1024)
  → make_cn_step_jit(α, dt, dx, dy)     # one-time JIT compile
    → for step in 5000:
        Q = build_laser_source(X, Y, ..., t)
        T = step_fn(T, Q)               # fused XLA kernel per step
```

**Characteristics:**
- DOF: $1024 \times 1024 = 1{,}048{,}576$
- Fully accurate everywhere — every point has the same resolution as the laser focus
- Bottleneck: Python loop dispatches 5,000 kernels; IO writing full 1024×1024 VTK files adds overhead

---

### Model 2 — AMR-Overlay (`runs/run_amr.py`)

**What it is:** The same 1024×1024 solver as Model 1, but with a *post-process* AMR visualization layer. The AMR here is **not a coupled solver** — it is a retrospective analysis and rendering tool applied after each save step.

**Why it exists:** To validate that we can correctly identify where the thermal gradients are and represent them using an adaptive cell structure. This is the stepping stone to the true AMR solver.

**How it works:**
1. Run the identical 1024×1024 CN solver (same accuracy, same DOF as Model 1).
2. Every 100 steps, extract $T$ to NumPy and call `build_amr_cells(T, dx, dy, ...)`.
3. `build_amr_cells` divides the domain into a $16 \times 16$ macro-cell grid, computes max $|\nabla T|$ in each cell, and assigns a refinement level (1–3) based on thresholds $[2.0, 4.0, 16.0]$.
4. Cells at level $L$ are subdivided into $2^{L-1} \times 2^{L-1}$ sub-cells for visualization/VTK output.

**Code path:**
```
# Same solver as Model 1
T = step_fn(T, Q)

# Extra post-process every save_every steps:
cells, level_map = build_amr_cells(T_np, dx, dy, Lx, Ly,
                                    macro=16, tiers=[2.0, 4.0, 16.0])
write_amr_legacy_vtk(path, cells)
```

**Characteristics:**
- Solver DOF: same as uniform (1,048,576) — no compute savings in the PDE solve
- AMR cells: variable — denser near the laser front, coarse everywhere else
- The speed advantage over Model 1 comes primarily from writing smaller AMR VTK files instead of dense 1024×1024 arrays
- Peak temperature is identical to Model 1 (same solver, same grid)

---

### Model 3 — Composite JIT-AMR (`runs/run_composite_amr.py`)

**What it is:** A genuinely coupled two-level solver. The domain is split into a **coarse background grid** and a **fine patch** that stays locked onto the laser zone. Both grids advance together every time step via bidirectional communication.

**This is the actual AMR solver** — not a visualization post-process, but a different PDE discretization that reduces total DOF while preserving accuracy where it matters.

**Grid Layout:**

```
Full domain [0,1]²:
┌─────────────────────────────┐
│  Coarse grid: 256×256        │
│   dx_c = 1/255 ≈ 3.9 mm     │
│                              │
│    ┌──────────────┐          │
│    │ Fine patch   │          │
│    │ 512×512      │          │
│    │ [0.3, 0.7]²  │          │
│    │ dx_f=0.78 mm │          │
│    └──────────────┘          │
└─────────────────────────────┘
```

| Grid | Resolution | DOF |
| :--- | :--- | :--- |
| Coarse | 256×256 | 65,536 |
| Fine patch | 512×512 | 262,144 |
| **Total** | — | **327,680** |
| Uniform 1024 | 1024×1024 | 1,048,576 |
| **Reduction** | — | **3.2× fewer DOF** |

**One Composite Time Step (`amr/composite_step.py`):**

```
1. Advance coarse grid:
   T_coarse_new = cn_step(T_coarse, Q_coarse, α, dt, dx_c, dy_c)

2. Interpolate coarse → fine boundary:
   T_boundary = bilinear_interp(T_coarse_new, patch.Xf, patch.Yf)

3. Advance fine patch with coarse-derived boundary conditions:
   T_patch_new = patch_cn_step(T_patch, Q_patch, T_boundary, α, dt, dx_f, dy_f)

4. Inject fine → coarse (update coarse cells inside patch):
   T_coarse_final = jnp.where(patch.mask, interp(T_patch_new), T_coarse_new)
```

**Bilinear Interpolation (`amr/interpolate.py`):**

Fine grid points inside the patch get boundary conditions from the coarse grid via `jax.scipy.ndimage.map_coordinates` with `order=1` (bilinear). This maps physical coordinates to fractional coarse-grid indices and performs weighted averaging.

**Injection (`amr/patch.py`):**

After the fine patch advances, its solution is projected back onto the coarse grid using `jnp.where` with a boolean mask:

```python
T_coarse_final = jnp.where(patch.mask, T_fine_at_coarse_all, T_coarse)
```

The mask covers coarse cells in $[0.3, 0.7]^2$ — the pre-defined patch region.

**`lax.scan` for JIT efficiency:**

The composite solver runs in chunks of 100 steps using `lax.scan` inside `@jax.jit`:

```python
@jax.jit
def run_chunk(state, t_start):
    def body(carry, step_idx):
        Tc, Tp = carry
        t = t_start + step_idx * dt
        Qc = build_laser_source(Xc, Yc, ..., t)
        Qf = build_laser_source(patch.Xf, patch.Yf, ..., t)
        return composite_step(Tc, Tp, Qc, Qf, ...), None
    return lax.scan(body, state, jnp.arange(chunk_size))
```

Each call to `run_chunk` compiles 100 steps into a single XLA operation. The entire loop over 50 chunks runs the XLA kernel 50 times versus 5,000 Python dispatches for the uniform solver.

---

## 5. The Static-Shape Constraint and Why It Enables JIT

Traditional AMR in C++ or Fortran uses **dynamic quadtrees** — patches are created, destroyed, and resized at runtime based on error indicators. Each mesh change triggers a re-allocation and re-initialization. This is flexible but inherently sequential and impossible to differentiate.

JAX requires **static shapes at compile time**. When you call `@jax.jit`, XLA traces the function once, emitting machine code specialized to the exact array shapes it sees. If a shape changes on the next call, JAX re-traces — which takes seconds and defeats the purpose.

Our design embraces this constraint:

- **Pre-define** the patch location and size before any time-stepping begins.
- **Never change** the coarse or fine grid shapes during the simulation.
- Accept the **trade-off**: if the laser moves completely outside the patch $[0.3, 0.7]^2$, fine-grid accuracy degrades in the laser zone.

For this simulation, the laser's circular orbit stays within the patch bounds throughout:

```
laser center: (0.5 + 0.2·cos(ωt), 0.5 + 0.2·sin(ωt))
max excursion: [0.3, 0.7] in both axes  ✓
patch covers:  [0.3, 0.7]²              ✓
```

The patch is sized to fully contain the orbit with a safety margin. This converts the "infinite flexibility" of dynamic meshes into "extreme speed" — one compilation, 5,000 steps at hardware speed.

---

## 6. Differentiability

This is where JAX-AMR differs fundamentally from any C++-based AMR code.

### The Problem with Traditional AMR

Classical AMR uses conditional logic: *if the gradient exceeds a threshold, refine; else coarsen.* These if-else branches create **non-differentiable discontinuities** in the computational graph. Gradient-based optimization or sensitivity analysis breaks at these jumps.

### How JAX-AMR Stays Differentiable

Every operation in the composite solver is implemented in pure `jnp`:

| Operation | Implementation | Differentiable? |
| :--- | :--- | :--- |
| 5-point Laplacian | `jnp.pad` + array slicing | Yes |
| Dirichlet BC | `jnp.ndarray.at[...].set(...)` | Yes |
| Bilinear interpolation | `jax.scipy.ndimage.map_coordinates(order=1)` | Yes |
| Fine→coarse injection | `jnp.where(mask, ...)` | Yes |
| lax.scan time loop | `jax.lax.scan` | Yes |
| Laser source | `jnp.exp(...)` | Yes |

There are **no Python if-statements, no NumPy calls, no dynamic control flow** inside the JIT-compiled region. The computation graph is a continuous function from inputs to outputs.

This means you can do:

```python
def peak_temperature(laser_power):
    res = run_simulation(laser_power=laser_power)
    return jnp.max(res["T_final"])

# Exact gradient: d(T_max)/d(P)
dT_dP = jax.grad(peak_temperature)(2500.0)
```

The gradient flows back through 5,000 time steps, through all interpolation and injection operations, through the Crank-Nicolson fixed-point iterations — exactly, not via finite differences. This enables **physics-informed optimization**: minimize peak temperature over beam parameters, run inverse problems, or train neural network surrogates with ground-truth Jacobians.

---

## 7. Benchmark Results

All runs: 5,000 steps, $\Delta t = 10^{-4}$ s, moving circular laser, Apple M2 CPU.

| Metric | Model 1: Uniform 1024 | Model 2: AMR-Overlay | Model 3: Composite AMR |
| :--- | :---: | :---: | :---: |
| **Wallclock Time** | 147.02 s | 22.98 s | **31.60 s** |
| **Speedup vs Uniform** | 1× | **6.4×** | **4.65×** |
| **Peak Temperature** | 118.7011 K | 118.7011 K | 118.6312 K |
| **Temperature Error** | 0% (reference) | 0% | **< 0.06%** |
| **Solver DOF** | 1,048,576 | 1,048,576 (same as uniform) | **327,680** |
| **DOF Reduction** | 1× | **none** | **3.2×** |
| **AMR in PDE Solve?** | No | **No** (visualization only) | **Yes** |
| **Differentiable?** | Yes (partial) | No | **Yes (full)** |

### Reading the Results

**Model 2 (AMR-Overlay) is fastest** because it runs the same 1024×1024 JAX solver but writes compact AMR VTK cell lists instead of full 1M-point dense arrays. The compute is identical; the IO is radically smaller.

**Model 3 (Composite AMR) saves the most compute** because it genuinely reduces DOF by 3.2× and uses `lax.scan` to batch 100 steps per XLA dispatch. The 4.65× wallclock improvement comes from: (a) fewer floating-point operations per step, (b) less data movement in/out of cache, and (c) fewer kernel launches.

**Temperature accuracy:** The composite solver produces peak temperature within 0.06% of the uniform reference. The small discrepancy arises from the coarser background grid outside the patch region. Inside the patch (where the laser is), the 512×512 fine grid resolves the Gaussian source at ~0.78 mm spacing versus ~0.98 mm for the uniform 1024 grid — comparable resolution where it counts.

---

## 8. Module Architecture

```
JAX-amR/
├── config/
│   └── params.py           # All hyperparameters: grid sizes, laser, dt, AMR thresholds
├── solver/
│   ├── grid.py             # build_grid(), build_laser_source() (Gaussian, moving/stationary)
│   ├── ops.py              # laplacian() (5-point FD), apply_bc() (Dirichlet)
│   └── cn_step.py          # cn_step(), make_cn_step_jit() (Crank-Nicolson + lax.scan)
├── amr/
│   ├── interpolate.py      # bilinear_interp() via map_coordinates
│   ├── patch.py            # PatchInfo, build_patch_info(), interpolate_coarse_to_fine(),
│   │                       #   inject_fine_to_coarse()
│   ├── composite_step.py   # composite_step() — one coupled coarse+fine CN step
│   ├── cells.py            # build_amr_cells() — gradient-based cell hierarchy (overlay)
│   ├── gradient.py         # compute_gradient_magnitude() for overlay AMR
│   └── thresholds.py       # assign_levels_array() — level assignment from gradient tiers
├── runs/
│   ├── run_uniform.py      # Model 1 driver
│   ├── run_amr.py          # Model 2 driver (overlay)
│   └── run_composite_amr.py# Model 3 driver (coupled composite)
├── viz/
│   ├── snapshots.py        # Multi-panel temperature plots
│   ├── animate.py          # GIF animation from frame list
│   ├── heatmap.py          # Single-frame heatmap
│   ├── crosssection.py     # 1D temperature profiles
│   └── amr_overlay.py      # AMR cell boundary visualization
├── analysis/
│   ├── metrics.py          # Timer context manager, peak-T extraction
│   ├── convergence.py      # Grid convergence analysis
│   └── comparison.py       # Multi-model result comparison
└── ioutils/
    ├── vtk_writer.py       # Legacy VTK + PVD for ParaView
    └── checkpoint.py       # .npz checkpoint save/load
```

### Key Data Flow (Model 3)

```
config/params.py
    ↓ Nc=256, Nf=512, patch=[0.3,0.7]², laser params
build_patch_info()  →  PatchInfo (Xf, Yf, Xc, Yc, mask)
    ↓
@jax.jit run_chunk(state, t_start)
    ↓
lax.scan over 100 steps:
    build_laser_source(Xc, Yc, ..., t)  →  Q_coarse
    build_laser_source(Xf, Yf, ..., t)  →  Q_fine
    composite_step(Tc, Tp, Qc, Qf, patch)
        ├── cn_step(Tc, Qc, dx_c, dy_c)          → T_coarse_new
        ├── bilinear_interp(T_coarse_new, Xf, Yf) → T_boundary
        ├── patch_cn_step(Tp, Qf, T_boundary)     → T_patch_new
        └── jnp.where(mask, T_patch_new, T_coarse_new) → T_coarse_final
    ↓
(T_coarse_final, T_patch_new)
```

---

## 9. Replication Guide

### 9.1 Environment Setup

```bash
git clone <repo>
cd JAX-amR
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Dependencies** (`requirements.txt`):
```
jax>=0.4.20
jaxlib>=0.4.20
numpy>=1.26
matplotlib>=3.8
imageio>=2.33
Pillow>=10.0
scipy>=1.11
```

No CUDA required. Tested on Apple M2 with the CPU backend. JAX will auto-detect available hardware.

### 9.2 Running the Three Models

```bash
# Model 1: Uniform reference (slow — ~147s)
PYTHONPATH=. python runs/run_uniform.py

# Model 2: AMR-Overlay (fast IO — ~23s)
PYTHONPATH=. python runs/run_amr.py

# Model 3: Composite JIT-AMR (fewer DOF — ~32s)
PYTHONPATH=. python runs/run_composite_amr.py
```

Output lands in `output/uniform/`, `output/amr_overlay/`, and `output/amr/` respectively — each containing `snapshots.png`, `animation.gif`, VTK files, and `.npz` checkpoints.

### 9.3 Key Configuration (`config/params.py`)

To change resolution or laser mode, edit `config/params.py`:

```python
LASER_MODE = "circular"    # or "stationary"
laser_power = 2500.0       # W/m²
n_steps = 5000             # time steps
dt = 1e-4                  # seconds per step

# Composite AMR patch (must contain laser orbit)
patch_x0, patch_x1 = 0.3, 0.7
patch_y0, patch_y1 = 0.3, 0.7
```

The patch bounds must contain the laser's full orbit — for the circular mode, the laser sweeps $[0.3, 0.7]^2$, which exactly fills the default patch.

---

## 10. Limitations and Future Work

**Static patch location.** If a problem requires the laser to traverse the entire domain, the composite solver degrades — the fine patch cannot follow. One mitigation is to re-define the patch between large time intervals and re-compile (acceptable if the patch moves rarely). True dynamic remeshing in JAX would require arrays padded to a maximum size and a JAX-native "active cell" mask — complex but feasible.

**Boundary condition coupling.** The current composite step uses one-way coupling at the fine-patch boundary: coarse → fine BCs are applied, then fine → coarse injection. A tighter approach would iterate the coupling to convergence within each time step (Schwarz iteration), improving accuracy at the patch interface.

**GPU scaling.** The current implementation runs on CPU. On GPU, `lax.scan` unrolls all 100 steps in a single device dispatch, and the XLA compiler can pipeline coarse/fine kernel execution. The differentiability property becomes especially powerful on GPU — full Jacobians for inverse design at GPU speed.

**Higher-order interpolation.** Bilinear (order=1) interpolation at the patch boundary introduces a small error proportional to $(\Delta x_c)^2$. Replacing with cubic (`order=3`) in `map_coordinates` would reduce this at minimal runtime cost, since it remains fully differentiable.

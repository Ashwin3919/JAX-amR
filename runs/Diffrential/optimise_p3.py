"""
P3 — Constrained dose maximisation.
Maximise total delivered laser energy subject to peak temperature never
exceeding T_MAX.  Uses a soft penalty (relu²) on constraint violations.

Run: python runs/Diffrential/optimise_p3.py
"""
import sys, os
_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "runs"))
os.environ.setdefault("JAX_PLATFORMS", "")
import jax
import jax.numpy as jnp
from jax import lax
import numpy as np
import config.params as p
from solver.grid import build_grid
from solver.laser_source import build_laser_source
from solver.cn_step import cn_step

# ── Config ────────────────────────────────────────────────────────────────────
N, N_STEPS, CHUNK = 64, 500, 50
N_CHUNKS = N_STEPS // CHUNK   # 10 chunks
T_MAX = 15.0       # safety ceiling (K)
PENALTY = 5000.0   # weight on constraint violations
N_ITERS = 300
LR = 100.0
MAX_POW = 20000.0

# ── Static setup ──────────────────────────────────────────────────────────────
X, Y = build_grid(N, N, p.Lx, p.Ly)
dx, dy = p.Lx / (N - 1), p.Ly / (N - 1)
T0 = jnp.zeros((N, N))

# ── Differentiable simulation ─────────────────────────────────────────────────
@jax.jit
def simulate(schedule):
    """Power schedule (N_CHUNKS,) → per-chunk peak T array (N_CHUNKS,)."""
    def outer(T, chunk_idx):
        power = schedule[chunk_idx]
        t_start = chunk_idx * CHUNK * p.dt
        def inner(Tk, step_idx):
            t = t_start + step_idx * p.dt
            Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, power, t)
            return cn_step(Tk, Q, p.alpha, p.dt, dx, dy, p.T_wall), None
        T_new, _ = lax.scan(inner, T, jnp.arange(CHUNK))
        return T_new, jnp.max(T_new)
    _, peaks = lax.scan(outer, T0, jnp.arange(N_CHUNKS))
    return peaks

def loss(schedule):
    peaks = simulate(schedule)
    reward = -jnp.sum(schedule) / 1000.0              # maximise total dose
    violation = PENALTY * jnp.mean(jnp.maximum(0.0, peaks - T_MAX)**2)
    return reward + violation

grad_fn = jax.jit(jax.grad(loss))

# ── Adam optimiser ────────────────────────────────────────────────────────────
schedule = jnp.zeros(N_CHUNKS)
m, v = jnp.zeros(N_CHUNKS), jnp.zeros(N_CHUNKS)

print(f"P3 — dose maximisation | T_MAX = {T_MAX} K | {N_CHUNKS} chunks × {CHUNK} steps")
print(f"{'iter':>5}  {'max T (K)':>10}  {'total dose':>12}  {'loss':>12}")

for i in range(N_ITERS):
    g = grad_fn(schedule)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g ** 2
    mh = m / (1 - 0.9 ** (i + 1))
    vh = v / (1 - 0.999 ** (i + 1))
    schedule = jnp.clip(schedule - LR * mh / (jnp.sqrt(vh) + 1e-8), 0.0, MAX_POW)
    if (i + 1) % 20 == 0:
        pks = simulate(schedule)
        l = float(loss(schedule))
        print(f"{i+1:5d}  {float(jnp.max(pks)):10.4f}  {float(jnp.sum(schedule)):12.1f}  {l:12.4e}")

# ── Results ───────────────────────────────────────────────────────────────────
peaks_f = np.asarray(simulate(schedule))
sched_f = np.asarray(schedule)
times = [(k + 1) * CHUNK * p.dt for k in range(N_CHUNKS)]

print(f"\n{'chunk':>6}  {'t (s)':>8}  {'power (W/m²)':>14}  {'peak T (K)':>10}")
for k, (t, pw, pk) in enumerate(zip(times, sched_f, peaks_f)):
    print(f"{k+1:6d}  {t:8.4f}  {pw:14.2f}  {pk:10.4f}")

total_dose = float(jnp.sum(schedule))
max_t = peaks_f.max()
print(f"\nTotal dose: {total_dose:.1f} W/m²")
print(f"Max peak T: {max_t:.4f} K  (ceiling {T_MAX} K)")

if max_t <= T_MAX + 0.1:
    print(f"Constraint satisfied (margin = {T_MAX - max_t:.4f} K).")
else:
    print(f"Constraint violated by {max_t - T_MAX:.4f} K. Consider increasing PENALTY or N_ITERS.")

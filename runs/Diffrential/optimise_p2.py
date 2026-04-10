"""
P2 — Inverse thermal profile tracking.
Find a per-chunk power schedule that makes the domain's peak temperature follow
a prescribed non-linear curve: T(k) = 15 * (k/N_CHUNKS)^1.5.

Run: python runs/Diffrential/optimise_p2.py
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
N_ITERS = 300
LR = 100.0
MAX_POW = 20000.0

# ── Static setup ──────────────────────────────────────────────────────────────
X, Y = build_grid(N, N, p.Lx, p.Ly)
dx, dy = p.Lx / (N - 1), p.Ly / (N - 1)
T0 = jnp.zeros((N, N))

target_curve = 15.0 * (jnp.arange(1, N_CHUNKS + 1) / float(N_CHUNKS)) ** 1.5

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
    return jnp.mean((simulate(schedule) - target_curve) ** 2)

grad_fn = jax.jit(jax.grad(loss))

# ── Adam optimiser ────────────────────────────────────────────────────────────
schedule = jnp.zeros(N_CHUNKS)
m, v = jnp.zeros(N_CHUNKS), jnp.zeros(N_CHUNKS)

print(f"P2 — profile tracking | T(k) = 15*(k/{N_CHUNKS})^1.5 | {N_CHUNKS} chunks × {CHUNK} steps")
print(f"{'iter':>5}  {'RMSE (K)':>10}  {'loss':>12}")

for i in range(N_ITERS):
    g = grad_fn(schedule)
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g ** 2
    mh = m / (1 - 0.9 ** (i + 1))
    vh = v / (1 - 0.999 ** (i + 1))
    schedule = jnp.clip(schedule - LR * mh / (jnp.sqrt(vh) + 1e-8), 0.0, MAX_POW)
    if (i + 1) % 20 == 0:
        l = float(loss(schedule))
        print(f"{i+1:5d}  {float(jnp.sqrt(l)):10.4f}  {l:12.4e}")

# ── Results ───────────────────────────────────────────────────────────────────
peaks_f = np.asarray(simulate(schedule))
target_f = np.asarray(target_curve)
sched_f = np.asarray(schedule)
times = [(k + 1) * CHUNK * p.dt for k in range(N_CHUNKS)]

print(f"\n{'chunk':>6}  {'t (s)':>8}  {'power (W/m²)':>14}  {'peak T':>10}  {'target':>10}  {'err':>8}")
for k, (t, pw, pk, tg) in enumerate(zip(times, sched_f, peaks_f, target_f)):
    print(f"{k+1:6d}  {t:8.4f}  {pw:14.2f}  {pk:10.4f}  {tg:10.4f}  {abs(pk-tg):8.4f}")

rmse = float(np.sqrt(np.mean((peaks_f - target_f) ** 2)))
if rmse < 0.01:
    print(f"\nRMSE = {rmse:.4f} K — converged.")
elif rmse < 0.1:
    print(f"\nRMSE = {rmse:.4f} K — close to target.")
else:
    print(f"\nRMSE = {rmse:.4f} K — consider more iterations or tuning LR.")

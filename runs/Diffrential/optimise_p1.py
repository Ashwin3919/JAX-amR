"""
P1 — Scalar power inversion.
Find the laser power that produces a target peak temperature at the final step.
Uses jax.grad through lax.scan (all 500 CN steps are differentiated).

Run: python runs/Diffrential/optimise_p1.py
"""
import sys, os

# Precision Toggle: Use True for float64 (high accuracy), False for float32 (fast)
USE_64BIT = False
os.environ["JAX_ENABLE_X64"] = "1" if USE_64BIT else "0"

_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "runs"))
os.environ.setdefault("JAX_PLATFORMS", "")
import jax
import jax.numpy as jnp
from jax import lax
import config.params as p
from solver.grid import build_grid
from solver.laser_source import build_laser_source
from solver.cn_step import cn_step

# ── Config ────────────────────────────────────────────────────────────────────
N        = 64         # grid resolution (faster than 1024 for optimisation)
N_STEPS  = 500        # number of time steps
TARGET_T = 25.0       # K — desired peak temperature at t_final
N_ITERS  = 200        # gradient descent iterations
LR       = 100.0      # Adam learning rate
MAX_POW  = 1e5        # upper bound on laser power

# ── Static setup ──────────────────────────────────────────────────────────────
X, Y = build_grid(N, N, p.Lx, p.Ly)
dx   = p.Lx / (N - 1)
dy   = p.Ly / (N - 1)
T0   = jnp.zeros((N, N))

# ── Differentiable simulation ─────────────────────────────────────────────────
@jax.jit
def peak_final(power):
    """Scalar power → peak T at final step."""
    def body(T, step_idx):
        t = step_idx * p.dt
        Q = build_laser_source(X, Y, p.laser_cx, p.laser_cy, p.laser_sigma, power, t)
        return cn_step(T, Q, p.alpha, p.dt, dx, dy, p.T_wall), None
    T_final, _ = lax.scan(body, T0, jnp.arange(N_STEPS))
    return jnp.max(T_final)

def loss(power):
    return (peak_final(power) - TARGET_T) ** 2

grad_fn = jax.jit(jax.grad(loss))

# ── Adam optimiser ────────────────────────────────────────────────────────────
power = jnp.array(p.laser_power)
m = jnp.zeros(())
v = jnp.zeros(())

print(f"P1 — target peak T = {TARGET_T} K  |  init power = {float(power):.1f} W/m²")
print(f"{'iter':>5}  {'power (W/m²)':>14}  {'peak T (K)':>12}  {'loss':>12}")

for i in range(N_ITERS):
    g  = grad_fn(power)
    m  = 0.9  * m + 0.1   * g
    v  = 0.999 * v + 0.001 * g ** 2
    mh = m / (1 - 0.9  ** (i + 1))
    vh = v / (1 - 0.999 ** (i + 1))
    power = jnp.clip(power - LR * mh / (jnp.sqrt(vh) + 1e-8), 0.0, MAX_POW)
    if (i + 1) % 10 == 0:
        pk = float(peak_final(power))
        print(f"{i+1:5d}  {float(power):14.2f}  {pk:12.4f}  {float(loss(power)):12.4e}")

pk_final = float(peak_final(power))
err = abs(pk_final - TARGET_T)
print(f"\nResult:  power = {float(power):.2f} W/m²  |  peak T = {pk_final:.4f} K  (target {TARGET_T} K)")

if err < 0.01:
    print(f"Converged (err = {err:.4f} K).")
elif err < 0.1:
    print(f"Close to target (err = {err:.4f} K).")
else:
    print(f"Did not fully converge (err = {err:.4f} K). Consider more iterations or tuning LR.")

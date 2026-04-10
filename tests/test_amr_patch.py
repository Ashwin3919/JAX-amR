"""
Tests for the dynamic AMR internals:
  - gradient_centroid
  - make_fine_coords (including boundary clamping)
  - reinit_patch (thermal history preservation)
  - adaptive_step (shape, zero-IC stability, differentiability)
"""
import pytest
import jax
import jax.numpy as jnp
import numpy as np

from amr.adaptive_patch import (
    gradient_centroid,
    make_fine_coords,
    coarse_to_fine,
    reinit_patch,
)
from amr.adaptive_step import adaptive_step
from solver.grid import build_grid
import config.params as p


# ── gradient_centroid ─────────────────────────────────────────────────────────

def test_gradient_centroid_uniform_falls_back_to_domain_centre():
    """Uniform field → zero gradient → centroid defaults to (0.5, 0.5)."""
    Xc, Yc = build_grid(16, 16, p.Lx, p.Ly)
    T = jnp.ones((16, 16)) * 5.0
    cx, cy = gradient_centroid(T, Xc, Yc)
    assert float(cx) == pytest.approx(0.5, abs=1e-6)
    assert float(cy) == pytest.approx(0.5, abs=1e-6)


def test_gradient_centroid_tracks_hotspot():
    """Gradient-weighted centroid should be near the peak of a Gaussian hotspot."""
    Xc, Yc = build_grid(64, 64, p.Lx, p.Ly)
    T = jnp.exp(-((Xc - 0.7) ** 2 + (Yc - 0.3) ** 2) / (2 * 0.05 ** 2))
    cx, cy = gradient_centroid(T, Xc, Yc)
    assert abs(float(cx) - 0.7) < 0.05
    assert abs(float(cy) - 0.3) < 0.05


def test_gradient_centroid_is_differentiable():
    """jax.grad should not raise through gradient_centroid."""
    Xc, Yc = build_grid(16, 16, p.Lx, p.Ly)

    def loss(scale):
        T = jnp.exp(-((Xc - 0.5) ** 2 + (Yc - 0.5) ** 2) / (2 * (0.1 * scale) ** 2))
        cx, cy = gradient_centroid(T, Xc, Yc)
        return cx + cy

    g = jax.grad(loss)(jnp.array(1.0))
    assert not jnp.isnan(g)


# ── make_fine_coords ──────────────────────────────────────────────────────────

def test_make_fine_coords_basic_shape_and_bounds():
    """Output shapes are (Nf, Nf) and bounds are (cx±half_w) unclamped."""
    Nf = 32
    Xf, Yf, x0, x1, y0, y1 = make_fine_coords(
        jnp.array(0.5), jnp.array(0.5), 0.2, Nf, Nf, p.Lx, p.Ly
    )
    assert Xf.shape == (Nf, Nf)
    assert Yf.shape == (Nf, Nf)
    assert float(x0) == pytest.approx(0.3, abs=1e-6)
    assert float(x1) == pytest.approx(0.7, abs=1e-6)
    assert float(y0) == pytest.approx(0.3, abs=1e-6)
    assert float(y1) == pytest.approx(0.7, abs=1e-6)


def test_make_fine_coords_clamping_near_left_boundary():
    """Patch centred near x=0 should not extend below x=0."""
    Xf, Yf, x0, x1, y0, y1 = make_fine_coords(
        jnp.array(0.05), jnp.array(0.5), 0.2, 16, 16, p.Lx, p.Ly
    )
    assert float(x0) >= 0.0
    assert float(x1) <= p.Lx
    # Width must be preserved exactly
    assert float(x1) - float(x0) == pytest.approx(0.4, abs=1e-6)


def test_make_fine_coords_clamping_near_right_boundary():
    """Patch centred near x=Lx should not extend beyond x=Lx."""
    Xf, Yf, x0, x1, y0, y1 = make_fine_coords(
        jnp.array(0.95), jnp.array(0.5), 0.2, 16, 16, p.Lx, p.Ly
    )
    assert float(x1) <= p.Lx + 1e-6
    assert float(x1) - float(x0) == pytest.approx(0.4, abs=1e-6)


def test_make_fine_coords_invalid_half_w():
    with pytest.raises(ValueError, match="half_w"):
        make_fine_coords(jnp.array(0.5), jnp.array(0.5), 0.0, 16, 16, p.Lx, p.Ly)


def test_make_fine_coords_invalid_Nf():
    with pytest.raises(ValueError, match="Nf_x"):
        make_fine_coords(jnp.array(0.5), jnp.array(0.5), 0.2, 1, 16, p.Lx, p.Ly)


# ── reinit_patch ─────────────────────────────────────────────────────────────

def test_reinit_patch_stationary_preserves_fine_values():
    """When bounds do not change, reinit_patch must return the old patch (near-identity)."""
    Nf = 16
    Xf, Yf, x0, x1, y0, y1 = make_fine_coords(
        jnp.array(0.5), jnp.array(0.5), 0.2, Nf, Nf, p.Lx, p.Ly
    )
    T_patch = jnp.ones((Nf, Nf)) * 300.0 + jnp.arange(Nf * Nf).reshape(Nf, Nf) * 0.01
    T_coarse = jnp.zeros((32, 32))

    T_new = reinit_patch(
        T_patch, x0, x1, y0, y1,
        Xf, Yf, x0, x1, y0, y1,
        Nf, Nf, T_coarse, 32, 32, p.Lx, p.Ly,
    )
    # Bilinear re-interpolation at the same grid points should be near-exact
    assert jnp.allclose(T_new, T_patch, atol=1e-3)


def test_reinit_patch_nonoverlap_falls_back_to_coarse():
    """Non-overlapping old/new bounds → all new fine points come from coarse."""
    Nf = 16
    Xf_new, Yf_new, x0_new, x1_new, y0_new, y1_new = make_fine_coords(
        jnp.array(0.1), jnp.array(0.1), 0.05, Nf, Nf, p.Lx, p.Ly
    )
    # Old patch far away — no overlap with new patch
    x0_old = jnp.array(0.7)
    x1_old = jnp.array(0.9)
    y0_old = jnp.array(0.7)
    y1_old = jnp.array(0.9)
    T_patch_old = jnp.ones((Nf, Nf)) * 999.0  # should never appear in output

    T_coarse = jnp.ones((32, 32)) * 42.0
    T_new = reinit_patch(
        T_patch_old, x0_old, x1_old, y0_old, y1_old,
        Xf_new, Yf_new, x0_new, x1_new, y0_new, y1_new,
        Nf, Nf, T_coarse, 32, 32, p.Lx, p.Ly,
    )
    # All new fine points are in fresh territory → coarse value (42.0)
    assert jnp.allclose(T_new, 42.0, atol=1e-5)


def test_reinit_patch_is_differentiable():
    """jax.grad should not raise through reinit_patch."""
    Nf = 8
    Xf, Yf, x0, x1, y0, y1 = make_fine_coords(
        jnp.array(0.5), jnp.array(0.5), 0.2, Nf, Nf, p.Lx, p.Ly
    )

    def loss(scale):
        T_patch = jnp.ones((Nf, Nf)) * scale
        T_coarse = jnp.zeros((16, 16))
        T_new = reinit_patch(
            T_patch, x0, x1, y0, y1,
            Xf, Yf, x0, x1, y0, y1,
            Nf, Nf, T_coarse, 16, 16, p.Lx, p.Ly,
        )
        return T_new.sum()

    g = jax.grad(loss)(jnp.array(1.0))
    assert not jnp.isnan(g)


# ── adaptive_step ─────────────────────────────────────────────────────────────

def _make_adaptive_state(Nc=16, Nf=32, half_w=0.25):
    Xc, Yc = build_grid(Nc, Nc, p.Lx, p.Ly)
    dx_c = p.Lx / (Nc - 1)
    dy_c = p.Ly / (Nc - 1)
    T_coarse = jnp.zeros((Nc, Nc))
    _, _, x0, x1, y0, y1 = make_fine_coords(
        jnp.array(0.5), jnp.array(0.5), half_w, Nf, Nf, p.Lx, p.Ly
    )
    T_patch = jnp.zeros((Nf, Nf))
    return T_coarse, T_patch, x0, x1, y0, y1, Xc, Yc, dx_c, dy_c, Nc, Nf


def test_adaptive_step_output_shapes():
    """adaptive_step must return arrays with the correct shapes."""
    Nc, Nf = 16, 32
    T_coarse, T_patch, x0, x1, y0, y1, Xc, Yc, dx_c, dy_c, Nc, Nf = _make_adaptive_state(Nc, Nf)

    Tc_new, Tp_new, x0n, x1n, y0n, y1n = adaptive_step(
        T_coarse, T_patch, x0, x1, y0, y1,
        jnp.zeros((Nc, Nc)), lambda Xf, Yf: jnp.zeros_like(Xf),
        Xc, Yc, 0.25,
        Nc, Nc, Nf, Nf, p.Lx, p.Ly,
        p.alpha, p.dt, dx_c, dy_c, p.T_wall,
    )
    assert Tc_new.shape == (Nc, Nc)
    assert Tp_new.shape == (Nf, Nf)


def test_adaptive_step_zero_ic_zero_source_stays_zero():
    """Zero IC + zero source + Dirichlet walls = temperature stays at zero."""
    Nc, Nf = 16, 32
    T_coarse, T_patch, x0, x1, y0, y1, Xc, Yc, dx_c, dy_c, Nc, Nf = _make_adaptive_state(Nc, Nf)

    Tc_new, Tp_new, *_ = adaptive_step(
        T_coarse, T_patch, x0, x1, y0, y1,
        jnp.zeros((Nc, Nc)), lambda Xf, Yf: jnp.zeros_like(Xf),
        Xc, Yc, 0.25,
        Nc, Nc, Nf, Nf, p.Lx, p.Ly,
        p.alpha, p.dt, dx_c, dy_c, p.T_wall,
    )
    assert jnp.allclose(Tc_new, 0.0, atol=1e-10)
    assert jnp.allclose(Tp_new, 0.0, atol=1e-10)


def test_adaptive_step_is_differentiable():
    """jax.grad should not raise through a full adaptive_step."""
    Nc, Nf = 8, 16
    T_coarse, T_patch, x0, x1, y0, y1, Xc, Yc, dx_c, dy_c, Nc, Nf = _make_adaptive_state(Nc, Nf)

    def loss(power):
        Tc_new, Tp_new, *_ = adaptive_step(
            T_coarse, T_patch, x0, x1, y0, y1,
            jnp.ones((Nc, Nc)) * power,
            lambda Xf, Yf: jnp.ones_like(Xf) * power,
            Xc, Yc, 0.25,
            Nc, Nc, Nf, Nf, p.Lx, p.Ly,
            p.alpha, p.dt, dx_c, dy_c, p.T_wall,
        )
        return Tc_new.sum()

    g = jax.grad(loss)(jnp.array(1.0))
    assert not jnp.isnan(g)
    assert g > 0

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from amr.patch import build_patch_info, interpolate_coarse_to_fine, inject_fine_to_coarse
from amr.interpolate import bilinear_interp
from amr.composite_step import composite_step

def test_bilinear_interp_exact():
    """Test that bilinear interpolation is exact for a linear field."""
    Nc_x, Nc_y = 10, 10
    Lx, Ly = 1.0, 1.0
    
    # Linear field T = x + y
    x = jnp.linspace(0, Lx, Nc_x)
    y = jnp.linspace(0, Ly, Nc_y)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    T_coarse = X + Y
    
    # Fine points
    xf = jnp.array([0.25, 0.75])
    yf = jnp.array([0.25, 0.75])
    Xf, Yf = jnp.meshgrid(xf, yf, indexing="ij")
    
    T_interp = bilinear_interp(T_coarse, Xf, Yf, Lx, Ly, Nc_x, Nc_y)
    T_expected = Xf + Yf
    
    assert jnp.allclose(T_interp, T_expected, atol=1e-5)

def test_patch_info_mask():
    """Test that the patch mask correctly identifies points."""
    patch = build_patch_info(0.2, 0.8, 0.2, 0.8, 10, 10, 5, 5, 1.0, 1.0)
    # Coarse grid (5x5): 0.0, 0.25, 0.5, 0.75, 1.0
    # Inside [0.2, 0.8]: 0.25, 0.5, 0.75
    # So mask should be True for indices 1, 2, 3 in both axes
    expected_mask = jnp.zeros((5, 5), dtype=bool)
    expected_mask = expected_mask.at[1:4, 1:4].set(True)
    
    assert jnp.all(patch.mask == expected_mask)

def test_inject_fine_to_coarse_exact():
    """Test that injection correctly updates the coarse grid."""
    patch = build_patch_info(0.2, 0.8, 0.2, 0.8, 10, 10, 5, 5, 1.0, 1.0)
    
    T_coarse = jnp.zeros((5, 5))
    T_fine = jnp.ones((10, 10)) * 5.0
    
    T_injected = inject_fine_to_coarse(patch, T_coarse, T_fine)
    
    # Points inside should be 5.0, points outside should be 0.0
    assert jnp.all(T_injected[patch.mask] == 5.0)
    assert jnp.all(T_injected[~patch.mask] == 0.0)

def test_composite_step_differentiability():
    """Test that we can take gradients through a composite step."""
    patch = build_patch_info(0.4, 0.6, 0.4, 0.6, 16, 16, 8, 8, 1.0, 1.0)
    
    def loss(power):
        Tc = jnp.zeros((8, 8))
        Tp = jnp.zeros((16, 16))
        Qc = jnp.ones((8, 8)) * power
        Qp = jnp.ones((16, 16)) * power
        
        Tc_next, Tp_next = composite_step(
            Tc, Tp, Qc, Qp, patch, 1e-3, 1e-3, 0.1, 0.1, 0.01, 0.01
        )
        return Tc_next.sum()
    
    grad_fn = jax.grad(loss)
    g = grad_fn(100.0)
    
    assert not jnp.isnan(g)
    assert g > 0

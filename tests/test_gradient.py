"""Tests for amr/gradient.py — verifies JAX differentiability."""
import jax
import jax.numpy as jnp
import numpy as np
from amr.gradient import compute_gradient_magnitude


def test_gradient_uniform_field_is_zero():
    """Gradient of a constant field should be zero everywhere."""
    T = jnp.ones((16, 16))
    mag = compute_gradient_magnitude(T, 0.1, 0.1)
    assert jnp.allclose(mag, 0.0, atol=1e-6), f"Expected zero gradient, got max={mag.max()}"


def test_gradient_linear_field():
    """Gradient of T=x should give |grad T| ≈ 1 in the interior."""
    Nx, Ny = 32, 32
    x = jnp.linspace(0, 1, Nx)
    y = jnp.linspace(0, 1, Ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    T = X
    dx = 1.0 / (Nx - 1)
    dy = 1.0 / (Ny - 1)
    mag = compute_gradient_magnitude(T, dx, dy)
    assert jnp.allclose(mag[1:-1, 1:-1], 1.0, atol=1e-5), f"Interior gradient wrong: {mag[1:-1,1:-1].mean()}"


def test_gradient_is_differentiable():
    """jax.grad must not raise — proves the function is JAX-compatible."""
    def total_grad(scale: float) -> float:
        T = jnp.linspace(0, scale, 8).reshape(8, 1) * jnp.ones((8, 8))
        return compute_gradient_magnitude(T, 0.1, 0.1).sum()
    g = jax.grad(total_grad)(1.0)
    assert jnp.isfinite(g), f"Gradient is not finite: {g}"


def test_gradient_output_shape():
    """Output shape must match input shape."""
    T = jnp.ones((12, 15))
    mag = compute_gradient_magnitude(T, 0.05, 0.05)
    assert mag.shape == T.shape, f"Shape mismatch: {mag.shape} vs {T.shape}"

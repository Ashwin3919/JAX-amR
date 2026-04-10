"""Tests for config/params.py — validates physical parameter consistency."""
import config.params as p


def test_positive_physical_params():
    assert p.alpha > 0, "Thermal diffusivity must be positive"
    assert p.dt > 0, "Time step must be positive"
    assert p.laser_sigma > 0, "Laser sigma must be positive"
    assert p.laser_power > 0, "Laser power must be positive"
    assert p.grad_epsilon > 0, "Gradient epsilon must be positive"


def test_grid_size_valid():
    assert p.Nx >= 2, "Nx must be >= 2"
    assert p.Ny >= 2, "Ny must be >= 2"


def test_cfl_sanity():
    """CN is unconditionally stable, but extreme CFL is a sign of misconfiguration."""
    dx = p.Lx / (p.Nx - 1)
    cfl = p.alpha * p.dt / dx ** 2
    assert cfl < 10.0, f"CFL number {cfl:.2f} suspiciously large — check dt or alpha"


def test_laser_mode_valid():
    assert p.LASER_MODE in ("stationary", "circular"), f"Unknown LASER_MODE: {p.LASER_MODE}"


def test_new_constants_present():
    assert hasattr(p, "laser_omega"), "laser_omega missing from config"
    assert hasattr(p, "laser_radius"), "laser_radius missing from config"
    assert hasattr(p, "grad_epsilon"), "grad_epsilon missing from config"

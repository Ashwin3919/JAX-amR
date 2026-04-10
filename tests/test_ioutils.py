"""Tests for ioutils/ — checkpoint and VTK writer."""
from __future__ import annotations
import os
import tempfile

import numpy as np
import pytest

from ioutils.checkpoint import save_checkpoint, load_checkpoint
from ioutils.vtk_writer import write_legacy_vtk


def test_checkpoint_roundtrip(tmp_path):
    T = np.random.rand(16, 16).astype(np.float32)
    path = str(tmp_path / "ckpt.npz")
    save_checkpoint(path, T, step=10, t=0.001)
    T2, step, t = load_checkpoint(path)
    assert np.allclose(T, T2, atol=1e-5), "Roundtrip failed"
    assert step == 10
    assert abs(t - 0.001) < 1e-9


def test_checkpoint_missing_file():
    with pytest.raises(FileNotFoundError):
        load_checkpoint("/nonexistent/path/ckpt.npz")


def test_save_checkpoint_bad_shape(tmp_path):
    with pytest.raises(ValueError, match="2-D"):
        save_checkpoint(str(tmp_path / "x.npz"), np.ones((4, 4, 4)), step=0, t=0.0)


def test_vtk_shape_mismatch():
    X = np.ones((4, 4))
    Y = np.ones((4, 5))  # deliberate mismatch
    T = np.ones((4, 4))
    with pytest.raises(ValueError, match="shape mismatch"):
        write_legacy_vtk("/tmp/test_shape.vtk", X, Y, T)


def test_vtk_creates_file(tmp_path):
    n = 8
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    T = np.ones((n, n))
    path = str(tmp_path / "test.vtk")
    write_legacy_vtk(path, X, Y, T)
    assert os.path.exists(path)
    assert os.path.getsize(path) > 0

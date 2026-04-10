"""Smoke tests for viz/ — headless matplotlib."""
from __future__ import annotations
import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _dummy_field(n: int = 16) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, x, indexing="ij")
    T = np.random.rand(n, n)
    return T, X, Y


def test_plot_heatmap_returns_image():
    from viz.heatmap import plot_heatmap
    T, X, Y = _dummy_field()
    fig, ax = plt.subplots()
    im = plot_heatmap(ax, T, X, Y, dark=False)
    assert im is not None
    plt.close(fig)


def test_plot_crosssection_returns_line():
    from viz.crosssection import plot_crosssection
    T, X, Y = _dummy_field()
    fig, ax = plt.subplots()
    line = plot_crosssection(ax, T, X, Y, dark=False)
    assert line is not None
    plt.close(fig)

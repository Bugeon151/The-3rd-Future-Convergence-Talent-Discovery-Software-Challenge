"""Tests for core geometry (homography, foot points)."""
from __future__ import annotations

import numpy as np
import pytest

from nearzero.core.geometry import Homography, foot_point


def test_homography_identity():
    h = Homography()
    p = (3.0, 4.0)
    back = tuple(float(v) for v in h.to_world(*h.to_pixel(*p)))
    assert np.allclose(back, p, atol=1e-6)


def test_homography_from_points_roundtrip():
    px = [[0, 0], [640, 0], [640, 480], [0, 480]]
    world = [[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]]
    h = Homography.from_points(px, world)
    for p, w in zip(px, world):
        assert np.allclose(h.to_world(*p), w, atol=1e-3)


def test_foot_point_center():
    box = (10, 20, 110, 220)
    fx, fy = foot_point(box)
    assert fx == 60
    assert fy == 220
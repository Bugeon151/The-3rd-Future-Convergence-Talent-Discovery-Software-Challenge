"""Floor-plane geometry: image<->world homography and world-space helpers.

The system maps feet points of detected objects to a ground plane expressed in
meters. This is more stable in demo conditions than monocular depth estimation.
"""
from __future__ import annotations

import json
import os
from typing import Sequence

import numpy as np

try:  # pragma: no cover - cv2 is a hard dependency in practice
    import cv2
except Exception:  # pragma: no cover
    cv2 = None


def _h(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    v = m @ np.array([x, y, 1.0])
    return float(v[0] / v[2]), float(v[1] / v[2])


class Homography:
    """4-point floor homography (image pixel -> world meter)."""

    def __init__(self, h: np.ndarray | None = None):
        self._h = h if h is not None else np.eye(3, dtype=np.float64)
        if not self.is_identity():
            self._inv = np.linalg.inv(self._h)
        else:
            self._inv = self._h

    @classmethod
    def from_points(cls, src: Sequence[Sequence[float]], dst: Sequence[Sequence[float]]) -> "Homography":
        if cv2 is None:  # pragma: no cover
            raise RuntimeError("cv2 required for homography")
        src = np.asarray(src, dtype=np.float32)
        dst = np.asarray(dst, dtype=np.float32)
        h, _ = cv2.findHomography(src, dst)
        return cls(h)

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        return _h(self._h, float(x), float(y))

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        return _h(self._inv, float(x), float(y))

    def distance(self, w1: tuple[float, float], w2: tuple[float, float]) -> float:
        return float(np.hypot(w1[0] - w2[0], w1[1] - w2[1]))

    def is_identity(self) -> bool:
        return bool(np.allclose(self._h, np.eye(3)))

    @classmethod
    def identity(cls) -> "Homography":
        return cls()

    def to_json(self) -> list[list[float]]:
        return self._h.tolist()

    @classmethod
    def from_json(cls, h_json) -> "Homography":
        return cls(np.asarray(h_json, dtype=np.float64))


def save_homography(h: Homography, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(h.to_json(), fh)


def load_homography(path: str) -> Homography | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return Homography.from_json(json.load(fh))


def foot_point(box: Sequence[float]) -> tuple[float, float]:
    """Ground-plane reference point (feet/mid-bottom of a detection box)."""
    x1, y1, x2, y2 = [float(v) for v in box]
    return (x1 + x2) / 2.0, max(y1, y2)


def line_segment_min_distance(
    p0: np.ndarray, p1: np.ndarray, q0: np.ndarray, q1: np.ndarray
) -> float:
    """Minimum distance between two 2D line segments (used for analytic checks)."""
    return _seg_seg(p0, p1, q0, q1)


def _seg_seg(p1, p2, q1, q2) -> float:
    d1 = q2[0] - q1[0]
    d2 = q2[1] - q1[1]
    d0 = p2[0] - p1[0]
    d3 = p2[1] - p1[1]
    denom = d0 * d2 - d3 * d1
    best = 1e18
    pts = [p1, p2, q1, q2]
    for i in range(2):
        for j in range(2, 4):
            d = pts[i] - pts[j]
            best = min(best, float(np.dot(d, d)))
    if abs(denom) < 1e-9:
        return float(best ** 0.5)
    t = ((q1[0] - p1[0]) * d2 - (q1[1] - p1[1]) * d1) / denom
    u = ((q1[0] - p1[0]) * d3 - (q1[1] - p1[1]) * d0) / denom
    if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
        return 0.0
    for seg in [(p1, p2), (q1, q2)]:
        a, b = seg
        for pt in (p1, p2, q1, q2):
            if pt is a or pt is b:
                continue
            v = b - a
            if np.dot(v, v) < 1e-12:
                continue
            proj = a + v * np.clip(np.dot(pt - a, v) / np.dot(v, v), 0.0, 1.0)
            d = float(np.dot(proj - pt, proj - pt))
            best = min(best, d)
    return float(best ** 0.5)


def point_segment_min_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    v = b - a
    if np.dot(v, v) < 1e-12:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, v) / np.dot(v, v), 0.0, 1.0)
    return float(np.linalg.norm(p - (a + t * v)))


def wrap_angle(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)
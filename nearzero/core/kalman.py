"""Constant-velocity Kalman filter for 2D ground-plane tracking.

State vector: [x, y, vx, vy]. Tracks supply both filtered positions and a
covariance that feeds the uncertainty-aware trajectory sampler.
"""
from __future__ import annotations

import numpy as np


class CVKalmanFilter:
    def __init__(
        self,
        x: float,
        y: float,
        vx: float = 0.0,
        vy: float = 0.0,
        process_noise: float = 0.6,
        measurement_noise: float = 0.08,
    ):
        self._x = np.array([float(x), float(y), float(vx), float(vy)])
        self._p = np.eye(4) * 1.0
        self._q = process_noise * np.diag([0.02, 0.02, 1.0, 1.0])
        self._r = measurement_noise * np.eye(2)

    def predict(self, dt: float) -> None:
        f = np.array(
            [[1.0, 0.0, dt, 0.0], [0.0, 1.0, 0.0, dt], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        )
        self._x = f @ self._x
        self._p = f @ self._p @ f.T + self._q

    def update(self, zx: float, zy: float) -> None:
        z = np.array([float(zx), float(zy)])
        h = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        s = h @ self._p @ h.T + self._r
        k = self._p @ h.T @ np.linalg.inv(s)
        self._x = self._x + k @ (z - h @ self._x)
        self._p = (np.eye(4) - k @ h) @ self._p

    def set_position_velocity(self, x: float, y: float, vx: float, vy: float) -> None:
        self._x = np.array([float(x), float(y), float(vx), float(vy)])

    def get(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._x.copy(), self._x[0:2].copy(), self._x[2:4].copy()

    def covariance(self) -> np.ndarray:
        return self._p.copy()

    @staticmethod
    def estimate_velocity(times: np.ndarray, pos: np.ndarray) -> tuple[float, float]:
        if len(times) < 2:
            return 0.0, 0.0
        dt = (times[-1] - times[0]) or 1e-9
        v = (pos[-1] - pos[0]) / dt
        return float(v[0]), float(v[1])


def ema_velocity(history: list[tuple[float, float, float]], window: int = 5) -> tuple[float, float]:
    """Median-smoothed finite-difference velocity from (t, x, y) history."""
    if len(history) < 2:
        return 0.0, 0.0
    recent = history[-window:]
    vx = np.median(
        [(recent[i][1] - recent[i - 1][1]) / max(recent[i][0] - recent[i - 1][0], 1e-6) for i in range(1, len(recent))]
    )
    vy = np.median(
        [(recent[i][2] - recent[i - 1][2]) / max(recent[i][0] - recent[i - 1][0], 1e-6) for i in range(1, len(recent))]
    )
    return float(vx), float(vy)
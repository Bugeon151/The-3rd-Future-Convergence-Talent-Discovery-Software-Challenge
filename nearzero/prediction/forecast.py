"""Probabilistic trajectory forecasting.

A Trajectory holds an expected path plus Monte-Carlo samples that represent
future-position uncertainty (growing with horizon). Sample sets of two objects
are matched by index for dependent collision probability estimation.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    t0: float
    times: np.ndarray
    positions: np.ndarray
    samples: np.ndarray | None = None

    @property
    def has_uncertainty(self) -> bool:
        return self.samples is not None and self.samples.shape[0] > 0


class Forecaster(abc.ABC):
    name: str = "base"

    def __init__(self, horizon_s: float = 4.0, dt_s: float = 0.1, n_samples: int = 100):
        self.horizon_s = horizon_s
        self.dt_s = dt_s
        self.n_samples = n_samples
        self._times: np.ndarray | None = None

    def times(self) -> np.ndarray:
        if self._times is None:
            steps = int(round(self.horizon_s / self.dt_s)) + 1
            self._times = np.arange(steps, dtype=float) * self.dt_s
        return self._times

    @abc.abstractmethod
    def forecast(self, pos: np.ndarray, vel: np.ndarray, cov: np.ndarray | None, t0: float) -> Trajectory:
        pass

    def sample_uncertainty(self, cov, scale: float) -> np.ndarray:
        if cov is None:
            sigma = np.array([[0.05, 0.0], [0.0, 0.05]])
        else:
            sigma = cov[0:2, 0:2].copy() * scale
            sigma = np.clip(sigma, 0.0004, 4.0)
        return sigma


class ConstantVelocityForecaster(Forecaster):
    """Baseline A forecast: linear extrapolation of current velocity."""

    name = "constant_velocity"

    def forecast(self, pos, vel, cov, t0):
        ts = self.times()
        expected = pos[None, :] + np.outer(ts, vel)
        sigma = self.sample_uncertainty(cov, 1.0)
        ang_noise = 0.30
        samples = np.empty((self.n_samples, ts.size, 2))
        rng = np.random.default_rng(0)
        for i in range(self.n_samples):
            v = vel + rng.multivariate_normal(np.zeros(2), sigma) + rng.normal(0.0, 0.04, 2)
            ang = rng.normal(0.0, ang_noise)
            speed = np.hypot(v[0], v[1])
            if speed > 1e-6:
                ca, sa = np.cos(ang), np.sin(ang)
                v = np.array([v[0] * ca - v[1] * sa, v[0] * sa + v[1] * ca])
            drift = ts[:, None] * rng.multivariate_normal(np.zeros(2), sigma)
            samples[i] = pos[None, :] + np.outer(ts, v) + drift
        return Trajectory(t0, t0 + ts, expected, samples)


class KalmanForecaster(Forecaster):
    """Baseline B forecast: propagate the Kalman state and covariance."""

    name = "kalman"

    def __init__(self, horizon_s=4.0, dt_s=0.1, n_samples=100, process_scale: float = 1.2):
        super().__init__(horizon_s, dt_s, n_samples)
        self.process_scale = process_scale

    def forecast(self, pos, vel, cov, t0):
        ts = self.times()
        f = np.array(
            [
                [1.0, 0.0, self.dt_s, 0.0],
                [0.0, 1.0, 0.0, self.dt_s],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        x = np.concatenate([pos, vel])
        step = np.eye(4) * 0.02
        if cov is None:
            p = np.eye(4) * 1.0
        else:
            p = cov * self.process_scale
        expected_list = []
        xk = x
        pk = p
        for _ in ts:
            expected_list.append(xk[0:2])
            xk = f @ xk
            pk = f @ pk @ f.T + step
        expected = np.asarray(expected_list)
        sigma = p[0:2, 0:2]
        sigma = np.clip(sigma, 0.0004, 4.0)
        rng = np.random.default_rng(0)
        samples = np.empty((self.n_samples, ts.size, 2))
        for i in range(self.n_samples):
            v = vel + rng.multivariate_normal(np.zeros(2), sigma * 1.5)
            drift = np.stack([rng.multivariate_normal(np.zeros(2), sigma * (k + 1)) for k in range(len(ts))])
            line = pos[None, :] + np.outer(ts, v)
            samples[i] = line + drift
        return Trajectory(t0, t0 + ts, expected, samples)


class EnsembleForecaster(Forecaster):
    """NEAR-ZERO X multi-future forecaster.

    Combines independent hypotheses (linear, kalman-propagated, laterally
    perturbed maneuver hypotheses) into one sample set and expected path.
    Hypothesis blending avoids a single-model overconfidence.
    """

    name = "ensemble"

    def __init__(self, horizon_s=4.0, dt_s=0.1, n_samples=100, lateral_maneuvers=0.9):
        super().__init__(horizon_s, dt_s, n_samples)
        self.lateral_maneuvers = lateral_maneuvers
        self._cv = ConstantVelocityForecaster(horizon_s, dt_s, n_samples)
        self._kf = KalmanForecaster(horizon_s, dt_s, n_samples)

    def forecast(self, pos, vel, cov, t0):
        ts = self.times()
        if self.n_samples <= 0:
            expected = pos[None, :] + np.outer(ts, vel)
            return Trajectory(t0, t0 + ts, expected, None)
        base_s = self.n_samples * 3 // 4
        self._cv.n_samples = base_s
        self._kf.n_samples = self.n_samples // 4
        cv_t = self._cv.forecast(pos, vel, cov, t0)
        kf_t = self._kf.forecast(pos, vel, cov, t0)
        samples = np.concatenate([cv_t.samples, kf_t.samples], axis=0)
        rng = np.random.default_rng(1)
        speed = float(np.hypot(vel[0], vel[1]))
        if speed > 0.3:
            lat = np.array([-vel[1], vel[0]]) / max(speed, 1e-6) * self.lateral_maneuvers
            for i in range(min(4, max(1, self.n_samples - samples.shape[0]))):
                tau = rng.uniform(0.4, 1.2)
                shift = np.zeros((cv_t.times.size, 2))
                mask = cv_t.times > tau
                shift[mask] = (cv_t.times[mask] - tau)[:, None] * lat[None, :]
                samples = np.concatenate([samples, (cv_t.positions + shift)[None, ...]], axis=0)
        samples = samples[: self.n_samples]
        expected = np.mean(samples, axis=0) if samples.shape[0] else cv_t.positions
        return Trajectory(t0, cv_t.times, expected, samples)
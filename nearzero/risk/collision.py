"""Probabilistic pair risk: collision probability, TTC, PMD, collision point.

Estimates treat the forecast sample sets as an approximate posterior over future
trajectories. Nothing here claims to be a true Bayesian posterior; the UI and docs
label these as *model-based estimates*.
"""
from __future__ import annotations

import numpy as np

from ..prediction.forecast import Trajectory


def expected_distance_profile(a: Trajectory, b: Trajectory) -> np.ndarray:
    return np.linalg.norm(a.positions - b.positions, axis=1)


def pair_minimum_distance(a: Trajectory, b: Trajectory) -> tuple[float, float]:
    d = expected_distance_profile(a, b)
    i = int(np.argmin(d))
    return float(d[i]), float(a.times[i])


def relative_kinematics(a_pos, a_vel, b_pos, b_vel):
    sep = np.asarray(a_pos) - np.asarray(b_pos)
    rel = np.asarray(a_vel) - np.asarray(b_vel)
    d = float(np.linalg.norm(sep))
    closing = -float(np.dot(sep, rel)) / max(d, 1e-9)
    return d, closing


def analytic_ttc(a_pos, a_vel, b_pos, b_vel, r_sum: float) -> float | None:
    d, closing = relative_kinematics(a_pos, a_vel, b_pos, b_vel)
    if closing <= 0.0:
        return None
    eff = max(d - r_sum, 0.0)
    return eff / closing


def ttc_from_profiles(a: Trajectory, b: Trajectory, r_sum: float) -> float | None:
    d = expected_distance_profile(a, b)
    first = np.where(d <= r_sum)[0]
    if first.size == 0:
        return None
    i = int(first[0])
    if i == 0:
        return float(a.times[0])
    d0, d1 = d[i - 1], d[i]
    t0, t1 = a.times[i - 1], a.times[i]
    if d1 >= d0 + 1e-12:
        return None
    frac = (d0 - r_sum) / max(d0 - d1, 1e-9)
    return float(np.clip(t0 + frac * (t1 - t0), 0.0, a.times[-1]))


def _matched_samples(a: Trajectory, b: Trajectory):
    if a.samples is None or b.samples is None:
        return None
    n = min(a.samples.shape[0], b.samples.shape[0])
    if n == 0:
        return None
    return a.samples[:n], b.samples[:n]


def collision_probability(a: Trajectory, b: Trajectory, r_sum: float) -> float:
    s = _matched_samples(a, b)
    if s is None:
        d = expected_distance_profile(a, b)
        return 0.0 if float(d.min()) > r_sum else 1.0
    sa, sb = s
    dists = np.linalg.norm(sa - sb, axis=2)
    min_d = dists.min(axis=1)
    return float(np.mean(min_d < r_sum))


def zone_intrusion_probability(traj: Trajectory, polygons: list[np.ndarray]) -> float:
    if not polygons:
        return 0.0
    if traj.samples is None:
        pts = traj.positions
        inside = [_inside_any(p, polygons) for p in pts]
        return float(np.any(inside))
    flags = np.zeros(traj.samples.shape[0], dtype=bool)
    for i in range(traj.samples.shape[0]):
        flags[i] = any(_inside_any(p, polygons) for p in traj.samples[i])
    return float(np.mean(flags))


def pmd_world_uncertain(traj: Trajectory) -> float:
    """Expected PMD from the sample set (mean of per-sample minima)."""
    if traj.samples is None:
        return 0.05
    min_d = np.linalg.norm(traj.samples - traj.samples.mean(axis=(1, 2), keepdims=True), axis=2).min(axis=1)
    return float(np.mean(min_d))


def _inside_any(p: np.ndarray, polygons: list[np.ndarray]) -> bool:
    return any(_point_in_poly(p, poly) for poly in polygons)


def _point_in_poly(p: np.ndarray, poly: np.ndarray) -> bool:
    x, y = float(p[0]), float(p[1])
    n = len(poly)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y) and x < (xj - xi) * (y - yi) / (yj - yi) + xi:
            inside = not inside
        j = i
    return inside
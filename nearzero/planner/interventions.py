"""Counterfactual interventions: physics-based application to agent kinematics.

Each intervention transforms the affected agent's velocity profile. Deceleration
is integrated over the prediction horizon so the resulting trajectory is
physically plausible (reaction time + ramp), which keeps risk estimates honest.
"""
from __future__ import annotations

import numpy as np


def _avg_velocity_factor(v0: float, reaction_s: float, horizon_s: float, cadence: float, brake: float) -> float:
    """Mean speed factor produced by braking at cadence after a reaction delay."""
    speed = max(abs(v0), 1e-6)
    if horizon_s <= 1e-9:
        return 1.0
    t_stop = speed / max(cadence, 1e-6)
    acc = 0.0
    n = 100
    dt = horizon_s / n
    for i in range(n):
        t = (i + 0.5) * dt
        if t < reaction_s:
            v = speed
        else:
            tau = t - reaction_s
            v = max(0.0, speed - cadence * min(tau, t_stop))
        acc += min(v, speed)
    return float(acc / (n * speed))


def apply_intervention(
    intervention: str,
    subject_vel: np.ndarray,
    target_vel: np.ndarray,
    reaction_s: float,
    stop_decel: float,
    lateral_shift_m: float,
    horizon_s: float,
    dodge_time_s: float = 1.2,
):
    """Return (subject_vel, target_vel) after applying the intervention."""
    s = subject_vel.copy()
    t = target_vel.copy()
    if intervention == "NO_ACTION":
        return s, t
    if intervention == "STOP_VEHICLE":
        f = _avg_velocity_factor(np.linalg.norm(t), reaction_s, horizon_s, stop_decel, stop_decel)
        return s, t * f
    if intervention == "SLOW_VEHICLE_50":
        return s, t * 0.5
    if intervention == "SLOW_VEHICLE_30":
        return s, t * 0.3
    if intervention == "HUMAN_STOP":
        f = _avg_velocity_factor(np.linalg.norm(s), reaction_s, horizon_s, 1.5, 1.5)
        return s * f, t
    if intervention in ("HUMAN_LEFT", "HUMAN_RIGHT"):
        speed = np.linalg.norm(s)
        lat = np.array([-s[1], s[0]]) if speed > 1e-4 else np.array([0.0, 1.0])
        if speed < 1e-4:
            lat = np.array([-1.0, 0.0])
        if intervention == "HUMAN_RIGHT":
            lat = -lat
        dodge = lateral_shift_m / max(dodge_time_s, 1e-6)
        return s + lat * dodge, t
    if intervention in ("VEHICLE_LEFT", "VEHICLE_RIGHT"):
        speed = np.linalg.norm(t)
        lat = np.array([-t[1], t[0]]) if speed > 1e-4 else np.array([0.0, 1.0])
        if speed < 1e-4:
            lat = np.array([-1.0, 0.0])
        if intervention == "VEHICLE_LEFT":
            lat = -lat
        dodge = lateral_shift_m / max(dodge_time_s, 1e-6)
        return s, t + lat * dodge
    return s, t
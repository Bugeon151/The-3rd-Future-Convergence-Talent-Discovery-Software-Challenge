"""Tests for probabilistic risk estimation and scoring."""
from __future__ import annotations

import numpy as np
import pytest

from nearzero.prediction.forecast import EnsembleForecaster, Trajectory
from nearzero.risk import collision as col
from nearzero.risk.engine import RiskEngine, TrackContext
from nearzero.risk.scoring import combine_components, component_collision, component_pmd, component_ttc
from nearzero.risk.zones import circle_zone, rectangle_zone

H = 4.0
DT = 0.1
N = 60


def _traj(pos, vel, n=N, seed=0):
    fc = EnsembleForecaster(H, DT, n)
    rng = np.random.default_rng(seed)
    ts = np.arange(int(H / DT) + 1) * DT
    base = np.asarray(pos) + np.outer(ts, np.asarray(vel))
    samples = np.stack([base + rng.normal(0, 0.05, base.shape) for _ in range(n)])
    return Trajectory(0.0, ts, base, samples)


def test_analytic_ttc_headon():
    a = (0.0, 0.0)
    b = (4.0, 0.0)
    va = (2.0, 0.0)
    vb = (-1.0, 0.0)
    ttc = col.analytic_ttc(a, va, b, vb, r_sum=1.0)
    assert ttc is not None
    assert abs(ttc - 1.0) < 1e-6


def test_analytic_ttc_diverging_is_none():
    a = (0.0, 0.0)
    b = (4.0, 0.0)
    assert col.analytic_ttc(a, (-0.5, 0.0), b, (1.0, 0.0), r_sum=1.0) is None


def test_collision_probability_deterministic():
    a = _traj((0.0, 0.0), (1.0, 0.0))
    b = _traj((2.0, 0.0), (-1.0, 0.0))
    far = _traj((0.0, 5.0), (1.0, 0.0))
    p_collide = col.collision_probability(a, b, 0.95)
    p_far = col.collision_probability(a, far, 0.95)
    assert p_collide > 0.5
    assert p_far < 0.05


def test_ttc_from_profiles_present_and_absent():
    a = _traj((0.0, 0.0), (1.0, 0.0))
    b = _traj((2.5, 0.0), (-1.0, 0.0))
    assert col.ttc_from_profiles(a, b, 0.95) is not None
    c = _traj((0.0, 4.0), (0.0, 0.0))
    assert col.ttc_from_profiles(a, c, 0.95) is None


def test_component_monotonicity():
    assert component_collision(0.9) > component_collision(0.1)
    assert component_ttc(0.5, 2.5, 1.4) > component_ttc(5.0, 2.5, 1.4)
    assert component_pmd(0.3, 1.5, 3.0) > component_pmd(2.0, 1.5, 3.0)


def test_combined_risk_bounds():
    weights = {"collision_probability": 1.0, "ttc": 0.9, "predicted_min_distance": 0.7,
               "zone_intrusion": 0.6, "rule_violation": 0.5}
    comps_high = {"collision_probability": 0.9, "ttc": 0.9, "predicted_min_distance": 0.9,
                  "zone_intrusion": 0.0, "rule_violation": 0.0}
    comps_low = {"collision_probability": 0.0, "ttc": 0.0, "predicted_min_distance": 0.0,
                 "zone_intrusion": 0.0, "rule_violation": 0.0}
    assert 0.0 <= combine_components(comps_high, weights) <= 100.0
    assert combine_components(comps_low, weights) < 5.0


def test_point_in_polygon_and_zone():
    zone = circle_zone("test", 5.0, 5.0, 1.0)
    assert zone.active
    assert col.zone_intrusion_probability(
        _traj((5.0, 5.0), (0.0, 0.0)), [zone.polygon]) > 0.9
    far = _traj((0.0, 0.0), (0.0, 0.0))
    assert col.zone_intrusion_probability(far, [zone.polygon]) == 0.0
    rect = rectangle_zone("r", 0.0, 0.0, 4.0, 4.0)
    inside = _traj((2.0, 2.0), (0.0, 0.0))
    assert col.zone_intrusion_probability(inside, [rect.polygon]) > 0.9


def test_risk_engine_pair_ordering_and_confidence():
    cfg = __import__("nearzero.config", fromlist=["Config"]).Config()
    eng = RiskEngine(cfg.risk)
    a = TrackContext(1, "person", np.array([0.0, 0.0]), np.array([1.0, 0.0]), 0.35,
                     _traj((0.0, 0.0), (1.0, 0.0)), hits=10)
    b = TrackContext(2, "vehicle", np.array([3.0, 0.0]), np.array([-1.0, 0.0]), 0.6,
                     _traj((3.0, 0.0), (-1.0, 0.0)), hits=10)
    pr = eng._pair_risk(a, b)
    assert pr.subject_id == 1 and pr.target_id == 2
    assert 0.0 < pr.confidence <= 1.0
    assert pr.ttc is not None
"""Tests for the counterfactual planner and interventions."""
from __future__ import annotations

import numpy as np

from nearzero.config import Config
from nearzero.planner.counterfactual import CounterfactualPlanner
from nearzero.planner.interventions import apply_intervention
from nearzero.risk.engine import RiskEngine, TrackContext


def _ctx(cfg, tid, cls, pos, vel, radius, hits=12):
    from nearzero.prediction.forecast import EnsembleForecaster

    fc = EnsembleForecaster(cfg.prediction.horizon_s, cfg.prediction.dt_s, cfg.prediction.n_samples)
    traj = fc.forecast(np.asarray(pos, dtype=float), np.asarray(vel, dtype=float),
                       np.eye(4) * 0.05, 0.0)
    return TrackContext(tid, cls, np.asarray(pos, dtype=float), np.asarray(vel, dtype=float),
                        radius, traj, hits=hits)


def test_intervention_velocity_unchanged_for_no_action():
    s = np.array([1.0, 0.0])
    t = np.array([-1.0, 0.0])
    cfg = Config()
    sv, tv = apply_intervention("NO_ACTION", s, t, cfg.planner.reaction_s,
                                cfg.planner.stop_decel, cfg.planner.lateral_shift_m,
                                4.0, cfg.planner.dodge_time_s)
    assert np.allclose(sv, s) and np.allclose(tv, t)


def test_stop_reduces_vehicle_speed():
    cfg = Config()
    sv, tv = apply_intervention("STOP_VEHICLE", np.array([1.0, 0.0]), np.array([-2.0, 0.0]),
                                cfg.planner.reaction_s, cfg.planner.stop_decel,
                                cfg.planner.lateral_shift_m, 4.0, cfg.planner.dodge_time_s)
    assert -2.0 < tv[0] <= 0.0
    assert abs(np.linalg.norm(tv)) < 2.0
    assert sv[0] == 1.0


def test_lateral_dodge_changes_lateral_velocity():
    cfg = Config()
    sv, _ = apply_intervention("HUMAN_LEFT", np.array([1.0, 0.0]), np.array([-2.0, 0.0]),
                               cfg.planner.reaction_s, cfg.planner.stop_decel,
                               cfg.planner.lateral_shift_m, 4.0, cfg.planner.dodge_time_s)
    assert sv[1] > 0.0
    sr, _ = apply_intervention("HUMAN_RIGHT", np.array([1.0, 0.0]), np.array([-2.0, 0.0]),
                               cfg.planner.reaction_s, cfg.planner.stop_decel,
                               cfg.planner.lateral_shift_m, 4.0, cfg.planner.dodge_time_s)
    assert sr[1] < 0.0


def test_planner_finds_effective_intervention_head_on():
    cfg = Config()
    eng = RiskEngine(cfg.risk)
    fc_kind = _ctx  # placeholder to force shape
    a = _ctx(cfg, 1, "person", [0.0, 5.0], [0.9, 0.0], 0.35)
    b = _ctx(cfg, 2, "vehicle", [4.0, 5.0], [-1.9, 0.0], 0.6)
    planner = CounterfactualPlanner(cfg.planner, _forecaster(cfg), eng)
    pr = eng._pair_risk(a, b)
    rows = planner.plan(a, b, pr.score)
    best = planner.best(rows)
    assert rows
    assert best.intervention != "NO_ACTION" or pr.score < cfg.risk.warn_risk
    no_action = next(r for r in rows if r.intervention == "NO_ACTION")
    assert no_action.resulting_risk <= pr.score + 1e-6


def _forecaster(cfg):
    from nearzero.prediction.forecast import EnsembleForecaster

    return EnsembleForecaster(cfg.prediction.horizon_s, cfg.prediction.dt_s, cfg.prediction.n_samples)
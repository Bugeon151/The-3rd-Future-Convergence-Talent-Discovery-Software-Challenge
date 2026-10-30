"""Tests for evaluation metrics and baseline policies."""
from __future__ import annotations

from nearzero.config import Config
from nearzero.eval.metrics import Decision, alarm_episodes, aggregate_summaries, summarize_policy
from nearzero.eval.policies import DistanceThresholdPolicy, NearZeroPolicy, TTCPolicy


def test_alarm_episodes():
    decs = [
        Decision(0.0, False),
        Decision(0.1, True),
        Decision(0.2, True),
        Decision(0.3, False),
        Decision(0.4, True),
    ]
    eps = alarm_episodes(decs)
    assert eps == [(0.1, 0.3), (0.4, 0.4)]


def test_summarize_and_aggregate():
    decs = [
        Decision(0.0, False),
        Decision(1.0, True, recommended="HUMAN_LEFT", risk=50, resulting_risk=10),
        Decision(2.0, True, recommended="HUMAN_LEFT", risk=50, resulting_risk=10),
        Decision(3.0, False),
    ]
    gt = [{"t0": 2.0, "t1": 2.5, "subject": "P0", "target": "V0"}]
    s = summarize_policy("X", decs, gt, duration_s=4.0)
    assert s["recall"] == 1.0
    assert s["n_recommendations"] == 2
    assert s["intervention_success_rate"] == 1.0
    assert aggregate_summaries([s])["false_alarms_total"] == 0


def test_distance_policy_alarms_when_close():
    states = [
        {"t": 0.0, "agents": [
            {"name": "P0", "cls": "person", "x": 0.0, "y": 0.0},
            {"name": "V0", "cls": "vehicle", "x": 10.0, "y": 0.0},
        ]},
        {"t": 1.0, "agents": [
            {"name": "P0", "cls": "person", "x": 0.0, "y": 0.0},
            {"name": "V0", "cls": "vehicle", "x": 1.0, "y": 0.0},
        ]},
    ]
    decs = DistanceThresholdPolicy(2.0).decisions(states)
    assert decs[0].alarm is False
    assert decs[1].alarm is True


def test_ttc_policy_needs_velocities():
    states = [
        {"t": 0.0, "agents": [
            {"name": "P0", "cls": "person", "x": 0.0, "y": 0.0, "vx": 1.0, "vy": 0.0},
            {"name": "V0", "cls": "vehicle", "x": 3.0, "y": 0.0, "vx": -1.0, "vy": 0.0},
        ]},
    ]
    decs = TTCPolicy(3.0).decisions(states)
    assert decs[0].alarm is True


def test_nearzero_policy_runs(cfg=None):
    from nearzero.eval.compare import _pipeline

    cfg = cfg or Config()
    pipe, _ = _pipeline(cfg, enable_planning=True, enable_verification=False)
    from nearzero.sim.scenarios import make_scenario

    _, states, _ = make_scenario("head_on")
    decs = NearZeroPolicy(cfg, pipe).decisions(states[:25])
    assert len(decs) == 25
    assert any(d.risk is not None for d in decs)
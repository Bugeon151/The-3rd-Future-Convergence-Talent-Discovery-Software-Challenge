"""Tests for the simulator and end-to-end pipeline integration."""
from __future__ import annotations

import numpy as np

from nearzero.sim.scenarios import SCENARIO_NAMES, make_scenario
from nearzero.sim.world import Agent, World

PERSON_R = 0.35
CART_R = 0.60


def test_scenarios_deterministic():
    for name in SCENARIO_NAMES:
        _, s1, e1 = make_scenario(name)
        _, s2, e2 = make_scenario(name)
        assert s1 == s2
        assert e1 == e2
        assert len(s1) > 20


def test_scenario_head_on_records_collision():
    _, states, events = make_scenario("head_on")
    assert len(events) == 1
    ev = events[0]
    assert "t0" in ev and "t1" in ev


def test_world_person_radius_constants():
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 0.0, 0.0, 1.0, 0.0, PERSON_R))
    assert len(w.agents) == 1


def test_pipeline_demo_avoided_resolves():
    from nearzero.app.pipeline import SafetyPipeline
    from nearzero.config import Config
    from nearzero.events.store import EventStore
    from nearzero.sim.scenarios import make_scenario

    cfg = Config()
    store = EventStore(":memory:")
    pipe = SafetyPipeline(cfg, store=store, enable_planning=True, enable_verification=True)
    _, states, _ = make_scenario("demo_avoided")
    risks = []
    for st in states[:40]:
        objs = _objects(st)
        snap = pipe.step(None, objs, float(st["t"]), cfg.prediction.dt_s)
        risks.append(snap.risk.score)
    store.close()
    assert max(risks) > 40
    assert risks[-1] < 20


def _objects(st):
    return [
        type("O", (), dict(cls=a["cls"], conf=0.99, x=a["x"], y=a["y"], t=float(st["t"]), box=None))()
        for a in st["agents"]
        if a["cls"] in ("person", "vehicle")
    ]


def test_snapshot_state_json_safe():
    from nearzero.app.pipeline import SafetyPipeline
    from nearzero.app.state import snapshot_to_state
    from nearzero.config import Config
    from nearzero.sim.scenarios import make_scenario

    import json

    cfg = Config()
    pipe = SafetyPipeline(cfg)
    _, states, _ = make_scenario("crossing")
    snaps = []
    for st in states[:21]:
        snap = pipe.step(None, _objects(st), float(st["t"]), cfg.prediction.dt_s)
        snaps.append(snap)
        json.dumps(snapshot_to_state(snap))  # must not raise
    assert max(s.risk.score for s in snaps) >= 0
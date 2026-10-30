"""Tests for verification lifecycle and the event store."""
from __future__ import annotations

import os

import numpy as np

from nearzero.risk.engine import PairRisk
from nearzero.verify.monitor import (
    VerificationMonitor,
    STATUS_PREDICTED,
    STATUS_WARNING,
    STATUS_RECOMMENDED,
)


def _risk(score, ttc=5.0):
    return PairRisk(subject_id=1, target_id=2, collision_probability=0.5, ttc=ttc,
                    predicted_min_distance=1.0, collision_point=None, score=score,
                    components={}, confidence=0.9)


def test_monitor_lifecycle_prevented(tdir):
    from nearzero.events.store import EventStore

    store = EventStore(os.path.join(tdir, "e.db"))
    mon = VerificationMonitor(warn_risk=45.0, intervention_risk=50.0, resolve_drop=0.5,
                              verify_window_s=3.0, store=store)
    key = (1, 2)

    # open
    trans = mon.update(0.0, {key: _risk(60.0)}, None, "", {1, 2})
    assert any(x["to"] == STATUS_PREDICTED for x in trans) or mon.open_events

    # recommend
    trans = mon.update(0.5, {key: _risk(62.0)}, key, "HUMAN_LEFT", {1, 2})
    assert any(x["to"] == STATUS_RECOMMENDED for x in trans)
    assert mon.open_events[0].recommended == "HUMAN_LEFT"

    # resolve (risk crashes then stays low for 5 frames)
    for k in range(6):
        mon.update(0.6 + 0.1 * k, {key: _risk(5.0)}, key, "HUMAN_LEFT", {1, 2})
    msgs = [e["status"] for e in store.query(status="PREVENTED")]
    assert msgs, "expected a PREVENTED event in store"
    store.close()


def test_monitor_closes_as_false_alarm_when_track_lost():
    mon = VerificationMonitor(warn_risk=45.0, intervention_risk=50.0)
    key = (1, 2)
    mon.update(0.0, {key: _risk(60.0)}, None, "", {1, 2})
    for k in range(5):
        mon.update(0.1 * (k + 1), {}, None, "", set())
    assert not mon.open_events
    assert len(mon.recent_history) == 1
    assert mon.state_dict()["last_event"]["state"] == "PREDICTED"


def test_event_store_roundtrip(tdir):
    from nearzero.events.store import EventStore

    db = os.path.join(tdir, "e2.db")
    store = EventStore(db)
    eid = store.add_event("PREDICTED", "collision_risk", 1, 2, 55.0, 1.0,
                          details={"ttc": 1.2})
    store.update_event(eid, status="PREVENTED", risk_final=5.0, recommended="HUMAN_LEFT")
    rows = store.query()
    assert rows and rows[0]["status"] == "PREVENTED"
    assert rows[0]["recommended"] == "HUMAN_LEFT"
    assert store.summary()["PREVENTED"] == 1
    store.clear()
    assert store.summary() == {}
    store.close()
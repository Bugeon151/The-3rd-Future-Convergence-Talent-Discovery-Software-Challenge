"""Tests for tracking (Kalman filter, track manager, object association)."""
from __future__ import annotations

import numpy as np

from nearzero.config import Config
from nearzero.core.kalman import CVKalmanFilter
from nearzero.core.tracks import Object, TrackManager


def _obj(cls, x, y, t=0.0, conf=0.9):
    return Object(cls=cls, conf=conf, x=x, y=y, t=t)


def test_kalman_velocity_estimate():
    kf = CVKalmanFilter(0.0, 0.0)
    for k in range(30):
        kf.predict(0.1)
        kf.update(0.5 * k, 0.0)
    _, _, v = kf.get()
    assert 4.0 < v[0] < 6.0
    assert abs(v[1]) < 0.5


def test_kalman_static_velocity_estimate():
    times = np.arange(10, dtype=float) * 0.1
    pos = np.stack([0.5 * times, np.zeros_like(times)], axis=1)
    vx, vy = CVKalmanFilter.estimate_velocity(times, pos)
    assert abs(vx - 0.5) < 1e-6 and abs(vy) < 1e-6


def test_track_manager_association_stable():
    cfg = Config()
    tm = TrackManager(gating_m=cfg.perception.gating_meters,
                      max_age=cfg.perception.max_age_frames,
                      min_hits=cfg.perception.min_hits)
    tm.step([_obj("person", 1.0, 1.0, t=0.0)], t_now=0.0, dt=0.1)
    tm.step([_obj("person", 1.1, 1.0, t=0.1)], t_now=0.1, dt=0.1)
    tm.step([_obj("person", 1.2, 1.0, t=0.2)], t_now=0.2, dt=0.1)
    assert len(tm.tracks) == 1
    t = tm.tracks[0]
    assert t.cls == "person"
    assert len(t.history) == 3


def test_track_manager_death_and_reassignment():
    cfg = Config()
    tm = TrackManager(gating_m=cfg.perception.gating_meters,
                      max_age=cfg.perception.max_age_frames,
                      min_hits=cfg.perception.min_hits)
    tm.step([_obj("person", 1.0, 1.0, t=0.0)], t_now=0.0, dt=0.1)
    tm.step([_obj("person", 1.1, 1.0, t=0.1)], t_now=0.1, dt=0.1)
    old_id = tm.tracks[0].track_id
    for k in range(cfg.perception.max_age_frames + 2):
        tm.step([], t_now=0.1 * (2 + k), dt=0.1)
    assert all(tr.track_id != old_id for tr in tm.tracks) or len(tm.tracks) == 0
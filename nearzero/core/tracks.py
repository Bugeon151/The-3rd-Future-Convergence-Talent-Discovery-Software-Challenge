"""Detection objects, tracklets, and the multi-object tracker.

Tracking is done in world coordinates (ground plane), which is robust against
pixel-space ID switches and keeps trajectory history stable for risk math.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .kalman import CVKalmanFilter, ema_velocity


@dataclass
class Object:
    """A single detection in one frame."""

    cls: str
    conf: float
    x: float
    y: float
    box: Optional[tuple[float, float, float, float]] = None
    t: float = 0.0


@dataclass
class Track:
    track_id: int
    cls: str
    kalman: CVKalmanFilter
    history: list[tuple[float, float, float]] = field(default_factory=list)
    last_seen: float = 0.0
    hits: int = 0
    missed: int = 0
    active: bool = True
    box: Optional[tuple[float, float, float, float]] = None
    conf: float = 0.0

    @property
    def position(self) -> tuple[float, float]:
        _, p, _ = self.kalman.get()
        return float(p[0]), float(p[1])

    @property
    def velocity(self) -> tuple[float, float]:
        _, _, v = self.kalman.get()
        return float(v[0]), float(v[1])

    def speed(self) -> float:
        vx, vy = self.velocity
        return float(np.hypot(vx, vy))

    def age(self, t_now: float) -> float:
        return float(t_now - self.last_seen)

    def append(self, t: float, x: float, y: float) -> None:
        self.history.append((float(t), float(x), float(y)))
        if len(self.history) > 120:
            self.history.pop(0)

    def update(self, obj: Object, dt: float = 0.1) -> None:
        self.cls = obj.cls
        self.conf = obj.conf
        self.box = obj.box
        self.last_seen = obj.t
        self.hits += 1
        self.missed = 0
        if self.history:
            dt = max(min(dt, obj.t - self.history[-1][0]), 1e-3) if obj.t > self.history[-1][0] else dt
        self.kalman.predict(dt)
        self.kalman.update(obj.x, obj.y)
        self.append(obj.t, obj.x, obj.y)

    def mark_missed(self, t_now: float, dt: float) -> None:
        self.missed += 1
        self.last_seen = t_now
        self.kalman.predict(dt)
        px, py = self.position
        self.append(t_now, px, py)

    def smooth_velocity(self) -> tuple[float, float]:
        if len(self.history) >= 3:
            vx, vy = ema_velocity(self.history)
            return vx, vy
        return self.velocity


class TrackManager:
    def __init__(self, gating_m: float = 2.5, max_age: int = 25, min_hits: int = 3):
        self._tracks: dict[int, Track] = {}
        self._next_id = 1
        self.gating_m = gating_m
        self.max_age = max_age
        self.min_hits = min_hits

    def step_forced(self, objects: list[Object], ext_ids: list[int], t_now: float, dt: float) -> list[Track]:
        """Association variant for external trackers (e.g. ByteTrack) whose IDs
        we trust for continuity. New external ids map onto our own id space."""
        if not hasattr(self, "_ext_map"):
            self._ext_map: dict[int, int] = {}
        alive = {tr.track_id: tr for tr in self._tracks.values() if tr.active}
        used = set()
        for obj, ext in zip(objects, ext_ids):
            my_id = self._ext_map.get(ext)
            if my_id is not None and my_id in alive and alive[my_id].cls == obj.cls:
                alive[my_id].update(obj, dt)
                used.add(my_id)
            else:
                tr = self._spawn(obj)
                self._ext_map[ext] = tr.track_id
                used.add(tr.track_id)
        for tr_id, tr in alive.items():
            if tr_id not in used:
                tr.mark_missed(t_now, dt)
                if tr.missed > self.max_age:
                    tr.active = False
        self._tracks = {t.track_id: t for t in self._tracks.values() if t.active}
        return sorted(
            [t for t in self._tracks.values() if t.active and t.hits >= self.min_hits],
            key=lambda t: t.track_id,
        )

    def step(self, objects: list[Object], t_now: float, dt: float) -> list[Track]:
        active = [t for t in self._tracks.values() if t.active]
        assigned = set()
        used = set()

        order = sorted(objects, key=lambda o: -o.conf)
        for obj in order:
            best_id, best_d = None, None
            for tr in active:
                if tr.track_id in used:
                    continue
                if tr.cls != obj.cls:
                    continue
                px, py = tr.position
                d = float(np.hypot(px - obj.x, py - obj.y))
                box_cost = 0.0
                if tr.box is not None and obj.box is not None:
                    bc = self._box_iou(tr.box, obj.box)
                    box_cost = 1.0 - bc
                    d = 0.6 * d + 0.4 * box_cost * self.gating_m
                if d < self.gating_m and (best_d is None or d < best_d):
                    best_d = d
                    best_id = tr.track_id
            if best_id is not None:
                self._tracks[best_id].update(obj, dt)
                assigned.add(best_id)
                used.add(best_id)
            else:
                tr = self._spawn(obj)
                assigned.add(tr.track_id)

        for tr in active:
            if tr.track_id not in assigned:
                tr.mark_missed(t_now, dt)
                if tr.missed > self.max_age:
                    tr.active = False

        alive = sorted(
            [t for t in self._tracks.values() if t.active and t.hits >= self.min_hits],
            key=lambda t: t.track_id,
        )
        self._tracks = {t.track_id: t for t in self._tracks.values() if t.active}
        return alive

    def _spawn(self, obj: Object) -> Track:
        tr = Track(track_id=self._next_id, cls=obj.cls, kalman=CVKalmanFilter(obj.x, obj.y))
        self._next_id += 1
        tr.update(obj)
        self._tracks[tr.track_id] = tr
        return tr

    @staticmethod
    def _box_iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        return inter / union if union > 0 else 0.0

    @property
    def tracks(self) -> list[Track]:
        return sorted(
            (t for t in self._tracks.values() if t.active), key=lambda t: t.track_id
        )
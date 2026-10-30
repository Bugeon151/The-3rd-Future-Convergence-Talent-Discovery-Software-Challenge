"""Deterministic scenario suite for the 2D simulator.

Each scenario returns (world, states, ground_truth_events) where
ground_truth_events are dicts {t0, t1, subject, target} (contact intervals).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..risk.zones import RestrictedZone, circle_zone
from .world import Agent, World, const_velocity, momentum, abrupt_turn, steer_to

PERSON_R = 0.35
CART_R = 0.60


@dataclass
class Scenario:
    name: str
    duration_s: float
    world: World
    description: str = ""
    interval: list = field(default_factory=lambda: [0.1, 0.1])

    def run(self) -> list[dict]:
        steps = max(1, int(self.duration_s / self.world.dt))
        return self.world.grow(steps)


@dataclass
class ScenarioResult:
    name: str
    states: list[dict]
    collisions: list[dict]
    interval: float

    def agent_trace(self, name: str) -> list[tuple[float, float, float]]:
        out = []
        for st in self.states:
            for a in st["agents"]:
                if a["name"] == name:
                    out.append((st["t"], a["x"], a["y"]))
                    break
        return out


def _contact_events(states: list[dict], r_sum: float) -> list[dict]:
    events = []
    start = None
    prev_t = 0.0
    for st in states:
        agents = {a["name"]: a for a in st["agents"]}
        names = list(agents)
        close = False
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = agents[names[i]], agents[names[j]]
                d = np.hypot(a["x"] - b["x"], a["y"] - b["y"])
                if d < r_sum:
                    close = True
        if close and start is None:
            start = st["t"]
        if not close and start is not None:
            events.append({"t0": start, "t1": st["t"], "subject": names[0], "target": names[1]})
            start = None
        prev_t = st["t"]
    if start is not None:
        events.append({"t0": start, "t1": prev_t, "subject": names[0], "target": names[1]})
    return events


def make_scenario(name: str, seed: int = 2026) -> tuple[World, list[dict], list[dict]]:
    if name in _FACTORY:
        return _FACTORY[name](seed)
    raise KeyError(f"unknown scenario: {name}")


SCENARIO_NAMES = ("head_on", "crossing", "parallel", "safe_pass", "sudden_stop",
                  "sudden_turn", "zone_intrusion", "demo_avoided")


def _head_on(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 1.0, 5.0, 0.9, 0.0, PERSON_R))
    w.add(Agent("V0", "vehicle", 9.0, 5.0, -1.8, 0.0, CART_R))
    states = w.grow(70)
    return _wrap("head_on", w, 0.1, states)


def _crossing(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 1.0, 5.0, 1.5, 0.0, PERSON_R))
    w.add(Agent("V0", "vehicle", 5.0, 8.0, 0.0, -1.125, CART_R))
    states = w.grow(70)
    return _wrap("crossing", w, 0.1, states)


def _parallel(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 1.0, 6.0, 1.0, 0.0, PERSON_R))
    w.add(Agent("V0", "vehicle", 0.5, 4.0, 1.2, 0.0, CART_R))
    states = w.grow(70)
    return _wrap("parallel", w, 0.1, states)


def _safe_pass(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 4.0, 5.0, 0.8, 0.0, PERSON_R))
    w.add(Agent("V0", "vehicle", 2.0, 6.4, 2.4, 0.0, CART_R))
    states = w.grow(60)
    return _wrap("safe_pass", w, 0.1, states)


def _sudden_stop(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 1.0, 5.0, 0.9, 0.0, PERSON_R))
    w.add(Agent("V0", "vehicle", 9.0, 5.0, -1.8, 0.0, CART_R,
                controller=momentum(3.6, -1.8, 0.0, decay=3.0)))
    states = w.grow(70)
    return _wrap("sudden_stop", w, 0.1, states)


def _sudden_turn(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 1.0, 5.0, 0.9, 0.0, PERSON_R,
                controller=abrupt_turn(2.0, dvy_delta=2.0)))
    w.add(Agent("V0", "vehicle", 9.0, 5.0, -1.8, 0.0, CART_R))
    states = w.grow(70)
    return _wrap("sudden_turn", w, 0.1, states)


def _zone_intrusion(_seed):
    w = World(dt=0.1)
    w.add(Agent("V0", "vehicle", 6.0, 5.0, 0.0, 0.0, CART_R))
    w.add(Agent("P0", "person", 1.0, 5.0, 1.1, 0.0, PERSON_R))
    w.zones.append(circle_zone("cart-guard", 6.0, 5.0, 1.4))
    states = w.grow(70)
    return _wrap("zone_intrusion", w, 0.1, states)


def _demo_avoided(_seed):
    w = World(dt=0.1)
    w.add(Agent("P0", "person", 1.0, 5.0, 0.9, 0.0, PERSON_R))
    w.add(Agent("V0", "vehicle", 9.5, 5.0, -1.9, 0.0, CART_R))
    states = w.grow(90)
    return _wrap("demo_avoided", w, 0.1, states)


def apply_vehicle_dodge(states: list[dict], idx: int, dt: float, lateral_speed: float = 2.2,
                        ramp_s: float = 0.3, max_offset_m: float = 1.4) -> None:
    """Execute a committed evasive lane shift on the world timeline (deterministic).

    The vehicle ramps sideways and holds the new offset for the rest of the
    episode, so the executed maneuver stays clear of the pedestrian.
    """
    if idx + 1 >= len(states):
        return

    def _veh(s):
        for a in s["agents"]:
            if a["name"] == "V0":
                return a
        raise KeyError("V0")

    vx = float(_veh(states[idx])["vx"])
    y0 = float(_veh(states[idx])["y"])
    offset = 0.0
    for j in range(1, len(states) - idx):
        k = idx + j
        if k >= len(states):
            break
        t_rel = j * dt
        prev = _veh(states[k - 1])
        cur = _veh(states[k])
        veh_x = float(prev["x"]) + vx * dt
        if offset >= max_offset_m:
            vy = 0.0
        else:
            vy = lateral_speed * min(1.0, t_rel / max(ramp_s, 1e-9))
        cur["x"] = veh_x
        cur["y"] = prev["y"] + vy * dt
        cur["vx"] = vx
        cur["vy"] = vy
        offset = max(offset, abs(cur["y"] - y0))


_FACTORY = {
    "head_on": _head_on,
    "crossing": _crossing,
    "parallel": _parallel,
    "safe_pass": _safe_pass,
    "sudden_stop": _sudden_stop,
    "sudden_turn": _sudden_turn,
    "zone_intrusion": _zone_intrusion,
    "demo_avoided": _demo_avoided,
}


def _wrap(name, world, dt, states):
    agents = world.agents
    r_sum = agents[0].radius + agents[1].radius
    collisions = _contact_events(states, r_sum)
    return world, states, collisions
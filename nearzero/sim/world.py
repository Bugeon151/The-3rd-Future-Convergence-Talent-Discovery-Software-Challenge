"""2D ground-plane simulator used for offline evaluation and deterministic demos.

Agents are circles in a world measured in meters. Controllers update velocity
each step, allowing scripted behaviors (braking, turning, dodging).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

Controller = Callable[["Agent", float], tuple[float, float]]


@dataclass
class Agent:
    name: str
    cls: str
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = 0.35
    controller: Controller | None = None
    color: tuple[int, int, int] = (255, 166, 46)

    def step(self, t: float, dt: float) -> None:
        if self.controller is not None:
            self.vx, self.vy = self.controller(self, t)
        self.x += self.vx * dt
        self.y += self.vy * dt

    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "cls": self.cls,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "radius": self.radius,
        }


@dataclass
class World:
    t: float = 0.0
    dt: float = 0.1
    agents: list[Agent] = field(default_factory=list)
    zones: list = field(default_factory=list)

    def add(self, agent: Agent) -> None:
        self.agents.append(agent)

    def state(self) -> dict:
        return {"t": self.t, "agents": [a.as_dict() for a in self.agents]}

    def step(self) -> None:
        self.t += self.dt
        for a in self.agents:
            a.step(self.t, self.dt)

    def grow(self, steps: int) -> list[dict]:
        states = []
        for _ in range(steps):
            self.step()
            states.append(self.state())
        return states


def const_velocity(sx: float, sy: float) -> Controller:
    def _ctrl(agent: Agent, t: float) -> tuple[float, float]:
        return sx, sy

    return _ctrl


def momentum(duration_s: float, sx: float, sy: float, decay: float = 2.0) -> Controller:
    def _ctrl(agent: Agent, t: float) -> tuple[float, float]:
        if t < duration_s:
            return sx, sy
        f = max(0.0, 1.0 - (t - duration_s) * decay)
        return sx * f, sy * f

    return _ctrl


def steer_to(t_when: float, heading: float, speed: float, rate: float = 1.5) -> Controller:
    def _ctrl(agent: Agent, t: float) -> tuple[float, float]:
        if t < t_when:
            return agent.vx, agent.vy
        cur = math.atan2(agent.vy, agent.vx)
        diff = (heading - cur + np.pi) % (2 * np.pi) - np.pi
        new = cur + np.clip(diff, -rate * 0.1, rate * 0.1)
        return speed * math.cos(new), speed * math.sin(new)

    return _ctrl


def abrupt_turn(t_when: float, dvy_delta: float = 0.0, dvx_delta: float = 0.0,
                factor: float = 1.0) -> Controller:
    """Lateral dodge impulse applied at t_when (adds velocity, scales existing)."""
    def _ctrl(agent: Agent, t: float) -> tuple[float, float]:
        if t < t_when:
            return agent.vx, agent.vy
        return agent.vx * factor + dvx_delta, agent.vy * factor + dvy_delta

    return _ctrl
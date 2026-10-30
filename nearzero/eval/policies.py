"""Baseline safety policies (A/B/C) and the NEAR-ZERO X policy (D).

All operate on the same simulator state streams. A/B/C are threshold systems;
D runs the real forecasting + probabilistic risk + counterfactual + verification
pipeline, so "D better than baseline" is measured, not assumed.
"""
from __future__ import annotations

import numpy as np

from ..config import Config
from ..core.tracks import Object
from ..risk import collision as col
from ..risk.engine import RiskEngine
from .metrics import Decision

PERSON_R = 0.35
CART_R = 0.60
SAFETY_MARGIN = 0.20
R_SUM = PERSON_R + CART_R + SAFETY_MARGIN


class DistanceThresholdPolicy:
    """Baseline A: alarm when current center distance < threshold."""

    name = "A: distance_threshold"

    def __init__(self, threshold_m: float = 2.0):
        self.threshold = threshold_m

    def decisions(self, states: list[dict]) -> list[Decision]:
        out = []
        for st in states:
            agents = {a["name"]: a for a in st["agents"] if a["cls"] in ("person", "vehicle")}
            names = list(agents)
            close = False
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = agents[names[i]], agents[names[j]]
                    if np.hypot(a["x"] - b["x"], a["y"] - b["y"]) < self.threshold:
                        close = True
            out.append(Decision(t=st["t"], alarm=close, risk=None))
        return out


class TTCPolicy:
    """Baseline B: alarm when analytic time-to-collision is below threshold."""

    name = "B: ttc_threshold"

    def __init__(self, ttc_threshold_s: float = 3.0):
        self.threshold = ttc_threshold_s

    def decisions(self, states: list[dict]) -> list[Decision]:
        out = []
        for st in states:
            agents = {a["name"]: a for a in st["agents"] if a["cls"] in ("person", "vehicle")}
            alarm = False
            names = list(agents)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = agents[names[i]], agents[names[j]]
                    ttc = col.analytic_ttc(
                        (a["x"], a["y"]), (a["vx"], a["vy"]),
                        (b["x"], b["y"]), (b["vx"], b["vy"]), R_SUM,
                    )
                    if ttc is not None and ttc <= self.threshold:
                        alarm = True
            out.append(Decision(t=st["t"], alarm=alarm, risk=None))
        return out


class PredictionPolicy:
    """Baseline C: constant-velocity future extrapolation with PMD threshold."""

    name = "C: trajectory_prediction"

    def __init__(self, horizon_s: float = 4.0, pmd_threshold_m: float = R_SUM):
        self.horizon = horizon_s
        self.threshold = pmd_threshold_m
        self._times = None

    def decisions(self, states: list[dict]) -> list[Decision]:
        out = []
        prev = {}
        for st in states:
            agents = {a["name"]: a for a in st["agents"] if a["cls"] in ("person", "vehicle")}
            alarm = False
            names = list(agents)
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    a, b = agents[names[i]], agents[names[j]]
                    pa = np.array([a["x"], a["y"]])
                    pb = np.array([b["x"], b["y"]])
                    va = np.array([a["vx"], a["vy"]])
                    vb = np.array([b["vx"], b["vy"]])
                    ts = np.arange(0, self.horizon + 1e-6, 0.1)
                    if self._times is None:
                        self._times = ts
                    da = pa[None, :] + np.outer(ts, va)
                    db = pb[None, :] + np.outer(ts, vb)
                    dist = np.linalg.norm(da - db, axis=1)
                    ttc = col.analytic_ttc((a["x"], a["y"]), (a["vx"], a["vy"]),
                                           (b["x"], b["y"]), (b["vx"], b["vy"]), R_SUM)
                    if (dist.min() < self.threshold) or (ttc is not None and ttc <= self.horizon):
                        alarm = True
            out.append(Decision(t=st["t"], alarm=alarm, risk=None))
        return out


class NearZeroPolicy:
    """Baseline D: NEAR-ZERO X full pipeline (real components)."""

    name = "D: nearzero_x"

    def __init__(self, cfg: Config, pipeline):
        self.cfg = cfg
        self.pipeline = pipeline

    def decisions(self, states: list[dict], dt: float = 0.1) -> list[Decision]:
        out = []
        running_t = 0.0
        for st in states:
            t = float(st["t"])
            objects = []
            for a in st["agents"]:
                if a["cls"] in ("person", "vehicle"):
                    objects.append(
                        Object(cls=a["cls"], conf=0.99, x=a["x"], y=a["y"], t=t)
                    )
            snap = self.pipeline.step(None, objects, t, dt)
            risk = snap.risk.score
            alarm = risk >= self.cfg.risk.warn_risk
            rec = ""
            resulting = None
            if snap.best_intervention is not None and snap.best_intervention.intervention != "NO_ACTION":
                rec = snap.best_intervention.intervention
                resulting = snap.best_intervention.resulting_risk
            out.append(
                Decision(
                    t=t,
                    alarm=alarm,
                    recommended=rec,
                    risk=risk,
                    resulting_risk=resulting,
                )
            )
        return out
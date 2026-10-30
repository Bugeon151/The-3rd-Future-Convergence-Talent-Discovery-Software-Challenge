"""The NEAR-ZERO X risk engine.

Evaluates the safety of the current multi-object scene by combining:
  - collision probability (Monte-Carlo over forecast samples)
  - time-to-collision (TTC)
  - predicted minimum distance (PMD)
  - restricted-zone intrusion probability
  - natural-language rule violations

All components feed a documented weighted score (docs/RISK_MODEL.md).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..config import RiskConfig
from ..core import geometry
from ..prediction.forecast import Trajectory
from . import collision, scoring
from .zones import RestrictedZone


@dataclass
class TrackContext:
    track_id: int
    cls: str
    pos: np.ndarray
    vel: np.ndarray
    radius: float
    traj: Trajectory
    hits: int = 1
    cov: np.ndarray | None = None


@dataclass
class PairRisk:
    subject_id: int
    target_id: int
    collision_probability: float
    ttc: float | None
    predicted_min_distance: float
    collision_point: tuple[float, float] | None
    score: float
    components: dict[str, float]
    confidence: float = 1.0

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.collision_point:
            d["collision_point"] = [float(self.collision_point[0]), float(self.collision_point[1])]
        return d


@dataclass
class ScenarioRisk:
    score: float
    state: str
    confidence: float
    collision_probability: float
    ttc: float | None
    predicted_min_distance: float
    collision_point: tuple[float, float] | None
    zone_probability: float
    rule_violation: float
    components: dict[str, float] = field(default_factory=dict)
    pairs: list[PairRisk] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dataclasses.asdict(self)
        if self.collision_point:
            d["collision_point"] = [float(self.collision_point[0]), float(self.collision_point[1])]
        d["pairs"] = [p.to_dict() for p in self.pairs]
        return d


class RiskEngine:
    def __init__(self, cfg: RiskConfig):
        self.cfg = cfg

    def radius_for(self, cls: str) -> float:
        if cls == "person":
            return self.cfg.person_radius_m
        if cls.lower() in ("forklift", "machine", "cart", "vehicle"):
            return self.cfg.vehicle_radius_m
        return self.cfg.vehicle_radius_m

    def evaluate(
        self,
        contexts: list[TrackContext],
        zones: list[RestrictedZone] | None = None,
        rule_violations: list[dict] | None = None,
    ) -> ScenarioRisk:
        zones = zones or []
        zone_probs: dict[int, float] = {}
        for ctx in contexts:
            polys = [z.polygon for z in zones if z.active]
            zone_probs[ctx.track_id] = collision.zone_intrusion_probability(ctx.traj, polys)

        pair_risks: list[PairRisk] = []
        components: dict[str, float] = {
            "collision_probability": 0.0,
            "ttc": 0.0,
            "predicted_min_distance": 0.0,
            "zone_intrusion": 0.0,
            "rule_violation": 0.0,
        }

        for a in contexts:
            for b in contexts:
                if b.track_id <= a.track_id:
                    continue
                if not self._relevant_pair(a.cls, b.cls):
                    continue
                pr = self._pair_risk(a, b)
                pair_risks.append(pr)
                for k in components:
                    components[k] = max(components[k], pr.components.get(k, 0.0))

        zone_component = max(zone_probs.values(), default=0.0)
        rule_component = self._rule_component(rule_violations)
        components["zone_intrusion"] = zone_component
        components["rule_violation"] = rule_component

        score = scoring.combine_components(components, self.cfg.score_weights)
        state = scoring.classify_state(score, self.cfg.thresholds)

        worst = max(pair_risks, key=lambda p: p.score, default=None)
        confidence = float(np.clip(worst.confidence, 0.0, 1.0)) if worst else 1.0

        return ScenarioRisk(
            score=score,
            state=state,
            confidence=confidence,
            collision_probability=components["collision_probability"],
            ttc=worst.ttc if worst else None,
            predicted_min_distance=worst.predicted_min_distance if worst else None,
            collision_point=worst.collision_point if worst else None,
            zone_probability=zone_component,
            rule_violation=rule_component,
            components=components,
            pairs=pair_risks,
        )

    def _pair_risk(self, a: TrackContext, b: TrackContext) -> PairRisk:
        r_sum = a.radius + b.radius
        prob = collision.collision_probability(a.traj, b.traj, r_sum)
        pmd, _ = collision.pair_minimum_distance(a.traj, b.traj)
        ttc = collision.ttc_from_profiles(a.traj, b.traj, r_sum)

        c_point = None
        if ttc is not None:
            ia = int(np.argmin(np.abs(a.traj.times - (a.traj.t0 + ttc))))
            ib = int(np.argmin(np.abs(b.traj.times - (b.traj.t0 + ttc))))
            c_point = tuple(((a.traj.positions[ia] + b.traj.positions[ib]) / 2.0).tolist())

        comps = {
            "collision_probability": scoring.component_collision(prob),
            "ttc": scoring.component_ttc(ttc, self.cfg.ttc_x0, self.cfg.ttc_k),
            "predicted_min_distance": scoring.component_pmd(pmd, self.cfg.pmd_x0, self.cfg.pmd_k),
            "zone_intrusion": 0.0,
            "rule_violation": 0.0,
        }
        score = scoring.combine_components(comps, self.cfg.score_weights)
        return PairRisk(
            subject_id=a.track_id,
            target_id=b.track_id,
            collision_probability=prob,
            ttc=ttc,
            predicted_min_distance=pmd,
            collision_point=c_point,
            score=score,
            components=comps,
            confidence=self._pair_confidence(a, b),
        )

    @staticmethod
    def _relevant_pair(cls_a: str, cls_b: str) -> bool:
        if cls_a == cls_b:
            return False
        return True

    def _pair_confidence(self, a: TrackContext, b: TrackContext) -> float:
        ca = self._track_confidence(a)
        cb = self._track_confidence(b)
        return min(ca, cb)

    def _track_confidence(self, ctx: TrackContext) -> float:
        maturity = float(np.clip(ctx.hits / 8.0, 0.0, 1.0))
        speed = float(np.linalg.norm(ctx.vel))
        spread = 0.0
        if ctx.traj.samples is not None and ctx.traj.samples.shape[0] > 1:
            stds = ctx.traj.samples.std(axis=0).mean()
            spread = float(np.clip(stds / max(speed, 0.5), 0.0, 1.0))
        return float(np.clip(0.6 * maturity + 0.4 * (1.0 - spread), 0.05, 1.0))

    def _rule_component(self, violations: list[dict] | None) -> float:
        if not violations:
            return 0.0
        best = 0.0
        for v in violations:
            sev = float(v.get("severity_value", 0.0))
            intensity = float(v.get("intensity", 0.0))
            best = max(best, scoring.component_rule(sev, intensity))
        return best
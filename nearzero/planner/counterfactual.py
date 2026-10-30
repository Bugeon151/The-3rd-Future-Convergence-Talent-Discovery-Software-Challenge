"""Counterfactual Prevention Planner.

Simulates each candidate intervention on the critical pair, re-forecasts the
affected trajectories and re-runs the same risk engine. Ranking minimizes risk
while weighing intervention cost and feasibility, selecting the *minimum
effective intervention* that drives risk below the prevention target.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np

from ..config import PlannerConfig
from ..prediction.forecast import Forecaster
from ..risk.engine import RiskEngine, TrackContext
from .interventions import apply_intervention


@dataclass
class InterventionResult:
    intervention: str
    original_risk: float
    resulting_risk: float
    resulting_collision_probability: float
    risk_reduction: float
    feasibility: float
    intervention_cost: float
    utility: float

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


class CounterfactualPlanner:
    def __init__(self, cfg: PlannerConfig, forecaster: Forecaster, risk_engine: RiskEngine):
        self.cfg = cfg
        self.forecaster = forecaster
        self.risk_engine = risk_engine

    def plan(
        self, subject: TrackContext, target: TrackContext, current_risk: float
    ) -> list[InterventionResult]:
        results: list[InterventionResult] = []
        for name in self.cfg.interventions:
            s_vel, t_vel = apply_intervention(
                name,
                subject.vel,
                target.vel,
                self.cfg.reaction_s,
                self.cfg.stop_decel,
                self.cfg.lateral_shift_m,
                self.forecaster.horizon_s,
                self.cfg.dodge_time_s,
            )
            new_sub = self._with_velocity(subject, s_vel)
            new_tgt = self._with_velocity(target, t_vel)
            pr = self.risk_engine._pair_risk(new_sub, new_tgt)
            resulting = pr.score
            reduction = max(0.0, current_risk - resulting)
            feasibility = float(np.clip(self.cfg.feasibility_table.get(name, 0.5), 0.0, 1.0))
            cost = float(self.cfg.cost_table.get(name, 0.0))
            reduction_norm = reduction / 100.0
            achieved = resulting <= self.risk_engine.cfg.prevention_target_risk
            utility = reduction_norm * (1.5 if achieved else 1.0) - self.cfg.cost_weight * cost
            if name == "HUMAN_LEFT" and current_risk:
                utility += 0.005
            results.append(
                InterventionResult(
                    intervention=name,
                    original_risk=float(current_risk),
                    resulting_risk=float(resulting),
                    resulting_collision_probability=float(pr.collision_probability),
                    risk_reduction=float(reduction),
                    feasibility=float(feasibility),
                    intervention_cost=float(cost),
                    utility=float(utility),
                )
            )
        return results

    def best(self, results: list[InterventionResult]) -> InterventionResult | None:
        feasible = [r for r in results if r.feasibility >= self.cfg.feasibility_min]
        if not feasible:
            return None
        target = self.risk_engine.cfg.prevention_target_risk
        effective = [r for r in feasible if r.resulting_risk <= target]
        pool = effective if effective else feasible
        return max(pool, key=lambda r: r.utility)

    def _with_velocity(self, ctx: TrackContext, vel: np.ndarray) -> TrackContext:
        traj = self.forecaster.forecast(ctx.pos, vel, ctx.cov, ctx.traj.t0)
        return TrackContext(
            track_id=ctx.track_id,
            cls=ctx.cls,
            pos=ctx.pos,
            vel=vel,
            radius=ctx.radius,
            traj=traj,
            hits=ctx.hits,
            cov=ctx.cov,
        )
"""Safety pipeline: PERCEIVE -> PREDICT -> PREVENT -> VERIFY.

One step() ingests a frame + detections and yields a Snapshot with risk state,
forecasts, counterfactual plans, and verification transitions.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ..config import Config
from ..core.geometry import Homography
from ..core.tracks import Object, Track, TrackManager
from ..events.store import EventStore
from ..planner.counterfactual import CounterfactualPlanner, InterventionResult
from ..prediction.forecast import EnsembleForecaster, Trajectory
from ..risk.engine import PairRisk, RiskEngine, ScenarioRisk, TrackContext
from ..rules import engine as rule_engine
from ..rules.schema import SafetyRule, load_rules
from ..verify.monitor import VerificationMonitor

PERSON_CLS = "person"
VEHICLE_CLS = "vehicle"


@dataclass
class Snapshot:
    t: float
    frame: np.ndarray
    risk: ScenarioRisk
    tracks: list[dict]
    forecasts: list[dict]
    interventions: list[InterventionResult] = field(default_factory=list)
    best_intervention: InterventionResult | None = None
    recommended_pair: tuple[int, int] | None = None
    verify: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    rule_violations: list[dict] = field(default_factory=list)


class SafetyPipeline:
    def __init__(self, cfg: Config, homography: Homography | None = None,
                 zones=None, rules: list[SafetyRule] | None = None,
                 store: EventStore | None = None,
                 enable_planning: bool = True,
                 enable_verification: bool = True,
                 use_uncertainty: bool = True):
        self.cfg = cfg
        self.homography = homography or Homography()
        self.enable_planning = enable_planning
        self.enable_verification = enable_verification
        self.use_uncertainty = use_uncertainty
        self._forecaster = EnsembleForecaster(
            cfg.prediction.horizon_s, cfg.prediction.dt_s,
            cfg.prediction.n_samples if use_uncertainty else 0,
        )
        self._risk_engine = RiskEngine(cfg.risk)
        self._planner = CounterfactualPlanner(cfg.planner, self._forecaster, self._risk_engine)
        self._track_manager = TrackManager(
            gating_m=cfg.perception.gating_meters,
            max_age=cfg.perception.max_age_frames,
            min_hits=cfg.perception.min_hits,
        )
        self._monitor = VerificationMonitor(
            cfg.risk.warn_risk,
            cfg.risk.intervention_risk,
            cfg.verification.resolved_on_risk_drop,
            cfg.verification.verification_window_s,
            store if enable_verification else None,
        )
        self._store = store
        self.zones = list(zones or [])
        self.rules = rules if rules is not None else load_rules(cfg.rules_path)
        self._last_frame_t = 0.0
        self._last_t = 0.0
        self._fps = 0.0
        self._last_risk = None

    def set_zones(self, zones) -> None:
        self.zones = list(zones)

    def set_rules(self, rules: list[SafetyRule]) -> None:
        self.rules = rules

    def step(self, frame: np.ndarray | None, objects: list[Object],
             t: float, dt: float, ext_ids: list[int] | None = None) -> Snapshot:
        t0 = time.perf_counter()
        tracks = self._update_tracks(objects, t, dt, ext_ids)
        t_perceive = time.perf_counter()

        contexts = self._build_contexts(tracks)
        rule_inputs = [
            rule_engine.RuleInput(track.track_id, self._cls(track.cls), *track.position,
                                  track.speed(), True)
            for track in tracks
        ]
        violations = rule_engine.evaluate_rules(
            self.rules, rule_inputs, lambda a, b: self.homography.distance(a, b)
        )
        risk = self._risk_engine.evaluate(contexts, self.zones, violations)
        t_risk = time.perf_counter()

        forecasts = self._forecasts_to_dicts(contexts)

        interventions: list[InterventionResult] = []
        best: InterventionResult | None = None
        recommended_pair: tuple[int, int] | None = None
        if self.enable_planning and risk.score >= self.cfg.risk.intervention_risk:
            worst = max(risk.pairs, key=lambda p: p.score, default=None)
            if worst and worst.score > self.cfg.risk.warn_risk and worst.confidence >= self.cfg.verification.required_confidence:
                ctx_map = {c.track_id: c for c in contexts}
                sub = ctx_map.get(worst.subject_id)
                tgt = ctx_map.get(worst.target_id)
                if sub is not None and tgt is not None:
                    interventions = self._planner.plan(sub, tgt, worst.score)
                    best = self._planner.best(interventions)
                    if best and best.intervention != "NO_ACTION":
                        recommended_pair = (worst.subject_id, worst.target_id)
        elif risk.score >= self.cfg.risk.warn_risk:
            worst = max(risk.pairs, key=lambda p: p.score, default=None)
            if self.enable_planning and worst:
                interventions, best = self._quick_plan(contexts, worst)
                if best and best.intervention != "NO_ACTION":
                    recommended_pair = (worst.subject_id, worst.target_id)
        t_plan = time.perf_counter()

        pair_map: dict[tuple[int, int], PairRisk] = {}
        for pr in risk.pairs:
            pair_map[(pr.subject_id, pr.target_id)] = pr
        alive_ids = {c.track_id for c in contexts}
        rec_key = None
        if recommended_pair is not None and best is not None:
            rec_key = recommended_pair
        transitions = []
        if self.enable_verification:
            transitions = self._monitor.update(
                t,
                pair_map,
                rec_key,
                best.intervention if best else "",
                alive_ids,
            )
        t_verify = time.perf_counter()

        now = time.perf_counter()
        metrics = {
            "perceive_ms": round((t_perceive - t0) * 1000, 2),
            "risk_ms": round((t_risk - t_perceive) * 1000, 2),
            "plan_ms": round((t_plan - t_risk) * 1000, 2),
            "verify_ms": round((t_verify - t_plan) * 1000, 2),
            "total_ms": round((now - t0) * 1000, 2),
            "fps": round(self._fps, 1),
            "tracks_active": len(tracks),
        }

        snap = Snapshot(
            t=t,
            frame=frame,
            risk=risk,
            tracks=self._tracks_to_dicts(tracks),
            forecasts=forecasts,
            interventions=interventions,
            best_intervention=best,
            recommended_pair=recommended_pair,
            verify=self._monitor.state_dict(),
            metrics=metrics,
            rule_violations=violations,
        )
        self._last_risk = risk
        self._track_fps(t)
        return snap

    def _quick_plan(self, contexts, worst_pair: PairRisk):
        ctx_map = {c.track_id: c for c in contexts}
        sub = ctx_map.get(worst_pair.subject_id)
        tgt = ctx_map.get(worst_pair.target_id)
        if sub is None or tgt is None:
            return [], None
        results = self._planner.plan(sub, tgt, worst_pair.score)
        return results, self._planner.best(results)

    def _update_tracks(self, objects: list[Object], t: float, dt: float, ext_ids=None):
        if ext_ids and len(ext_ids) == len(objects):
            return self._track_manager.step_forced(objects, list(ext_ids), t, max(dt, 1e-4))
        return self._track_manager.step(objects, t, max(dt, 1e-4))

    def _build_contexts(self, tracks: list[Track]) -> list[TrackContext]:
        contexts = []
        for tr in tracks:
            pos = tr.position
            vel = tr.smooth_velocity()
            cov = tr.kalman.covariance()
            traj = self._forecaster.forecast(np.asarray(pos), np.asarray(vel), cov, self._last_t)
            contexts.append(
                TrackContext(
                    track_id=tr.track_id,
                    cls=self._cls(tr.cls),
                    pos=np.asarray(pos),
                    vel=np.asarray(vel),
                    radius=self._risk_engine.radius_for(self._cls(tr.cls)),
                    traj=traj,
                    hits=tr.hits,
                    cov=cov,
                )
            )
        return contexts

    @staticmethod
    def _cls(raw: str) -> str:
        if raw == PERSON_CLS:
            return PERSON_CLS
        return VEHICLE_CLS

    def _forecasts_to_dicts(self, contexts: list[TrackContext]) -> list[dict]:
        out = []
        for c in contexts:
            s = c.traj.samples
            samples = None
            if s is not None:
                samples = s[:: max(1, s.shape[0] // 12)].tolist()
            out.append(
                {
                    "track_id": c.track_id,
                    "cls": c.cls,
                    "pos": c.pos.tolist(),
                    "vel": c.vel.tolist(),
                    "positions": c.traj.positions[::2].tolist(),
                    "samples": samples,
                    "std": (float(s.std(axis=0).mean()) if s is not None and s.shape[0] > 1 else 0.0),
                }
            )
        return out

    @staticmethod
    def _tracks_to_dicts(tracks: list[Track]) -> list[dict]:
        return [
            {
                "track_id": tr.track_id,
                "cls": tr.cls,
                "conf": tr.conf,
                "box": tr.box,
                "pos": list(tr.position),
                "vel": list(tr.velocity),
                "speed": tr.speed(),
                "hits": tr.hits,
                "history_len": len(tr.history),
            }
            for tr in tracks
        ]

    def _track_fps(self, t: float) -> None:
        if self._last_frame_t:
            dt = t - self._last_frame_t
            if dt > 1e-4:
                inst = 1.0 / dt
                self._fps = 0.9 * self._fps + 0.1 * inst
        self._last_frame_t = t
        self._last_t = t

    def annotate(self, frame: np.ndarray | None, snap: Snapshot,
                 homography: Homography | None = None, simulation: bool = False) -> np.ndarray | None:
        if frame is None or not self.cfg.pipeline.annotate_ui:
            return frame
        h = homography or self.homography
        if simulation:
            return frame
        from ..vision import annotate
        from ..risk import scoring

        for tdict in snap.tracks:
            box = tdict.get("box")
            if box:
                color = annotate._color("orange") if tdict["cls"] == "person" else annotate._color("cyan")
                label = f"{tdict['cls']} #{tid} {tdict['speed']:.1f}m/s"
                annotate.draw_box(frame, box, label, color)
        for fc in snap.forecasts:
            color = annotate._color("orange") if fc["cls"] == "person" else annotate._color("cyan")
            annotate.draw_predictions(frame, h, fc, color)
            annotate.draw_velocity(frame, h, fc["pos"], fc["vel"], color)
        annotate.draw_zones(frame, h, self.zones)
        if snap.risk.collision_point and snap.risk.score >= self.cfg.risk.warn_risk:
            annotate.draw_collision_point(frame, h, snap.risk.collision_point[0], snap.risk.collision_point[1])
        states = scoring.confidence_label(snap.risk.confidence)
        lines = [
            ("STATE", f"{snap.risk.state} / {states}"),
            ("RISK", f"{snap.risk.score:.0f}/100"),
            ("COLLISION PROB", f"{snap.risk.collision_probability * 100:.0f}%"),
            ("TTC", f"{snap.risk.ttc:.1f}s" if snap.risk.ttc is not None else "-"),
            ("BEST INTERVENTION", snap.best_intervention.intervention if snap.best_intervention else "-"),
        ]
        annotate.draw_top_bar(frame, lines)
        return frame
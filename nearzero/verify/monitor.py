"""Closed-loop verification: monitors whether predicted danger was mitigated.

Event lifecycle:
    PREDICTED -> WARNING -> INTERVENTION_RECOMMENDED -> PREVENTED
                                                      |-> UNRESOLVED
                +-> FALSE_ALARM (danger never materialized)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..events.store import EventStore
from ..risk.engine import PairRisk

STATUS_PREDICTED = "PREDICTED"
STATUS_WARNING = "WARNING"
STATUS_RECOMMENDED = "INTERVENTION_RECOMMENDED"
STATUS_PREVENTED = "PREVENTED"
STATUS_UNRESOLVED = "UNRESOLVED"
STATUS_FALSE_ALARM = "FALSE_ALARM"
STATUS_CLEARED = "CLEARED"


@dataclass
class OpenEvent:
    event_id: int
    pair: tuple[int, int]
    kind: str
    initial_risk: float
    peak_risk: float
    t_open: float
    state: str = STATUS_PREDICTED
    recommended: str = ""
    low_count: int = 0
    miss_count: int = 0
    recommended_at: float | None = None


class VerificationMonitor:
    def __init__(
        self,
        warn_risk: float,
        intervention_risk: float,
        resolve_drop: float = 0.5,
        verify_window_s: float = 3.0,
        store: EventStore | None = None,
    ):
        self.warn_risk = warn_risk
        self.intervention_risk = intervention_risk
        self.resolve_drop = resolve_drop
        self.verify_window_s = verify_window_s
        self.store = store
        self._open: dict[tuple[int, int], OpenEvent] = {}
        self._history: list[OpenEvent] = []
        self._rec_key: set[tuple[int, int]] = set()

    def update(
        self,
        t: float,
        pair_risks: dict[tuple[int, int], PairRisk],
        recommended: tuple[int, int] | None,
        best_intervention: str,
        alive_ids: set[int],
    ) -> list[dict]:
        """Advance verification state machine and persist state transitions."""
        transitions: list[dict] = []
        for key in list(self._open.keys()):
            subj, tgt = key
            present = subj in alive_ids and tgt in alive_ids
            event = self._open[key]
            risk = pair_risks.get(key)
            risk_score = risk.score if risk else 0.0
            event.peak_risk = max(event.peak_risk, risk_score)

            if recommended and recommended == key and best_intervention:
                if event.state in (STATUS_PREDICTED, STATUS_WARNING):
                    old = event.state
                    event.state = STATUS_RECOMMENDED
                    event.recommended = best_intervention
                    if self.store:
                        self.store.update_event(event.event_id, status=STATUS_RECOMMENDED,
                                                recommended=best_intervention)
                    transitions.append({"event_id": event.event_id, "from": old,
                                        "to": STATUS_RECOMMENDED, "t": t})

            dropped = risk_score <= self.resolve_drop * event.initial_risk
            if dropped:
                event.low_count += 1
            else:
                event.low_count = 0

            resolved = False
            if not present:
                event.miss_count += 1
                if event.miss_count >= 4:
                    resolved = True
            elif risk is not None and risk.ttc is not None and risk.ttc <= 0.5:
                resolved = True
            elif event.low_count >= 5 and event.state == STATUS_RECOMMENDED:
                resolved = True

            if resolved:
                hood = "PREVENTED" if event.state == STATUS_RECOMMENDED else "FALSE_ALARM"
                if self.store:
                    self.store.update_event(event.event_id, status=hood, risk_final=risk_score,
                                            details={"peak_risk": event.peak_risk,
                                                     "verified_at": t})
                self._history.append(event)
                del self._open[key]
                transitions.append({"event_id": event.event_id, "from": event.state,
                                    "to": hood, "t": t})
            elif risk is not None and risk.ttc is not None and risk.ttc <= 0.0:
                if self.store:
                    self.store.update_event(event.event_id, status=STATUS_UNRESOLVED,
                                            risk_final=risk_score)
                self._history.append(event)
                del self._open[key]
                transitions.append({"event_id": event.event_id, "from": event.state,
                                    "to": STATUS_UNRESOLVED, "t": t})
            elif t - event.t_open > self.verify_window_s and event.state == STATUS_PREDICTED:
                if self.store:
                    self.store.update_event(event.event_id, status=STATUS_FALSE_ALARM,
                                            risk_final=risk_score)
                self._history.append(event)
                del self._open[key]
                transitions.append({"event_id": event.event_id, "from": event.state,
                                    "to": STATUS_FALSE_ALARM, "t": t})

        for key, risk in pair_risks.items():
            if key in self._open:
                continue
            if risk.score >= self.warn_risk:
                event_id = None
                if self.store:
                    event_id = self.store.add_event(
                        STATUS_PREDICTED, "collision_risk", key[0], key[1],
                        float(risk.score), t, origin="risk_engine",
                        details={"collision_probability": float(risk.collision_probability),
                                 "ttc": float(risk.ttc) if risk.ttc is not None else None},
                    )
                ev = OpenEvent(event_id or -1, key, "collision_risk", float(risk.score),
                               float(risk.score), t)
                self._open[key] = ev
        return transitions

    @property
    def open_events(self) -> list[OpenEvent]:
        return list(self._open.values())

    @property
    def recent_history(self) -> list[OpenEvent]:
        return self._history[-20:]

    def state_dict(self) -> dict:
        return {
            "open": [
                {
                    "subject_id": e.pair[0],
                    "target_id": e.pair[1],
                    "state": e.state,
                    "initial_risk": e.initial_risk,
                    "peak_risk": e.peak_risk,
                    "recommended": e.recommended,
                }
                for e in self._open.values()
            ],
            "last_event": self._history[-1].__dict__ if self._history else None,
        }
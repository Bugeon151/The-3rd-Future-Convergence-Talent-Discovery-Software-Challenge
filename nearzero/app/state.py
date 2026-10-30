"""JSON-safe snapshot serialization for the dashboard API."""
from __future__ import annotations

from .pipeline import Snapshot


def intervention_rows(snap: Snapshot) -> list[dict]:
    if not snap.interventions:
        return []
    ordered = sorted(snap.interventions, key=lambda r: r.resulting_risk)
    best_name = snap.best_intervention.intervention if snap.best_intervention else None
    out = []
    for r in ordered:
        row = r.to_dict()
        row["best"] = r.intervention == best_name and r.resulting_risk == min(
            x.resulting_risk for x in ordered
        )
        out.append(row)
    return out


def snapshot_to_state(snap: Snapshot, mode: str = "sim") -> dict:
    risk = snap.risk
    track_info = []
    for tr in snap.tracks:
        track_info.append(
            {
                "track_id": tr["track_id"],
                "cls": tr["cls"],
                "speed": tr["speed"],
                "pos": tr["pos"],
                "vel": tr["vel"],
                "hits": tr["hits"],
            }
        )
    best = snap.best_intervention.to_dict() if snap.best_intervention else None
    rec = None
    if snap.recommended_pair is not None:
        rec = {"subject": snap.recommended_pair[0], "target": snap.recommended_pair[1]}
    return {
        "t": round(snap.t, 2),
        "mode": mode,
        "risk": {
            "score": round(risk.score, 1),
            "state": risk.state,
            "collision_probability": round(risk.collision_probability * 100.0, 1),
            "ttc": round(risk.ttc, 2) if risk.ttc is not None else None,
            "predicted_min_distance": round(risk.predicted_min_distance, 2) if risk.predicted_min_distance is not None else None,
            "zone_probability": round(risk.zone_probability * 100.0, 1),
            "rule_violation": round(risk.rule_violation, 3),
            "collision_point": list(risk.collision_point) if risk.collision_point else None,
            "confidence": round(risk.confidence, 2),
            "components": risk.components,
        },
        "tracks": track_info,
        "forecasts": snap.forecasts,
        "counterfactual": intervention_rows(snap),
        "best_intervention": best,
        "recommended_pair": rec,
        "verify": snap.verify,
        "metrics": snap.metrics,
        "rule_violations": [
            {k: (round(v, 3) if isinstance(v, float) else v) for k, v in vv.items()}
            for vv in snap.rule_violations
        ],
    }
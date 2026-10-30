"""Evaluation metrics for safety policies.

Ground truth: contact intervals from the simulator. Decisions: per-frame
(alarm, recommendation). We report recall of GT events, false alarms, warning
lead time, intervention success, and mean risk reduction.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Decision:
    t: float
    alarm: bool
    recommended: str = ""
    risk: float | None = None
    resulting_risk: float | None = None


def alarm_episodes(decisions: list[Decision]) -> list[tuple[float, float]]:
    episodes = []
    start = None
    for d in decisions:
        if d.alarm and start is None:
            start = d.t
        elif not d.alarm and start is not None:
            episodes.append((start, d.t))
            start = None
    if start is not None:
        episodes.append((start, decisions[-1].t))
    return episodes


def evaluate_collision_event(
    event: dict, decisions: list[Decision], lead_window_s: float = 6.0
) -> dict:
    t0 = event["t0"]
    prior = [d for d in decisions if d.t <= t0 - 1e-6]
    last = [d for d in prior if d.alarm]
    hit = len(last) > 0
    lead = None
    if hit:
        first = min(d.t for d in last)
        if first >= max(0.0, t0 - lead_window_s):
            lead = t0 - first
        else:
            hit = False
    return {"hit": hit, "lead_time": lead}


def summarize_policy(
    name: str,
    decisions: list[Decision],
    gt_events: list[dict],
    duration_s: float,
    lead_window_s: float = 6.0,
) -> dict:
    episodes = alarm_episodes(decisions)
    matched = set()
    leads = []
    for ev in gt_events:
        r = evaluate_collision_event(ev, decisions, lead_window_s)
        if r["hit"] and r["lead_time"] is not None:
            matched.add((ev["t0"], ev["subject"], ev["target"]))
            leads.append(r["lead_time"])
    false_alarms = 0
    for (s, e) in episodes:
        if not any(abs(max(0.0, ev["t0"]) - s) <= lead_window_s + 1e-6 or s <= ev["t1"] + 1e-6 for ev in gt_events):
            false_alarms += 1
    recs = [d for d in decisions if d.recommended]
    inter_success = [d for d in recs if d.resulting_risk is not None and d.resulting_risk <= 25.0]
    keep = [d for d in recs if d.resulting_risk is not None]
    return {
        "policy": name,
        "n_events": len(gt_events),
        "recall": (len(matched) / len(gt_events)) if gt_events else None,
        "mean_lead_time": (sum(leads) / len(leads)) if leads else None,
        "median_lead_time": float(sorted(leads)[len(leads) // 2]) if leads else None,
        "max_lead_time": max(leads) if leads else None,
        "min_lead_time": min(leads) if leads else None,
        "false_alarms": false_alarms,
        "false_alarm_rate_min": false_alarms / max(duration_s / 60.0, 1e-6),
        "n_recommendations": len(recs),
        "intervention_success_rate": (len(inter_success) / len(keep)) if keep else None,
        "mean_risk_reduction": (
            (sum(d.risk - d.resulting_risk for d in keep) / len(keep)) if keep else None
        ),
        "episodes": episodes,
    }


def aggregate_summaries(sums: list[dict], weights: list[float] | None = None) -> dict:
    w = weights or [1.0] * len(sums)
    def wmean(key):
        vals = [s[key] for s, w_ in zip(sums, w) if s.get(key) is not None]
        ws = [w_ for s, w_ in zip(sums, w) if s.get(key) is not None]
        if not vals:
            return None
        return sum(v * ww for v, ww in zip(vals, ws)) / sum(ws)
    out = {
        "n_scenarios": len(sums),
        "total_events": sum(s["n_events"] for s in sums),
        "recall_mean": wmean("recall"),
        "mean_lead_time": wmean("mean_lead_time"),
        "median_lead_time": wmean("median_lead_time"),
        "false_alarms_total": sum(s["false_alarms"] for s in sums),
        "false_alarm_rate_min_mean": wmean("false_alarm_rate_min"),
        "intervention_success_rate": wmean("intervention_success_rate"),
        "mean_risk_reduction": wmean("mean_risk_reduction"),
        "unresolved_events": sum(s.get("unresolved", 0) for s in sums),
    }
    return out
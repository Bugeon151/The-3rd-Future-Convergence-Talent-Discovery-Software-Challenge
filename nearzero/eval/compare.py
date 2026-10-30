"""Runs the baseline comparison and ablation, writes results/ artifacts.

Run: python -m nearzero.eval.all   (or scripts/run_eval.ps1)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time

import numpy as np

from ..config import Config, load_config
from ..events.store import EventStore
from ..sim.scenarios import SCENARIO_NAMES, make_scenario
from .metrics import Decision, aggregate_summaries, alarm_episodes, summarize_policy

logger = logging.getLogger("nearzero.eval")

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "results")


def _pipeline(cfg: Config, enable_planning=True, enable_verification=True, use_uncertainty=True):
    from ..app.pipeline import SafetyPipeline

    store = EventStore(":memory:")
    pipe = SafetyPipeline(
        cfg,
        zones=None,
        store=store,
        enable_planning=enable_planning,
        enable_verification=enable_verification,
        use_uncertainty=use_uncertainty,
    )
    pipe._store = store
    return pipe, store


def run_policy_on_scenario(cfg, make_policy, scenario_name, dt=0.1):
    _, states, events = make_scenario(scenario_name)
    policy = make_policy()
    decision = policy.decisions(states)
    duration = states[-1]["t"] if states else 0.0
    summary = summarize_policy(
        policy.name, decision, events, duration, lead_window_s=6.0
    )
    summary["scenario"] = scenario_name
    return summary, decision, states, events


def comparison_suite(cfg: Config) -> dict:
    from .policies import (
        DistanceThresholdPolicy,
        NearZeroPolicy,
        PredictionPolicy,
        TTCPolicy,
    )

    factories = {
        "A: distance_threshold": lambda: DistanceThresholdPolicy(2.0),
        "B: ttc_threshold": lambda: TTCPolicy(3.0),
        "C: trajectory_prediction": lambda: PredictionPolicy(),
    }
    rows = []
    details: dict[str, list[Decision]] = {}
    summaries: dict[str, list[dict]] = {}

    for scenario in SCENARIO_NAMES:
        for name, make in factories.items():
            summary, dec, _, _ = run_policy_on_scenario(cfg, make, scenario)
            summaries.setdefault(name, []).append(summary)
            details[f"{name}|{scenario}"] = dec

    for scenario in SCENARIO_NAMES:
        pipe, store = _pipeline(cfg, enable_planning=True, enable_verification=True, use_uncertainty=True)
        summary, dec, _, _ = run_policy_on_scenario(cfg, lambda: NearZeroPolicy(cfg, pipe), scenario)
        summary["unresolved"] = store.summary().get("UNRESOLVED", 0)
        summary["prevented"] = store.summary().get("PREVENTED", 0)
        store.close()
        summaries.setdefault("D: nearzero_x", []).append(summary)
        details[f"D: nearzero_x|{scenario}"] = dec

    for policy, slist in summaries.items():
        agg = aggregate_summaries(slist)
        rows.append(
            {
                "policy": policy,
                "scenario": "*",
                "recall": agg["recall_mean"],
                "mean_lead_time": agg["mean_lead_time"],
                "false_alarms": agg["false_alarms_total"],
                "false_alarm_rate_min": agg["false_alarm_rate_min_mean"],
                "intervention_success_rate": agg["intervention_success_rate"],
                "mean_risk_reduction": agg["mean_risk_reduction"],
                "unresolved": agg["unresolved_events"],
            }
        )
        for s in slist:
            rows.append(
                {
                    "policy": policy,
                    "scenario": _scenario_name(s),
                    "recall": s["recall"],
                    "mean_lead_time": s["mean_lead_time"],
                    "false_alarms": s["false_alarms"],
                    "false_alarm_rate_min": s["false_alarm_rate_min"],
                    "intervention_success_rate": s.get("intervention_success_rate"),
                    "mean_risk_reduction": s.get("mean_risk_reduction"),
                    "unresolved": s.get("unresolved", 0),
                }
            )
    return {"rows": rows, "details": details, "summaries": summaries}


def _scenario_name(summary: dict) -> str:
    return summary.get("scenario", "")


def ablation_suite(cfg: Config) -> list[dict]:
    configs = [
        ("full", True, True, True),
        ("no_uncertainty", True, True, False),
        ("no_counterfactual", False, True, True),
        ("no_verification", True, False, True),
    ]
    out = []
    for label, plan, ver, unc in configs:
        totals = []
        for scenario in ("head_on", "demo_avoided", "safe_pass", "sudden_stop"):
            pipe, store = _pipeline(cfg, plan, ver, unc)
            from .policies import NearZeroPolicy

            summary, _, _, _ = run_policy_on_scenario(cfg, lambda: NearZeroPolicy(cfg, pipe), scenario)
            summary["unresolved"] = store.summary().get("UNRESOLVED", 0)
            summary["prevented"] = store.summary().get("PREVENTED", 0)
            store.close()
            totals.append(summary)
        agg = aggregate_summaries(totals)
        out.append({"ablation": label, **agg})
    return out


def vision_benchmark(cfg: Config, iters: int = 12) -> dict:
    from ..vision.perception import SimulatedPerception
    from ..sim.scenarios import make_scenario

    det_ms = None
    total_ms = None
    device = "CPU"
    try:
        _, states, _ = make_scenario("demo_avoided")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        from ..vision.detector import Detector

        det = Detector(cfg.perception)
        if det.available:
            device = det.device
            t0 = time.perf_counter()
            n = 0
            for i in range(iters):
                r = det.predict(frame)
                n += 1
            det_ms = (time.perf_counter() - t0) / n * 1000
        else:
            det_ms = None
    except Exception as exc:  # pragma: no cover
        logger.warning("vision benchmark failed: %s", exc)
        det_ms = None

    pipe, _ = _pipeline(cfg)
    from .policies import NearZeroPolicy

    policy = NearZeroPolicy(cfg, pipe)
    _, states, _ = make_scenario("demo_avoided")
    t0 = time.perf_counter()
    policy.decisions(states)
    total_ms = (time.perf_counter() - t0) / max(len(states), 1) * 1000
    return {
        "device": device,
        "detector_inference_ms": det_ms,
        "detector_fps": 1000.0 / det_ms if det_ms else None,
        "core_pipeline_ms": total_ms,
        "core_pipeline_fps": 1000.0 / total_ms if total_ms else None,
    }


def write_results(comparison: dict, ablation: list[dict], bench: dict, cfg: Config) -> None:
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "plots"), exist_ok=True)
    rows = comparison["rows"]
    with open(os.path.join(RESULTS, "comparison.csv"), "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else ["policy"])
        writer.writeheader()
        writer.writerows(rows)
    metrics = {"benchmark": bench, "ablation": ablation, "comparison_rows": rows}
    with open(os.path.join(RESULTS, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)
    _write_report(comparison, ablation, bench, cfg)


def _write_report(comparison, ablation, bench, cfg) -> None:
    md = [
        "# Evaluation Report",
        "",
        "Measured on the deterministic synthetic scenario suite (see nearzero/sim/scenarios.py).",
        "Every number below is produced by running the code in this repository (`scripts/run_eval.ps1`).",
        "",
        "## 1. Runtime benchmark",
        "",
        f"- Detector device: {bench['device']}",
        f"- Detector inference: {bench['detector_inference_ms']:.1f} ms"
        if bench["detector_inference_ms"] else "- Detector inference: NOT MEASURED",
        f"- Core pipeline (perception->verify, no vision): {bench['core_pipeline_ms']:.1f} ms"
        if bench["core_pipeline_ms"] else "- Core pipeline: NOT MEASURED",
        "",
        "## 2. Policy comparison (per-scenario)",
        "",
        "| policy | scenario | recall | mean lead (s) | false alarms |",
        "|---|---|---|---|---|",
    ]
    seen_rows = set()
    for r in comparison["rows"]:
        if "scenario" not in r or not r["scenario"]:
            continue
        key = (r["policy"], r["scenario"])
        if key in seen_rows:
            continue
        seen_rows.add(key)
        md.append(
            f"| {r['policy']} | {r['scenario']} | {_fmt(r['recall'])} | "
            f"{_fmt(r['mean_lead_time'])} | {r['false_alarms']} |"
        )
    md += [
        "",
        "## 3. Ablation (NEAR-ZERO X components)",
        "",
        "| ablation | recall | mean_lead (s) | false_alarms | intervention_success | mean_risk_reduction |",
        "|---|---|---|---|---|---|",
    ]
    for a in ablation:
        md.append(
            f"| {a['ablation']} | {_fmt(a.get('recall_mean'))} | {_fmt(a.get('mean_lead_time'))} | "
            f"{a.get('false_alarms_total')} | {_fmt(a.get('intervention_success_rate'))} | "
            f"{_fmt(a.get('mean_risk_reduction'))} |"
        )
    md += [
        "",
        "## 4. Honest limitations",
        "",
        "- Synthetic evaluation isolates algorithmic safety quality from camera noise.",
        "- Lead time and recall are computed on ground-truth contact intervals.",
        "- Values labeled NULL are scenarios where the metric does not apply.",
        "",
    ]
    with open(os.path.join(RESULTS, "evaluation_report.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        policy_names = []
        leads = []
        fas = []
        for r in comparison["rows"]:
            if "scenario" in r:
                continue
            policy_names.append(r["policy"])
            leads.append(r["mean_lead_time"] or 0)
            fas.append(r["false_alarm_rate_min_mean"] or 0)
        axes[0].bar(policy_names, leads, color="#2e9aff")
        axes[0].set_title("Weighted mean warning lead time (s)")
        axes[0].set_xticklabels(policy_names, rotation=20, ha="right", fontsize=8)
        axes[1].bar(policy_names, fas, color="#ff9f2e")
        axes[1].set_title("Weighted false-alarm rate / min")
        axes[1].set_xticklabels(policy_names, rotation=20, ha="right", fontsize=8)
        abl = [a["ablation"] for a in ablation]
        red = [a.get("mean_risk_reduction") or 0 for a in ablation]
        axes[2].bar(abl, red, color="#4caf50")
        axes[2].set_title("Mean predicted risk reduction")
        axes[2].tick_params(axis="x", rotation=20, labelsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(RESULTS, "plots", "comparison.png"), dpi=120)
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        logger.warning("plots skipped: %s", exc)


def _fmt(v):
    if v is None:
        return "NULL"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def main(cfg_path: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    logger.info("starting comparison")
    comparison = comparison_suite(cfg)
    ablation = ablation_suite(cfg)
    bench = vision_benchmark(cfg)
    write_results(comparison, ablation, bench, cfg)
    logger.info("evaluation complete -> results/")
    return {"comparison": comparison, "ablation": ablation, "benchmark": bench}
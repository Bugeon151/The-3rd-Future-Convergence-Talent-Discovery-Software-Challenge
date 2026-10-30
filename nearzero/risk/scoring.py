"""Risk score normalization and state classification.

Every raw component (collision prob, TTC, PMD, zone intrusion, rule violation)
is mapped to [0,1] with a documented monotonic function, then combined with
weights. The final score is 100 * weighted mean.
"""
from __future__ import annotations

import numpy as np


def sigmoid(x: float, x0: float, k: float) -> float:
    return float(1.0 / (1.0 + np.exp(-k * (float(x) - x0))))


def component_collision(p: float) -> float:
    return float(np.clip(p, 0.0, 1.0))


def component_ttc(ttc: float | None, x0: float, k: float) -> float:
    if ttc is None or not np.isfinite(ttc):
        return 0.0
    return 1.0 - sigmoid(ttc, x0, k)


def component_pmd(pmd: float | None, x0: float, k: float) -> float:
    if pmd is None:
        return 0.0
    return 1.0 - sigmoid(pmd, x0, k)


def component_zone(p: float) -> float:
    return float(np.clip(p, 0.0, 1.0))


def component_rule(severity_value: float, violation_intensity: float) -> float:
    return float(np.clip(severity_value * violation_intensity, 0.0, 1.0))


def combine_components(
    components: dict[str, float], weights: dict[str, float]
) -> float:
    total_w = sum(weights.get(k, 0.0) for k in components)
    if total_w <= 0.0:
        return 0.0
    value = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)
    score = 100.0 * value / total_w
    return float(np.clip(score, 0.0, 100.0))


def classify_state(score: float, thresholds: dict[str, float]) -> str:
    if score < thresholds["safe"]:
        return "SAFE"
    if score < thresholds["caution"]:
        return "CAUTION"
    if score < thresholds["warning"]:
        return "WARNING"
    return "CRITICAL"


def confidence_label(confidence: float) -> str:
    if confidence < 0.35:
        return "LOW CONFIDENCE"
    if confidence < 0.7:
        return "MEDIUM CONFIDENCE"
    return "HIGH CONFIDENCE"
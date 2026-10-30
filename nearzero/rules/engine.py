"""Deterministic rule engine: evaluates DSL rules against scene state.

Produces violation records consumed by the risk engine. All evaluation is
numeric/geometric; an LLM never decides safety.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .schema import SafetyRule


@dataclass
class RuleInput:
    track_id: int
    cls: str
    x: float
    y: float
    speed: float
    active: bool = True


def evaluate_rules(
    rules: Iterable[SafetyRule],
    inputs: list[RuleInput],
    distance: callable,
) -> list[dict]:
    violations: list[dict] = []
    for rule in rules:
        matched = [i for i in inputs if i.cls == rule.subject or rule.subject == "any"]
        targets = [i for i in inputs if _target_matches(i, rule)]
        for sub in matched:
            for tgt in targets:
                if sub.track_id == tgt.track_id:
                    continue
                if rule.condition.get("machine_active") and not _target_matches(tgt, rule):
                    continue
                d = distance((sub.x, sub.y), (tgt.x, tgt.y))
                intensity = _intensity(rule, d)
                if intensity is None:
                    continue
                violations.append(
                    {
                        "rule_id": rule.rule_id or rule.text,
                        "subject": sub.track_id,
                        "target": tgt.track_id,
                        "severity": rule.severity,
                        "severity_value": rule.severity_value(),
                        "intensity": intensity,
                        "distance": d,
                    }
                )
    return violations


def _target_is_active(tgt: RuleInput, rule: SafetyRule) -> bool:
    if "machine_active" not in rule.condition:
        return True
    is_machine = tgt.cls in ("vehicle", "machine", "cart", "forklift", "truck", "car", "bus", "motorcycle", "bicycle")
    return tgt.active and (not is_machine or tgt.speed > 0.02)


def _target_cls(tgt: RuleInput, target_spec: str) -> bool:
    if target_spec in ("any", "machine"):
        return tgt.cls in ("vehicle", "machine", "cart", "forklift", "truck", "car", "bus")
    return tgt.cls == target_spec or (target_spec == "vehicle" and tgt.cls != "person")


def _target_matches(tgt: RuleInput, rule: SafetyRule) -> bool:
    if not _target_cls(tgt, rule.target):
        return False
    return _target_is_active(tgt, rule)


def _intensity(rule: SafetyRule, d: float) -> float | None:
    lt = rule.condition.get("distance_lt")
    gt = rule.condition.get("distance_gt")
    if lt is not None and d < lt:
        return min(1.0, (lt - d) / lt)
    if gt is not None and d > gt:
        return min(1.0, (d - gt) / gt)
    if lt is None and gt is None:
        return 1.0
    return None
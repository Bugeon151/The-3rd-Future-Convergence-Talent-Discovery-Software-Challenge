"""Natural-language safety rules compiled to a validated DSL (JSON).

Scope: rules are parsed/validated to a structured schema, then evaluated purely
deterministically. No LLM is involved in safety decisions.
"""
from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field

SEVERITY_VALUES = {"low": 0.35, "medium": 0.65, "high": 1.0}
VALID_CONDITIONS = {"machine_active", "distance_lt", "distance_gt", "zone"}
VALID_SEVERITY = set(SEVERITY_VALUES)


@dataclass
class SafetyRule:
    text: str
    subject: str
    target: str
    condition: dict = field(default_factory=dict)
    severity: str = "high"
    rule_id: str = ""

    def severity_value(self) -> float:
        return SEVERITY_VALUES[self.severity]

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SafetyRule":
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def validate_rule(d: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if "subject" not in d or not isinstance(d["subject"], str):
        errors.append("subject must be a string")
    if "target" not in d or not isinstance(d["target"], str):
        errors.append("target must be a string")
    cond = d.get("condition", {})
    if not isinstance(cond, dict):
        errors.append("condition must be an object")
    else:
        for k in cond:
            if k not in VALID_CONDITIONS:
                errors.append(f"unknown condition key: {k}")
        for dist_key in ("distance_lt", "distance_gt"):
            if dist_key in cond:
                v = cond[dist_key]
                if not isinstance(v, (int, float)) or v <= 0:
                    errors.append(f"{dist_key} must be a positive number")
    sev = d.get("severity", "high")
    if sev not in VALID_SEVERITY:
        errors.append(f"severity must be one of {sorted(VALID_SEVERITY)}")
    return len(errors) == 0, errors


def load_rules(path: str | None = None) -> list[SafetyRule]:
    if not path or not os.path.exists(path):
        return default_rules()
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    rules = []
    for item in raw:
        ok, _ = validate_rule(item)
        if ok:
            rules.append(SafetyRule.from_dict(item))
    return rules


def save_rules(rules: list[SafetyRule], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([r.to_dict() for r in rules], fh, ensure_ascii=False, indent=2)


def default_rules() -> list[SafetyRule]:
    return [
        SafetyRule(
            text="기계 작동 중에는 작업자가 1.5m 안에 들어오면 안 된다.",
            subject="person",
            target="machine",
            condition={"machine_active": True, "distance_lt": 1.5},
            severity="high",
            rule_id="default-ko-1",
        ),
        SafetyRule(
            text="No person may come within 1.5 m of an active machine.",
            subject="person",
            target="vehicle",
            condition={"machine_active": True, "distance_lt": 1.5},
            severity="high",
            rule_id="default-en-1",
        ),
    ]
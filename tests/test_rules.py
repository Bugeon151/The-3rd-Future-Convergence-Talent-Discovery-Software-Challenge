"""Tests for the rules DSL: schema validation, Ko/En natural-language parser, engine."""
from __future__ import annotations

from nearzero.rules.engine import RuleInput, evaluate_rules
from nearzero.rules.natural import parse_rule
from nearzero.rules.schema import SafetyRule, default_rules, validate_rule


def test_default_rules_valid():
    for rule in default_rules():
        ok, errors = validate_rule(rule.to_dict())
        assert ok, (rule.name if hasattr(rule, "name") else rule.text, errors)


def test_parse_english():
    r = parse_rule("keep 2.0 m distance between person and vehicle")
    assert r is not None
    assert r.subject == "person"
    assert r.target == "vehicle"
    assert r.condition.get("distance_gt") == 2.0


def test_parse_korean():
    r = parse_rule("기계 작동 중에는 작업자가 1.5m 안에 들어오면 안 된다.")
    assert r is not None
    assert r.subject == "person"
    assert r.condition.get("machine_active") is True
    assert r.condition.get("distance_lt") == 1.5
    assert "stop" in " ".join(getattr(r, "actions", [])) or r.target == "vehicle"


def test_parse_garbage_returns_none():
    assert parse_rule("jsjdfkjsdflk 9933 ㅋㅋㅋ") is None


def test_evaluate_rules_detects_violation():
    rule = SafetyRule(text="no person within 1.5m of active vehicle",
                      subject="person", target="vehicle",
                      condition={"machine_active": True, "distance_lt": 1.5},
                      rule_id="r1")
    inputs = [
        RuleInput(1, "person", 2.0, 0.0, 1.0, True),
        RuleInput(2, "vehicle", 2.5, 0.0, 30.0, True),
    ]
    violations = evaluate_rules([rule], inputs, lambda a, b: abs(a[0] - b[0]))
    assert any("r1" in v["rule_id"] for v in violations)


def test_evaluate_rules_safe_no_violation():
    rule = SafetyRule(text="no person within 1.5m of active vehicle",
                      subject="person", target="vehicle",
                      condition={"machine_active": True, "distance_lt": 1.5},
                      rule_id="r2")
    inputs = [
        RuleInput(1, "person", 2.0, 0.0, 1.0, True),
        RuleInput(2, "vehicle", 6.0, 0.0, 30.0, True),
    ]
    assert evaluate_rules([rule], inputs, lambda a, b: abs(a[0] - b[0])) == []
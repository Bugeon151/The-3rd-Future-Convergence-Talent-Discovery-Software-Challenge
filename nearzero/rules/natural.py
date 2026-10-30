"""Deterministic pattern-based parser for safety-rule sentences (Ko/En).

If an LLM parser is available later it can expand coverage; this parser keeps the
core system functional offline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .schema import SafetyRule, SEVERITY_VALUES, VALID_SEVERITY

_SUBJECTS = {
    "person": ("person", "worker", "작업자", "사람", "작업인"),
    "vehicle": ("cart", "vehicle", "기계", "카트", "운반대", "이동대"),
}
_SUBJECT_ALIAS = {alias: cls for cls, aliases in _SUBJECTS.items() for alias in aliases}

_ACTION_RULES = [
    (r"(do|shall) not come within\s+([\d.]+)\s*m\s+of\s+(an\s+|a\s+)?([A-Za-z]+)\s+while\s+(?:it|the\s+\w+)\s+is\s+(active|operating|running)", None),
    (r"(no|never)\s+(allow\s+)?([A-Za-z]+)\s+within\s+([\d.]+)\s*m\s+of\s+(?:an?\s+|a\s+)?([A-Za-z]+)\s+(?:while\s+(active|operating|moving))?", None),
    (r"keep\s+(?:a\s+)?([\d.]+)\s*m\s+distance\s+between\s+([A-Za-z]+)\s+and\s+([A-Za-z]+)", None),
]


def _ko_distance_pattern(text: str) -> tuple[str | None, float | None]:
    m = re.search(r"([\d.]+)\s*m", text)
    if not m:
        return None, None
    return str(m.group(1)), float(m.group(1)) / 100.0 if m.group(1).startswith("0.") else float(m.group(1))


def parse_rule(text: str) -> SafetyRule | None:
    s = text.strip()
    subj = _match_subject(s)
    low = s.lower()
    if "안에 들어오면" in s or "안으로 들어오면" in s:
        dist = _ko_distance_pattern(s)[1]
        on_, target = match_ko_when(s)
        return SafetyRule(
            text=s,
            subject=subj or "person",
            target=target or (subj and "vehicle") or "vehicle",
            condition={"machine_active": on_, "distance_lt": dist} if dist else {"machine_active": on_},
            severity=_ko_severity(s),
            rule_id="nl-ko",
        )
    for pat, _tar in _ACTION_RULES:
        m = re.search(pat, low)
        if m:
            groups = m.groups()
            if "within" in low and "m of" in low:
                dist = float(re.search(r"within\s+([\d.]+)\s*m", low).group(1))
                sub2 = _match_subject_in(low, groups)
                tar = sub2[1] if len(sub2) > 1 else "vehicle"
                cond = {"distance_lt": dist}
                if " active" in low or " active" in s or "operating" in low or "running" in low:
                    cond["machine_active"] = True
                return SafetyRule(
                    text=s, subject=sub2[0], target=tar, condition=cond,
                    severity=_en_severity(s), rule_id="nl-en",
                )
            if "distance between" in low:
                d = float(groups[0])
                return SafetyRule(
                    text=s, subject=_map_subj(groups[1]), target=_map_subj(groups[2]),
                    condition={"distance_gt": d}, severity=_en_severity(s), rule_id="nl-en",
                )
    return None


def _map_subj(word: str) -> str:
    return _SUBJECT_ALIAS.get(word, "person" if word in ("person", "worker") else "vehicle")


def _match_subject(text: str) -> str | None:
    for alias, cls in _SUBJECT_ALIAS.items():
        if alias in text.lower():
            return cls
    return None


def _match_subject_in(text: str, groups: tuple) -> tuple[str, str] | None:
    words = [g for g in groups if g and g.strip()]
    parsed = [_map_subj(w) for w in words if w.isalnum()]
    if len(parsed) >= 2:
        return parsed[0], parsed[1]
    return ("person", "vehicle")


def match_ko_when(text: str) -> tuple[bool, str]:
    active = any(k in text for k in ("작동 중", "운전 중", "가동 중", "운행 중", "움직이는", "작동하는"))
    for alias, cls in _SUBJECT_ALIAS.items():
        if alias in text and cls != "person":
            return active, cls
    return active, "machine"


def _ko_severity(text: str) -> str:
    if any(k in text for k in ("절대", "반드시", "무조건", "엄격히")):
        return "high"
    return "medium" if any(k in text for k in ("가능하면", "되도록")) else "medium"


def _en_severity(text: str) -> str:
    low = text.lower()
    if any(k in low for k in ("must never", "strictly", "absolute")):
        return "high"
    if any(k in low for k in ("avoid", "preferably")):
        return "medium"
    return "medium"
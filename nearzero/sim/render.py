"""Top-down renderer for the simulated world (demo "camera")."""
from __future__ import annotations

import numpy as np
import cv2

from ..risk.engine import ScenarioRisk
from ..risk.zones import RestrictedZone

PPX = 60


def _to_px(x: float, y: float, x0: float, y0: float) -> tuple[int, int]:
    return int((x - x0) * PPX), int((y - y0) * PPX)


def render_scene(
    states: list[dict],
    zones: list[RestrictedZone] | None = None,
    risk: ScenarioRisk | None = None,
    forecasts: dict[int, dict] | None = None,
    assignments: dict[str, int] | None = None,
) -> np.ndarray:
    """Draw the current world frame (top-down) into a BGR image."""
    current = states[-1]
    xs = [a["x"] for a in current["agents"]]
    ys = [a["y"] for a in current["agents"]]
    if xs:
        x0, x1 = min(xs) - 2.0, max(xs) + 2.0
        y0, y1 = min(ys) - 2.0, max(ys) + 2.0
    else:
        x0, y0, x1, y1 = -2.0, -2.0, 12.0, 10.0
    w = int((x1 - x0) * PPX)
    h = int((y1 - y0) * PPX)
    frame = np.full((max(h, 2), max(w, 2), 3), 30, dtype=np.uint8)

    gx0 = int((0 - x0) * PPX)
    gy0 = int((0 - y0) * PPX)
    for i in range(0, max(w, 1), PPX):
        cv2.line(frame, (i, 0), (i, h), (45, 45, 48), 1)
    for j in range(0, max(h, 1), PPX):
        cv2.line(frame, (0, j), (w, j), (45, 45, 48), 1)
    if 0 <= gx0 < w and 0 <= gy0 < h:
        cv2.line(frame, (gx0, 0), (gx0, h), (55, 55, 60), 2)
        cv2.line(frame, (0, gy0), (w, gy0), (55, 55, 60), 2)
        cv2.putText(frame, "0", (gx0 + 3, gy0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 160), 1)

    for z in zones or []:
        pts = np.array([_to_px(p[0], p[1], x0, y0) for p in z.polygon]).reshape(-1, 1, 2)
        cv2.polylines(frame, [pts], True, (0, 80, 255), 2)
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], (70, 40, 90))
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    for a in current["agents"]:
        px, py = _to_px(a["x"], a["y"], x0, y0)
        r = max(6, int(a["radius"] * PPX))
        color = (46, 46, 235) if a["cls"] == "person" else (46, 190, 245)
        cv2.circle(frame, (px, py), r, color, -1)
        cv2.circle(frame, (px, py), r, (0, 0, 0), 1)
        sp = np.hypot(a["vx"], a["vy"])
        if sp > 0.02:
            ex = int(px + a["vx"] / sp * (r + 14))
            ey = int(py + a["vy"] / sp * (r + 14))
            cv2.arrowedLine(frame, (px, py), (ex, ey), (255, 255, 255), 2, cv2.LINE_AA, tipLength=0.3)
        tag = assignments.get(a["name"], a["name"]) if assignments else a["name"]
        cv2.putText(frame, f"{a['cls'][0].upper()}#{tag}", (px - 20, py - r - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 225), 1, cv2.LINE_AA)

    if forecasts:
        for tid, traj in forecasts.items():
            pts = np.array([_to_px(p[0], p[1], x0, y0) for p in traj["positions"][::max(1, len(traj["positions"]) // 14)]])
            color = (46, 46, 235) if traj["cls"] == "person" else (46, 190, 245)
            cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, color, 1, cv2.LINE_AA)

    if risk and risk.collision_point:
        px, py = _to_px(risk.collision_point[0], risk.collision_point[1], x0, y0)
        rr = 14
        cv2.line(frame, (px - rr, py - rr), (px + rr, py + rr), (40, 40, 230), 3)
        cv2.line(frame, (px - rr, py + rr), (px + rr, py - rr), (40, 40, 230), 3)
        cv2.putText(frame, "IMPACT", (px - 24, py - rr - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 40, 230), 1, cv2.LINE_AA)

    return frame
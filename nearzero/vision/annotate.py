"""Live-frame annotation: detection boxes, IDs, velocity, forecasts, zones."""
from __future__ import annotations

import numpy as np
import cv2

from ..core.geometry import Homography


HEX2RGB = {
    "red": (40, 40, 230),
    "orange": (0, 140, 255),
    "amber": (0, 200, 255),
    "green": (80, 220, 80),
    "cyan": (230, 220, 40),
    "purple": (200, 60, 180),
}


def _color(name: str) -> tuple[int, int, int]:
    return HEX2RGB.get(name, (255, 255, 255))


def draw_predictions(
    frame: np.ndarray,
    homography: Homography,
    traj: dict,
    color: tuple[int, int, int],
) -> None:
    """Draw expected trajectory + uncertainty band in world->pixel space."""
    pos = traj["positions"]
    if len(pos) < 2:
        return
    pts = np.array([homography.to_pixel(float(x0), float(y0)) for x0, y0 in pos])
    pts = pts.astype(int)
    for i in range(len(pts) - 1):
        cv2.line(frame, tuple(pts[i]), tuple(pts[i + 1]), color, 2, cv2.LINE_AA)
        cv2.circle(frame, tuple(pts[i]), 2, color, -1)
    samples = traj.get("samples")
    if samples is not None and samples.shape[0] > 1:
        std = samples.std(axis=0)[:, :2].mean(axis=1)
        for k in range(0, len(pos), max(1, len(pos) // 5)):
            px, py = homography.to_pixel(float(pos[k][0]), float(pos[k][1]))
            s = max(2, int(std[min(k, std.size - 1)] * (frame.shape[1] / 10.0)))
            cv2.circle(frame, (int(px), int(py)), max(4, min(s, 80)), _alpha(color, 0.25), 1)


def _alpha(color: tuple[int, int, int], f: float) -> tuple[int, int, int]:
    return tuple(int(c * f + 255 * (1 - f)) for c in color)


def draw_box(
    frame: np.ndarray,
    box: tuple[float, float, float, float],
    label: str,
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = [int(v) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (20, 20, 20), 1, cv2.LINE_AA)


def draw_velocity(
    frame: np.ndarray,
    homography: Homography,
    wpos: tuple[float, float],
    vel: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    px, py = homography.to_pixel(wpos[0], wpos[1])
    scale = frame.shape[1] / 12.0
    ex = int(px + vel[0] * scale * 0.4)
    ey = int(py + vel[1] * scale * 0.4)
    cv2.arrowedLine(frame, (int(px), int(py)), (ex, ey), color, 2, cv2.LINE_AA, tipLength=0.25)


def draw_zones(frame: np.ndarray, homography: Homography, zones) -> None:
    for z in zones:
        if not getattr(z, "active", True):
            continue
        pts = np.array([homography.to_pixel(float(p[0]), float(p[1])) for p in z.polygon], dtype=int)
        cv2.polylines(frame, [pts], True, (0, 60, 255), 2, cv2.LINE_AA)
        cv2.fillPoly(frame, [pts], (55, 45, 90))
        cx, cy = pts.mean(axis=0)
        cv2.putText(frame, z.name, (int(cx), int(cy)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (220, 220, 220), 1, cv2.LINE_AA)


def draw_collision_point(frame: np.ndarray, homography: Homography, wx: float, wy: float) -> None:
    px, py = homography.to_pixel(wx, wy)
    r = 12
    cv2.line(frame, (int(px - r), int(py - r)), (int(px + r), int(py + r)), (40, 40, 230), 3)
    cv2.line(frame, (int(px - r), int(py + r)), (int(px + r), int(py - r)), (40, 40, 230), 3)
    cv2.circle(frame, (int(px), int(py)), r, (40, 40, 230), 2)


def draw_top_bar(frame: np.ndarray, lines: list[tuple[str, str]]) -> None:
    h = 34 + 20 * len(lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], h), (14, 16, 20), -1)
    cv2.addWeighted(overlay, 0.92, frame, 0.08, 0, frame)
    for i, (k, v) in enumerate(lines):
        y = 24 + 20 * i
        cv2.putText(frame, k, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 170, 190), 1, cv2.LINE_AA)
        cv2.putText(frame, v, (150, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (240, 246, 255), 1, cv2.LINE_AA)
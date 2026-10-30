"""Restricted / safety zones in world coordinates."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RestrictedZone:
    name: str
    polygon: np.ndarray
    active: bool = True
    owner_cls: str | None = None


def rectangle_zone(name: str, x0: float, y0: float, w: float, h: float) -> RestrictedZone:
    return RestrictedZone(
        name, np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]])
    )


def circle_zone(name: str, cx: float, cy: float, r: float, n: int = 24) -> RestrictedZone:
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return RestrictedZone(name, np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles)], axis=1))
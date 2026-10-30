"""Perception entry points: pixel (camera) perception and simulated perception.

PixelPerception maps detection boxes to the ground plane through the floor
homography and drives tracklets; SimulatedPerception feeds ground-truth world
states (optionally with noise) straight into the same tracking core, which is
why the whole P0 stack is testable and demo-able without a camera.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

from ..config import PerceptionConfig
from ..core.geometry import Homography, foot_point
from ..core.tracks import Object
from ..sim.world import World
from .detector import Detector, RawDetection

logger = logging.getLogger("nearzero.vision")


def is_person(cls: str) -> bool:
    return cls == "person"


class Perception:
    """Base interface: one step of perception yields Objects (world coords)."""

    def step(self, frame: np.ndarray | None, t: float, dt: float, **kw) -> list[Object]:
        raise NotImplementedError


class PixelPerception(Perception):
    def __init__(self, cfg: PerceptionConfig, homography: Homography):
        self.cfg = cfg
        self.homography = homography
        self.detector = Detector(cfg)

    def step(self, frame, t, dt):
        if frame is None or not self.detector.available:
            return []
        if self.cfg.use_bytetrack:
            dets = self.detector.track(frame)
        else:
            dets = self.detector.predict(frame)
        return self._to_objects(dets, t)

    def _to_objects(self, dets: list[RawDetection], t: float) -> list[Object]:
        objs: list[Object] = []
        for det in dets:
            fx, fy = foot_point(det.box)
            wx, wy = self.homography.to_world(fx, fy)
            objs.append(
                Object(
                    cls=det.cls,
                    conf=det.conf,
                    x=wx,
                    y=wy,
                    box=det.box,
                    t=float(t),
                )
            )
        return objs


def blind_homography() -> Homography:
    """Identity homography used in simulated mode: world meters == pixel meters."""
    return Homography()


class SimulatedPerception(Perception):
    """Wraps a simulator world; detections are the true agent states (+ noise)."""

    def __init__(self, world: World, noise_xy: float = 0.04):
        self.world = world
        self.noise_xy = noise_xy
        self._rng = np.random.default_rng(0)

    def step(self, frame, t, dt, states: list[dict] | None = None):
        current = states[-1] if states else self.world.state()
        objs: list[Object] = []
        for a in current["agents"]:
            if a["cls"] not in ("person", "vehicle", "cart", "forklift", "machine"):
                continue
            nx = a["x"] + self._rng.normal(0.0, self.noise_xy)
            ny = a["y"] + self._rng.normal(0.0, self.noise_xy)
            objs.append(
                Object(cls=a["cls"], conf=0.99, x=float(nx), y=float(ny), t=float(t))
            )
        return objs


class VideoSource:
    """Threaded cv2.VideoCapture wrapper pacing playback to a target FPS."""

    def __init__(self, path_or_index=0, fps: float = 25.0, loop: bool = False):
        self._cap = cv2.VideoCapture(path_or_index)
        self.fps = fps
        self.loop = loop
        self._open = bool(self._cap.isOpened())

    @property
    def is_open(self) -> bool:
        return self._open

    @property
    def size(self) -> tuple[int, int]:
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return w, h

    def read(self, t: float) -> np.ndarray | None:
        if not self._open:
            return None
        ok, frame = self._cap.read()
        if not ok:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
            if not ok:
                self._open = False
                return None
        return frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()


def has_webcam() -> bool:
    cap = cv2.VideoCapture(0)
    ok = cap.isOpened()
    cap.release()
    return ok
"""Runtime drivers: simulated-scenario runner and camera/video runner."""
from __future__ import annotations

import logging
import threading
import time

import numpy as np

from ..config import Config
from ..events.store import EventStore
from ..sim.render import render_scene
from ..sim.scenarios import SCENARIO_NAMES, make_scenario
from ..vision.perception import PixelPerception, SimulatedPerception, VideoSource
from ..app.pipeline import SafetyPipeline

logger = logging.getLogger("nearzero.runner")


class BaseRunner:
    def __init__(self, cfg: Config, mode: str, store: EventStore | None = None):
        self.cfg = cfg
        self.mode = mode
        self.pipeline = SafetyPipeline(cfg, store=store)
        self.latest_snapshot = None
        self.latest_jpeg = None
        self._stop = False
        self._t = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop = True

    def _run(self) -> None:
        raise NotImplementedError

    def _set_frame(self, snapshot, jpeg_quality: int) -> None:
        self.latest_snapshot = snapshot
        if snapshot.frame is not None:
            import cv2

            ok, buf = cv2.imencode(
                ".jpg", snapshot.frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            )
            if ok:
                self.latest_jpeg = buf.tobytes()


class SimRunner(BaseRunner):
    """Deterministic scenario playback through the full pipeline."""

    def __init__(self, cfg: Config, scenario: str = "demo_avoided", store=None, loop: bool = True):
        super().__init__(cfg, "sim", store)
        self.scenario = scenario
        self.loop = loop
        self._i = 0
        self._assignments: dict[str, int] = {}
        self._executed: str | None = None
        self._executed_at: int | None = None
        if scenario not in SCENARIO_NAMES:
            raise KeyError(f"unknown scenario '{scenario}'; available: {SCENARIO_NAMES}")
        _, self.states, self.collisions = make_scenario(scenario)

    def reset(self) -> None:
        _, self.states, self.collisions = make_scenario(self.scenario)
        self._i = 0
        self._executed = None
        self._executed_at = None

    def tick_for(self, scenario_t: float) -> bool:
        idx = int(round(scenario_t / self.cfg.prediction.dt_s))
        if idx >= len(self.states):
            return False
        self._advance(idx)
        return True

    def _advance(self, idx: int) -> None:
        self._i = idx
        state = self.states[idx]
        t = float(state["t"])
        dt = self.cfg.prediction.dt_s
        frame = render_scene(self.states[: idx + 1], self.pipeline.zones)
        perception = SimulatedPerception.__new__(SimulatedPerception)
        perception.world = None
        perception.noise_xy = 0.04
        perception._rng = np.random.default_rng(0)
        objects = perception.step(frame, t, dt, self.states[: idx + 1])
        snap = self.pipeline.step(frame, objects, t, dt)
        self._maybe_execute(snap, idx, dt)
        self._assignments = _assign_names(self.states[: idx + 1], snap.tracks)
        overlay = render_scene(self.states[: idx + 1], self.pipeline.zones, snap.risk,
                               {f["track_id"]: f for f in snap.forecasts}, self._assignments)
        snap.frame = overlay
        self._set_frame(snap, self.cfg.server.mjpeg_quality)

    def _maybe_execute(self, snap, idx: int, dt: float) -> None:
        """Execute a recommended vehicle maneuver on the world timeline (PREVENT)."""
        if self._executed:
            return
        best = snap.best_intervention
        if not best:
            return
        if not best.intervention.startswith("VEHICLE_"):
            return
        if snap.risk.score < self.cfg.risk.intervention_risk:
            return
        if idx + 1 >= len(self.states):
            return
        direction = -1.0 if best.intervention.endswith("RIGHT") else 1.0
        from ..sim.scenarios import apply_vehicle_dodge

        apply_vehicle_dodge(self.states, idx, dt, lateral_speed=direction * 1.9)
        self._executed = best.intervention
        self._executed_at = idx

    def _run(self) -> None:
        while not self._stop:
            dt = self.cfg.prediction.dt_s
            cycle_t = 0.0
            while not self._stop:
                idx = int(round(cycle_t / dt))
                if idx >= len(self.states):
                    break
                self._advance(idx)
                if self.cfg.server.port <= 0:
                    time.sleep(dt)
                else:
                    time.sleep(dt * 0.15)
                cycle_t += dt
            if not self.loop:
                break
            self.reset()
            time.sleep(1.0)


class CameraRunner(BaseRunner):
    """Webcam or video-file pipeline (needs PyTorch/ultralytics stack and
    optional floor homography calibration)."""

    def __init__(self, cfg: Config, source: str | int = 0, store=None, loop: bool = False):
        super().__init__(cfg, "webcam" if isinstance(source, int) else "video", store)
        self.source = source
        self.loop = loop
        self.cap = VideoSource(source, fps=cfg.pipeline.fps_in, loop=loop)
        self.calibrated = False
        self._perception = None

    def _ensure_perception(self) -> None:
        if self._perception is None:
            from ..core.geometry import load_homography

            h = load_homography(self.cfg.perception.homography_json)
            if h is not None:
                self.pipeline.homography = h
                self.calibrated = True
            else:
                h = self.pipeline.homography
                self.calibrated = False
            self._perception = PixelPerception(self.cfg.perception, h)

    def _run(self) -> None:
        self._ensure_perception()
        frame_rate = self.cfg.prediction.dt_s
        t = 0.0
        while not self._stop:
            t0 = time.perf_counter()
            frame = self.cap.read(t)
            if frame is None:
                if self.loop:
                    continue
                break
            frame = self.cfg.pipeline.privacy_mode and _blur(frame) or frame
            objects = self._perception.step(frame, t, frame_rate)
            snap = self.pipeline.step(frame, objects, t, frame_rate, ext_ids=None)
            snap.frame = self.pipeline.annotate(frame, snap)
            self._set_frame(snap, self.cfg.server.mjpeg_quality)
            t += frame_rate
            elapsed = time.perf_counter() - t0
            sleep = max(0.0, frame_rate - elapsed)
            time.sleep(sleep)
        self.cap.release()


def _blur(frame):
    from ..vision.privacy import blur_faces

    return blur_faces(frame)


def _assign_names(states: list[dict], track_dicts: list[dict]) -> dict[str, int]:
    current = states[-1]["agents"]
    assignments: dict[str, int] = {}
    for agent in current:
        best_d = 1e9
        best_id = None
        for tr in track_dicts:
            d = float(np.hypot(tr["pos"][0] - agent["x"], tr["pos"][1] - agent["y"]))
            if d < best_d and tr["cls"] == agent["cls"]:
                best_d = d
                best_id = tr["track_id"]
        if best_id is not None:
            assignments[agent["name"]] = best_id
    return assignments
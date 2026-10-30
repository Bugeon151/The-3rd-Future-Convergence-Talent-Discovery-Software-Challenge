"""Central configuration for NEAR-ZERO X.

All tunable numbers live here (no magic numbers in modules). Values are chosen
from published safety/simulation literature where possible and documented in
docs/RISK_MODEL.md.
"""
from __future__ import annotations

import dataclasses
import json
import os
from typing import Any

SEED = 2026


@dataclasses.dataclass
class PerceptionConfig:
    model_name: str = "yolov8n.pt"
    conf_thres: float = 0.35
    iou_thres: float = 0.45
    image_size: int = 640
    person_classes: tuple[str, ...] = ("person",)
    vehicle_classes: tuple[str, ...] = (
        "car",
        "truck",
        "bus",
        "motorcycle",
        "bicycle",
    )
    object_classes: tuple[str, ...] = ("forklift", "machine")
    max_age_frames: int = 25
    min_hits: int = 3
    gating_meters: float = 2.5
    use_bytetrack: bool = True
    device: str = "auto"
    homography_json: str = "configs/homography.json"


@dataclasses.dataclass
class PredictionConfig:
    horizon_s: float = 4.0
    dt_s: float = 0.1
    n_samples: int = 100
    velocity_noise: tuple[float, float] = (0.12, 0.12)
    angular_noise: float = 0.30
    accel_noise: float = 0.25
    kalman_process_scale: float = 1.2


@dataclasses.dataclass
class RiskConfig:
    person_radius_m: float = 0.35
    vehicle_radius_m: float = 0.60
    safety_margin_m: float = 0.20
    ttc_threshold_s: float = 3.0
    dist_threshold_m: float = 2.0
    mc_samples: int = 200
    ttc_x0: float = 2.5
    ttc_k: float = 1.4
    pmd_x0: float = 1.5
    pmd_k: float = 3.0
    score_weights: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "collision_probability": 1.0,
            "ttc": 0.9,
            "predicted_min_distance": 0.7,
            "zone_intrusion": 0.6,
            "rule_violation": 0.5,
        }
    )
    thresholds: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "safe": 25.0,
            "caution": 50.0,
            "warning": 75.0,
        }
    )
    warn_risk: float = 45.0
    intervention_risk: float = 48.0
    prevention_target_risk: float = 25.0


@dataclasses.dataclass
class PlannerConfig:
    interventions: tuple[str, ...] = (
        "NO_ACTION",
        "STOP_VEHICLE",
        "SLOW_VEHICLE_50",
        "SLOW_VEHICLE_30",
        "HUMAN_STOP",
        "HUMAN_LEFT",
        "HUMAN_RIGHT",
        "VEHICLE_LEFT",
        "VEHICLE_RIGHT",
    )
    reaction_s: float = 0.7
    stop_decel: float = 2.0
    lateral_shift_m: float = 1.2
    dodge_time_s: float = 0.6
    cost_weight: float = 0.35
    feasibility_min: float = 0.15
    feasibility_table: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "NO_ACTION": 1.0,
            "STOP_VEHICLE": 0.95,
            "SLOW_VEHICLE_50": 0.9,
            "SLOW_VEHICLE_30": 0.9,
            "HUMAN_STOP": 0.75,
            "HUMAN_LEFT": 0.6,
            "HUMAN_RIGHT": 0.6,
            "VEHICLE_LEFT": 0.45,
            "VEHICLE_RIGHT": 0.45,
        }
    )
    cost_table: dict[str, float] = dataclasses.field(
        default_factory=lambda: {
            "NO_ACTION": 0.0,
            "STOP_VEHICLE": 0.55,
            "SLOW_VEHICLE_50": 0.35,
            "SLOW_VEHICLE_30": 0.25,
            "HUMAN_STOP": 0.3,
            "HUMAN_LEFT": 0.18,
            "HUMAN_RIGHT": 0.18,
            "VEHICLE_LEFT": 0.42,
            "VEHICLE_RIGHT": 0.42,
        }
    )


@dataclasses.dataclass
class VerificationConfig:
    verification_window_s: float = 3.0
    resolved_on_risk_drop: float = 0.5
    required_confidence: float = 0.35


@dataclasses.dataclass
class EventConfig:
    db_path: str = "data/nearzero.db"


@dataclasses.dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 5000
    mjpeg_quality: int = 70


@dataclasses.dataclass
class PipelineConfig:
    fps_in: int = 20
    annotate_ui: bool = True
    privacy_mode: bool = False
    log_events: bool = True
    record_raw_video: bool = False


@dataclasses.dataclass
class Config:
    perception: PerceptionConfig = dataclasses.field(default_factory=PerceptionConfig)
    prediction: PredictionConfig = dataclasses.field(default_factory=PredictionConfig)
    risk: RiskConfig = dataclasses.field(default_factory=RiskConfig)
    planner: PlannerConfig = dataclasses.field(default_factory=PlannerConfig)
    verification: VerificationConfig = dataclasses.field(default_factory=VerificationConfig)
    events: EventConfig = dataclasses.field(default_factory=EventConfig)
    server: ServerConfig = dataclasses.field(default_factory=ServerConfig)
    pipeline: PipelineConfig = dataclasses.field(default_factory=PipelineConfig)
    rules_path: str = "configs/rules.json"
    zones_path: str = "configs/zones.json"

    @classmethod
    def from_json(cls, path: str) -> "Config":
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Config":
        cfg = cls()
        for field in dataclasses.fields(cls):
            if field.name in data and isinstance(data[field.name], dict):
                sub = getattr(cfg, field.name)
                for k, v in data[field.name].items():
                    if hasattr(sub, k):
                        setattr(sub, k, v)
        return cfg

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    def to_dict(self) -> dict[str, Any]:
        out = {}
        for field in dataclasses.fields(cls if False is False else self.__class__):
            sub = getattr(self, field.name)
            out[field.name] = dataclasses.asdict(sub)
        return out


def load_config(path: str | None = None) -> Config:
    if path and os.path.exists(path):
        return Config.from_json(path)
    return Config()
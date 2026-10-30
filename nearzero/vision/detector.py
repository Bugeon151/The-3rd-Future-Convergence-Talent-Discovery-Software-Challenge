"""Object detection via Ultralytics YOLO with CPU/CUDA auto-selection."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np

from ..config import PerceptionConfig

logger = logging.getLogger("nearzero.vision.detector")


@dataclass
class RawDetection:
    box: tuple[float, float, float, float]
    cls: str
    conf: float
    ext_id: int | None = None


def _ensure_torchvision_nms() -> None:
    """Fallback pure-torch NMS when the compiled torchvision op is unavailable."""
    try:
        import torch
        import torchvision.ops  # noqa: F401
    except Exception:
        return
    if getattr(torch.ops, "torchvision", None) is not None and hasattr(
        torch.ops.torchvision, "nms"
    ):
        return

    def _box_iou_pure(boxes_a, boxes_b):
        lt = torch.max(boxes_a[:, None, :2], boxes_b[None, :, :2])
        rb = torch.min(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        wh = (rb - lt).clamp(min=0.0)
        inter = wh[..., 0] * wh[..., 1]
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / union.clamp(min=1e-9)

    def _nms_pure(boxes, scores, iou_threshold):
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.int64, device=boxes.device)
        order = torch.argsort(scores, descending=True)
        keep = []
        idx = order
        while idx.numel() > 0:
            current = int(idx[0].item())
            keep.append(current)
            if idx.numel() == 1:
                break
            rest = idx[1:]
            iou = _box_iou_pure(boxes[current : current + 1], boxes[rest])[0]
            idx = rest[iou <= iou_threshold]
        if not keep:
            return torch.empty(0, dtype=torch.int64, device=boxes.device)
        return torch.tensor(keep, dtype=torch.int64, device=boxes.device)

    import torchvision.ops

    torchvision.ops.nms = _nms_pure
    if hasattr(torchvision.ops, "boxes"):
        torchvision.ops.boxes.nms = _nms_pure


class Detector:
    def __init__(self, cfg: PerceptionConfig, weights: str | None = None):
        self.cfg = cfg
        self.available = False
        self.device = "cpu"
        self._model = None
        self._names: list[str] = []
        _ensure_torchvision_nms()
        self._load(weights or cfg.model_name)

    def _load(self, weights: str) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            logger.warning("ultralytics unavailable: %s", exc)
            return
        if not os.path.exists(weights):
            alt = os.path.join(os.path.dirname(__file__), "models", weights)
            if os.path.exists(alt):
                weights = alt
        try:
            model = YOLO(weights)
            self._model = model
            self._names = list(model.names.values())
            self.device = self._pick_device()
            self.available = True
            logger.info("detector loaded %s on %s", weights, self.device)
        except Exception as exc:  # pragma: no cover
            logger.warning("failed to load detector %s: %s", weights, exc)
            self.available = False

    def _pick_device(self) -> str:
        if self.cfg.device == "cpu":
            return "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda:0"
        except Exception:
            pass
        return "cpu"

    def _keep(self, cls_name: str) -> bool:
        if cls_name in self.cfg.person_classes:
            return True
        if cls_name in self.cfg.vehicle_classes:
            return True
        return cls_name in self.cfg.object_classes

    def predict(self, frame: np.ndarray) -> list[RawDetection]:
        if not self.available:
            return []
        res = self._model.predict(
            frame,
            conf=self.cfg.conf_thres,
            iou=self.cfg.iou_thres,
            imgsz=self.cfg.image_size,
            device=self.device,
            verbose=False,
        )
        dets = extract_raw(res, self._names, self._keep)
        return dets

    def track(self, frame: np.ndarray) -> list[RawDetection]:
        if not self.available:
            return []
        try:
            res = self._model.track(
                frame,
                conf=self.cfg.conf_thres,
                iou=self.cfg.iou_thres,
                imgsz=self.cfg.image_size,
                device=self.device,
                persist=True,
                tracker="bytetrack.yaml",
                classes=_class_indices(self._names, self.cfg),
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("bytetrack failed (%s), falling back to predict", exc)
            return self.predict(frame)
        dets = extract_raw(res, self._names, self._keep)
        return dets


def _class_indices(names: list[str], cfg: PerceptionConfig) -> list[int]:
    keep = set(cfg.person_classes) | set(cfg.vehicle_classes) | set(cfg.object_classes)
    return [i for i, n in enumerate(names) if n in keep]


def extract_raw(results, names: list[str], keep: callable) -> list[RawDetection]:
    dets: list[RawDetection] = []
    for r in results:
        if r.boxes is None:
            continue
        for bi, box in enumerate(r.boxes.xyxy.cpu().numpy()):
            cls_name = names[int(r.boxes.cls[bi].item())]
            if not keep(cls_name):
                continue
            conf = float(r.boxes.conf[bi]) if r.boxes.conf is not None else 0.0
            tid = None
            if hasattr(r.boxes, "id") and r.boxes.id is not None:
                tid = int(r.boxes.id.cpu().numpy()[bi])
            dets.append(
                RawDetection(
                    box=tuple(float(v) for v in box),
                    cls=cls_name,
                    conf=conf,
                    ext_id=tid,
                )
            )
    return dets
from __future__ import annotations

from typing import Dict, List
import numpy as np

from counter.backends.base import DetectorBackend
from counter.domain.types import Detection

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover
    YOLO = None

class YoloBackend(DetectorBackend):
    def __init__(self, weights: str, device: str = 'cpu', conf: float = 0.35, iou: float = 0.35):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Install extras: uv pip install -e '.[predict]'")
        self.model = YOLO(weights)
        self.device = device
        self.conf = conf
        self.iou = iou

    def labels(self) -> Dict[int, str]:
        try:
            return dict(self.model.names)
        except Exception:
            return {}

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        res = self.model.predict(frame_bgr, verbose=False, device=self.device, conf=self.conf, iou=self.iou)
        out: List[Detection] = []
        if not res or res[0].boxes is None:
            return out
        names = self.labels()
        boxes = res[0].boxes.xyxy.cpu().numpy()
        cls = res[0].boxes.cls.cpu().numpy().astype(int)
        confs = res[0].boxes.conf.cpu().numpy()
        for bbox, cid, score in zip(boxes, cls, confs):
            out.append(Detection(tuple(map(float, bbox)), float(score), int(cid), names.get(int(cid), str(cid))))
        return out

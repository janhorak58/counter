from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from counter.backends.base import DetectorBackend
from counter.domain.types import Detection

try:  # pragma: no cover
    from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano  # type: ignore
except Exception:  # pragma: no cover
    RFDETRBase = RFDETRLarge = RFDETRMedium = RFDETRSmall = RFDETRNano = None


def _pick_model_cls(size: str | None):
    size = (size or "base").lower()
    if size == "nano":
        return RFDETRNano
    if size == "small":
        return RFDETRSmall
    if size == "medium":
        return RFDETRMedium
    if size == "large":
        return RFDETRLarge
    return RFDETRBase


class RfDetrBackend(DetectorBackend):
    """RF-DETR inference wrapper (Roboflow `rfdetr` library).

    This backend returns per-frame detections WITHOUT tracking IDs.
    For counting you should use TrackProviders (see `tracking/rfdetr_roboflow.py`).
    """

    def __init__(self, weights: str = "", size: str | None = None, conf: float = 0.35):
        if RFDETRBase is None:
            raise ImportError("rfdetr is not installed. Install extras: uv pip install -e '.[predict]'")
        ModelCls = _pick_model_cls(size)
        if ModelCls is None:
            raise ImportError("rfdetr model classes not found. Check your rfdetr install/version.")

        w = (weights or "").strip()
        if w and w.lower() not in {"pretrained", "default"}:
            self.model = ModelCls(pretrain_weights=w)
        else:
            self.model = ModelCls()
        self.conf = float(conf)

        self._names: Optional[Dict[int, str]] = None
        try:
            names = getattr(self.model, "class_names", None) or getattr(self.model, "class_names_", None)
            if isinstance(names, (list, tuple)):
                self._names = {i: str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self._names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            self._names = None

    def labels(self) -> Dict[int, str]:
        return self._names or {}

    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        dets = self.model.predict(frame_rgb, threshold=self.conf)
        if not hasattr(dets, "xyxy"):
            return []
        xyxy = dets.xyxy
        class_id = getattr(dets, "class_id", None)
        confidence = getattr(dets, "confidence", None)
        if class_id is None:
            class_id = np.zeros((len(xyxy),), dtype=int)
        if confidence is None:
            confidence = np.ones((len(xyxy),), dtype=float)

        out: List[Detection] = []
        for box, cid, sc in zip(xyxy, class_id, confidence):
            cid_i = int(cid) if cid is not None else -1
            name = self._names.get(cid_i, str(cid_i)) if self._names else str(cid_i)
            out.append(
                Detection(
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(sc),
                    raw_class_id=cid_i,
                    raw_class_name=name,
                )
            )
        return out

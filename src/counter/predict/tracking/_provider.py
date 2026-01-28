from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from counter.predict.tracking.providers import TrackProvider
from counter.predict.types import RawTrack

try:  # pragma: no cover
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None


def _norm_optional_str(x: Optional[str]) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    if s.lower() in {"none", "null"}:
        return None
    return s


class UltralyticsYoloTrackProvider(TrackProvider):
    def __init__(
        self,
        *,
        weights: str,
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.35,
        tracking_enabled: bool = True,
        tracker_yaml: Optional[str] = None,
        tracker_params: Optional[Dict[str, Any]] = None,
    ):
        if YOLO is None:
            raise ImportError(
                "Ultralytics is not installed. Install it, e.g.: uv pip install ultralytics"
            )

        self._weights = str(weights)
        self._device = str(device)
        self._conf = float(conf)
        self._iou = float(iou)

        self._tracking_enabled = bool(tracking_enabled)
        self._tracker_yaml = _norm_optional_str(tracker_yaml)
        self._tracker_params = tracker_params or {}

        self.model = YOLO(self._weights)
        self.model.to(self._device)

    def reset(self) -> None:
        # Ultralytics keeps tracking state internally; simplest is to re-init the model.
        self.model = YOLO(self._weights)
        self.model.to(self._device)

    def get_label_map(self) -> Optional[Dict[int, str]]:
        try:
            names = getattr(self.model, "names", None)
            if isinstance(names, dict):
                return {int(k): str(v) for k, v in names.items()}
        except Exception:
            return None
        return None

    def update(self, frame_bgr) -> List[RawTrack]:
        # --- run model ---
        if self._tracking_enabled:
            kwargs: Dict[str, Any] = dict(
                conf=self._conf,
                iou=self._iou,
                persist=True,
                verbose=False,
                **self._tracker_params,
            )
            # IMPORTANT: don't pass tracker at all if it's None/empty, otherwise Ultralytics tries to open file "None"
            if self._tracker_yaml is not None:
                kwargs["tracker"] = self._tracker_yaml

            res_list = self.model.track(frame_bgr, **kwargs)
        else:
            res_list = self.model.predict(
                frame_bgr,
                conf=self._conf,
                iou=self._iou,
                verbose=False,
            )

        out: List[RawTrack] = []
        if not res_list:
            return out

        res = res_list[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            return out

        xyxy = boxes.xyxy.cpu().numpy()  # (N,4)
        confs = boxes.conf.cpu().numpy().astype(float).tolist()
        class_ids = boxes.cls.cpu().numpy().astype(int).tolist()

        # tracking ids (may be missing)
        if getattr(boxes, "id", None) is not None:
            track_ids = boxes.id.cpu().numpy().astype(int).tolist()
        else:
            # fallback: stable per-frame ids (not real tracking)
            track_ids = list(range(len(class_ids)))

        names = self.get_label_map() or {}

        for tid, bb, sc, cid in zip(track_ids, xyxy, confs, class_ids):
            x1, y1, x2, y2 = [float(x) for x in bb]
            out.append(
                RawTrack(
                    track_id=int(tid),
                    bbox=(x1, y1, x2, y2),
                    score=float(sc),
                    raw_class_id=int(cid),
                    raw_class_name=str(names.get(int(cid), str(cid))),
                )
            )
        return out

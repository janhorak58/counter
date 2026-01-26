from __future__ import annotations

from typing import List, Optional

import numpy as np

from counter.tracking.providers import TrackProvider, RawTrack

try:  # pragma: no cover
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None

class UltralyticsYoloTrackProvider(TrackProvider):
    """Uses ultralytics YOLO.track() to get track IDs + boxes.

    This intentionally reuses Ultralytics' built-in ByteTrack integration.
    It is pragmatic and matches your legacy behavior.
    """

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        conf: float = 0.35,
        iou: float = 0.35,
        tracker_yaml: str = "bytetrack.yaml",
    ):
        if YOLO is None:
            raise ImportError("ultralytics is not installed. Install: uv pip install -e '.[predict]'")
        self.model = YOLO(weights)
        self.device = device
        self.conf = float(conf)
        self.iou = float(iou)
        self.tracker_yaml = tracker_yaml

    def update(self, frame_bgr: np.ndarray) -> List[RawTrack]:
        # Ultralytics returns a list (batch). We always pass a single frame.
        results = self.model.track(
            frame_bgr,
            persist=True,
            verbose=False,
            device=self.device,
            conf=self.conf,
            iou=self.iou,
            tracker=self.tracker_yaml,
        )

        out: List[RawTrack] = []
        r0 = results[0]
        if r0.boxes is None or r0.boxes.id is None:
            return out

        boxes = r0.boxes.xyxy.cpu().numpy()
        track_ids = r0.boxes.id.cpu().numpy().astype(int)
        class_ids = r0.boxes.cls.cpu().numpy().astype(int)
        confs = r0.boxes.conf.cpu().numpy()

        # names mapping is in r0.names (dict)
        names = getattr(r0, "names", {}) or {}

        for bbox, tid, cid, sc in zip(boxes, track_ids, class_ids, confs):
            name = str(names.get(int(cid), str(int(cid))))
            out.append(
                RawTrack(
                    track_id=int(tid),
                    bbox=(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])),
                    score=float(sc),
                    raw_class_id=int(cid),
                    raw_class_name=name,
                )
            )
        return out

from __future__ import annotations

from typing import List, Optional
import numpy as np

from counter.tracking.providers import TrackProvider, RawTrack

try:  # pragma: no cover
    import supervision as sv  # type: ignore
except Exception:  # pragma: no cover
    sv = None

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


class RoboflowRfDetrTrackProvider(TrackProvider):
    """RF-DETR via Roboflow's `rfdetr` library + Supervision ByteTrack.

    - Inference: `model.predict(image, threshold=...)` returns `supervision.Detections`.
      This is exactly how RF-DETR docs show it.  
    - Tracking: Supervision ByteTrack is the reference implementation used in Roboflow ecosystem. 

    Notes:
    - We intentionally *don't* implement ByteTrack ourselves.
    - `weights` may be an empty string for official pre-trained weights (library downloads them).
      For fine-tuned checkpoints, pass the checkpoint path via `pretrain_weights=...`. 
    """

    def __init__(
        self,
        weights: str,
        size: str | None = None,
        conf: float = 0.35,
        bytetrack_min_hits: int = 3,
        bytetrack_track_thresh: float = 0.25,
        bytetrack_match_thresh: float = 0.8,
        bytetrack_buffer: int = 30,
    ):
        if sv is None or RFDETRBase is None:
            raise ImportError(
                "Missing dependencies for RF-DETR tracking. Install extras: "
                "uv pip install -e '.[predict]' (needs rfdetr + supervision + torch)."
            )
        self.conf = float(conf)

        ModelCls = _pick_model_cls(size)
        if ModelCls is None:
            raise ImportError("rfdetr model classes not found. Check your rfdetr install/version.")

        w = (weights or "").strip()
        if w and w.lower() not in {"pretrained", "default"}:
            self.model = ModelCls(pretrain_weights=w)
        else:
            self.model = ModelCls()

        # Supervision ByteTrack tracker (no manual implementation)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=float(bytetrack_track_thresh),
            lost_track_buffer=int(bytetrack_buffer),
            minimum_matching_threshold=float(bytetrack_match_thresh),
            minimum_consecutive_frames=int(bytetrack_min_hits),
        )

        # RF-DETR exposes class names
        self._names: Optional[dict[int, str]] = None
        try:
            names = getattr(self.model, "class_names", None) or getattr(self.model, "class_names_", None)
            if isinstance(names, (list, tuple)):
                self._names = {i: str(n) for i, n in enumerate(names)}
            elif isinstance(names, dict):
                self._names = {int(k): str(v) for k, v in names.items()}
        except Exception:
            self._names = None

    def update(self, frame_bgr: np.ndarray) -> List[RawTrack]:
        # RF-DETR docs use RGB input; OpenCV gives BGR.
        frame_rgb = frame_bgr[:, :, ::-1].copy()

        detections = self.model.predict(frame_rgb, threshold=self.conf)

        # Ensure we have a Supervision Detections object
        if not hasattr(detections, "xyxy"):
            return []

        tracked = self.tracker.update_with_detections(detections)

        # Supervision stores tracker ids in `tracker_id` (np.ndarray) after tracking.
        if getattr(tracked, "tracker_id", None) is None:
            return []

        out: List[RawTrack] = []

        xyxy = tracked.xyxy
        class_id = getattr(tracked, "class_id", None)
        conf = getattr(tracked, "confidence", None)
        tid = tracked.tracker_id

        if class_id is None:
            class_id = np.zeros((len(xyxy),), dtype=int)
        if conf is None:
            conf = np.ones((len(xyxy),), dtype=float)

        for box, cid, sc, track_id in zip(xyxy, class_id, conf, tid):
            cid_i = int(cid) if cid is not None else -1
            name = str(cid_i)
            if self._names and cid_i in self._names:
                name = self._names[cid_i]
            out.append(
                RawTrack(
                    track_id=int(track_id),
                    bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                    score=float(sc),
                    raw_class_id=cid_i,
                    raw_class_name=name,
                )
            )
        return out

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional
import platform

import numpy as np
import supervision as sv

from rfdetr import RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano, RFDETRXLarge, RFDETR2XLarge

from counter.core.types import BBoxXYXY
from counter.predict.tracking.providers import TrackProvider
from counter.predict.types import RawTrack


def _make_rfdetr_model(model_size: str | None, weights: str | None):
    """Instantiate an RF-DETR model with optional pretrained weights."""
    size = (model_size or "small").lower().strip()
    cls = {
        "nano": RFDETRNano,
        "small": RFDETRSmall,
        "medium": RFDETRMedium,
        "large": RFDETRLarge,
        "xlarge": RFDETRLarge,
        "2xlarge": RFDETRLarge,
    }.get(size)

    if cls is None:
        raise ValueError(f"Unsupported rfdetr_size={model_size!r}. Use one of: nano/small/medium/large/base")

    # Pretrained: weights=None -> default pretrain (download/cache handled by rfdetr).
    if weights is None:
        return cls()

    # Tuned: weights=local checkpoint -> pretrain_weights=<path>.
    p = Path(weights)
    if not p.exists():
        raise FileNotFoundError(f"RF-DETR weights file not found: {weights!r}")

    return cls(pretrain_weights=str(p))


@dataclass
class RfDetrTrackProvider(TrackProvider):
    """RF-DETR-based track provider with optional ByteTrack."""

    variant: str  # "tuned" | "pretrained"
    weights: Optional[str]  # tuned: local .pth, pretrained: None
    model_size: str  # "nano" | "small" | "medium" | "large" | "xlarge" | "2xlarge"
    device: str  # "cpu" | "cuda" (RF-DETR maps internally)
    conf: float
    tracking_type: str  # "none" | "bytetrack"
    tracking_params: dict[str, Any]

    # Best-effort JIT/tracing speedup (often fails on Windows).
    optimize_for_inference: bool = True

    def __post_init__(self) -> None:
        self.model = _make_rfdetr_model(self.model_size, self.weights)

        # Best-effort inference optimization; may fail depending on torch/rfdetr version.
        if self.optimize_for_inference and hasattr(self.model, "optimize_for_inference"):
            try:
                # Skip on CPU and Windows; it is rarely worth it and often fails.
                if self.device.lower() == "cpu":
                    raise RuntimeError("Skip optimize_for_inference on CPU (not worth it).")
                if platform.system().lower().startswith("win"):
                    raise RuntimeError("Skip optimize_for_inference on Windows (torch.jit.trace often fails).")

                self.model.optimize_for_inference()
            except Exception as e:
                print(f"[WARN] RF-DETR optimize_for_inference() skipped: {type(e).__name__}: {e}")

        self._new_tracker()

    def _new_tracker(self) -> None:
        if self.tracking_type == "none":
            self.tracker = None
            return

        self.tracker = sv.ByteTrack(**(self.tracking_params or {}))

    def reset(self) -> None:
        """Reset tracking state (reinitialize tracker)."""
        self._new_tracker()

    def update(self, frame_bgr: np.ndarray) -> list[RawTrack]:
        """Run detection (and optional tracking) on a single frame."""
        frame_rgb = frame_bgr[:, :, ::-1].copy()
        detections: sv.Detections = self.model.predict(frame_rgb, threshold=float(self.conf))

        if self.tracker is not None:
            detections = self.tracker.update_with_detections(detections)

        if detections is None or len(detections) == 0:
            return []

        xyxy = detections.xyxy
        scores = detections.confidence
        cls_ids = detections.class_id
        tids = getattr(detections, "tracker_id", None)

        out: list[RawTrack] = []
        n = len(detections)
        for i in range(n):
            x1, y1, x2, y2 = map(int, xyxy[i].tolist())
            score = float(scores[i]) if scores is not None else 1.0
            raw_id = int(cls_ids[i]) if cls_ids is not None else -1
            tid = int(tids[i]) if tids is not None else -1

            out.append(
                RawTrack(
                    track_id=tid,
                    bbox=(float(x1), float(y1), float(x2), float(y2)),
                    score=score,
                    raw_class_id=raw_id,
                    raw_class_name=str(raw_id),
                )
            )

        return out

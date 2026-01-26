from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from counter.config.schema import TrackingCfg
from counter.domain.model_spec import ModelSpec
from counter.tracking.providers import TrackProvider
from counter.tracking.rfdetr_roboflow import RoboflowRfDetrTrackProvider
from counter.tracking.yolo_ultralytics import UltralyticsYoloTrackProvider


def _resolve_bytetrack_yaml(
    tracking: Optional[TrackingCfg],
    work_dir: Optional[Path],
) -> str:
    """Return a tracker YAML path for Ultralytics.

    Priority:
      1) tracking.tracker_yaml (explicit path)
      2) generate YAML from tracking.params into work_dir
      3) default 'bytetrack.yaml' (Ultralytics built-in preset)

    Note: Ultralytics expects specific keys; we map your legacy params
    (track_thresh, match_thresh) to the common ByteTrack YAML keys.
    """

    if tracking is None:
        return "bytetrack.yaml"

    if tracking.tracker_yaml:
        return str(tracking.tracker_yaml)

    params = dict(tracking.params or {})
    if not params:
        return "bytetrack.yaml"

    # Map legacy keys -> Ultralytics ByteTrack keys
    if "track_thresh" in params and "track_high_thresh" not in params:
        try:
            th = float(params["track_thresh"])
        except Exception:
            th = 0.25
        params.setdefault("track_high_thresh", th)
        params.setdefault("new_track_thresh", th)
        params.setdefault("track_low_thresh", min(0.1, th / 2.0))
        params.pop("track_thresh", None)

    # Provide sensible defaults for keys Ultralytics typically expects
    base: Dict[str, Any] = {
        "tracker_type": "bytetrack",
        "track_high_thresh": float(params.pop("track_high_thresh", 0.25)),
        "track_low_thresh": float(params.pop("track_low_thresh", 0.1)),
        "new_track_thresh": float(params.pop("new_track_thresh", 0.25)),
        "track_buffer": int(params.pop("track_buffer", 30)),
        "match_thresh": float(params.pop("match_thresh", 0.8)),
        "fuse_score": bool(params.pop("fuse_score", True)),
    }

    # Keep any remaining keys as-is (advanced users)
    base.update(params)

    out_dir = work_dir or Path(".")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "tracker.bytetrack.generated.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return str(out_path)


def _infer_rfdetr_size(spec: ModelSpec) -> str:
    if spec.rfdetr_size:
        return str(spec.rfdetr_size)
    name = str(getattr(spec, "model_id", "") or "").lower()
    for size in ("small", "medium", "large"):
        if size in name:
            return size
    return "medium"


class TrackerFactory:
    """Pragmatic factory: returns TrackProvider that yields RawTrack (track_id + bbox + raw class)."""

    @staticmethod
    def create(
        spec: ModelSpec,
        conf: float,
        iou: float,
        device: str,
        tracking: Optional[TrackingCfg] = None,
        work_dir: Optional[Path] = None,
    ) -> TrackProvider:
        if spec.backend == "yolo":
            if not spec.weights:
                raise ValueError(f"YOLO model '{spec.model_id}' missing weights in models.yaml")

            tracker_yaml = _resolve_bytetrack_yaml(tracking=tracking, work_dir=work_dir)
            return UltralyticsYoloTrackProvider(
                weights=spec.weights,
                device=device,
                conf=float(conf),
                iou=float(iou),
                tracker_yaml=tracker_yaml,
            )

        if spec.backend == "rfdetr":
            # RF-DETR library can run with pretrained weights even when weights is empty,
            # but for your registry we keep it explicit.
            if spec.weights is None:
                raise ValueError(
                    f"RF-DETR model '{spec.model_id}' missing weights in models.yaml (use '' for pretrained)"
                )
            return RoboflowRfDetrTrackProvider(
                weights=str(spec.weights),
                model_size=_infer_rfdetr_size(spec),
                device=device,
                conf=float(conf),
                iou=float(iou),
            )

        raise ValueError(f"Unsupported backend: {spec.backend}")

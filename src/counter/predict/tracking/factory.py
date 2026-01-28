# src/counter/predict/tracking/factory.py
from __future__ import annotations

from counter.core.schema import PredictConfig, ModelSpecCfg
from counter.predict.tracking.providers import TrackProvider
from counter.predict.tracking._provider import UltralyticsYoloTrackProvider
from counter.predict.tracking.rfdetr_provider import RfDetrTrackProvider


def make_provider(cfg: PredictConfig, spec: ModelSpecCfg) -> TrackProvider:
    if spec.backend == "yolo":
        return UltralyticsYoloTrackProvider(
            weights=spec.weights,
            device=cfg.device,
            conf=cfg.thresholds.conf,
            iou=cfg.thresholds.iou,
            tracking_enabled=cfg.tracking.enabled,
            tracker_yaml=cfg.tracking.tracker_yaml,
            tracker_params=cfg.tracking.params,
        )

    if spec.backend == "rfdetr":
        return RfDetrTrackProvider(
            variant=spec.variant,
            weights=spec.weights,  # tuned: cesta k .pth, pretrained: None
            model_size=spec.rfdetr_size or "small",
            device=cfg.device,
            conf=cfg.thresholds.conf,
            tracking_type=cfg.tracking.type,
            tracking_params=cfg.tracking.params,
        )

    raise ValueError(f"Unsupported backend: {spec.backend}")

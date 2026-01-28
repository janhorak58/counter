from __future__ import annotations

"""Pydantic schema definitions for prediction, evaluation, and model registry."""

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LineCfg(BaseModel):
    """Counting line definition in pixel coordinates."""

    name: str = "Line_1"
    coords: List[int] = Field(
        default_factory=lambda: [846, 404, 1328, 456],
        description="Line coordinates as [x1, y1, x2, y2]",
    )
    default_resolution: Tuple[int, int] = Field(
        default_factory=lambda: (1920, 1080),
        description="Default resolution as [width, height]",
    )


class ThresholdsCfg(BaseModel):
    """Score and IoU thresholds for detection filtering."""

    conf: float = 0.35
    iou: float = 0.5


class TrackingCfg(BaseModel):
    """Tracking configuration for prediction."""

    type: Literal["none", "bytetrack"] = "bytetrack"

    # Ultralytics YOLO tracker configuration file (YAML).
    tracker_yaml: Optional[str] = None

    # Supervision ByteTrack parameters (and general tracker params).
    params: Dict[str, Any] = Field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.type != "none"


class ExportCfg(BaseModel):
    """Output export settings for prediction."""

    save_video: bool = True
    save_raw: bool = True
    save_counts_json: bool = True
    out_dir: str = "runs/predict/export"


class PreviewCfg(BaseModel):
    """Preview rendering settings for quick inspection."""

    enabled: bool = False
    every_n_frames: int = 1
    max_width: int = 800


class PredictConfig(BaseModel):
    """Prediction configuration loaded from YAML."""

    run_id: str = "local_predict"
    model_id: str
    output_dir: str = "runs/predict"
    device: str = "cpu"
    debug: bool = False

    thresholds: ThresholdsCfg = Field(default_factory=ThresholdsCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)

    videos_dir: str
    videos: List[str]

    probe_frames: int = 0

    line: LineCfg = Field(default_factory=LineCfg)
    greyzone_px: float = 0.0
    preview: PreviewCfg = Field(default_factory=PreviewCfg)
    save_video: bool = True


class ChartsCfg(BaseModel):
    """Optional chart generation settings for evaluation."""

    enabled: bool = True


class EvalFiltersCfg(BaseModel):
    """Filters limiting which runs or models are included in evaluation."""

    # Empty lists mean "do not filter".
    backends: List[Literal["yolo", "rfdetr"]] = Field(default_factory=list)
    variants: List[Literal["tuned", "pretrained"]] = Field(default_factory=list)
    model_ids: List[str] = Field(default_factory=list)
    run_ids: List[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    """Evaluation configuration for scoring and ranking runs."""

    gt_dir: str = "data/counts_gt"
    runs_dir: str = "runs"
    out_dir: str = "runs"
    only_completed: bool = True

    videos_dir: Optional[str] = None

    rank_by: Literal["video_mae_total", "event_wape_total", "rate_mae_total"] = "event_wape_total"

    filters: EvalFiltersCfg = Field(default_factory=EvalFiltersCfg)
    charts: ChartsCfg = Field(default_factory=ChartsCfg)
    timestamp: str = Field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"),
        description="Timestamp used to isolate evaluation output folders.",
    )


class MappingCfg(BaseModel):
    """Class mapping configuration."""

    tourist: int
    skier: int
    cyclist: int
    tourist_dog: int

    def _get_dict(self) -> Dict[int, int]:
        return {
            "tourist": self.tourist,
            "skier": self.skier,
            "cyclist": self.cyclist,
            "tourist_dog": self.tourist_dog,
        }

    def _get(self, raw_class_id: int) -> Optional[int]:
        for canonical_id, name in enumerate(["tourist", "skier", "cyclist", "tourist_dog"]):
            if getattr(self, name) == raw_class_id:
                return canonical_id
        return None


class ModelSpecCfg(BaseModel):
    """Single model entry in the registry."""

    model_id: str
    backend: Literal["yolo", "rfdetr"]
    variant: Literal["pretrained", "tuned"] = "tuned"
    weights: Optional[str] = None
    mapping: Optional[MappingCfg] = None
    rfdetr_size: Optional[Literal["base", "small", "medium", "large", "nano", "xlarge", "2xlarge"]] = None

    model_config = ConfigDict(extra="allow")


class ModelsRegistry(BaseModel):
    """Supports both YAML shapes:
    A) flat dict: <id>: {...}
    B) wrapped: models: <id>: {...}
    """

    models: Dict[str, ModelSpecCfg] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any):
        # Allow a flat dict and wrap it to the new shape.
        if isinstance(data, dict) and "models" not in data:
            data = {"models": data}

        if not isinstance(data, dict):
            return data

        models = data.get("models")
        if isinstance(models, dict):
            fixed = {}
            for model_id, spec in models.items():
                if isinstance(spec, dict):
                    spec = dict(spec)
                    spec.setdefault("model_id", model_id)
                fixed[model_id] = spec
            data["models"] = fixed

        return data

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LineCfg(BaseModel):
    name: str = "Line_1"
    start: List[int] = Field(default_factory=lambda: [0, 0])
    end: List[int] = Field(default_factory=lambda: [100, 100])


class ThresholdsCfg(BaseModel):
    """Detection thresholds."""

    conf: float = 0.35
    iou: float = 0.35


class TrackingCfg(BaseModel):
    """Tracking configuration.

    For Ultralytics YOLO this maps to the built-in tracker YAML (ByteTrack).
    """

    type: Literal["bytetrack"] = "bytetrack"
    tracker_yaml: Optional[str] = None
    params: Dict[str, Any] = Field(default_factory=dict)


class ExportCfg(BaseModel):
    out_dir: str = "runs"
    save_video: bool = False

class PreviewCfg(BaseModel):
    """Optional UI preview during prediction."""
    enabled: bool = False
    every_n_frames: int = 50          # jak často posílat preview
    max_width: int = 960              # zmenšení pro rychlost (jen náhled)


class PredictConfig(BaseModel):
    """Predict configuration.

    Supported YAML shapes:

    New (recommended):
      thresholds: {conf: 0.35, iou: 0.35}
      tracking: {type: bytetrack, params: {...}}
      export: {out_dir: runs, save_video: false}

    Legacy (still accepted):
      conf: 0.35
      iou: 0.35
      out_dir: runs
      save_video: false
    """

    model_config = ConfigDict(extra="ignore")

    model_id: str
    videos_dir: str
    videos: List[str] = Field(default_factory=list)

    device: str = "cpu"

    line: LineCfg = Field(default_factory=LineCfg)
    greyzone_px: float = 20.0

    thresholds: ThresholdsCfg = Field(default_factory=ThresholdsCfg)
    tracking: TrackingCfg = Field(default_factory=TrackingCfg)
    export: ExportCfg = Field(default_factory=ExportCfg)
    preview: PreviewCfg = Field(default_factory=PreviewCfg)


    @field_validator("videos", mode="before")
    @classmethod
    def _coerce_videos(cls, v):
        # YAML sometimes contains videos: null
        if v is None:
            return []
        return v

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_shape(cls, data: Any):
        if not isinstance(data, dict):
            return data

        # thresholds legacy
        if "thresholds" not in data:
            thr: Dict[str, Any] = {}
            if "conf" in data:
                thr["conf"] = data.pop("conf")
            if "iou" in data:
                thr["iou"] = data.pop("iou")
            if thr:
                data["thresholds"] = thr

        # export legacy
        if "export" not in data:
            exp: Dict[str, Any] = {}
            if "out_dir" in data:
                exp["out_dir"] = data.pop("out_dir")
            if "save_video" in data:
                exp["save_video"] = data.pop("save_video")
            if exp:
                data["export"] = exp

        return data

    # Back-compat convenience accessors used across the codebase
    @property
    def conf(self) -> float:
        return float(self.thresholds.conf)

    @property
    def iou(self) -> float:
        return float(self.thresholds.iou)

    @property
    def out_dir(self) -> str:
        return str(self.export.out_dir)

    @property
    def save_video(self) -> bool:
        return bool(self.export.save_video)


class ChartsCfg(BaseModel):
    enabled: bool = True


class EvalFiltersCfg(BaseModel):
    # prázdné listy = nefiltrovat (evaluate všechno)
    backends: List[Literal["yolo", "rfdetr"]] = Field(default_factory=list)
    variants: List[Literal["tuned", "pretrained"]] = Field(default_factory=list)
    model_ids: List[str] = Field(default_factory=list)
    run_ids: List[str] = Field(default_factory=list)


class EvalConfig(BaseModel):
    gt_dir: str = "data/counts_gt"
    runs_dir: str = "runs"
    out_dir: str = "runs"
    only_completed: bool = True

    videos_dir: Optional[str] = None

    rank_by: Literal["video_mae_total", "event_wape_total", "rate_mae_total"] = "event_wape_total"

    filters: EvalFiltersCfg = Field(default_factory=EvalFiltersCfg)
    charts: ChartsCfg = Field(default_factory=ChartsCfg)
    timestamp: str = Field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))


class BenchmarkSpec(BaseModel):
    model_ids: List[str]
    videos_dir: str
    gt_dir: str
    out_dir: str = "runs"


class ModelSpecCfg(BaseModel):
    model_config = ConfigDict(extra="allow")

    backend: Literal["yolo", "rfdetr"]
    variant: Literal["tuned", "pretrained"]

    weights: Optional[str] = None
    mapping_policy: Optional[str] = None
    rfdetr_size: Optional[str] = None

    class_map: Optional[Dict[int, int]] = None
    coco_ids: Optional[Dict[str, int]] = None

    @field_validator("class_map", mode="before")
    @classmethod
    def _coerce_class_map_keys(cls, v):
        if v is None:
            return None
        if isinstance(v, dict):
            out = {}
            for k, val in v.items():
                out[int(k)] = int(val)
            return out
        return v


class ModelsRegistry(BaseModel):
    """Supports both YAML shapes:

    A) flat dict:
       yolo11m_tuned:
         backend: yolo
         variant: tuned
         weights: ...

    B) wrapped:
       models:
         yolo11m_tuned: {...}
    """

    models: Dict[str, ModelSpecCfg] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _normalize(cls, data: Any):
        if isinstance(data, dict) and "models" not in data:
            return {"models": data}
        return data

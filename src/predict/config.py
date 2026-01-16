from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "video_folder": "data/videos/",
        "output_folder": "data/output/",
        "results_folder": "data/results/predicted",
        "model_path": "models/yolov8s/weights/best.pt",
        "video_filename": "",
    },
    "parameters": {
        "confidence_threshold": 0.4,
        "iou_threshold": 0.5,
        "grey_zone_size": 20.0,
        "device": "cpu",
        "mode": "custom",
        "model_type": "yolo",
        "track_iou_threshold": 0.3,
        "track_max_lost": 15,
        "track_match_classes": True,
        "rfdetr_box_format": "xyxy",
        "rfdetr_box_normalized": "auto",
        "use_interactive_lines": True,
        "num_lines": 1,
        "lines": [
            {"start": [846, 404], "end": [1328, 456], "name": "Line_1"},
        ],
        "show_window": True,
        "progress_every_n_frames": 100,
    },
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_predict_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if isinstance(raw, dict):
        if "prediction" in raw and isinstance(raw["prediction"], dict):
            raw = raw["prediction"]
        elif "predict" in raw and isinstance(raw["predict"], dict):
            raw = raw["predict"]

    cfg = {
        "paths": dict(DEFAULT_CONFIG["paths"]),
        "parameters": dict(DEFAULT_CONFIG["parameters"]),
    }
    if isinstance(raw, dict):
        _deep_update(cfg, raw)
    return cfg

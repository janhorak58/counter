from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model_type": "yolo",
    "model_path": "",
    "data_yaml": "data/dataset_yolo/data.yaml",
    "dataset_dir": "data/dataset_coco",
    "device": "cpu",
    "conf": 0.25,
    "iou": 0.6,
    "imgsz": 640,
    "split": "val",
    "output_dir": "eval/model",
    "rfdetr_box_format": "xyxy",
    "rfdetr_box_normalized": "auto",
    "visuals": True,
    "visuals_samples": 8,
    "visuals_iou": 0.5,
    "coco_eval": True,
}


def _deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_eval_model_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if isinstance(raw, dict):
        if "eval_model" in raw and isinstance(raw["eval_model"], dict):
            raw = raw["eval_model"]
        elif "eval" in raw and isinstance(raw["eval"], dict):
            raw = raw["eval"]

    cfg = dict(DEFAULT_CONFIG)
    if isinstance(raw, dict):
        _deep_update(cfg, raw)
    return cfg

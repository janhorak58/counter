from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "gt_folder": "data/results/gt",
    "pred_folder": "data/results/predicted",
    "out_dir": "data/results/analysis",
    "plots": False,
    "map_pretrained_counts": False,
    "run_yolo_eval": False,
    "yolo_mode": "custom",
    "model_path": "models/yolov8s/weights/best.pt",
    "data_yaml": "data/data.yaml",
    "device": "cpu",
    "conf": 0.25,
    "iou": 0.6,
    "split": "val",
}


def load_eval_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if isinstance(raw, dict) and "eval" in raw and isinstance(raw["eval"], dict):
        raw = raw["eval"]

    cfg = DEFAULT_CONFIG.copy()
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg

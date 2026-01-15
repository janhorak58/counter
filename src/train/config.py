from typing import Any, Dict

import yaml


DEFAULT_CONFIG: Dict[str, Any] = {
    "model": "yolov8s.pt",
    "data_yaml": "data/data.yaml",
    "epochs": 100,
    "imgsz": 960,
    "batch": 16,
    "workers": 8,
    "device": "cpu",
    "patience": 15,
    "project": "models",
    "name": "yolov8s",
    "plots": True,
    "save": True,
    "cos_lr": True,
}


def load_train_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f) or {}

    if isinstance(raw, dict) and "train" in raw and isinstance(raw["train"], dict):
        raw = raw["train"]

    cfg = DEFAULT_CONFIG.copy()
    if isinstance(raw, dict):
        cfg.update(raw)
    return cfg

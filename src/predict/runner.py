from typing import Dict, Optional

from src.predict.base_runner import run_prediction_base
from src.predict.rfdetr_runner import run_rfdetr_prediction


def run_prediction(cfg: Dict) -> Optional[Dict[str, Dict]]:
    params = cfg.get("parameters", {})
    model_type = (params.get("model_type") or cfg.get("model_type") or "yolo").lower()
    if model_type == "rfdetr":
        return run_rfdetr_prediction(cfg)
    if model_type == "yolo":
        return run_prediction_base(cfg, model_type="yolo")
    raise ValueError(f"Unsupported model_type: {model_type}")

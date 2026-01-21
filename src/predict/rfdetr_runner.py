from typing import Dict, Optional

from src.predict.base_runner import run_prediction_base


def run_rfdetr_prediction(cfg: Dict) -> Optional[Dict[str, Dict]]:
    return run_prediction_base(cfg, model_type="rfdetr")

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import yaml

from .schema import PredictConfig, EvalConfig, ModelsRegistry

def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f'Config not found: {p}')
    with p.open('r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config must be a mapping at top-level: {p}')
    return data

def load_predict_config(path: str | Path) -> PredictConfig:
    data = load_yaml(path)
    if data.get('videos', []) is None:
        data['videos'] = []
    return PredictConfig.model_validate(data)


def load_eval_config(path: str | Path) -> EvalConfig:
    return EvalConfig.model_validate(load_yaml(path))

def load_models_registry(path: str | Path) -> ModelsRegistry:
    return ModelsRegistry.model_validate(load_yaml(path))

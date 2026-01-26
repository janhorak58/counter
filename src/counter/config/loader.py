from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from counter.config.schema import EvalConfig, ModelsRegistry, PredictConfig


def load_yaml(path: str | Path) -> Any:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_models_registry(path: str | Path) -> ModelsRegistry:
    data = load_yaml(path)
    return ModelsRegistry.model_validate(data)


def load_predict_config(path: str | Path) -> PredictConfig:
    data = load_yaml(path)
    return PredictConfig.model_validate(data)


def load_eval_config(path: str | Path) -> EvalConfig:
    data = load_yaml(path)
    return EvalConfig.model_validate(data)

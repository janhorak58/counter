from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import yaml

from counter.core.schema import EvalConfig, ModelsRegistry, PredictConfig


class ConfigValidationError(Exception):
    pass


def load_yaml_dict(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def dump_yaml_text(data: Dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def parse_yaml_text(text: str) -> Dict[str, Any]:
    try:
        obj = yaml.safe_load(text)
    except Exception as exc:
        raise ConfigValidationError(f"Invalid YAML: {exc}") from exc
    if not isinstance(obj, dict):
        raise ConfigValidationError("YAML root must be a mapping/object.")
    return obj


def validate_predict_yaml_text(text: str) -> Tuple[bool, str]:
    try:
        obj = parse_yaml_text(text)
        PredictConfig.model_validate(obj)
        return True, "Predict config is valid."
    except Exception as exc:
        return False, str(exc)


def validate_eval_yaml_text(text: str) -> Tuple[bool, str]:
    try:
        obj = parse_yaml_text(text)
        EvalConfig.model_validate(obj)
        return True, "Eval config is valid."
    except Exception as exc:
        return False, str(exc)


def validate_models_yaml_text(text: str) -> Tuple[bool, str]:
    try:
        obj = parse_yaml_text(text)
        ModelsRegistry.model_validate(obj)
        return True, "Models registry is valid."
    except Exception as exc:
        return False, str(exc)


def write_yaml_file(path: str | Path, data: Dict[str, Any]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
    return p

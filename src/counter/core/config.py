from __future__ import annotations

"""YAML and Pydantic config loaders for the core package."""

from pathlib import Path
from typing import Any, Type, TypeVar

try:  
    import yaml  # type: ignore
except Exception:  
    yaml = None

from counter.core.schema import EvalConfig, ModelsRegistry, PredictConfig

T = TypeVar("T")


def load_yaml(path: str | Path) -> Any:
    """Load YAML from file.

    Raises an informative error if PyYAML is missing.
    """
    if yaml is None:
        raise ImportError(
            "PyYAML is not installed. Install it, e.g.: uv pip install pyyaml"
        )

    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pydantic(path: str | Path, cls: Type[T]) -> T:
    """Load YAML and validate it against a Pydantic v2 model."""
    data = load_yaml(path)
    return cls.model_validate(data)  # type: ignore[attr-defined]


def load_models_registry(path: str | Path) -> ModelsRegistry:
    """Load the models registry YAML."""
    return load_pydantic(path, ModelsRegistry)


def load_predict_config(path: str | Path) -> PredictConfig:
    """Load prediction configuration YAML."""
    return load_pydantic(path, PredictConfig)


def load_eval_config(path: str | Path) -> EvalConfig:
    """Load evaluation configuration YAML."""
    return load_pydantic(path, EvalConfig)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Literal, Any

import yaml

from counter.config.schema import ModelsRegistry


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    backend: Literal["yolo", "rfdetr"]
    variant: Literal["tuned", "pretrained"]

    weights: Optional[str] = None
    mapping_policy: Optional[str] = None
    rfdetr_size: Optional[str] = None

    # optional helpers for mapping
    class_map: Optional[Dict[str, int]] = None
    coco_ids: Optional[Dict[str, int]] = None


def load_models(models_yaml: str | Path) -> Dict[str, ModelSpec]:
    """
    Loads configs/models.yaml (supports both shapes):
      A) flat dict:
         yolo11m_tuned: {backend: yolo, variant: tuned, ...}
      B) wrapped:
         models: { yolo11m_tuned: {...} }
    """
    p = Path(models_yaml)
    if not p.exists():
        raise FileNotFoundError(f"models.yaml not found: {p}")

    data: Any = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    registry = ModelsRegistry.model_validate(data)

    out: Dict[str, ModelSpec] = {}
    for model_id, cfg in registry.models.items():
        out[model_id] = ModelSpec(
            model_id=model_id,
            backend=cfg.backend,
            variant=cfg.variant,
            weights=cfg.weights,
            mapping_policy=cfg.mapping_policy,
            rfdetr_size=cfg.rfdetr_size,
            class_map=cfg.class_map,
            coco_ids=cfg.coco_ids,
        )
    return out

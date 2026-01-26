from __future__ import annotations

from counter.domain.model_spec import ModelSpec
from counter.mapping.base import MappingPolicy
from counter.mapping.tuned import TunedMapping
from counter.mapping.pretrained import CocoBaselineMapping


class MappingFactory:
    """Pragmatic factory: returns MappingPolicy (Detection -> mapped class id)."""

    @staticmethod
    def create(spec: ModelSpec) -> MappingPolicy:
        # Optional explicit override (if you decide to use it in models.yaml later)
        policy = (spec.mapping_policy or "").strip().lower()

        # Tuned models: usually already in canonical ids 0..3
        if policy in {"tuned", "identity", "identity4"} or spec.variant == "tuned":
            if not spec.class_map:
                # Minimal-friction default (keep only 0..3, ignore the rest)
                return TunedMapping(class_map={0: 0, 1: 1, 2: 2, 3: 3})

            # Coerce keys to int defensively (YAML keys may be strings).
            class_map = {int(k): int(v) for k, v in (spec.class_map or {}).items()}
            return TunedMapping(class_map=class_map)

        # Pretrained COCO models: convert COCO classes -> your canonical 0..3 at the end
        if policy in {"coco", "pretrained", "coco_baseline"} or spec.variant == "pretrained":
            coco_ids = spec.coco_ids or {"person": 0, "bicycle": 1, "dog": 16, "skis": 30}
            # NOTE: Ultralytics COCO ids are 0-based by default (person=0). If you use 1-based somewhere,
            # set coco_ids in models.yaml explicitly.
            return CocoBaselineMapping(coco_ids=coco_ids)

        # Fallback: be strict (better fail than silently count garbage)
        raise ValueError(
            f"Cannot infer mapping policy for model '{spec.model_id}'. "
            f"variant='{spec.variant}', mapping_policy='{spec.mapping_policy}'."
        )

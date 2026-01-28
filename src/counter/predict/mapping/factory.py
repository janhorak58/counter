from __future__ import annotations

from typing import Dict, Optional

from counter.core.schema import ModelSpecCfg
from counter.predict.mapping.pretrained import CocoBaselineMapping
from counter.predict.mapping.tuned import TunedMapping
from counter.predict.mapping.track_mapper import TrackMapper
from counter.core.types import CanonicalClass


def make_mapper(*, spec: ModelSpecCfg, label_map: Optional[Dict[int, str]], log=None) -> TrackMapper:
    """Choose mapping policy.

    Priority:
      1) spec.mapping (explicit)
      2) spec.variant heuristics

    For tuned: uses spec.mapping or inferred mapping.
    For pretrained: uses spec.coco_ids or inferred coco ids (needs label_map).
    """

    variant = (spec.variant or "").lower().strip()

    if variant == "tuned":
        return TrackMapper(TunedMapping(mapping=spec.mapping))
    else:
        return TrackMapper(CocoBaselineMapping(mapping=spec.mapping))


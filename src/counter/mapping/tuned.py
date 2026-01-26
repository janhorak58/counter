from __future__ import annotations

from typing import Dict, Optional, Tuple

from counter.domain.types import Detection
from counter.mapping.base import MappingPolicy

class TunedMapping(MappingPolicy):
    def __init__(self, class_map: Dict[int, int]):
        self.class_map = {int(k): int(v) for k, v in class_map.items()}

    def map_detection(self, det: Detection) -> Optional[int]:
        return self.class_map.get(int(det.raw_class_id))

    def finalize_counts(self, in_counts: Dict[int, int], out_counts: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
        return dict(in_counts), dict(out_counts)

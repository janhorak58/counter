from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from counter.core.schema import MappingCfg, ModelSpecCfg
from counter.predict.mapping.base import MappingPolicy


@dataclass(frozen=True)
class TunedMapping(MappingPolicy):
    """Direct map raw class ids -> canonical class ids (0..3)."""
    mapping: Optional[MappingCfg] = None

    def map_raw(self, *, raw_class_id: int, raw_class_name: str) -> Optional[int]:
        return self.mapping._get(raw_class_id) if self.mapping else None

    def finalize_counts(
        self,
        in_counts: Dict[int, int],
        out_counts: Dict[int, int],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        # Already canonical
        return dict(in_counts), dict(out_counts)

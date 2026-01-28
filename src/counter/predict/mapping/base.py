from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from counter.core.schema import MappingCfg, ModelSpecCfg


class MappingPolicy(ABC):
    """Maps raw detections to internal class IDs.

    Notes:
    - For tuned models, mapped ids are canonical (0..3).
    - For COCO-style models, mapped ids may be intermediate (PERSON=100, ...).
      Finalization turns those into canonical counts.
    """
    mapping: Optional[MappingCfg] = None

    @abstractmethod
    def map_raw(self, *, raw_class_id: int, raw_class_name: str) -> Optional[int]:
        """Map raw class information to a (possibly intermediate) class id."""
        ...

    @abstractmethod
    def finalize_counts(
        self,
        in_counts: Dict[int, int],
        out_counts: Dict[int, int],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Finalize counts into canonical class IDs."""
        ...

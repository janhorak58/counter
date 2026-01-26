from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from counter.domain.types import Detection

class MappingPolicy(ABC):
    @abstractmethod
    def map_detection(self, det: Detection) -> Optional[int]:
        ...

    @abstractmethod
    def finalize_counts(self, in_counts: Dict[int, int], out_counts: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
        ...

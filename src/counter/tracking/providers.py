from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from counter.domain.types import BBoxXYXY

@dataclass(frozen=True)
class RawTrack:
    track_id: int
    bbox: BBoxXYXY
    score: float
    raw_class_id: int
    raw_class_name: str

class TrackProvider(ABC):
    @abstractmethod
    def update(self, frame_bgr: np.ndarray) -> List[RawTrack]:
        ...

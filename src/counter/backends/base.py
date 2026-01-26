from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List
import numpy as np

from counter.domain.types import Detection

class DetectorBackend(ABC):
    @abstractmethod
    def infer(self, frame_bgr: np.ndarray) -> List[Detection]:
        ...

    @abstractmethod
    def labels(self) -> Dict[int, str]:
        ...

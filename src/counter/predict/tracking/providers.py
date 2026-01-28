from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from counter.predict.types import RawTrack


class TrackProvider(ABC):
    """Backend-specific track provider.

    update(frame_bgr) returns list of RawTrack.

    Optionally exposes class-name mapping for heuristics.
    """

    @abstractmethod
    def reset(self) -> None:
        ...

    @abstractmethod
    def update(self, frame_bgr) -> List[RawTrack]:
        ...

    def get_label_map(self) -> Optional[Dict[int, str]]:
        return None

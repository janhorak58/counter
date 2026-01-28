from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from counter.core.types import BBoxXYXY
from counter.core.types import Side

@dataclass(frozen=True)
class RawTrack:
    track_id: int
    bbox: BBoxXYXY
    score: float
    raw_class_id: int
    raw_class_name: str



@dataclass
class MappedTrack:
    track_id: int
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int
    raw_class_id: int
    raw_class_name: str
    initial_side: Side = Side.UNKNOWN
    current_side: Side = Side.UNKNOWN
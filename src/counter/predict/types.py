from __future__ import annotations

"""Prediction-time data structures."""

from dataclasses import dataclass

from counter.core.types import BBoxXYXY
from counter.core.types import Side


@dataclass(frozen=True)
class RawTrack:
    """Raw tracker output before class mapping."""
    track_id: int
    bbox: BBoxXYXY
    score: float
    raw_class_id: int
    raw_class_name: str


@dataclass
class MappedTrack:
    """Tracked object with mapped class and line side state."""
    track_id: int
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int
    raw_class_id: int
    raw_class_name: str
    initial_side: Side = Side.UNKNOWN
    current_side: Side = Side.UNKNOWN

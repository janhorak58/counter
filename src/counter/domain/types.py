from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

BBoxXYXY = Tuple[float, float, float, float]

class CanonicalClass(IntEnum):
    TOURIST = 0
    SKIER = 1
    CYCLIST = 2
    TOURIST_DOG = 3

class Side(IntEnum):
    OUT = -1
    IN = 1
    UNKNOWN = 0

@dataclass(frozen=True)
class Detection:
    bbox: BBoxXYXY
    score: float
    raw_class_id: int
    raw_class_name: str

@dataclass(frozen=True)
class MappedDetection:
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int

@dataclass(frozen=True)
class Track:
    track_id: int
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int

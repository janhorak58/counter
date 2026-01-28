from __future__ import annotations

"""Shared type aliases and small data containers."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Tuple

# Bounding box in (x1, y1, x2, y2) pixel coordinates.
BBoxXYXY = Tuple[float, float, float, float]

@dataclass(frozen=True)
class LineCoords:
    """Normalized line coordinates in float form."""
    x1: float
    y1: float
    x2: float
    y2: float

    @staticmethod
    def from_start_end(start: Tuple[float, float], end: Tuple[float, float]) -> "LineCoords":
        """Build line coordinates from two (x, y) tuples."""
        return LineCoords(float(start[0]), float(start[1]), float(end[0]), float(end[1]))

    @staticmethod
    def from_coords(coords: Tuple[float, float, float, float]) -> "LineCoords":
        """Build line coordinates from a 4-tuple of floats."""
        return LineCoords(float(coords[0]), float(coords[1]), float(coords[2]), float(coords[3]))


class CanonicalClass(IntEnum):
    """Project-wide canonical class IDs."""
    TOURIST = 0
    SKIER = 1
    CYCLIST = 2
    TOURIST_DOG = 3

class Side(IntEnum):
    """Which side of a line a point lies on."""
    OUT = -1
    IN = 1
    ON = 0
    UNKNOWN = 99

@dataclass(frozen=True)
class Detection:
    """Raw detector output for a single object."""
    bbox: BBoxXYXY
    score: float
    raw_class_id: int
    raw_class_name: str

@dataclass(frozen=True)
class MappedDetection:
    """Detection after class remapping."""
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int

@dataclass(frozen=True)
class Track:
    """Tracked object state."""
    track_id: int
    bbox: BBoxXYXY
    score: float
    mapped_class_id: int

    @property
    def x1(self) -> float:
        """Left x coordinate of the bounding box."""
        return self.bbox[0]

    @property
    def y1(self) -> float:
        """Top y coordinate of the bounding box."""
        return self.bbox[1]

    @property
    def x2(self) -> float:
        """Right x coordinate of the bounding box."""
        return self.bbox[2]

    @property
    def y2(self) -> float:
        """Bottom y coordinate of the bounding box."""
        return self.bbox[3]

    @property
    def class_id(self) -> int:
        """Alias for mapped_class_id for compatibility."""
        return self.mapped_class_id

    @property
    def conf(self) -> float:
        """Alias for score for compatibility."""
        return self.score

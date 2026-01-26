from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from counter.domain.types import Side

Point = Tuple[float, float]

@dataclass(frozen=True)
class Line:
    start: Point
    end: Point
    name: str = 'Line_1'

def signed_distance(point: Point, line: Line) -> float:
    ax, ay = line.start
    bx, by = line.end
    px, py = point
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    cross = abx * apy - aby * apx
    norm = (abx * abx + aby * aby) ** 0.5
    if norm == 0:
        return 0.0
    return cross / norm

def side_of(point: Point, line: Line, greyzone_px: float) -> Side:
    d = signed_distance(point, line)
    if abs(d) < greyzone_px:
        return Side.UNKNOWN
    return Side.IN if d > 0 else Side.OUT

def bottom_center(bbox_xyxy: Tuple[float, float, float, float]) -> Point:
    x1, y1, x2, y2 = bbox_xyxy
    return ((x1 + x2) / 2.0, y2)

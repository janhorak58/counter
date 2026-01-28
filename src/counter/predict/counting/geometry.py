from __future__ import annotations

from typing import Tuple

from counter.core.types import BBoxXYXY, LineCoords, Side


def bottom_center(bbox: BBoxXYXY) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y2))


def _sign(p1: Tuple[float, float], p2: Tuple[float, float], p3: Tuple[float, float]) -> float:
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def classify_point(line: LineCoords, point: Tuple[float, float], vid_resolution: Tuple[int, int], line_base_resolution: Tuple[int, int]) -> Side:
    x1, y1, x2, y2 = line
    vid_w, vid_h = vid_resolution
    base_w, base_h = line_base_resolution
    x1 = x1 * vid_w / base_w
    y1 = y1 * vid_h / base_h
    x2 = x2 * vid_w / base_w
    y2 = y2 * vid_h / base_h

    p1 = (x1, y1)
    p2 = (x2, y2)
    p3 = point

    v = _sign(p1, p2, p3)
    if v > 0:
        return Side.IN
    if v < 0:
        return Side.OUT
    return Side.ON

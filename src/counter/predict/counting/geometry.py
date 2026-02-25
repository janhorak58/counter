from __future__ import annotations

from typing import Tuple

from counter.core.types import BBoxXYXY, LineCoords, Side


def bottom_center(bbox: BBoxXYXY) -> Tuple[float, float]:
    """Return the bottom-center point of a bounding box."""
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y2))


def classify_point(
    line: LineCoords,
    point: Tuple[float, float],
    vid_resolution: Tuple[int, int],
    line_base_resolution: Tuple[int, int],
    greyzone_px: float = 0.0,
) -> Side:
    """Classify a point as IN/OUT/ON relative to a line in video coordinates."""
    x1, y1, x2, y2 = line
    vid_w, vid_h = vid_resolution
    base_w, base_h = line_base_resolution
    if (vid_w, vid_h) != (base_w, base_h):
        sx = vid_w / base_w
        sy = vid_h / base_h
        x1 *= sx
        y1 *= sy
        x2 *= sx
        y2 *= sy

    px, py = point
    dx = x2 - x1
    dy = y2 - y1
    v = dx * (py - y1) - dy * (px - x1)

    if greyzone_px > 0.0:
        len2 = dx * dx + dy * dy
        if len2 == 0.0:
            return Side.ON
        tol2 = greyzone_px * greyzone_px * len2
        if v * v <= tol2:
            return Side.ON
    if v > 0:
        return Side.IN
    if v < 0:
        return Side.OUT
    return Side.ON

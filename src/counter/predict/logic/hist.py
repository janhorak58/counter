from __future__ import annotations

from collections import Counter
from typing import Dict, List

from counter.predict.types import MappedTrack, RawTrack


def raw_hist(tracks: List[RawTrack]) -> Dict[str, int]:
    # raw_class_name is informative for YOLO; fallback to id
    c = Counter([str(t.raw_class_name) for t in tracks])
    return {str(k): int(v) for k, v in c.items()}


def raw_id_hist(tracks: List[RawTrack]) -> Dict[int, int]:
    c = Counter([int(t.raw_class_id) for t in tracks])
    return {int(k): int(v) for k, v in c.items()}


def mapped_hist(tracks: List[MappedTrack]) -> Dict[int, int]:
    c = Counter([int(t.mapped_class_id) for t in tracks])
    return {int(k): int(v) for k, v in c.items()}

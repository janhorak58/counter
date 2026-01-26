from __future__ import annotations

from typing import Dict, List, Tuple

from counter.domain.types import Detection, Track
from counter.mapping.base import MappingPolicy
from counter.tracking.providers import RawTrack


class TrackMapper:
    """Adapter: RawTrack -> Track using MappingPolicy."""

    def __init__(self, policy: MappingPolicy):
        self._policy = policy

    def map_tracks(self, raw_tracks: List[RawTrack]) -> List[Track]:
        mapped: List[Track] = []
        for tr in raw_tracks:
            det = Detection(
                bbox=tr.bbox,
                score=tr.score,
                raw_class_id=tr.raw_class_id,
                raw_class_name=tr.raw_class_name,
            )
            mapped_id = self._policy.map_detection(det)
            if mapped_id is None:
                continue
            mapped.append(
                Track(
                    track_id=tr.track_id,
                    bbox=tr.bbox,
                    score=tr.score,
                    mapped_class_id=int(mapped_id),
                )
            )
        return mapped

    def finalize_counts(
        self, in_counts: Dict[int, int], out_counts: Dict[int, int]
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        return self._policy.finalize_counts(in_counts, out_counts)

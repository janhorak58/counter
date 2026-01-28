from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from counter.core.schema import MappingCfg, ModelSpecCfg
from counter.predict.mapping.base import MappingPolicy
from counter.predict.types import MappedTrack, RawTrack


class TrackMapper:
    """Adapter: RawTrack -> MappedTrack using MappingPolicy."""

    def __init__(self, policy: MappingPolicy):
        """Create a track mapper with the given policy."""
        self._policy = policy

    def map_tracks(self, raw_tracks: List[RawTrack]) -> List[MappedTrack]:
        """Map raw tracks to canonical tracks, skipping unknown classes."""
        out: List[MappedTrack] = []
        for rt in raw_tracks:
            mapped_id = self._policy.map_raw(raw_class_id=rt.raw_class_id, raw_class_name=rt.raw_class_name)
            if mapped_id is None:
                continue
            out.append(
                MappedTrack(
                    track_id=int(rt.track_id),
                    bbox=rt.bbox,
                    score=float(rt.score),
                    mapped_class_id=int(mapped_id),
                    raw_class_id=int(rt.raw_class_id),
                    raw_class_name=str(rt.raw_class_name),
                )
            )
        return out

    def finalize_counts(
        self,
        in_count: Dict[int, int],
        out_count: Dict[int, int],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Finalize raw counts using the mapping policy."""
        return self._policy.finalize_counts(in_count, out_count)

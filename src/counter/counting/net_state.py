from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from counter.domain.types import Side, Track
from counter.counting.geometry import Line, bottom_center, side_of

@dataclass
class TrackState:
    track_id: int
    initial_side: Optional[Side] = None
    last_side: Optional[Side] = None
    counted: bool = False
    counted_direction: Optional[str] = None
    class_hist: Dict[int, int] = field(default_factory=dict)

    def vote_class(self, mapped_class_id: int):
        self.class_hist[mapped_class_id] = self.class_hist.get(mapped_class_id, 0) + 1

    def majority_class(self) -> Optional[int]:
        if not self.class_hist:
            return None
        return sorted(self.class_hist.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

class NetStateCounter:
    def __init__(self, line: Line, greyzone_px: float):
        self.line = line
        self.greyzone_px = float(greyzone_px)
        self.states: Dict[int, TrackState] = {}

    def observe(self, tracks: List[Track]):
        for tr in tracks:
            st = self.states.get(tr.track_id)
            if st is None:
                st = TrackState(track_id=tr.track_id)
                self.states[tr.track_id] = st

            st.vote_class(tr.mapped_class_id)

            pt = bottom_center(tr.bbox)
            current_side = side_of(pt, self.line, self.greyzone_px)
            if current_side == Side.UNKNOWN:
                continue

            if st.initial_side is None:
                st.initial_side = current_side
                st.last_side = current_side
                continue

            if st.last_side is None:
                st.last_side = current_side
                continue

            if current_side != st.last_side:
                if not st.counted:
                    direction = 'in' if (st.initial_side == Side.OUT and current_side == Side.IN) else 'out'
                    st.counted = True
                    st.counted_direction = direction
                else:
                    if current_side == st.initial_side:
                        st.counted = False
                        st.counted_direction = None
                st.last_side = current_side


    def snapshot_raw_counts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Best-effort live counts for UI progress.

        It uses the same logic as finalize_raw_counts(), but does not modify state.
        Counts can still change later if a track returns back across the line.
        """
        return self.finalize_raw_counts()

    def finalize_raw_counts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        in_counts: Dict[int, int] = {}
        out_counts: Dict[int, int] = {}
        for st in self.states.values():
            if not st.counted or st.counted_direction is None:
                continue
            cls = st.majority_class()
            if cls is None:
                continue
            if st.counted_direction == 'in':
                in_counts[cls] = in_counts.get(cls, 0) + 1
            else:
                out_counts[cls] = out_counts.get(cls, 0) + 1
        return in_counts, out_counts

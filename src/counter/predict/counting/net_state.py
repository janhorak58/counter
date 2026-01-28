from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Tuple

from counter.core.types import LineCoords, Side
from counter.predict.counting.geometry import classify_point
from counter.predict.types import MappedTrack


class NetState(Enum):
    UNKNOWN = "unknown"
    IN = "in"
    OUT = "out"


@dataclass
class TrackState:
    side: Side = Side.ON
    net_state: NetState = NetState.UNKNOWN
    last_xy: Optional[Tuple[float, float]] = None
    voted_class_id: Optional[int] = None

    def vote_class(self, class_id: int) -> None:
        if self.voted_class_id is None:
            self.voted_class_id = int(class_id)


def _net_transition(prev: NetState, current_side: Side) -> NetState:
    if prev == NetState.UNKNOWN:
        if current_side == Side.IN:
            return NetState.IN
        if current_side == Side.OUT:
            return NetState.OUT
        return NetState.UNKNOWN

    if prev == NetState.IN:
        if current_side in (Side.IN, Side.ON):
            return NetState.IN
        if current_side == Side.OUT:
            return NetState.OUT

    if prev == NetState.OUT:
        if current_side in (Side.OUT, Side.ON):
            return NetState.OUT
        if current_side == Side.IN:
            return NetState.IN

    return prev


@dataclass
class NetStateCounter:
    line: LineCoords
    line_base_resolution: Tuple[int, int] = (1920, 1080)
    greyzone_px: float = 0.0

    # internal state
    states: Dict[int, TrackState] = None  # type: ignore

    def __post_init__(self) -> None:
        if self.states is None:
            self.states = {}

    def reset(self) -> None:
        self.states = {}

    def update(self, track: MappedTrack, xy: Tuple[float, float], class_id: int, video_resolution: Tuple[int, int]) -> Optional[str]:
        """Update state of one track and return event: 'in' | 'out' | None."""
        st = self.states.setdefault(int(track.track_id), TrackState())
        st.vote_class(int(class_id))

        side = classify_point(self.line, xy, video_resolution, self.line_base_resolution)
        
        prev_net = st.net_state
        new_net = _net_transition(prev_net, side)
        if track.initial_side == Side.UNKNOWN:
            track.initial_side = side
        track.current_side = side
        st.side = side
        st.net_state = new_net
        st.last_xy = xy

        if prev_net == NetState.IN and new_net == NetState.OUT:
            return "out"
        if prev_net == NetState.OUT and new_net == NetState.IN:
            return "in"
        return None

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple

from counter.core.types import LineCoords, Side
from counter.predict.counting.geometry import classify_point
from counter.predict.types import MappedTrack


class NetState(Enum):
    """Coarse net state of a track relative to its initial side."""

    UNKNOWN = "unknown"
    AT_INITIAL = "at_initial"
    AWAY = "away"


@dataclass
class TrackState:
    """Per-track state used for counting and optional visualization."""

    initial_side: Side = Side.UNKNOWN
    current_side: Side = Side.UNKNOWN

    net_state: NetState = NetState.UNKNOWN

    voted_class_id: Optional[int] = None

    # If the object has been counted away from its initial side, remember direction and frame.
    counted_dir: Optional[str] = None  # 'in' | 'out'
    counted_frame: Optional[int] = None

    # Last N points (bottom-center), for debug rendering.
    history: Deque[Tuple[float, float]] = field(default_factory=deque)

    def vote_class(self, class_id: int) -> None:
        if self.voted_class_id is None:
            self.voted_class_id = int(class_id)


@dataclass
class NetStateCounter:
    """Track line events relative to each track's initial side.

    Rules:
    - initial_side is set once (first non-ON side).
    - event is emitted when current_side becomes the opposite side of initial_side.
    - if the track returns back to initial_side within oscillation_window_frames, an undo event is emitted.

    Events:
    - 'in' | 'out'
    - 'undo_in' | 'undo_out'
    - None
    """

    line: LineCoords
    line_base_resolution: Tuple[int, int] = (1920, 1080)
    greyzone_px: float = 0.0

    oscillation_window_frames: int = 0
    trajectory_len: int = 40

    # Internal per-track state.
    states: Dict[int, TrackState] = field(default_factory=dict)

    def reset(self) -> None:
        self.states = {}

    def get_trajectories(self) -> Dict[int, List[Tuple[float, float]]]:
        """Return a copy of trajectories (track_id -> list of xy points)."""
        out: Dict[int, List[Tuple[float, float]]] = {}
        for tid, st in self.states.items():
            if st.history:
                out[int(tid)] = list(st.history)
        return out

    def update(
        self,
        track: MappedTrack,
        xy: Tuple[float, float],
        class_id: int,
        video_resolution: Tuple[int, int],
        *,
        frame_idx: int,
    ) -> Optional[str]:
        """Update one track and return an event: 'in' | 'out' | 'undo_in' | 'undo_out' | None."""
        tid = int(track.track_id)
        st = self.states.get(tid)
        if st is None:
            st = TrackState()
            st.history = deque(maxlen=int(self.trajectory_len))
            self.states[tid] = st

        st.vote_class(int(class_id))
        st.history.append((float(xy[0]), float(xy[1])))

        side = classify_point(self.line, xy, video_resolution, self.line_base_resolution, self.greyzone_px)
        st.current_side = side
        track.current_side = side

        # Initialize initial side once we get a stable side.
        if track.initial_side == Side.UNKNOWN and side in (Side.IN, Side.OUT):
            track.initial_side = side
        if st.initial_side == Side.UNKNOWN and track.initial_side in (Side.IN, Side.OUT):
            st.initial_side = track.initial_side

        # Still unknown -> cannot decide.
        if st.initial_side == Side.UNKNOWN:
            st.net_state = NetState.UNKNOWN
            return None

        # Ignore ON state to reduce jitter near the line.
        if side == Side.ON:
            return None

        if side == st.initial_side:
            st.net_state = NetState.AT_INITIAL

            # If we were counted away and we returned, we can undo.
            if st.counted_dir is not None and st.counted_frame is not None:
                dt = int(frame_idx) - int(st.counted_frame)
                if int(self.oscillation_window_frames) <= 0 or dt <= int(self.oscillation_window_frames):
                    ev = f"undo_{st.counted_dir}"
                    st.counted_dir = None
                    st.counted_frame = None
                    return ev
            return None

        # We are on the opposite side of initial.
        st.net_state = NetState.AWAY

        desired: Optional[str] = None
        if st.initial_side == Side.OUT and side == Side.IN:
            desired = "in"
        elif st.initial_side == Side.IN and side == Side.OUT:
            desired = "out"

        if desired is None:
            return None

        # Count only once while away.
        if st.counted_dir is None:
            st.counted_dir = desired
            st.counted_frame = int(frame_idx)
            return desired

        return None

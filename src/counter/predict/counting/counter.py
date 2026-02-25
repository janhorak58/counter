from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Tuple

from counter.core.types import CanonicalClass, LineCoords
from counter.predict.counting.geometry import bottom_center
from counter.predict.counting.net_state import NetStateCounter
from counter.predict.types import MappedTrack

FinalizeFn = Callable[[Dict[int, int], Dict[int, int]], Tuple[Dict[int, int], Dict[int, int]]]


def _fill_missing(counts: Dict[int, int], keys: Iterable[int]) -> Dict[int, int]:
    """Ensure all canonical class IDs exist in a counts dict."""
    return {int(k): int(counts.get(int(k), 0)) for k in keys}


@dataclass
class TrackCounter:
    """Count in/out events for tracked objects crossing a line.

    This counter is robust against short oscillations (crossing and returning)
    by undoing events when the track returns to its initial side within
    oscillation_window_frames.

    Trajectories of the last `trajectory_len` frames are stored inside NetStateCounter
    and can be used for visualization.
    """

    line: LineCoords
    finalize_fn: FinalizeFn
    greyzone_px: float = 0.0
    oscillation_window_frames: int = 40
    trajectory_len: int = 40
    line_base_resolution: Tuple[int, int] = (1920, 1080)
    log: Callable[..., None] = lambda *a, **k: None

    def __post_init__(self) -> None:
        self.net = NetStateCounter(
            line=self.line,
            line_base_resolution=self.line_base_resolution,
            greyzone_px=self.greyzone_px,
            oscillation_window_frames=int(self.oscillation_window_frames),
            trajectory_len=int(self.trajectory_len),
        )
        self.raw_in_counts: Dict[int, int] = {}
        self.raw_out_counts: Dict[int, int] = {}
        self._frame_idx: int = 0
        self.video_resolution: Tuple[int, int] = (1920, 1080)
        self._canon_ids: List[int] = [int(c.value) for c in CanonicalClass]

    def reset(self, video_resolution: Tuple[int, int]) -> None:
        """Reset internal counters for a new video."""
        self.net.reset()
        self.raw_in_counts = {}
        self.raw_out_counts = {}
        self._frame_idx = 0
        self.video_resolution = video_resolution

    def update(self, tracks: List[MappedTrack]) -> None:
        """Update counters with a batch of tracks from the current frame."""
        self._frame_idx += 1

        for tr in tracks:
            tid = int(tr.track_id)
            xy = bottom_center(tr.bbox)

            ev = self.net.update(
                track=tr,
                xy=xy,
                class_id=int(tr.mapped_class_id),
                video_resolution=self.video_resolution,
                frame_idx=int(self._frame_idx),
            )
            if not ev:
                continue

            voted = self.net.states[tid].voted_class_id if tid in self.net.states else None
            cid = int(voted) if voted is not None else int(tr.mapped_class_id)

            if ev == "in":
                self.log("Line crossed IN", {"track_id": tid, "class_id": cid, "frame_idx": self._frame_idx})
                self.raw_in_counts[cid] = int(self.raw_in_counts.get(cid, 0) + 1)
            elif ev == "out":
                self.log("Line crossed OUT", {"track_id": tid, "class_id": cid, "frame_idx": self._frame_idx})
                self.raw_out_counts[cid] = int(self.raw_out_counts.get(cid, 0) + 1)
            elif ev == "undo_in":
                self.log(
                    "Undo IN (returned)",
                    {"track_id": tid, "class_id": cid, "frame_idx": self._frame_idx},
                )
                self.raw_in_counts[cid] = max(0, int(self.raw_in_counts.get(cid, 0)) - 1)
            elif ev == "undo_out":
                self.log(
                    "Undo OUT (returned)",
                    {"track_id": tid, "class_id": cid, "frame_idx": self._frame_idx},
                )
                self.raw_out_counts[cid] = max(0, int(self.raw_out_counts.get(cid, 0)) - 1)

    def snapshot_counts(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Return current counts with missing classes filled in."""
        fin_in, fin_out = self.finalize_fn(dict(self.raw_in_counts), dict(self.raw_out_counts))
        return _fill_missing(fin_in, self._canon_ids), _fill_missing(fin_out, self._canon_ids)

    def finalize(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """Finalize and return counts (alias for snapshot_counts)."""
        return self.snapshot_counts()

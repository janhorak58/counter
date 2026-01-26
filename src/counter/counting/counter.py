from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Tuple

from counter.counting.geometry import Line
from counter.counting.net_state import NetStateCounter
from counter.domain.types import Track

Finalizer = Callable[[Dict[int, int], Dict[int, int]], Tuple[Dict[int, int], Dict[int, int]]]


def _identity_finalizer(
    in_counts: Dict[int, int], out_counts: Dict[int, int]
) -> Tuple[Dict[int, int], Dict[int, int]]:
    return dict(in_counts), dict(out_counts)


class TrackCounter:
    """Counts tracked objects crossing a line using net-state logic."""

    def __init__(
        self,
        line_start: Iterable[float],
        line_end: Iterable[float],
        greyzone_px: float,
        class_ids: Iterable[int],
        finalize_counts: Finalizer | None = None,
    ):
        self._line_start = tuple(line_start)
        self._line_end = tuple(line_end)
        self._greyzone_px = float(greyzone_px)
        self._class_ids = [int(c) for c in class_ids]
        self._finalize_counts = finalize_counts or _identity_finalizer
        self.reset()

    def reset(self) -> None:
        self._counter = NetStateCounter(Line(self._line_start, self._line_end), self._greyzone_px)

    def update(self, tracks: List[Track]) -> None:
        self._counter.observe(tracks)

    def finalize(self) -> Dict[str, Dict[int, int]]:
        in_counts, out_counts = self._counter.finalize_raw_counts()
        in_counts, out_counts = self._finalize_counts(in_counts, out_counts)
        return {
            "in_count": self._fill_missing(in_counts),
            "out_count": self._fill_missing(out_counts),
        }

    def _fill_missing(self, counts: Dict[int, int]) -> Dict[int, int]:
        if not self._class_ids:
            return dict(counts)
        return {int(cid): int(counts.get(int(cid), 0)) for cid in self._class_ids}

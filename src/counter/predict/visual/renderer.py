from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:  # pragma: no cover
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

from counter.core.types import CanonicalClass, LineCoords
from counter.predict.mapping.pretrained import INTERMEDIATE_NAMES
from counter.predict.types import MappedTrack, RawTrack


def _put_text(img, text: str, org: Tuple[int, int], scale: float = 0.6, thick: int = 1):
    """Render text onto an image if OpenCV is available."""
    if cv2 is None:
        return
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thick, cv2.LINE_AA)


def _resize_keep_aspect(frame, max_width: int):
    """Resize frame to max_width while preserving aspect ratio."""
    if cv2 is None:
        return frame
    h, w = frame.shape[:2]
    if max_width <= 0 or w <= max_width:
        return frame
    scale = float(max_width) / float(w)
    nh = int(h * scale)
    return cv2.resize(frame, (max_width, nh))


@dataclass
class FrameRenderer:
    """Render tracks, counts, and optional debug overlays on frames."""

    line: LineCoords
    show_boxes: bool = True
    show_stats: bool = True
    show_raw: bool = False
    show_dropped_raw: bool = False

    def render(
        self,
        frame_bgr,
        *,
        tracks: List[MappedTrack],
        in_counts: Dict[int, int],
        out_counts: Dict[int, int],
        frame_idx: int,
        fps: Optional[float] = None,
        raw_tracks: Optional[List[RawTrack]] = None,
        total_frames: Optional[int] = None,
    ):
        """Draw line, tracks, and counters onto the provided frame."""
        if cv2 is None:
            return frame_bgr

        x1, y1, x2, y2 = [int(v) for v in self.line]
        cv2.line(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)

        mapped_ids = {int(tr.track_id) for tr in tracks}

        # Debug overlay for raw detections dropped by the mapper.
        if self.show_dropped_raw and raw_tracks:
            for rt in raw_tracks:
                if int(rt.track_id) in mapped_ids:
                    continue
                bx1, by1, bx2, by2 = [int(v) for v in rt.bbox]
                cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                txt = f"RAW_DROP | {rt.score:.2f} | id:{rt.track_id} | raw:{rt.raw_class_id}:{rt.raw_class_name}"
                _put_text(frame_bgr, txt, (bx1, max(0, by1 - 6)), scale=0.5, thick=1)

        if self.show_boxes:
            for tr in tracks:
                bx1, by1, bx2, by2 = [int(v) for v in tr.bbox]
                cv2.rectangle(frame_bgr, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                cv2.circle(frame_bgr, (int((bx1 + bx2) / 2), by2), 3, (0, 255, 0), -1)
                label = INTERMEDIATE_NAMES.get(int(tr.mapped_class_id))
                if label is None:
                    try:
                        label = CanonicalClass(int(tr.mapped_class_id)).name.lower()
                    except Exception:
                        label = str(tr.mapped_class_id)

                parts = [label]
                parts.append(f"{tr.score:.2f}")
                parts.append(f"id:{tr.track_id}")
                if self.show_raw:
                    parts.append(f"raw:{tr.raw_class_id}:{tr.raw_class_name}")
                txt = " | ".join(parts)
                _put_text(frame_bgr, txt, (bx1, max(0, by1 - 6)), scale=0.5, thick=1)

        if self.show_stats:
            y = 26
            _put_text(
                frame_bgr,
                f"frame {frame_idx}/{total_frames if total_frames is not None else '?'}",
                (10, y),
                scale=0.6,
                thick=1,
            )
            y += 26
            if fps:
                _put_text(frame_bgr, f"fps {fps:.2f}", (10, y), scale=0.6, thick=1)
                y += 26

            for c in CanonicalClass:
                cid = int(c)
                name = c.name.lower()
                inc = int(in_counts.get(cid, 0))
                outc = int(out_counts.get(cid, 0))
                _put_text(frame_bgr, f"{name}: IN {inc} OUT {outc}", (10, y), scale=0.6, thick=1)
                y += 26

        return frame_bgr

    def preview_frame(self, frame_bgr, max_width: int):
        """Resize frame for preview display."""
        return _resize_keep_aspect(frame_bgr, max_width=max_width)

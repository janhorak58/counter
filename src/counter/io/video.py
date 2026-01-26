from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

@dataclass(frozen=True)
class VideoInfo:
    path: str
    fps: float
    frame_count: int
    width: int
    height: int

def open_video(path: str):
    if cv2 is None:
        raise ImportError("opencv-python not installed. Install extras: uv pip install -e '.[predict]'")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f'Cannot open video: {path}')
    return cap

def get_video_info(path: str) -> VideoInfo:
    cap = open_video(path)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return VideoInfo(path=path, fps=fps, frame_count=frame_count, width=width, height=height)

def iter_frames(path: str) -> Iterator[Tuple[int, "cv2.Mat"]]:
    cap = open_video(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()

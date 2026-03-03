from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2



def load_first_frame_rgb(video_path: str | Path) -> Tuple[Any, int, int]:
    p = Path(video_path)
    cap = cv2.VideoCapture(str(p))
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise ValueError(f"Failed to load first frame from video: {p}")

    h, w = frame.shape[:2]
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame_rgb, int(w), int(h)



def extract_two_points(canvas_json: Optional[Dict[str, Any]]) -> List[Tuple[int, int]]:
    if not canvas_json:
        return []

    objs = canvas_json.get("objects", []) if isinstance(canvas_json, dict) else []
    out: List[Tuple[int, int]] = []

    for obj in objs:
        if not isinstance(obj, dict):
            continue

        typ = str(obj.get("type", "")).lower()

        if typ == "circle":
            left = float(obj.get("left", 0.0))
            top = float(obj.get("top", 0.0))
            radius = float(obj.get("radius", 0.0))
            sx = float(obj.get("scaleX", 1.0))
            sy = float(obj.get("scaleY", 1.0))
            x = int(round(left + radius * sx))
            y = int(round(top + radius * sy))
            out.append((x, y))

        elif typ == "path":
            path = obj.get("path", [])
            if isinstance(path, list) and path:
                first = path[0]
                if isinstance(first, list) and len(first) >= 3:
                    try:
                        x = int(round(float(first[1])))
                        y = int(round(float(first[2])))
                        out.append((x, y))
                    except Exception:
                        pass

        if len(out) >= 2:
            break

    return out[:2]

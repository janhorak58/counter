from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _to_int_keys(d: Any) -> Dict[int, int]:
    if not isinstance(d, dict):
        return {}
    out: Dict[int, int] = {}
    for k, v in d.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def load_counts_json(path: str | Path) -> Dict[str, Any]:
    """Load *.counts.json and normalize in_count/out_count keys to ints."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    obj["in_count"] = _to_int_keys(obj.get("in_count", {}))
    obj["out_count"] = _to_int_keys(obj.get("out_count", {}))
    return obj


def load_gt_dir_counts(gt_dir: str | Path) -> Dict[str, Dict[str, Any]]:
    """Return map: video_stem -> GT object loaded from *.counts.json."""
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(gt_dir)
    if not p.exists():
        return out

    for fp in sorted(p.glob("*.counts.json")):
        obj = load_counts_json(fp)
        video = obj.get("video")
        if isinstance(video, str) and video:
            key = Path(video).stem
        else:
            key = fp.name.replace(".counts.json", "")
        out[key] = obj

    return out

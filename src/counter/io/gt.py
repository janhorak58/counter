from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import ast
import csv
import json


def load_gt_csv(path: str) -> Tuple[str, Dict[int, int], Dict[int, int]]:
    """
    Legacy formát:
      line_name,in_count,out_count
      Line_1,"{0: 59,...}","{0: 42,...}"
    """
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    line = row["line_name"]
    in_counts = ast.literal_eval(row["in_count"])
    out_counts = ast.literal_eval(row["out_count"])
    return line, dict(in_counts), dict(out_counts)


def load_counts_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

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

    obj["in_count"] = _to_int_keys(obj.get("in_count", {}))
    obj["out_count"] = _to_int_keys(obj.get("out_count", {}))
    return obj


def load_gt_dir_counts(gt_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Vrátí mapu: vid_stem -> GT objekt (z .counts.json)
    """
    out: Dict[str, Dict[str, Any]] = {}
    p = Path(gt_dir)
    if not p.exists():
        return out

    for fp in sorted(p.glob("*.counts.json")):
        obj = load_counts_json(str(fp))
        video = obj.get("video")
        if isinstance(video, str) and video:
            key = Path(video).stem
        else:
            # fallback: podle názvu souboru
            key = fp.name.replace(".counts.json", "")
        out[key] = obj
    return out

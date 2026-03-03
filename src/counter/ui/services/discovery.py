from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from counter.eval.logic.discover import discover_predict_runs as _discover_predict_runs


@dataclass
class EvalRunInfo:
    eval_dir: Path
    name: str



def discover_predict_runs(runs_root: str | Path) -> List[Dict[str, Any]]:
    infos = _discover_predict_runs(runs_root)
    out: List[Dict[str, Any]] = []

    for info in infos:
        out.append(
            {
                "run_id": info.run_id,
                "run_dir": Path(info.run_dir),
                "predict_dir": Path(info.predict_dir),
                "model_id": info.model_id,
                "backend": info.backend,
                "variant": info.variant,
                "status": info.status,
            }
        )

    out.sort(key=lambda r: str(r["run_id"]), reverse=True)
    return out



def discover_eval_runs(eval_root: str | Path) -> List[EvalRunInfo]:
    root = Path(eval_root)
    if not root.exists():
        return []

    out: List[EvalRunInfo] = []
    for p in root.rglob("per_run_metrics.csv"):
        d = p.parent
        out.append(EvalRunInfo(eval_dir=d, name=d.name))

    out.sort(key=lambda x: x.eval_dir.stat().st_mtime, reverse=True)
    return out



def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:
        return None



def load_counts_rows(predict_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for f in sorted(predict_dir.glob("*.counts.json")):
        if f.name == "aggregate.counts.json":
            continue

        obj = load_json(f)
        if not obj:
            continue

        in_count = obj.get("in_count", {}) or {}
        out_count = obj.get("out_count", {}) or {}

        rows.append(
            {
                "file": f.name,
                "video": obj.get("video", ""),
                "line_name": obj.get("line_name", ""),
                "in_total": sum(int(v) for v in in_count.values()),
                "out_total": sum(int(v) for v in out_count.values()),
                "path": str(f),
            }
        )

    return rows



def list_pred_videos(predict_dir: Path) -> List[Path]:
    vids = list(predict_dir.glob("*.pred.mp4")) + list(predict_dir.glob("*.avi"))
    vids.sort()
    return vids



def load_eval_tables(eval_dir: Path) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}

    for name, fname in [
        ("per_run", "per_run_metrics.csv"),
        ("per_video", "per_video_metrics.csv"),
        ("per_class", "per_class_metrics.csv"),
    ]:
        p = eval_dir / fname
        if p.exists():
            out[name] = pd.read_csv(p)
        else:
            out[name] = pd.DataFrame()

    return out



def list_chart_images(eval_dir: Path) -> List[Path]:
    charts = sorted((eval_dir / "charts").rglob("*.png")) if (eval_dir / "charts").exists() else []
    return charts

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from counter.core.io.json import read_json
from counter.core.schema import EvalConfig


@dataclass(frozen=True)
class PredictRunInfo:
    run_id: str
    run_dir: Path
    predict_dir: Path
    model_id: str
    backend: str
    variant: str
    status: str


def infer_from_dirname(dirname: str) -> Tuple[str, str, str]:
    """Supports names like:
      yolo_tuned__yolo11m_v11
      rfdetr_pretrained__rfdetr_small_pretrained
      20260125_224012
    """
    name = dirname.lower()

    if name.startswith("yolo"):
        backend = "yolo"
    elif name.startswith("rfdetr"):
        backend = "rfdetr"
    else:
        backend = "unknown"

    if "tuned" in name:
        variant = "tuned"
    elif "pretrained" in name:
        variant = "pretrained"
    else:
        variant = "unknown"

    model_id = dirname
    if "__" in dirname:
        model_id = dirname.split("__", 1)[1]

    return backend, variant, model_id


def mk_runinfo_from_run_dir(run_dir: Path) -> Optional[PredictRunInfo]:
    predict_dir = run_dir / "predict"
    if not predict_dir.exists():
        return None

    meta: Dict[str, Any] = {}
    run_json_path = run_dir / "run.json"
    if run_json_path.exists():
        obj = read_json(run_json_path)
        if isinstance(obj, dict):
            meta = obj

    backend_fallback, variant_fallback, model_fallback = infer_from_dirname(run_dir.name)
    status = str(meta.get("status", "completed"))

    return PredictRunInfo(
        run_id=str(meta.get("run_id", run_dir.name)),
        run_dir=run_dir,
        predict_dir=predict_dir,
        model_id=str(meta.get("model_id", model_fallback)),
        backend=str(meta.get("backend", backend_fallback)),
        variant=str(meta.get("variant", variant_fallback)),
        status=status,
    )


def discover_predict_runs(runs_dir: str | Path) -> List[PredictRunInfo]:
    p = Path(runs_dir)
    if not p.exists():
        return []

    # 1) runs/<id> (contains run.json + predict/)
    if (p / "run.json").exists() and (p / "predict").exists():
        one = mk_runinfo_from_run_dir(p)
        return [one] if one else []

    # 2) runs/<id>/predict passed directly
    if p.name == "predict" and (p.parent / "run.json").exists():
        one = mk_runinfo_from_run_dir(p.parent)
        return [one] if one else []

    # 3) runs/ directory
    out: List[PredictRunInfo] = []
    for run_dir in sorted([x for x in p.iterdir() if x.is_dir()]):
        info = mk_runinfo_from_run_dir(run_dir)
        if info:
            out.append(info)
    return out


def passes_filters(run: PredictRunInfo, cfg: EvalConfig) -> bool:
    f = cfg.filters
    if cfg.only_completed and run.status != "completed":
        return False
    if f.run_ids and run.run_id not in f.run_ids:
        return False
    if f.backends and run.backend not in f.backends:
        return False
    if f.variants and run.variant not in f.variants:
        return False
    if f.model_ids and run.model_id not in f.model_ids:
        return False
    return True

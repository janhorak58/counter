from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from counter.core.io import dump_json
from counter.core.pipeline.base import StageContext
from counter.core.schema import PredictConfig


@dataclass
class FinalizeRun:
    """Stage that writes run metadata and aggregates output paths."""
    name: str = "finalize_run"

    def run(self, ctx: StageContext) -> None:
        cfg: PredictConfig = ctx.cfg
        spec = ctx.state["model_spec"]
        run_id: str = ctx.state["run_id"]
        run_root: Path = ctx.state["run_root"]
        predict_dir: Path = ctx.state["predict_dir"]
        log = ctx.assets.get("log")

        counts_paths: List[str] = list(ctx.state.get("counts_paths", []))

        run_json: Dict[str, Any] = {
            "run_id": run_id,
            "status": "completed",
            "model_id": cfg.model_id,
            "backend": spec.backend,
            "variant": spec.variant,
            "weights": spec.weights,
            "mapping": spec.mapping._get_dict() if spec.mapping is not None else None,
            "thresholds": {"conf": cfg.thresholds.conf, "iou": cfg.thresholds.iou},
            "tracker": {"type": cfg.tracking.type, "params": cfg.tracking.params},
            "line": {"name": cfg.line.name, "coords": list(cfg.line.coords)},
            "greyzone_px": float(cfg.greyzone_px),
            "videos_dir": cfg.videos_dir,
            "videos": list(cfg.videos),
        }
        dump_json(run_root / "run.json", run_json)

        dump_json(predict_dir / "aggregate.counts.json", {"run_id": run_id, "counts": counts_paths})

        log("run_done", {"run_id": run_id, "run_root": str(run_root), "predict_dir": str(predict_dir)})

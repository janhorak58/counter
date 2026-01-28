from __future__ import annotations

from typing import Any, Dict, List

from counter.core.pipeline.base import StageContext
from counter.eval.logic.compute import evaluate_one_run


class EvaluateRuns:
    """Stage that evaluates all discovered runs."""
    name = "EvaluateRuns"

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        runs = ctx.state["runs"]
        gt_map = ctx.state["gt_map"]
        classes = ctx.state["classes"]
        class_names = ctx.state["class_names"]
        charts_dir = ctx.state["charts_dir"]

        per_run_rows: List[Dict[str, Any]] = ctx.state["per_run_rows"]
        per_video_rows: List[Dict[str, Any]] = ctx.state["per_video_rows"]
        per_class_rows: List[Dict[str, Any]] = ctx.state["per_class_rows"]

        log = ctx.assets.get("log")

        for run in runs:
            run_row, video_rows, class_rows = evaluate_one_run(
                cfg=cfg,
                run=run,
                gt_map=gt_map,
                classes=classes,
                class_names=class_names,
                charts_dir=charts_dir,
                log=log,
            )
            per_run_rows.append(run_row)
            per_video_rows.extend(video_rows)
            per_class_rows.extend(class_rows)

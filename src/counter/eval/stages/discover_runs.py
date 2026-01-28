from __future__ import annotations

from pathlib import Path
from typing import Optional

from counter.core.pipeline.base import StageContext
from counter.eval.logic.discover import discover_predict_runs, passes_filters


class DiscoverPredictRuns:
    name = "DiscoverPredictRuns"

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        p: Optional[Path] = ctx.state.get("predict_run_dir")

        runs = discover_predict_runs(p) if p is not None else discover_predict_runs(Path(cfg.runs_dir))
        runs = [r for r in runs if passes_filters(r, cfg)]

        if not runs:
            raise FileNotFoundError(f"No matching predict runs found in: {cfg.runs_dir}")

        ctx.state["runs"] = runs

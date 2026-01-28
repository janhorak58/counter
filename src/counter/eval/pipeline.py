from __future__ import annotations

from pathlib import Path
from typing import Optional

from counter.core.schema import EvalConfig  # :contentReference[oaicite:1]{index=1}
from counter.core.pipeline.base import PipelineRunner, StageContext
from counter.core.pipeline.log import JsonlLogger

from counter.eval.stages.load_gt import LoadGTCounts
from counter.eval.stages.discover_runs import DiscoverPredictRuns
from counter.eval.stages.init_output import InitOutput
from counter.eval.stages.eval_runs import EvaluateRuns
from counter.eval.stages.export_rank_charts import RankExportCharts


class EvalPipeline:
    def run(self, cfg: EvalConfig, predict_run_dir: str | Path | None = None) -> Path:
        runner = PipelineRunner(
            stages=[
                LoadGTCounts(),
                DiscoverPredictRuns(),
                InitOutput(),
                EvaluateRuns(),
                RankExportCharts(),
            ],
            fail_fast=True,
        )

        out_dir = Path(cfg.out_dir) / f"eval_{cfg.timestamp}"
        log = JsonlLogger(out_dir / "eval.log.jsonl")

        ctx = StageContext(
            cfg=cfg,
            state={"predict_run_dir": Path(predict_run_dir) if predict_run_dir else None},
            assets={"log": log},
        )
        runner.run(ctx)
        return ctx.state["out_root"]

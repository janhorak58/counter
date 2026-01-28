from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from counter.core.pipeline.base import StageContext
from counter.eval.logic.ranking import rank_runs
from counter.eval.logic.export import export_csvs, export_json_bundle
from counter.eval.logic.charts import export_summary_charts


class RankExportCharts:
    name = "RankExportCharts"

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg

        out_root: Path = ctx.state["out_root"]
        charts_dir: Path = ctx.state["charts_dir"]
        classes = ctx.state["classes"]
        class_names = ctx.state["class_names"]

        per_run_rows: List[Dict[str, Any]] = ctx.state["per_run_rows"]
        per_video_rows: List[Dict[str, Any]] = ctx.state["per_video_rows"]
        per_class_rows: List[Dict[str, Any]] = ctx.state["per_class_rows"]

        score_field, ranked = rank_runs(cfg, per_run_rows)

        export_json_bundle(
            out_root=out_root,
            score_field=score_field,
            classes=[{"id": int(c), "name": n} for c, n in zip(classes, class_names)],
            ranked_runs=ranked,
            per_run=per_run_rows,
            per_video=per_video_rows,
            per_class=per_class_rows,
            notes={
                "score_total_video_mae": "Avg MAE over (IN_total, OUT_total), each video equal weight.",
                "score_total_event_wape": "Avg class-aware (micro) WAPE over (IN, OUT). WAPE=sum_over_classes(|err|)/sum_over_classes(GT). Penalizes class swaps.",
                "score_total_rate_mae": "Avg abs error of passages/hour (IN & OUT). Needs durations.",
                "score_total_class_wape": "Avg macro-WAPE over classes (IN & OUT).",
            },
        )

        export_csvs(
            out_root=out_root,
            score_field=score_field,
            class_names=class_names,
            ranked_runs=ranked,
            per_video_rows=per_video_rows,
            per_class_rows=per_class_rows,
        )

        if cfg.charts.enabled:
            export_summary_charts(
                charts_dir=charts_dir,
                score_field=score_field,
                ranked_runs=ranked,
                per_video_rows=per_video_rows,
                per_class_rows=per_class_rows,
                class_names=class_names,
            )

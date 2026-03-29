from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from counter.core.io.csv import write_csv
from counter.core.io.json import dump_json


def export_json_bundle(
    *,
    out_root: Path,
    rank_by: str,
    score_field: str,
    classes: List[Dict[str, Any]],
    ranked_runs: List[Dict[str, Any]],
    per_run: List[Dict[str, Any]],
    per_video: List[Dict[str, Any]],
    per_class: List[Dict[str, Any]],
    notes: Dict[str, str],
) -> None:
    """Export evaluation results as JSON files."""
    dump_json(
        out_root / "benchmark.json",
        {
            "rank_by": rank_by,
            "score_field": score_field,
            "classes": classes,
            "ranked_runs": ranked_runs,
            "notes": notes,
        },
    )

    dump_json(out_root / "metrics.json", {"per_run": per_run, "per_video": per_video, "per_class": per_class})


def export_csvs(
    *,
    out_root: Path,
    score_field: str,
    class_names: List[str],
    ranked_runs: List[Dict[str, Any]],
    per_video_rows: List[Dict[str, Any]],
    per_class_rows: List[Dict[str, Any]],
) -> None:
    """Export evaluation results as CSV files."""
    per_run_cols = [
        "rank",
        "run_id",
        "model_id",
        "backend",
        "variant",
        score_field,
        "score_total_video_mae",
        "score_total_micro_wape",
        "score_total_macro_wape",
        "score_total_rate_mae",
        "mae_in_total",
        "rmse_in_total",
        "bias_in_total",
        "mae_out_total",
        "rmse_out_total",
        "bias_out_total",
        "mae_in_total_counts",
        "rmse_in_total_counts",
        "bias_in_total_counts",
        "mae_out_total_counts",
        "rmse_out_total_counts",
        "bias_out_total_counts",
        "wape_in_totalcounts",
        "wape_out_totalcounts",
        "wape_in_total_l1",
        "wape_out_total_l1",
        "wape_micro_in",
        "wape_micro_out",
        "wape_wmacro_gt_in",
        "wape_wmacro_gt_out",
        "class_wape_macro_in",
        "class_wape_macro_out",
        "sum_gt_in_total",
        "sum_gt_out_total",
        "sum_pred_in_total",
        "sum_pred_out_total",
        "sum_abs_err_in_total",
        "sum_abs_err_out_total",
        "sum_duration_s",
        "gt_in_per_h",
        "pred_in_per_h",
        "gt_out_per_h",
        "pred_out_per_h",
        "rate_mae_in_per_h",
        "rate_mae_out_per_h",
    ]

    seen = set()
    per_run_cols = [col for col in per_run_cols if not (col in seen or seen.add(col))]
    write_csv(out_root / "per_run_metrics.csv", ranked_runs, per_run_cols)

    per_video_cols = [
        "run_id",
        "model_id",
        "backend",
        "variant",
        "video",
        "duration_s",
        "gt_in_total",
        "pred_in_total",
        "abs_err_in_total",
        "err_in_total",
        "gt_out_total",
        "pred_out_total",
        "abs_err_out_total",
        "err_out_total",
        "gt_in_per_h",
        "pred_in_per_h",
        "gt_out_per_h",
        "pred_out_per_h",
        "mae_in_cls",
        "rmse_in_cls",
        "bias_in_cls",
        "mae_out_cls",
        "rmse_out_cls",
        "bias_out_cls",
    ]
    for cname in class_names:
        per_video_cols += [f"abs_err_in_{cname}", f"err_in_{cname}", f"abs_err_out_{cname}", f"err_out_{cname}"]
    write_csv(out_root / "per_video_metrics.csv", per_video_rows, per_video_cols)

    per_class_cols = [
        "run_id",
        "model_id",
        "backend",
        "variant",
        "direction",
        "class_id",
        "class_name",
        "mae",
        "rmse",
        "bias",
        "within1",
        "within2",
    ]
    write_csv(out_root / "per_class_metrics.csv", per_class_rows, per_class_cols)

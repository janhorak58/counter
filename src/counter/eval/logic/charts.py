from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib.pyplot as plt
from counter.eval.logic.metrics import safe_div

Counts = Union[Dict[int, int], Sequence[int]]


def _sort_labels_values(labels: Sequence[str], values: Sequence[float]) -> tuple[list[str], list[float]]:
    """Sort labels by value ascending, keeping labels aligned."""
    pairs = list(zip(labels, values))
    # Lower error is better, so sort ascending.
    pairs.sort(key=lambda t: (float("inf") if t[1] is None else float(t[1])))
    labs = [p[0] for p in pairs]
    vals = [float(p[1]) for p in pairs]
    return labs, vals


def _ensure_parent(path: Union[str, Path]) -> Path:
    """Ensure parent directory exists and return the Path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize(s: str) -> str:
    """Normalize whitespace for chart titles."""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s


def bar_counts(
    path: Union[str, Path],
    title: str,
    gt: Counts,
    pred: Counts,
    labels: Sequence[str],
) -> None:
    """Render a grouped bar chart for GT vs predictions."""
    p = _ensure_parent(path)

    gt_list = [int(gt.get(i, 0)) for i in range(len(labels))] if isinstance(gt, dict) else [int(x) for x in gt]
    pr_list = [int(pred.get(i, 0)) for i in range(len(labels))] if isinstance(pred, dict) else [int(x) for x in pred]

    x = list(range(len(labels)))
    width = 0.40

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.bar([i - width / 2 for i in x], gt_list, width=width, label="GT")
    ax.bar([i + width / 2 for i in x], pr_list, width=width, label="Pred")

    ax.set_title(_sanitize(title))
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=20, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)


def bar_metric(
    path: Union[str, Path],
    title: str,
    labels: Sequence[str],
    values: Sequence[float],
    ylabel: str = "",
) -> None:
    """Render a bar chart for a single metric across labels."""
    p = _ensure_parent(path)

    x = list(range(len(labels)))
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(111)
    ax.bar(x, list(values))
    ax.set_title(_sanitize(title))
    ax.set_xticks(x)
    ax.set_xticklabels(list(labels), rotation=20, ha="right")
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)


def scatter_xy(
    path: Union[str, Path],
    title: str,
    x: Sequence[float],
    y: Sequence[float],
    labels: Optional[Sequence[str]] = None,
    xlabel: str = "x",
    ylabel: str = "y",
) -> None:
    """Render a labeled scatter plot."""
    p = _ensure_parent(path)

    fig = plt.figure(figsize=(6.8, 5.6))
    ax = fig.add_subplot(111)

    ax.scatter(list(x), list(y))
    ax.set_title(_sanitize(title))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)

    if labels:
        for xi, yi, lab in zip(x, y, labels):
            ax.annotate(str(lab), (float(xi), float(yi)), fontsize=8, alpha=0.8)

    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)


def heatmap_matrix(
    path: Union[str, Path],
    title: str,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    matrix: Sequence[Sequence[float]],
    xlabel: str = "",
    ylabel: str = "",
    fmt: str = "{:.2f}",
) -> None:
    """Render a heatmap from a matrix of values."""
    p = _ensure_parent(path)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(matrix, aspect="auto")

    ax.set_title(_sanitize(title))
    ax.set_xticks(list(range(len(x_labels))))
    ax.set_xticklabels(list(x_labels), rotation=25, ha="right")
    ax.set_yticks(list(range(len(y_labels))))
    ax.set_yticklabels(list(y_labels))

    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)

    # Annotate values.
    for i in range(len(y_labels)):
        for j in range(len(x_labels)):
            v = matrix[i][j]
            if v is None:
                continue
            try:
                if isinstance(v, float) and math.isnan(v):
                    continue
            except Exception:
                pass
            ax.text(j, i, fmt.format(float(v)), ha="center", va="center", fontsize=7)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(str(p), dpi=150)
    plt.close(fig)


def export_summary_charts(
    *,
    charts_dir: Path,
    score_field: str,
    ranked_runs: List[Dict[str, Any]],
    per_video_rows: List[Dict[str, Any]],
    per_class_rows: List[Dict[str, Any]],
    class_names: List[str],
) -> None:
    """Export summary charts to the charts directory."""
    if len(ranked_runs) < 1:
        return

    charts_dir.mkdir(parents=True, exist_ok=True)
    main_dir = charts_dir / "main"
    leader_dir = charts_dir / "leaderboards"
    scatter_dir = charts_dir / "scatters"
    heat_dir = charts_dir / "heatmaps"

    for d in [main_dir, leader_dir, scatter_dir, heat_dir]:
        d.mkdir(parents=True, exist_ok=True)
    labels = [str(r["model_id"]) for r in ranked_runs]

    # Leaderboards (per-run metrics).
    run_metric_fields = [
        score_field,
        "score_total_event_wape",
        "score_total_video_mae",
        "score_total_class_wape",
        "score_total_rate_mae",
        "wape_micro_in",
        "wape_micro_out",
        "wape_wmacro_gt_in",
        "wape_wmacro_gt_out",
        "wape_in_totalcounts",
        "wape_out_totalcounts",
        "wape_in_total_l1",
        "wape_out_total_l1",
        "mae_in_total",
        "mae_out_total",
        "rmse_in_total",
        "rmse_out_total",
        "mae_in_total_counts",
        "mae_out_total_counts",
    ]

    # De-dup while preserving order.
    seen = set()
    run_metric_fields = [f for f in run_metric_fields if not (f in seen or seen.add(f))]

    for field in run_metric_fields:
        # Skip if field is not present at all.
        if not any(field in r for r in ranked_runs):
            continue
        vals = [float(r.get(field, 0.0) or 0.0) for r in ranked_runs]
        labs_sorted, vals_sorted = _sort_labels_values(labels, vals)

        bar_metric(
            leader_dir / f"leaderboard_{field}.png",
            f"Leaderboard ({field})",
            labs_sorted,
            vals_sorted,
            ylabel=field,
        )

    # Scatters (paired IN vs OUT).
    scatter_specs = [
        (
            "mae_in_total",
            "mae_out_total",
            "scatter_mae_total_in_vs_out.png",
            "MAE total: IN vs OUT",
            "MAE IN total",
            "MAE OUT total",
        ),
        (
            "rmse_in_total",
            "rmse_out_total",
            "scatter_rmse_total_in_vs_out.png",
            "RMSE total: IN vs OUT",
            "RMSE IN total",
            "RMSE OUT total",
        ),
        (
            "wape_micro_in",
            "wape_micro_out",
            "scatter_wape_micro_in_vs_out.png",
            "WAPE micro: IN vs OUT",
            "WAPE micro IN",
            "WAPE micro OUT",
        ),
        (
            "wape_wmacro_gt_in",
            "wape_wmacro_gt_out",
            "scatter_wape_wmacro_gt_in_vs_out.png",
            "WAPE weighted-macro(GT): IN vs OUT",
            "WAPE wmacro GT IN",
            "WAPE wmacro GT OUT",
        ),
        (
            "wape_in_totalcounts",
            "wape_out_totalcounts",
            "scatter_wape_totalcounts_in_vs_out.png",
            "WAPE total-counts: IN vs OUT",
            "WAPE total-counts IN",
            "WAPE total-counts OUT",
        ),
        (
            "wape_in_total_l1",
            "wape_out_total_l1",
            "scatter_wape_total_l1_in_vs_out.png",
            "WAPE L1-total: IN vs OUT",
            "WAPE L1 IN",
            "WAPE L1 OUT",
        ),
    ]

    for xf, yf, fname, title, xl, yl in scatter_specs:
        if not any((xf in r) or (yf in r) for r in ranked_runs):
            continue
        x = [float(r.get(xf, 0.0) or 0.0) for r in ranked_runs]
        y = [float(r.get(yf, 0.0) or 0.0) for r in ranked_runs]
        scatter_xy(
            scatter_dir / fname,
            title,
            x=x,
            y=y,
            labels=labels,
            xlabel=xl,
            ylabel=yl,
        )

    # Per-video heatmaps.
    videos = sorted({Path(r["video"]).stem for r in per_video_rows})
    if videos:
        run_labels = [str(r["model_id"]) for r in ranked_runs]
        run_idx = {str(r["run_id"]): i for i, r in enumerate(ranked_runs)}
        vid_idx = {v: j for j, v in enumerate(videos)}
        nan = float("nan")

        def build_matrix(value_fn):
            mat = [[nan for _ in videos] for _ in run_labels]
            for r in per_video_rows:
                ri = run_idx.get(str(r["run_id"]))
                vj = vid_idx.get(Path(str(r["video"])).stem)
                if ri is None or vj is None:
                    continue
                mat[ri][vj] = value_fn(r)
            return mat

        mat_in = build_matrix(lambda r: float(r.get("abs_err_in_total", nan)))
        mat_out = build_matrix(lambda r: float(r.get("abs_err_out_total", nan)))
        heatmap_matrix(
            heat_dir / "heatmap_abs_total_error_IN.png",
            "Abs total error (IN) per video",
            x_labels=videos,
            y_labels=run_labels,
            matrix=mat_in,
            xlabel="video",
            ylabel="model",
            fmt="{:.0f}",
        )
        heatmap_matrix(
            heat_dir / "heatmap_abs_total_error_OUT.png",
            "Abs total error (OUT) per video",
            x_labels=videos,
            y_labels=run_labels,
            matrix=mat_out,
            xlabel="video",
            ylabel="model",
            fmt="{:.0f}",
        )

        # Old-style abs error totals if present.
        if any("abs_err_in_total_counts" in r for r in per_video_rows):
            mat_in_cnt = build_matrix(lambda r: float(r.get("abs_err_in_total_counts", nan)))
            heatmap_matrix(
                heat_dir / "heatmap_abs_total_error_counts_IN.png",
                "Abs total-count error (IN) per video",
                x_labels=videos,
                y_labels=run_labels,
                matrix=mat_in_cnt,
                xlabel="video",
                ylabel="model",
                fmt="{:.0f}",
            )
        if any("abs_err_out_total_counts" in r for r in per_video_rows):
            mat_out_cnt = build_matrix(lambda r: float(r.get("abs_err_out_total_counts", nan)))
            heatmap_matrix(
                heat_dir / "heatmap_abs_total_error_counts_OUT.png",
                "Abs total-count error (OUT) per video",
                x_labels=videos,
                y_labels=run_labels,
                matrix=mat_out_cnt,
                xlabel="video",
                ylabel="model",
                fmt="{:.0f}",
            )

        # Per-video WAPE heatmaps (class-aware L1 / GT_total).
        if any(("abs_err_in_total" in r) and ("gt_in_total" in r) for r in per_video_rows):
            mat_wape_in = build_matrix(
                lambda r: safe_div(float(r.get("abs_err_in_total", nan)), float(r.get("gt_in_total", 0) or 0))
            )
            heatmap_matrix(
                charts_dir / "heatmap_wape_video_IN.png",
                "WAPE per video (IN) = abs_err_total / GT_total",
                x_labels=videos,
                y_labels=run_labels,
                matrix=mat_wape_in,
                xlabel="video",
                ylabel="model",
                fmt="{:.3f}",
            )
        if any(("abs_err_out_total" in r) and ("gt_out_total" in r) for r in per_video_rows):
            mat_wape_out = build_matrix(
                lambda r: safe_div(float(r.get("abs_err_out_total", nan)), float(r.get("gt_out_total", 0) or 0))
            )
            heatmap_matrix(
                charts_dir / "heatmap_wape_video_OUT.png",
                "WAPE per video (OUT) = abs_err_total / GT_total",
                x_labels=videos,
                y_labels=run_labels,
                matrix=mat_wape_out,
                xlabel="video",
                ylabel="model",
                fmt="{:.3f}",
            )

    # Class heatmaps.
    run_labels = [str(r["model_id"]) for r in ranked_runs]
    class_cols = class_names
    idx_run = {str(r["run_id"]): i for i, r in enumerate(ranked_runs)}
    idx_class = {cname: j for j, cname in enumerate(class_cols)}

    nan = float("nan")
    mat_cls_in = [[nan for _ in class_cols] for _ in run_labels]
    mat_cls_out = [[nan for _ in class_cols] for _ in run_labels]

    for row in per_class_rows:
        ri = idx_run.get(str(row["run_id"]))
        cj = idx_class.get(str(row["class_name"]))
        if ri is None or cj is None:
            continue
        if str(row["direction"]) == "IN":
            mat_cls_in[ri][cj] = float(row.get("mae", nan))
        else:
            mat_cls_out[ri][cj] = float(row.get("mae", nan))

    heatmap_matrix(
        charts_dir / "heatmap_mae_by_class_IN.png",
        "MAE by class (IN)",
        x_labels=class_cols,
        y_labels=run_labels,
        matrix=mat_cls_in,
        xlabel="class",
        ylabel="model",
        fmt="{:.2f}",
    )
    heatmap_matrix(
        charts_dir / "heatmap_mae_by_class_OUT.png",
        "MAE by class (OUT)",
        x_labels=class_cols,
        y_labels=run_labels,
        matrix=mat_cls_out,
        xlabel="class",
        ylabel="model",
        fmt="{:.2f}",
    )

    # Main set (3-5 key charts).
    main_fields = [
        "score_total_event_wape",
        "score_total_video_mae",
        "score_total_class_wape",
    ]

    for field in main_fields:
        if any(field in r for r in ranked_runs):
            vals = [float(r.get(field, 0.0) or 0.0) for r in ranked_runs]
            labs_sorted, vals_sorted = _sort_labels_values(labels, vals)
            bar_metric(
                main_dir / f"leaderboard_{field}.png",
                f"Leaderboard ({field})",
                labs_sorted,
                vals_sorted,
                ylabel=field,
            )

    # Overview scatter: MAE IN vs OUT.
    if any(("mae_in_total" in r) or ("mae_out_total" in r) for r in ranked_runs):
        x = [float(r.get("mae_in_total", 0.0) or 0.0) for r in ranked_runs]
        y = [float(r.get("mae_out_total", 0.0) or 0.0) for r in ranked_runs]
        scatter_xy(
            main_dir / "scatter_mae_total_in_vs_out.png",
            "MAE total: IN vs OUT",
            x=x,
            y=y,
            labels=labels,
            xlabel="MAE IN total",
            ylabel="MAE OUT total",
        )

    # Optionally include class heatmaps in the main folder by reusing matrices.
    if per_class_rows:
        heatmap_matrix(
            main_dir / "heatmap_mae_by_class_IN.png",
            "MAE by class (IN)",
            x_labels=class_cols,
            y_labels=run_labels,
            matrix=mat_cls_in,
            xlabel="class",
            ylabel="model",
            fmt="{:.2f}",
        )
        heatmap_matrix(
            main_dir / "heatmap_mae_by_class_OUT.png",
            "MAE by class (OUT)",
            x_labels=class_cols,
            y_labels=run_labels,
            matrix=mat_cls_out,
            xlabel="class",
            ylabel="model",
            fmt="{:.2f}",
        )

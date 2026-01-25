from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.core.DetectedObject import DetectedObject


@dataclass(frozen=True)
class PredictionFile:
    path: Path
    model: str
    video_id: int


def _parse_count_dict(value: object) -> Dict[int, int]:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return {}
    if isinstance(value, dict):
        return {int(k): int(v) for k, v in value.items()}
    text = str(value).strip()
    if not text or text == "{}":
        return {}
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return {}
    if isinstance(parsed, dict):
        return {int(k): int(v) for k, v in parsed.items()}
    return {}


def load_counts_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rows: List[Dict[str, int]] = []
    for _, row in df.iterrows():
        line_name = str(row.get("line_name", "")).strip()
        in_dict = _parse_count_dict(row.get("in_count"))
        out_dict = _parse_count_dict(row.get("out_count"))
        class_ids = sorted(set(in_dict) | set(out_dict))
        for class_id in class_ids:
            rows.append(
                {
                    "line_name": line_name,
                    "class_id": int(class_id),
                    "in_count": int(in_dict.get(class_id, 0)),
                    "out_count": int(out_dict.get(class_id, 0)),
                }
            )
    return pd.DataFrame(rows)


def discover_prediction_files(pred_dir: Path) -> List[PredictionFile]:
    pattern = re.compile(r"^vid(?P<video>\d+)_+(?P<model>.+)_results\.csv$", re.IGNORECASE)
    predictions: List[PredictionFile] = []
    for path in pred_dir.rglob("*.csv"):
        match = pattern.match(path.name)
        if not match:
            continue
        video_id = int(match.group("video"))
        model = match.group("model")
        predictions.append(PredictionFile(path=path, model=model, video_id=video_id))
    return predictions


def _prediction_variant(path: Path) -> str:
    parts = {part.lower() for part in path.parts}
    if "yolo_tuned" in parts:
        return "yolo_tuned"
    if "yolo_pretrained" in parts:
        return "yolo_pretrained"
    if "rfdetr_pretrained" in parts:
        return "rfdetr_pretrained"
    if "rfdetr_tuned" in parts:
        return "rfdetr_tuned"
    return "unknown"


def _remap_pred_classes(pred_df: pd.DataFrame, class_map: Dict[int, int]) -> pd.DataFrame:
    if pred_df.empty:
        return pred_df
    mapped = pred_df[pred_df["class_id"].isin(class_map)].copy()
    if mapped.empty:
        return mapped
    mapped["class_id"] = mapped["class_id"].map(class_map).astype(int)
    return (
        mapped.groupby(["line_name", "class_id"], as_index=False)[["in_count", "out_count"]]
        .sum()
    )


def _prepare_eval_rows(
    gt_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    model: str,
    video_id: int,
) -> pd.DataFrame:
    merged = pd.merge(
        gt_df,
        pred_df,
        on=["line_name", "class_id"],
        how="outer",
        suffixes=("_gt", "_pred"),
    )
    merged["in_count_gt"] = merged["in_count_gt"].fillna(0).astype(int)
    merged["out_count_gt"] = merged["out_count_gt"].fillna(0).astype(int)
    merged["in_count_pred"] = merged["in_count_pred"].fillna(0).astype(int)
    merged["out_count_pred"] = merged["out_count_pred"].fillna(0).astype(int)

    merged["gt_total"] = merged["in_count_gt"] + merged["out_count_gt"]
    merged["pred_total"] = merged["in_count_pred"] + merged["out_count_pred"]
    merged["abs_error"] = (merged["pred_total"] - merged["gt_total"]).abs()
    merged["sq_error"] = (merged["pred_total"] - merged["gt_total"]) ** 2
    merged["abs_error_in"] = (merged["in_count_pred"] - merged["in_count_gt"]).abs()
    merged["abs_error_out"] = (merged["out_count_pred"] - merged["out_count_gt"]).abs()
    merged["ape"] = merged.apply(
        lambda r: float(r["abs_error"]) / float(r["gt_total"])
        if r["gt_total"] > 0
        else np.nan,
        axis=1,
    )
    merged["ape_in"] = merged.apply(
        lambda r: float(r["abs_error_in"]) / float(r["in_count_gt"])
        if r["in_count_gt"] > 0
        else np.nan,
        axis=1,
    )
    merged["ape_out"] = merged.apply(
        lambda r: float(r["abs_error_out"]) / float(r["out_count_gt"])
        if r["out_count_gt"] > 0
        else np.nan,
        axis=1,
    )
    merged["model"] = model
    merged["video_id"] = video_id
    return merged


def _safe_mean(series: pd.Series) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.mean())


def _safe_std(series: pd.Series) -> float:
    values = series.dropna()
    if len(values) < 2:
        return float("nan")
    return float(values.std())


def _percentile(series: pd.Series, q: float) -> float:
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(np.percentile(values, q))


def compute_metrics(
    all_rows: pd.DataFrame,
    class_ids: Iterable[int],
    class_names: Dict[int, str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: List[Dict[str, float]] = []
    per_class_rows: List[Dict[str, float]] = []
    per_video_rows: List[Dict[str, float]] = []
    robustness_rows: List[Dict[str, float]] = []

    for model, model_df in all_rows.groupby("model"):
        mae = float(model_df["abs_error"].mean())
        mape_in = _safe_mean(model_df["ape_in"])
        mape_out = _safe_mean(model_df["ape_out"])
        rmse = math.sqrt(float(model_df["sq_error"].mean()))
        total_error = float(model_df["pred_total"].sum() - model_df["gt_total"].sum())

        weights = model_df["gt_total"]
        weighted_mae = (
            float((model_df["abs_error"] * weights).sum() / weights.sum())
            if weights.sum() > 0
            else float("nan")
        )

        summary_rows.append(
            {
                "model": model,
                "mae": mae,
                "mape_in": mape_in,
                "mape_out": mape_out,
                "rmse": rmse,
                "total_count_error": total_error,
                "total_count_error_abs": abs(total_error),
                "weighted_mae": weighted_mae,
            }
        )

        for class_id in class_ids:
            class_df = model_df[model_df["class_id"] == class_id]
            if class_df.empty:
                continue
            per_class_rows.append(
                {
                    "model": model,
                    "class_id": int(class_id),
                    "class_name": class_names.get(int(class_id), f"class_{class_id}"),
                    "mae": float(class_df["abs_error"].mean()),
                    "mape_in": _safe_mean(class_df["ape_in"]),
                    "mape_out": _safe_mean(class_df["ape_out"]),
                    "rmse": math.sqrt(float(class_df["sq_error"].mean())),
                }
            )

        for video_id, video_df in model_df.groupby("video_id"):
            per_video_rows.append(
                {
                    "model": model,
                    "video_id": int(video_id),
                    "mae": float(video_df["abs_error"].mean()),
                    "mape_in": _safe_mean(video_df["ape_in"]),
                    "mape_out": _safe_mean(video_df["ape_out"]),
                    "rmse": math.sqrt(float(video_df["sq_error"].mean())),
                    "total_count_error": float(video_df["pred_total"].sum() - video_df["gt_total"].sum()),
                }
            )

        per_video_df = pd.DataFrame([row for row in per_video_rows if row["model"] == model])
        if not per_video_df.empty:
            robustness_rows.append(
                {
                    "model": model,
                    "mae_std": _safe_std(per_video_df["mae"]),
                    "mae_worst": float(per_video_df["mae"].max()),
                    "mae_p50": _percentile(per_video_df["mae"], 50),
                    "mae_p90": _percentile(per_video_df["mae"], 90),
                    "mae_p95": _percentile(per_video_df["mae"], 95),
                }
            )

    return (
        pd.DataFrame(summary_rows),
        pd.DataFrame(per_class_rows),
        pd.DataFrame(per_video_rows),
        pd.DataFrame(robustness_rows),
    )


def _save_mae_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    x = np.arange(len(summary_df["model"]))
    width = 0.4
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#f7f5f0")
    ax.set_facecolor("#f7f5f0")
    ax.bar(x, summary_df["mae"], width, label="MAE", color="#2a6f97")
    ax.set_xticks(x, summary_df["model"], rotation=15, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison: MAE", pad=12)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_mape_chart(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    x = np.arange(len(summary_df["model"]))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor("#f7f5f0")
    ax.set_facecolor("#f7f5f0")
    ax.bar(x - width / 2, summary_df["mape_in"], width, label="MAPE In", color="#ff7a59")
    ax.bar(x + width / 2, summary_df["mape_out"], width, label="MAPE Out", color="#f7b267")
    ax.set_xticks(x, summary_df["model"], rotation=15, ha="right")
    ax.set_ylabel("Error")
    ax.set_title("Model Comparison: MAPE (In/Out)", pad=12)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_table_image(df: pd.DataFrame, output_path: Path, title: str) -> None:
    if df.empty:
        return
    max_rows = min(len(df), 25)
    preview = df.head(max_rows).copy()
    preview = preview.where(pd.notna(preview), "")
    fig_height = 1.2 + 0.35 * (len(preview) + 1)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    fig.patch.set_facecolor("#f7f5f0")
    ax.set_facecolor("#f7f5f0")
    ax.axis("off")
    table = ax.table(
        cellText=preview.values,
        colLabels=preview.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.2)
    ax.set_title(title, pad=12)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _write_readme(
    output_dir: Path,
    gt_dir: Path,
    pred_dir: Path,
    summary_df: pd.DataFrame,
    per_video_df: pd.DataFrame,
    per_class_df: pd.DataFrame,
    robustness_df: pd.DataFrame,
) -> None:
    def _df_to_markdown(df: pd.DataFrame) -> List[str]:
        if df.empty:
            return ["No data available."]
        def _format_value(value: object) -> str:
            if value is None or (isinstance(value, float) and math.isnan(value)):
                return ""
            return str(value)

        headers = [str(h) for h in df.columns]
        rows = [[_format_value(value) for value in row] for row in df.values.tolist()]
        widths = [len(h) for h in headers]
        for row in rows:
            for i, value in enumerate(row):
                widths[i] = max(widths[i], len(value))
        header_line = "| " + " | ".join(h.ljust(widths[i]) for i, h in enumerate(headers)) + " |"
        sep_line = "| " + " | ".join("-" * widths[i] for i in range(len(widths))) + " |"
        data_lines = [
            "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(widths))) + " |"
            for row in rows
        ]
        return [header_line, sep_line, *data_lines]

    analysis_lines: List[str] = []
    analysis_lines.append("# Evaluation Run Analysis")
    analysis_lines.append("")
    analysis_lines.append(f"Run timestamp: {output_dir.name}")
    analysis_lines.append(f"GT directory: {gt_dir}")
    analysis_lines.append(f"Predicted directory: {pred_dir}")
    analysis_lines.append("")
    analysis_lines.append("## Summary Metrics (per model)")
    if summary_df.empty:
        analysis_lines.append("No models were evaluated.")
    else:
        analysis_lines.extend(_df_to_markdown(summary_df))
    analysis_lines.append("")
    analysis_lines.append("## Robustness Metrics (per model)")
    if robustness_df.empty:
        analysis_lines.append("No robustness metrics available.")
    else:
        analysis_lines.extend(_df_to_markdown(robustness_df))
    analysis_lines.append("")
    analysis_lines.append("## Per-class Metrics")
    if per_class_df.empty:
        analysis_lines.append("No per-class metrics available.")
    else:
        analysis_lines.extend(_df_to_markdown(per_class_df))
    analysis_lines.append("")
    analysis_lines.append("## Per-video Breakdown")
    if per_video_df.empty:
        analysis_lines.append("No per-video metrics available.")
    else:
        analysis_lines.extend(_df_to_markdown(per_video_df))
    analysis_lines.append("")
    analysis_lines.append("## Plots")
    analysis_lines.append("![MAE bar chart](charts/bar_mae.png)")
    analysis_lines.append("![MAPE bar chart](charts/bar_mape.png)")
    analysis_lines.append("![Summary table](charts/table_summary.png)")
    analysis_lines.append("![Per-class table](charts/table_per_class.png)")
    analysis_lines.append("![Per-video table](charts/table_per_video.png)")
    output_dir.joinpath("ANALYSIS.md").write_text("\n".join(analysis_lines), encoding="utf-8")


def run_evaluation(
    gt_dir: Path,
    pred_dir: Path,
    output_root: Path,
    class_ids: Iterable[int],
) -> Path:
    yolo_pretrained_class_map = {4: 0, 5: 1, 6: 2, 7: 3}
    predictions = discover_prediction_files(pred_dir)
    if not predictions:
        raise FileNotFoundError(f"No prediction files found in {pred_dir}")

    rows: List[pd.DataFrame] = []
    for pred in predictions:
        variant = _prediction_variant(pred.path)
        if variant in {"rfdetr_pretrained", "rfdetr_tuned"}:
            # TODO: Add RF-DETR class-id mapping support once defined.
            continue
        gt_path = gt_dir / f"data_{pred.video_id}.csv"
        if not gt_path.exists():
            continue
        gt_df = load_counts_csv(gt_path)
        pred_df = load_counts_csv(pred.path)
        if variant == "yolo_pretrained":
            pred_df = _remap_pred_classes(pred_df, yolo_pretrained_class_map)
        rows.append(_prepare_eval_rows(gt_df, pred_df, pred.model, pred.video_id))

    if not rows:
        raise FileNotFoundError("No matching GT/prediction pairs found.")

    all_rows = pd.concat(rows, ignore_index=True)
    class_names = DetectedObject.class_names
    summary_df, per_class_df, per_video_df, robustness_df = compute_metrics(
        all_rows, class_ids, class_names
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    charts_dir = output_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(output_dir / "summary_metrics.csv", index=False)
    summary_df.to_json(output_dir / "summary_metrics.json", orient="records", indent=2)
    per_class_df.to_csv(output_dir / "per_class_metrics.csv", index=False)
    per_video_df.to_csv(output_dir / "per_video_metrics.csv", index=False)
    robustness_df.to_csv(output_dir / "robustness_metrics.csv", index=False)

    _save_mae_chart(summary_df, charts_dir / "bar_mae.png")
    _save_mape_chart(summary_df, charts_dir / "bar_mape.png")
    _save_table_image(summary_df, charts_dir / "table_summary.png", "Summary Metrics")
    _save_table_image(per_class_df, charts_dir / "table_per_class.png", "Per-class Metrics")
    _save_table_image(per_video_df, charts_dir / "table_per_video.png", "Per-video Breakdown")

    _write_readme(output_dir, gt_dir, pred_dir, summary_df, per_video_df, per_class_df, robustness_df)
    return output_dir

#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "models" / "eval"
OUT_DIR = EVAL_DIR / "thesis_figures"


plt.style.use("tableau-colorblind10")


@dataclass(frozen=True)
class PlotStyle:
    yolo: str = "#1f77b4"
    rfdetr: str = "#d62728"
    stage_v0: str = "#9ca3af"
    stage_v1: str = "#111827"


STYLE = PlotStyle()


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def short_label(model_rel: str) -> str:
    parts = model_rel.split("/")
    if parts[-1] in {"run", "run2", "run3"} and len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    if len(parts) >= 2 and parts[-2].startswith("yolov8m_v12") and parts[-1].startswith("yolov8m_v2"):
        return parts[-2]
    return parts[-1]


def extract_stage(model_rel: str) -> str:
    parts = model_rel.split("/")
    for part in parts:
        if re.fullmatch(r"v\d+", part):
            return part
    return "unknown"


def model_family(model_rel: str) -> str:
    label = short_label(model_rel)
    if label.startswith("yolov5"):
        return "YOLOv5"
    if label.startswith("yolov8"):
        return "YOLOv8"
    if label.startswith("yolo11"):
        return "YOLO11"
    if label.startswith("yolo26"):
        return "YOLO26"
    if label.startswith("rfdetr"):
        return "RF-DETR"
    return label


def artifact_timestamp(artifact_rel: str) -> datetime:
    path = ROOT / "models" / artifact_rel
    return datetime.fromtimestamp(path.stat().st_mtime)


def prepare_detection_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(EVAL_DIR / "detection_summary.csv")
    df["stage"] = df["model_rel"].map(extract_stage)
    df["short_label"] = df["model_rel"].map(short_label)
    df["family"] = df["model_rel"].map(model_family)
    df["artifact_time"] = df["artifact"].map(artifact_timestamp)
    yolo = df[df["backend"] == "yolo"].copy()
    yolo["primary"] = yolo["primary"].astype(float)
    yolo["precision"] = yolo["precision"].astype(float)
    yolo["recall"] = yolo["recall"].astype(float)
    yolo["hours"] = yolo["hours"].astype(float)
    yolo["epoch"] = yolo["epoch"].astype(int)

    rfdetr = df[(df["backend"] == "rfdetr") & (df["basis"] == "test")].copy()
    rfdetr["primary"] = rfdetr["primary"].astype(float)
    rfdetr["precision"] = rfdetr["precision"].astype(float)
    rfdetr["recall"] = rfdetr["recall"].astype(float)
    return yolo, rfdetr


def prepare_count_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    overview = pd.read_csv(EVAL_DIR / "count_benchmark_overview.csv")
    per_class = pd.read_csv(EVAL_DIR / "count_benchmark_per_class.csv")
    overview["empty_benchmark"] = overview["empty_benchmark"].astype(str) == "True"
    per_class["empty_benchmark"] = per_class["empty_benchmark"].astype(str) == "True"
    overview["stage"] = overview["model_rel"].map(extract_stage)
    overview["short_label"] = overview["model_rel"].map(short_label)
    overview["family"] = overview["model_rel"].map(model_family)
    overview = overview[~overview["empty_benchmark"]].copy()
    per_class = per_class[~per_class["empty_benchmark"]].copy()
    return overview, per_class


def plot_detection_ranking(yolo: pd.DataFrame, rfdetr: pd.DataFrame) -> None:
    top_yolo = yolo.sort_values("primary", ascending=False)
    top_rfdetr = rfdetr.sort_values("primary", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1.4, 1]})

    axes[0].barh(
        top_yolo["short_label"][::-1],
        top_yolo["primary"][::-1],
        color=STYLE.yolo,
        alpha=0.9,
    )
    axes[0].set_title("YOLO Detection Ranking")
    axes[0].set_xlabel("Best validation mAP50-95")
    axes[0].grid(axis="x", alpha=0.25)
    for i, value in enumerate(top_yolo["primary"][::-1]):
        axes[0].text(value + 0.005, i, f"{value:.3f}", va="center", fontsize=9)

    axes[1].barh(
        top_rfdetr["short_label"][::-1],
        top_rfdetr["primary"][::-1],
        color=STYLE.rfdetr,
        alpha=0.9,
    )
    axes[1].set_title("RF-DETR Test Ranking")
    axes[1].set_xlabel("Test mAP50-95")
    axes[1].grid(axis="x", alpha=0.25)
    for i, value in enumerate(top_rfdetr["primary"][::-1]):
        axes[1].text(value + 0.004, i, f"{value:.3f}", va="center", fontsize=9)

    fig.suptitle("Detection Quality of Trained Models", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "YOLO uses best validation epoch from results.csv. RF-DETR uses explicit test split from results.json.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    save(fig, "01_detection_ranking")


def plot_precision_recall_tradeoff(yolo: pd.DataFrame, rfdetr: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    markers = {"v0": "o", "v1": "s", "unknown": "^"}

    for stage, subset in yolo.groupby("stage"):
        ax.scatter(
            subset["recall"],
            subset["precision"],
            s=subset["primary"] * 700,
            marker=markers.get(stage, "^"),
            color=STYLE.yolo,
            alpha=0.75,
            edgecolor="white",
            linewidth=0.8,
            label=f"YOLO {stage}",
        )
        for _, row in subset.iterrows():
            ax.annotate(row["short_label"], (row["recall"], row["precision"]), xytext=(4, 4), textcoords="offset points", fontsize=8)

    for stage, subset in rfdetr.groupby("stage"):
        ax.scatter(
            subset["recall"],
            subset["precision"],
            s=subset["primary"] * 1200,
            marker=markers.get(stage, "^"),
            color=STYLE.rfdetr,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.8,
            label=f"RF-DETR {stage}",
        )
        for _, row in subset.iterrows():
            ax.annotate(row["short_label"], (row["recall"], row["precision"]), xytext=(4, -10), textcoords="offset points", fontsize=8)

    ax.set_title("Precision/Recall Trade-off Across Trained Models")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.text(
        0.5,
        0.01,
        "Point size is proportional to mAP50-95. Marker shape indicates repository stage (v0 or v1).",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    save(fig, "02_precision_recall_tradeoff")


def plot_timeline(yolo: pd.DataFrame, rfdetr: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 9), sharex=True, gridspec_kw={"hspace": 0.18})
    markers = {"v0": "o", "v1": "s", "unknown": "^"}
    panel_data = [("YOLO", yolo.sort_values("artifact_time"), STYLE.yolo), ("RF-DETR", rfdetr.sort_values("artifact_time"), STYLE.rfdetr)]

    for ax, (group, subset, color) in zip(axes, panel_data):
        ax.plot(subset["artifact_time"], subset["primary"], color=color, alpha=0.4, linewidth=1.4)
        for _, row in subset.iterrows():
            ax.scatter(
                row["artifact_time"],
                row["primary"],
                color=color,
                marker=markers.get(row["stage"], "^"),
                s=90,
                edgecolor="white",
                linewidth=0.9,
                zorder=3,
            )
            offset_y = 6 if group == "YOLO" else -12
            ax.annotate(row["short_label"], (row["artifact_time"], row["primary"]), xytext=(5, offset_y), textcoords="offset points", fontsize=8)
        ax.set_ylabel("mAP50-95")
        ax.set_title(group, loc="left", fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)

    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=5, maxticks=9))
    plt.setp(axes[-1].get_xticklabels(), rotation=25, ha="right")
    axes[-1].set_xlabel("Artifact modification date")
    fig.suptitle("Model Quality Over Time", fontsize=16, fontweight="bold", y=0.985)
    fig.legend(
        handles=[
            plt.Line2D([0], [0], color=STYLE.yolo, marker="o", linestyle="-", label="YOLO"),
            plt.Line2D([0], [0], color=STYLE.rfdetr, marker="o", linestyle="-", label="RF-DETR"),
            plt.Line2D([0], [0], color=STYLE.stage_v0, marker="o", linestyle="None", label="stage v0"),
            plt.Line2D([0], [0], color=STYLE.stage_v1, marker="s", linestyle="None", label="stage v1"),
        ],
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
    )
    fig.text(
        0.5,
        0.01,
        "Timeline uses artifact modification timestamps as the closest consistent chronology proxy across stored model outputs.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    save(fig, "03_model_timeline")


def plot_stage_progress(yolo: pd.DataFrame, rfdetr: pd.DataFrame) -> None:
    stage_rows: list[dict[str, object]] = []
    for backend, subset in [("YOLO", yolo), ("RF-DETR", rfdetr)]:
        for stage, stage_df in subset.groupby("stage"):
            stage_rows.append(
                {
                    "backend": backend,
                    "stage": stage,
                    "mean_map": stage_df["primary"].mean(),
                    "best_map": stage_df["primary"].max(),
                    "count": len(stage_df),
                }
            )
    df = pd.DataFrame(stage_rows).sort_values(["backend", "stage"])
    fig, ax = plt.subplots(figsize=(9.5, 6.5))

    x_labels = [f"{row.backend} {row.stage}" for row in df.itertuples()]
    x = range(len(df))
    ax.bar(x, df["mean_map"], color=[STYLE.yolo if b == "YOLO" else STYLE.rfdetr for b in df["backend"]], alpha=0.75, label="mean")
    ax.scatter(x, df["best_map"], color="#111827", s=70, label="best within stage", zorder=3)
    for idx, row in enumerate(df.itertuples()):
        ax.text(idx, row.mean_map + 0.01, f"n={row.count}", ha="center", fontsize=9, color="#4b5563")

    ax.set_xticks(list(x), x_labels, rotation=20, ha="right")
    ax.set_ylabel("mAP50-95")
    ax.set_title("Average Quality by Repository Stage")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.text(
        0.5,
        0.01,
        "Bars show mean quality per stage. Black markers show the strongest model found in that stage/backend slice.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    save(fig, "04_stage_progress")


def plot_count_benchmarks(overview: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.2, 1]})
    ranked = overview.sort_values("score_total_micro_wape")

    axes[0].barh(
        ranked["short_label"],
        ranked["score_total_micro_wape"],
        color=[STYLE.yolo if "yolo" in model else STYLE.rfdetr for model in ranked["model_rel"]],
        alpha=0.9,
    )
    axes[0].invert_yaxis()
    axes[0].set_title("Per-Class Counting Benchmark")
    axes[0].set_xlabel("Micro WAPE (lower is better)")
    axes[0].grid(axis="x", alpha=0.25)

    mae_df = overview.dropna(subset=["score_total_video_mae"]).sort_values("score_total_video_mae")
    axes[1].barh(
        mae_df["short_label"],
        mae_df["score_total_video_mae"],
        color=[STYLE.yolo if "yolo" in model else STYLE.rfdetr for model in mae_df["model_rel"]],
        alpha=0.9,
    )
    axes[1].invert_yaxis()
    axes[1].set_title("Video-Level Count Error")
    axes[1].set_xlabel("Score total video MAE (lower is better)")
    axes[1].grid(axis="x", alpha=0.25)

    fig.suptitle("Counting-Oriented Evaluation of Benchmarked Models", fontsize=16, fontweight="bold")
    fig.text(
        0.5,
        0.01,
        "Empty benchmark runs are excluded. These metrics reflect counting quality, not detection mAP.",
        ha="center",
        fontsize=10,
        color="#4b5563",
    )
    save(fig, "05_count_benchmark_overview")


def heatmap_data(per_class: pd.DataFrame, value_col: str, row_order: list[str], col_order: list[str]) -> pd.DataFrame:
    pivot = per_class.pivot(index="model_rel", columns="metric_key", values=value_col)
    pivot = pivot.loc[row_order, col_order]
    return pivot


def plot_rfdetr_per_class_heatmap() -> None:
    df = pd.read_csv(EVAL_DIR / "rfdetr_per_class_detection.csv")
    test_df = df[df["split"] == "test"].copy()
    row_order = list(test_df.groupby("model_rel")["mAP50-95"].mean().sort_values(ascending=False).index)
    col_order = ["tourist", "skier", "cyclist", "tourist_dog"]
    pivot = test_df.pivot(index="model_rel", columns="class_name", values="mAP50-95").loc[row_order, col_order]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.values, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(col_order)), [c.upper() for c in col_order])
    ax.set_yticks(range(len(row_order)), [short_label(r) for r in row_order])
    ax.set_title("RF-DETR Test Per-Class Detection")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mAP50-95")
    fig.text(0.5, 0.01, "Higher values indicate stronger class-specific detection quality.", ha="center", fontsize=10, color="#4b5563")
    save(fig, "06_rfdetr_per_class_detection")


def plot_count_per_class_heatmap(per_class: pd.DataFrame) -> None:
    per_class = per_class.copy()
    per_class["metric_key"] = per_class["direction"] + "_" + per_class["class_name"]
    row_order = list(per_class.groupby("model_rel")["mae"].mean().sort_values().index)
    col_order = [
        "IN_TOURIST",
        "IN_SKIER",
        "IN_CYCLIST",
        "IN_TOURIST_DOG",
        "OUT_TOURIST",
        "OUT_SKIER",
        "OUT_CYCLIST",
        "OUT_TOURIST_DOG",
    ]
    pivot = heatmap_data(per_class, "mae", row_order, col_order)

    fig, ax = plt.subplots(figsize=(12.2, 5.6))
    im = ax.imshow(pivot.values, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(col_order)), [c.replace("_", "\n") for c in col_order])
    ax.set_yticks(range(len(row_order)), [short_label(r) for r in row_order])
    ax.set_title("Per-Class Counting Error")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", color="white", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("MAE")
    fig.text(0.5, 0.01, "Lower values indicate more reliable class-specific counting performance.", ha="center", fontsize=10, color="#4b5563")
    save(fig, "07_count_per_class_mae")


def write_index(yolo: pd.DataFrame, rfdetr: pd.DataFrame, overview: pd.DataFrame) -> None:
    earliest = min(yolo["artifact_time"].min(), rfdetr["artifact_time"].min()).strftime("%Y-%m-%d")
    latest = max(yolo["artifact_time"].max(), rfdetr["artifact_time"].max()).strftime("%Y-%m-%d")
    lines = [
        "# Thesis Figures",
        "",
        "These plots were generated with `matplotlib` from the CSV summaries in `models/eval`.",
        "",
        "## Files",
        "",
        "- `01_detection_ranking.(png|pdf)`: overall detection ranking for YOLO and RF-DETR.",
        "- `02_precision_recall_tradeoff.(png|pdf)`: precision/recall view with point size proportional to mAP50-95.",
        "- `03_model_timeline.(png|pdf)`: chronological progression using artifact modification dates as timeline proxy.",
        "- `04_stage_progress.(png|pdf)`: average and best quality by repository stage.",
        "- `05_count_benchmark_overview.(png|pdf)`: counting-focused ranking for non-empty benchmark runs.",
        "- `06_rfdetr_per_class_detection.(png|pdf)`: RF-DETR test per-class detection heatmap.",
        "- `07_count_per_class_mae.(png|pdf)`: per-class count error heatmap for benchmarked models.",
        "",
        "## Notes",
        "",
        f"- Timeline coverage in the stored artifacts spans `{earliest}` to `{latest}`.",
        "- The timeline uses file modification dates because explicit training timestamps are not stored consistently for every model artifact.",
        f"- Detection ranking includes {len(yolo)} YOLO artifacts and {len(rfdetr)} RF-DETR test artifacts.",
        f"- Counting plots include {len(overview)} non-empty benchmarked models.",
        "- The latest all-zero benchmark files for `yolo11l_v11` and `yolo11m_v11` are intentionally excluded from counting visualizations because their GT totals are zero.",
    ]
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ensure_out_dir()
    yolo, rfdetr = prepare_detection_df()
    overview, per_class = prepare_count_df()

    plot_detection_ranking(yolo, rfdetr)
    plot_precision_recall_tradeoff(yolo, rfdetr)
    plot_timeline(yolo, rfdetr)
    plot_stage_progress(yolo, rfdetr)
    plot_count_benchmarks(overview)
    plot_rfdetr_per_class_heatmap()
    plot_count_per_class_heatmap(per_class)
    write_index(yolo, rfdetr, overview)


if __name__ == "__main__":
    main()

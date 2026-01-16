import argparse
import os
from datetime import datetime
from typing import Iterable, Optional

import pandas as pd

from src.eval.config import load_eval_config
from src.eval.counts import build_complete_results
from src.eval.metrics import diff_stats, scores_micro_macro, tracking_miss_rate
from src.eval.plots import (
    save_confusion_matrix_plot,
    save_counts_plot,
    save_diff_plot,
    save_diff_stats_plot,
    save_scores_plot,
    save_tmr_plot,
)
from src.eval.rfdetr_eval import evaluate_rfdetr_custom, evaluate_rfdetr_mapped
from src.eval.yolo_eval import evaluate_yolo_custom, evaluate_yolo_mapped, evaluate_yolo_standard


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate counting results and detector model metrics.")
    parser.add_argument("--config", default="config.yaml")

    args = parser.parse_args(list(argv) if argv is not None else None)
    cfg = load_eval_config(args.config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(cfg["out_dir"], timestamp)
    _ensure_dir(out_dir)

    complete = build_complete_results(
        cfg["gt_folder"],
        cfg["pred_folder"],
        map_pretrained_counts=cfg["map_pretrained_counts"],
    )
    if complete.empty:
        print("No comparison results found.")
        return 1

    complete_path = os.path.join(out_dir, "complete_results.csv")
    complete.to_csv(complete_path, index=False)
    print(f"Saved: {complete_path}")

    scores = scores_micro_macro(complete)
    scores_path = os.path.join(out_dir, "scores_micro_macro.csv")
    scores.to_csv(scores_path, index=False)
    print(f"Saved: {scores_path}")

    tmr = tracking_miss_rate(complete)
    tmr_path = os.path.join(out_dir, "tracking_miss_rate.csv")
    tmr.to_csv(tmr_path, index=False)
    print(f"Saved: {tmr_path}")

    stats = diff_stats(complete)
    stats_path = os.path.join(out_dir, "diff_stats.csv")
    stats.to_csv(stats_path)
    print(f"Saved: {stats_path}")

    if cfg["plots"]:
        plots_dir = os.path.join(out_dir, "plots")
        _ensure_dir(plots_dir)
        for (video_num, yolo_model), group in complete.groupby(["video_num", "yolo_model"]):
            title = f"Video {video_num} - {yolo_model}"
            out_path = os.path.join(plots_dir, f"vid{video_num}_{yolo_model}.png")
            save_counts_plot(group, out_path, title)
            diff_path = os.path.join(plots_dir, f"vid{video_num}_{yolo_model}_diff.png")
            save_diff_plot(group, diff_path, f"{title} (diff)")
        save_scores_plot(scores, os.path.join(plots_dir, "scores.png"), "Scores")
        save_tmr_plot(tmr, os.path.join(plots_dir, "tracking_miss_rate.png"), "Tracking miss rate")
        save_diff_stats_plot(stats, os.path.join(plots_dir, "diff_stats.png"), "Diff stats")
        print(f"Saved plots to: {plots_dir}")

    run_model_eval = bool(cfg.get("run_model_eval", False) or cfg.get("run_yolo_eval", False))
    if run_model_eval:
        model_path = cfg["model_path"]
        model_type = str(cfg.get("model_type", "yolo")).lower()
        model_mode = str(cfg.get("model_mode", cfg.get("yolo_mode", "custom"))).lower()
        rfdetr_box_format = cfg.get("rfdetr_box_format", "xyxy")
        rfdetr_box_normalized = cfg.get("rfdetr_box_normalized", "auto")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        if model_name == "best" and os.path.basename(os.path.dirname(model_path)) == "weights":
            model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
        if model_type == "yolo":
            if model_mode == "custom":
                metrics = evaluate_yolo_standard(
                    model_path=cfg["model_path"],
                    data_yaml=cfg["data_yaml"],
                    device=cfg["device"],
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                    split=cfg["split"],
                    project=os.path.abspath(out_dir),
                    name=f"yolo_val_{model_name}",
                )
                metrics_path = os.path.join(out_dir, "yolo_val_metrics.csv")
                pd.DataFrame([{"model_name": model_name, **metrics}]).to_csv(metrics_path, index=False)
                print(f"Saved: {metrics_path}")
                mapped, cm, labels = evaluate_yolo_custom(
                    model_path=cfg["model_path"],
                    data_yaml=cfg["data_yaml"],
                    device=cfg["device"],
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                )
                metrics_path = os.path.join(out_dir, "yolo_class_metrics.csv")
                mapped.insert(0, "model_name", model_name)
                mapped.to_csv(metrics_path, index=False)
                print(f"Saved: {metrics_path}")
                yolo_plots_dir = os.path.join(out_dir, f"yolo_plots_{model_name}")
                _ensure_dir(yolo_plots_dir)
                save_confusion_matrix_plot(
                    cm,
                    labels,
                    os.path.join(yolo_plots_dir, "confusion_matrix.png"),
                    f"Confusion matrix ({model_name})",
                )
                save_confusion_matrix_plot(
                    cm,
                    labels,
                    os.path.join(yolo_plots_dir, "confusion_matrix_norm.png"),
                    f"Confusion matrix (normalized, {model_name})",
                    normalize=True,
                )
                print(f"YOLO model: {model_name}")
            else:
                mapped, cm, labels = evaluate_yolo_mapped(
                    model_path=cfg["model_path"],
                    data_yaml=cfg["data_yaml"],
                    device=cfg["device"],
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                )
                metrics_path = os.path.join(out_dir, "yolo_class_metrics.csv")
                mapped.insert(0, "model_name", model_name)
                mapped.to_csv(metrics_path, index=False)
                print(f"Saved: {metrics_path}")
                yolo_plots_dir = os.path.join(out_dir, f"yolo_plots_{model_name}")
                _ensure_dir(yolo_plots_dir)
                save_confusion_matrix_plot(
                    cm,
                    labels,
                    os.path.join(yolo_plots_dir, "confusion_matrix.png"),
                    f"Confusion matrix ({model_name})",
                )
                save_confusion_matrix_plot(
                    cm,
                    labels,
                    os.path.join(yolo_plots_dir, "confusion_matrix_norm.png"),
                    f"Confusion matrix (normalized, {model_name})",
                    normalize=True,
                )
                print(f"YOLO model: {model_name}")
        elif model_type == "rfdetr":
            if model_mode == "custom":
                mapped, cm, labels = evaluate_rfdetr_custom(
                    model_path=cfg["model_path"],
                    data_yaml=cfg["data_yaml"],
                    device=cfg["device"],
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                    box_format=rfdetr_box_format,
                    box_normalized=rfdetr_box_normalized,
                )
            else:
                mapped, cm, labels = evaluate_rfdetr_mapped(
                    model_path=cfg["model_path"],
                    data_yaml=cfg["data_yaml"],
                    device=cfg["device"],
                    conf=cfg["conf"],
                    iou=cfg["iou"],
                    box_format=rfdetr_box_format,
                    box_normalized=rfdetr_box_normalized,
                )
            metrics_path = os.path.join(out_dir, "rfdetr_class_metrics.csv")
            mapped.insert(0, "model_name", model_name)
            mapped.to_csv(metrics_path, index=False)
            print(f"Saved: {metrics_path}")
            rfdetr_plots_dir = os.path.join(out_dir, f"rfdetr_plots_{model_name}")
            _ensure_dir(rfdetr_plots_dir)
            save_confusion_matrix_plot(
                cm,
                labels,
                os.path.join(rfdetr_plots_dir, "confusion_matrix.png"),
                f"Confusion matrix ({model_name})",
            )
            save_confusion_matrix_plot(
                cm,
                labels,
                os.path.join(rfdetr_plots_dir, "confusion_matrix_norm.png"),
                f"Confusion matrix (normalized, {model_name})",
                normalize=True,
            )
            summary = mapped[mapped["class_id"] == "all"] if "class_id" in mapped.columns else pd.DataFrame()
            if not summary.empty:
                summary_path = os.path.join(out_dir, "rfdetr_metrics.csv")
                summary.to_csv(summary_path, index=False)
                print(f"Saved: {summary_path}")
            print(f"RF-DETR model: {model_name}")
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

    return 0

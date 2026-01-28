from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from counter.core.schema import EvalConfig
from counter.core.io.counts import load_counts_json
from counter.core.io import get_video_info
from counter.eval.logic.charts import bar_counts
from counter.eval.logic.metrics import (
    agg_metrics,
    class_wape_macro,
    class_wape_micro,
    class_wape_weighted_macro_gt,
    diffs,
    duration_s_from_pred_meta,
    safe_div,
    vectorize_counts,
)


def evaluate_one_run(
    *,
    cfg: EvalConfig,
    run: Any,
    gt_map: Dict[str, Dict[str, Any]],
    classes: List[int],
    class_names: List[str],
    charts_dir: Path,
    log=None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Compute per-run, per-video, and per-class evaluation rows.

    `run` is expected to have fields: run_id, model_id, backend, variant, predict_dir.
    """

    if log:
        log(
            "run_start",
            {"run_id": run.run_id, "model_id": run.model_id, "backend": run.backend, "variant": run.variant},
        )

    # Per-run diffs (video-weighted).
    diffs_in_cls: List[float] = []
    diffs_out_cls: List[float] = []

    # Signed total-count diffs (can hide class swaps; kept for comparison).
    diffs_in_total_counts: List[float] = []
    diffs_out_total_counts: List[float] = []

    # Class-aware "total" per video = L1 across classes.
    abs_err_in_total_ca_list: List[float] = []
    abs_err_out_total_ca_list: List[float] = []

    diffs_in_by_class: Dict[int, List[float]] = {cid: [] for cid in classes}
    diffs_out_by_class: Dict[int, List[float]] = {cid: [] for cid in classes}

    # Per-run sums (event-weighted).
    sum_gt_in_total = 0
    sum_gt_out_total = 0
    sum_pred_in_total = 0
    sum_pred_out_total = 0

    # Class-aware L1 totals (sum over videos of sum_c |p-g|).
    sum_abs_err_in_total = 0.0
    sum_abs_err_out_total = 0.0

    # Total-count abs error (|sum_c p - sum_c g|), kept for comparison.
    sum_abs_err_in_total_counts = 0.0
    sum_abs_err_out_total_counts = 0.0

    sum_gt_in_cls: Dict[int, int] = {cid: 0 for cid in classes}
    sum_gt_out_cls: Dict[int, int] = {cid: 0 for cid in classes}
    sum_abs_err_in_cls: Dict[int, float] = {cid: 0.0 for cid in classes}
    sum_abs_err_out_cls: Dict[int, float] = {cid: 0.0 for cid in classes}

    # Duration sum (rate-based metrics).
    sum_duration_s = 0.0

    per_video_rows: List[Dict[str, Any]] = []

    for f in sorted(Path(run.predict_dir).glob("*.counts.json")):
        if f.name == "aggregate.counts.json":
            continue

        pred_obj = load_counts_json(f)
        video = str(pred_obj.get("video", f"{f.stem}.mp4"))

        gt_obj = gt_map.get(Path(video).stem)
        if gt_obj is None:
            if log:
                log("gt_missing", {"run_id": run.run_id, "video": video})
            continue

        pred_in = pred_obj.get("in_count", {}) or {}
        pred_out = pred_obj.get("out_count", {}) or {}
        gt_in = gt_obj.get("in_count", {}) or {}
        gt_out = gt_obj.get("out_count", {}) or {}

        pred_in_vec = vectorize_counts(pred_in, classes)
        pred_out_vec = vectorize_counts(pred_out, classes)
        gt_in_vec = vectorize_counts(gt_in, classes)
        gt_out_vec = vectorize_counts(gt_out, classes)

        d_in = diffs(pred_in_vec, gt_in_vec)
        d_out = diffs(pred_out_vec, gt_out_vec)

        diffs_in_cls.extend([float(x) for x in d_in])
        diffs_out_cls.extend([float(x) for x in d_out])

        for cid, d in zip(classes, d_in):
            diffs_in_by_class[cid].append(float(d))
        for cid, d in zip(classes, d_out):
            diffs_out_by_class[cid].append(float(d))

        pred_in_total = int(sum(pred_in_vec))
        gt_in_total = int(sum(gt_in_vec))
        pred_out_total = int(sum(pred_out_vec))
        gt_out_total = int(sum(gt_out_vec))

        # Signed totals (useful for bias/debug).
        d_in_total_counts = float(pred_in_total - gt_in_total)
        d_out_total_counts = float(pred_out_total - gt_out_total)
        diffs_in_total_counts.append(d_in_total_counts)
        diffs_out_total_counts.append(d_out_total_counts)

        # Class-aware total error per video (L1 across classes).
        abs_err_in_total_ca = float(sum(abs(int(p) - int(g)) for p, g in zip(pred_in_vec, gt_in_vec)))
        abs_err_out_total_ca = float(sum(abs(int(p) - int(g)) for p, g in zip(pred_out_vec, gt_out_vec)))
        abs_err_in_total_ca_list.append(abs_err_in_total_ca)
        abs_err_out_total_ca_list.append(abs_err_out_total_ca)

        # Sums for event-weighted metrics.
        sum_gt_in_total += gt_in_total
        sum_gt_out_total += gt_out_total
        sum_pred_in_total += pred_in_total
        sum_pred_out_total += pred_out_total
        sum_abs_err_in_total += abs_err_in_total_ca
        sum_abs_err_out_total += abs_err_out_total_ca

        # Total-count abs error (can hide class swaps).
        sum_abs_err_in_total_counts += abs(d_in_total_counts)
        sum_abs_err_out_total_counts += abs(d_out_total_counts)

        # Per-class sums for class-aware metrics.
        for cid, pi, gi in zip(classes, pred_in_vec, gt_in_vec):
            sum_gt_in_cls[cid] += int(gi)
            sum_abs_err_in_cls[cid] += float(abs(int(pi) - int(gi)))
        for cid, po, go in zip(classes, pred_out_vec, gt_out_vec):
            sum_gt_out_cls[cid] += int(go)
            sum_abs_err_out_cls[cid] += float(abs(int(po) - int(go)))

        # Duration (optional).
        duration_s: Optional[float] = duration_s_from_pred_meta(pred_obj)
        if duration_s is None and getattr(cfg, "videos_dir", None):
            try:
                vinfo = get_video_info(str(Path(cfg.videos_dir) / video))
                if vinfo.fps > 0 and vinfo.frame_count > 0:
                    duration_s = float(vinfo.frame_count / vinfo.fps)
            except Exception:
                duration_s = None
        if duration_s is not None:
            sum_duration_s += float(duration_s)

        m_in_cls = agg_metrics([float(x) for x in d_in])
        m_out_cls = agg_metrics([float(x) for x in d_out])

        row: Dict[str, Any] = {
            "run_id": run.run_id,
            "model_id": run.model_id,
            "backend": run.backend,
            "variant": run.variant,
            "video": video,
            "duration_s": float(duration_s) if duration_s is not None else "",
            "mae_in_cls": m_in_cls["mae"],
            "rmse_in_cls": m_in_cls["rmse"],
            "bias_in_cls": m_in_cls["bias"],
            "mae_out_cls": m_out_cls["mae"],
            "rmse_out_cls": m_out_cls["rmse"],
            "bias_out_cls": m_out_cls["bias"],
            "gt_in_total": gt_in_total,
            "pred_in_total": pred_in_total,
            "gt_out_total": gt_out_total,
            "pred_out_total": pred_out_total,
            "abs_err_in_total": abs_err_in_total_ca,
            "abs_err_out_total": abs_err_out_total_ca,
            "err_in_total": d_in_total_counts,
            "err_out_total": d_out_total_counts,
            "abs_err_in_total_counts": abs(d_in_total_counts),
            "abs_err_out_total_counts": abs(d_out_total_counts),
        }

        if duration_s is not None:
            hours = float(duration_s) / 3600.0
            row["gt_in_per_h"] = safe_div(gt_in_total, hours)
            row["pred_in_per_h"] = safe_div(pred_in_total, hours)
            row["gt_out_per_h"] = safe_div(gt_out_total, hours)
            row["pred_out_per_h"] = safe_div(pred_out_total, hours)
        else:
            row["gt_in_per_h"] = ""
            row["pred_in_per_h"] = ""
            row["gt_out_per_h"] = ""
            row["pred_out_per_h"] = ""

        for cname, di, do in zip(class_names, d_in, d_out):
            row[f"abs_err_in_{cname}"] = abs(float(di))
            row[f"err_in_{cname}"] = float(di)
            row[f"abs_err_out_{cname}"] = abs(float(do))
            row[f"err_out_{cname}"] = float(do)

        per_video_rows.append(row)

        if cfg.charts.enabled:
            import re

            def _slug(s: str) -> str:
                # Safe folder name for Windows paths.
                return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")

            vid_stem = Path(video).stem
            run_folder = charts_dir / "in_out_counts" / f"{_slug(run.run_id)}__{_slug(run.model_id)}"
            run_folder.mkdir(parents=True, exist_ok=True)

            for direction, gt_vec, pr_vec in [("IN", gt_in_vec, pred_in_vec), ("OUT", gt_out_vec, pred_out_vec)]:
                chart_path = run_folder / f"{vid_stem}__{direction}.png"
                bar_counts(
                    path=str(chart_path),
                    title=f"{run.model_id} ({run.variant}/{run.backend}) :: {vid_stem} :: {direction}",
                    gt=gt_vec,
                    pred=pr_vec,
                    labels=class_names,
                )

    # Total-count (not class-aware).
    m_run_in_total_counts = agg_metrics(diffs_in_total_counts)
    m_run_out_total_counts = agg_metrics(diffs_out_total_counts)

    # Class-aware: MAE over per-video L1 errors.
    mae_in_total = safe_div(sum(abs_err_in_total_ca_list), float(len(abs_err_in_total_ca_list)))
    mae_out_total = safe_div(sum(abs_err_out_total_ca_list), float(len(abs_err_out_total_ca_list)))

    # Optional RMSE over per-video L1 errors.
    rmse_in_total = (
        safe_div(sum((x * x) for x in abs_err_in_total_ca_list), float(len(abs_err_in_total_ca_list))) ** 0.5
    ) if abs_err_in_total_ca_list else 0.0
    rmse_out_total = (
        safe_div(sum((x * x) for x in abs_err_out_total_ca_list), float(len(abs_err_out_total_ca_list))) ** 0.5
    ) if abs_err_out_total_ca_list else 0.0

    score_total_video_mae = float((mae_in_total + mae_out_total) / 2.0)

    # Event-weighted WAPE (total counts, can hide class swaps).
    wape_in_totalcounts = safe_div(sum_abs_err_in_total_counts, float(sum_gt_in_total))
    wape_out_totalcounts = safe_div(sum_abs_err_out_total_counts, float(sum_gt_out_total))

    # Class-aware "L1 total" WAPE (sum_c |p-g| / sum GT).
    wape_in_total_l1 = safe_div(sum_abs_err_in_total, float(sum_gt_in_total))
    wape_out_total_l1 = safe_div(sum_abs_err_out_total, float(sum_gt_out_total))

    # Class-aware micro WAPE.
    wape_micro_in = class_wape_micro(sum_abs_err_in_cls, sum_gt_in_cls)
    wape_micro_out = class_wape_micro(sum_abs_err_out_cls, sum_gt_out_cls)

    # Weighted-macro with weights w_c = sum_gt[c] (equals micro).
    wape_wmacro_gt_in = class_wape_weighted_macro_gt(sum_abs_err_in_cls, sum_gt_in_cls)
    wape_wmacro_gt_out = class_wape_weighted_macro_gt(sum_abs_err_out_cls, sum_gt_out_cls)

    # Headline score (direction-macro).
    score_total_event_wape = float((wape_micro_in + wape_micro_out) / 2.0)

    # Per-run: class-aware macro WAPE by class.
    class_wape_macro_in = class_wape_macro(sum_abs_err_in_cls, sum_gt_in_cls)
    class_wape_macro_out = class_wape_macro(sum_abs_err_out_cls, sum_gt_out_cls)
    score_total_class_wape = float((class_wape_macro_in + class_wape_macro_out) / 2.0)

    # Per-run: rate-based.
    if sum_duration_s > 0.0:
        hours = float(sum_duration_s) / 3600.0
        gt_in_rate = safe_div(float(sum_gt_in_total), hours)
        pred_in_rate = safe_div(float(sum_pred_in_total), hours)
        gt_out_rate = safe_div(float(sum_gt_out_total), hours)
        pred_out_rate = safe_div(float(sum_pred_out_total), hours)
        rate_mae_in = abs(pred_in_rate - gt_in_rate)
        rate_mae_out = abs(pred_out_rate - gt_out_rate)
        score_total_rate_mae = float((rate_mae_in + rate_mae_out) / 2.0)
    else:
        gt_in_rate = pred_in_rate = gt_out_rate = pred_out_rate = 0.0
        rate_mae_in = rate_mae_out = 0.0
        score_total_rate_mae = 0.0

    per_run_row = {
        "run_id": run.run_id,
        "model_id": run.model_id,
        "backend": run.backend,
        "variant": run.variant,
        "mae_in_total": float(mae_in_total),
        "rmse_in_total": float(rmse_in_total),
        "bias_in_total": "",  # Bias doesn't apply to L1 totals.
        "mae_out_total": float(mae_out_total),
        "rmse_out_total": float(rmse_out_total),
        "bias_out_total": "",
        "mae_in_total_counts": m_run_in_total_counts["mae"],
        "rmse_in_total_counts": m_run_in_total_counts["rmse"],
        "bias_in_total_counts": m_run_in_total_counts["bias"],
        "mae_out_total_counts": m_run_out_total_counts["mae"],
        "rmse_out_total_counts": m_run_out_total_counts["rmse"],
        "bias_out_total_counts": m_run_out_total_counts["bias"],
        "score_total_video_mae": score_total_video_mae,
        "sum_gt_in_total": sum_gt_in_total,
        "sum_gt_out_total": sum_gt_out_total,
        "sum_pred_in_total": sum_pred_in_total,
        "sum_pred_out_total": sum_pred_out_total,
        "sum_abs_err_in_total": float(sum_abs_err_in_total),
        "sum_abs_err_out_total": float(sum_abs_err_out_total),
        "wape_in_totalcounts": float(wape_in_totalcounts),
        "wape_out_totalcounts": float(wape_out_totalcounts),
        "wape_in_total_l1": float(wape_in_total_l1),
        "wape_out_total_l1": float(wape_out_total_l1),
        "wape_micro_in": float(wape_micro_in),
        "wape_micro_out": float(wape_micro_out),
        "wape_wmacro_gt_in": float(wape_wmacro_gt_in),
        "wape_wmacro_gt_out": float(wape_wmacro_gt_out),
        "score_total_event_wape": score_total_event_wape,
        "class_wape_macro_in": float(class_wape_macro_in),
        "class_wape_macro_out": float(class_wape_macro_out),
        "score_total_class_wape": float(score_total_class_wape),
        "sum_duration_s": float(sum_duration_s) if sum_duration_s > 0 else "",
        "gt_in_per_h": float(gt_in_rate) if sum_duration_s else "",
        "pred_in_per_h": float(pred_in_rate) if sum_duration_s else "",
        "gt_out_per_h": float(gt_out_rate) if sum_duration_s else "",
        "pred_out_per_h": float(pred_out_rate) if sum_duration_s else "",
        "rate_mae_in_per_h": float(rate_mae_in) if sum_duration_s else "",
        "rate_mae_out_per_h": float(rate_mae_out) if sum_duration_s else "",
        "score_total_rate_mae": float(score_total_rate_mae) if sum_duration_s else "",
    }

    per_class_rows: List[Dict[str, Any]] = []
    for cid, cname in zip(classes, class_names):
        m = agg_metrics(diffs_in_by_class[cid])
        per_class_rows.append(
            {
                "run_id": run.run_id,
                "model_id": run.model_id,
                "backend": run.backend,
                "variant": run.variant,
                "direction": "IN",
                "class_id": cid,
                "class_name": cname,
                **m,
            }
        )
        m = agg_metrics(diffs_out_by_class[cid])
        per_class_rows.append(
            {
                "run_id": run.run_id,
                "model_id": run.model_id,
                "backend": run.backend,
                "variant": run.variant,
                "direction": "OUT",
                "class_id": cid,
                "class_name": cname,
                **m,
            }
        )

    if log:
        log("run_done", {"run_id": run.run_id})

    return per_run_row, per_video_rows, per_class_rows

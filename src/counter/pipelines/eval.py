from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import json
import math

from counter.config.schema import EvalConfig
from counter.domain.types import CanonicalClass
from counter.io.gt import load_counts_json, load_gt_dir_counts
from counter.io.export import ensure_dir, dump_json
from counter.io.video import get_video_info
from counter.reporting.charts import bar_counts, bar_metric, heatmap_matrix, scatter_xy


@dataclass(frozen=True)
class PredictRunInfo:
    run_id: str
    run_dir: Path
    predict_dir: Path
    model_id: str
    backend: str
    variant: str
    status: str


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _infer_from_dirname(dirname: str) -> Tuple[str, str, str]:
    """
    Supports names like:
      yolo_tuned__yolo11m_v11
      rfdetr_pretrained__rfdetr_small_pretrained
      20260125_224012
    """
    name = dirname.lower()

    if name.startswith("yolo"):
        backend = "yolo"
    elif name.startswith("rfdetr"):
        backend = "rfdetr"
    else:
        backend = "unknown"

    if "tuned" in name:
        variant = "tuned"
    elif "pretrained" in name:
        variant = "pretrained"
    else:
        variant = "unknown"

    # try to extract model_id after "__"
    model_id = dirname
    if "__" in dirname:
        model_id = dirname.split("__", 1)[1]

    return backend, variant, model_id

def _mk_runinfo_from_run_dir(run_dir: Path) -> PredictRunInfo | None:
    predict_dir = run_dir / "predict"
    if not predict_dir.exists():
        return None

    run_json_path = run_dir / "run.json"
    meta: Dict[str, Any] = {}
    if run_json_path.exists():
        obj = _read_json(run_json_path)
        if isinstance(obj, dict):
            meta = obj

    # NEFILTRUJ podle meta["type"] – legacy i ruční run.json jsou OK.
    backend_fallback, variant_fallback, model_fallback = _infer_from_dirname(run_dir.name)

    status = str(meta.get("status", "completed"))

    return PredictRunInfo(
        run_id=str(meta.get("run_id", run_dir.name)),
        run_dir=run_dir,
        predict_dir=predict_dir,
        model_id=str(meta.get("model_id", model_fallback)),
        backend=str(meta.get("backend", backend_fallback)),
        variant=str(meta.get("variant", variant_fallback)),
        status=status,
    )


def _discover_predict_runs(runs_dir: Path) -> List[PredictRunInfo]:
    out: List[PredictRunInfo] = []
    if not runs_dir.exists():
        return out

    # runs_dir je JEDEN run jen když obsahuje run.json + predict/
    if (runs_dir / "run.json").exists() and (runs_dir / "predict").exists():
        one = _mk_runinfo_from_run_dir(runs_dir)
        return [one] if one else []

    # allow passing predict dir directly: .../<run_id>/predict
    if runs_dir.name == "predict" and (runs_dir.parent / "run.json").exists():
        one = _mk_runinfo_from_run_dir(runs_dir.parent)
        return [one] if one else []

    for run_dir in sorted([p for p in runs_dir.iterdir() if p.is_dir()]):
        info = _mk_runinfo_from_run_dir(run_dir)
        if info:
            out.append(info)

    return out



def discover_predict_runs(runs_dir: str | Path) -> List[PredictRunInfo]:
    return _discover_predict_runs(Path(runs_dir))


def _passes_filters(run: PredictRunInfo, cfg: EvalConfig) -> bool:
    f = cfg.filters
    if cfg.only_completed and run.status != "completed":
        return False
    if f.run_ids and run.run_id not in f.run_ids:
        return False
    if f.backends and run.backend not in f.backends:
        return False
    if f.variants and run.variant not in f.variants:
        return False
    if f.model_ids and run.model_id not in f.model_ids:
        return False
    return True


def _vectorize_counts(d: Dict[int, int], classes: List[int]) -> List[int]:
    return [int(d.get(c, 0)) for c in classes]


def _diffs(pred_vec: List[int], gt_vec: List[int]) -> List[int]:
    return [int(p) - int(g) for p, g in zip(pred_vec, gt_vec)]


def _agg_metrics(diffs: List[float]) -> Dict[str, float]:
    if not diffs:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "within1": 0.0, "within2": 0.0}

    n = float(len(diffs))
    mae = sum(abs(d) for d in diffs) / n
    rmse = math.sqrt(sum((d * d) for d in diffs) / n)
    bias = sum(diffs) / n
    within1 = sum(1 for d in diffs if abs(d) <= 1.0) / n
    within2 = sum(1 for d in diffs if abs(d) <= 2.0) / n

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "within1": float(within1),
        "within2": float(within2),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def _write_csv(path: Path, rows: List[Dict[str, Any]], cols: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))
    path.write_text("\n".join(lines), encoding="utf-8")


def _duration_s_from_pred_meta(pred_obj: Dict[str, Any]) -> Optional[float]:
    meta = pred_obj.get("meta")
    if not isinstance(meta, dict):
        return None
    v = meta.get("video")
    if not isinstance(v, dict):
        return None
    try:
        fps = float(v.get("fps") or 0.0)
        frames = int(v.get("frame_count") or 0)
        if fps > 0.0 and frames > 0:
            return float(frames / fps)
    except Exception:
        return None
    return None


def _class_wape_macro(sum_abs_err: Dict[int, float], sum_gt: Dict[int, int]) -> float:
    # macro average over classes with gt>0
    vals: List[float] = []
    for cid, gt in sum_gt.items():
        if gt > 0:
            vals.append(_safe_div(float(sum_abs_err.get(cid, 0.0)), float(gt)))
    return float(sum(vals) / float(len(vals))) if vals else 0.0


class EvalPipeline:
    """
    Outputs:
      - benchmark.json (ranked runs)
      - per_run_metrics.csv
      - per_video_metrics.csv
      - per_class_metrics.csv
      - charts/ (leaderboard, scatter, heatmaps + per-video GT vs Pred bars)
    """

    def run(self, cfg: EvalConfig, predict_run_dir: str | Path | None = None) -> Path:
        gt_map = load_gt_dir_counts(cfg.gt_dir)
        if not gt_map:
            raise FileNotFoundError(f"No GT *.counts.json found in: {cfg.gt_dir}")

        # allow explicit single run dir/predict dir
        if predict_run_dir is not None:
            p = Path(predict_run_dir)
            if p.name == "predict":
                runs = _discover_predict_runs(p)
            else:
                # accept runs/<id> OR runs/<id>/predict
                runs = _discover_predict_runs(p)
                if not runs and (p / "predict").exists():
                    runs = _discover_predict_runs(p)
            runs = [r for r in runs if _passes_filters(r, cfg)]
        else:
            runs_dir = Path(cfg.runs_dir)
            runs = [r for r in _discover_predict_runs(runs_dir) if _passes_filters(r, cfg)]

        if not runs:
            raise FileNotFoundError(f"No matching predict runs found in: {cfg.runs_dir}")

        out_root = ensure_dir(Path(cfg.out_dir) / f"eval_{cfg.timestamp}")
        charts_dir = ensure_dir(out_root / "charts")

        classes = [int(c) for c in CanonicalClass]
        class_names = [c.name for c in CanonicalClass]

        per_run_rows: List[Dict[str, Any]] = []
        per_video_rows: List[Dict[str, Any]] = []
        per_class_rows: List[Dict[str, Any]] = []

        for run in runs:
            # per-run diffs (video-weighted)
            diffs_in_cls: List[float] = []
            diffs_out_cls: List[float] = []
            diffs_in_total: List[float] = []
            diffs_out_total: List[float] = []

            diffs_in_by_class: Dict[int, List[float]] = {cid: [] for cid in classes}
            diffs_out_by_class: Dict[int, List[float]] = {cid: [] for cid in classes}

            # per-run sums (event-weighted + class-wape)
            sum_gt_in_total = 0
            sum_gt_out_total = 0
            sum_pred_in_total = 0
            sum_pred_out_total = 0
            sum_abs_err_in_total = 0.0
            sum_abs_err_out_total = 0.0

            sum_gt_in_cls: Dict[int, int] = {cid: 0 for cid in classes}
            sum_gt_out_cls: Dict[int, int] = {cid: 0 for cid in classes}
            sum_abs_err_in_cls: Dict[int, float] = {cid: 0.0 for cid in classes}
            sum_abs_err_out_cls: Dict[int, float] = {cid: 0.0 for cid in classes}

            # duration (rate-based)
            sum_duration_s = 0.0

            for f in sorted(run.predict_dir.glob("*.counts.json")):
                if f.name == "aggregate.counts.json":
                    continue

                pred_obj = load_counts_json(str(f))
                video = str(pred_obj.get("video", f"{f.stem}.mp4"))

                gt_obj = gt_map.get(Path(video).stem)
                if gt_obj is None:
                    continue

                pred_in = pred_obj.get("in_count", {}) or {}
                pred_out = pred_obj.get("out_count", {}) or {}
                gt_in = gt_obj.get("in_count", {}) or {}
                gt_out = gt_obj.get("out_count", {}) or {}

                pred_in_vec = _vectorize_counts(pred_in, classes)
                pred_out_vec = _vectorize_counts(pred_out, classes)
                gt_in_vec = _vectorize_counts(gt_in, classes)
                gt_out_vec = _vectorize_counts(gt_out, classes)

                d_in = _diffs(pred_in_vec, gt_in_vec)
                d_out = _diffs(pred_out_vec, gt_out_vec)

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

                d_in_total = float(pred_in_total - gt_in_total)
                d_out_total = float(pred_out_total - gt_out_total)
                diffs_in_total.append(d_in_total)
                diffs_out_total.append(d_out_total)

                # sums for event-weighted metrics
                sum_gt_in_total += gt_in_total
                sum_gt_out_total += gt_out_total
                sum_pred_in_total += pred_in_total
                sum_pred_out_total += pred_out_total
                sum_abs_err_in_total += abs(d_in_total)
                sum_abs_err_out_total += abs(d_out_total)

                # per-class sums for class-aware metric
                for cid, pi, gi in zip(classes, pred_in_vec, gt_in_vec):
                    sum_gt_in_cls[cid] += int(gi)
                    sum_abs_err_in_cls[cid] += float(abs(int(pi) - int(gi)))
                for cid, po, go in zip(classes, pred_out_vec, gt_out_vec):
                    sum_gt_out_cls[cid] += int(go)
                    sum_abs_err_out_cls[cid] += float(abs(int(po) - int(go)))

                # duration (optional)
                duration_s: Optional[float] = _duration_s_from_pred_meta(pred_obj)
                if duration_s is None and getattr(cfg, "videos_dir", None):
                    try:
                        vinfo = get_video_info(str(Path(cfg.videos_dir) / video))
                        if vinfo.fps > 0 and vinfo.frame_count > 0:
                            duration_s = float(vinfo.frame_count / vinfo.fps)
                    except Exception:
                        duration_s = None
                if duration_s is not None:
                    sum_duration_s += float(duration_s)

                m_in_cls = _agg_metrics([float(x) for x in d_in])
                m_out_cls = _agg_metrics([float(x) for x in d_out])

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
                    "abs_err_in_total": abs(d_in_total),
                    "err_in_total": d_in_total,
                    "gt_out_total": gt_out_total,
                    "pred_out_total": pred_out_total,
                    "abs_err_out_total": abs(d_out_total),
                    "err_out_total": d_out_total,
                }

                if duration_s:
                    hours = float(duration_s) / 3600.0
                    row["gt_in_per_h"] = _safe_div(gt_in_total, hours)
                    row["pred_in_per_h"] = _safe_div(pred_in_total, hours)
                    row["gt_out_per_h"] = _safe_div(gt_out_total, hours)
                    row["pred_out_per_h"] = _safe_div(pred_out_total, hours)
                else:
                    row["gt_in_per_h"] = ""
                    row["pred_in_per_h"] = ""
                    row["gt_out_per_h"] = ""
                    row["pred_out_per_h"] = ""

                for cname, di, do in zip(class_names, d_in, d_out):
                    row[f"abs_err_in_{cname}"] = abs(int(di))
                    row[f"err_in_{cname}"] = int(di)
                    row[f"abs_err_out_{cname}"] = abs(int(do))
                    row[f"err_out_{cname}"] = int(do)

                per_video_rows.append(row)

                if cfg.charts.enabled:
                    vid_stem = Path(video).stem
                    for direction, gt_vec, pr_vec in [
                        ("IN", gt_in_vec, pred_in_vec),
                        ("OUT", gt_out_vec, pred_out_vec),
                    ]:
                        chart_path = charts_dir / f"{run.run_id}__{vid_stem}__{direction}.png"
                        bar_counts(
                            path=str(chart_path),
                            title=f"{run.model_id} ({run.variant}/{run.backend}) :: {vid_stem} :: {direction}",
                            gt=gt_vec,
                            pred=pr_vec,
                            labels=class_names,
                        )

            # per-run: video-weighted
            m_run_in_cls = _agg_metrics(diffs_in_cls)
            m_run_out_cls = _agg_metrics(diffs_out_cls)
            m_run_in_total = _agg_metrics(diffs_in_total)
            m_run_out_total = _agg_metrics(diffs_out_total)
            score_total_video_mae = float((m_run_in_total["mae"] + m_run_out_total["mae"]) / 2.0)

            # per-run: event-weighted (WAPE)
            wape_in_total = _safe_div(sum_abs_err_in_total, float(sum_gt_in_total))
            wape_out_total = _safe_div(sum_abs_err_out_total, float(sum_gt_out_total))
            score_total_event_wape = float((wape_in_total + wape_out_total) / 2.0)

            # per-run: class-aware (macro WAPE by class)
            class_wape_macro_in = _class_wape_macro(sum_abs_err_in_cls, sum_gt_in_cls)
            class_wape_macro_out = _class_wape_macro(sum_abs_err_out_cls, sum_gt_out_cls)
            score_total_class_wape = float((class_wape_macro_in + class_wape_macro_out) / 2.0)

            # per-run: rate-based (needs duration)
            if sum_duration_s > 0.0:
                hours = float(sum_duration_s) / 3600.0
                gt_in_rate = _safe_div(float(sum_gt_in_total), hours)
                pred_in_rate = _safe_div(float(sum_pred_in_total), hours)
                gt_out_rate = _safe_div(float(sum_gt_out_total), hours)
                pred_out_rate = _safe_div(float(sum_pred_out_total), hours)
                rate_mae_in = abs(pred_in_rate - gt_in_rate)
                rate_mae_out = abs(pred_out_rate - gt_out_rate)
                score_total_rate_mae = float((rate_mae_in + rate_mae_out) / 2.0)
            else:
                gt_in_rate = pred_in_rate = gt_out_rate = pred_out_rate = 0.0
                rate_mae_in = rate_mae_out = 0.0
                score_total_rate_mae = 0.0

            per_run_rows.append(
                {
                    "run_id": run.run_id,
                    "model_id": run.model_id,
                    "backend": run.backend,
                    "variant": run.variant,

                    "mae_in_cls": m_run_in_cls["mae"],
                    "rmse_in_cls": m_run_in_cls["rmse"],
                    "bias_in_cls": m_run_in_cls["bias"],
                    "within1_in_cls": m_run_in_cls["within1"],
                    "within2_in_cls": m_run_in_cls["within2"],
                    "mae_out_cls": m_run_out_cls["mae"],
                    "rmse_out_cls": m_run_out_cls["rmse"],
                    "bias_out_cls": m_run_out_cls["bias"],
                    "within1_out_cls": m_run_out_cls["within1"],
                    "within2_out_cls": m_run_out_cls["within2"],

                    "mae_in_total": m_run_in_total["mae"],
                    "rmse_in_total": m_run_in_total["rmse"],
                    "bias_in_total": m_run_in_total["bias"],
                    "mae_out_total": m_run_out_total["mae"],
                    "rmse_out_total": m_run_out_total["rmse"],
                    "bias_out_total": m_run_out_total["bias"],

                    "score_total_video_mae": score_total_video_mae,

                    "sum_gt_in_total": sum_gt_in_total,
                    "sum_gt_out_total": sum_gt_out_total,
                    "sum_pred_in_total": sum_pred_in_total,
                    "sum_pred_out_total": sum_pred_out_total,
                    "sum_abs_err_in_total": float(sum_abs_err_in_total),
                    "sum_abs_err_out_total": float(sum_abs_err_out_total),
                    "wape_in_total": float(wape_in_total),
                    "wape_out_total": float(wape_out_total),
                    "score_total_event_wape": score_total_event_wape,

                    "class_wape_macro_in": float(class_wape_macro_in),
                    "class_wape_macro_out": float(class_wape_macro_out),
                    "score_total_class_wape": float(score_total_class_wape),

                    "sum_duration_s": float(sum_duration_s) if sum_duration_s else "",
                    "gt_in_per_h": float(gt_in_rate) if sum_duration_s else "",
                    "pred_in_per_h": float(pred_in_rate) if sum_duration_s else "",
                    "gt_out_per_h": float(gt_out_rate) if sum_duration_s else "",
                    "pred_out_per_h": float(pred_out_rate) if sum_duration_s else "",
                    "rate_mae_in_per_h": float(rate_mae_in) if sum_duration_s else "",
                    "rate_mae_out_per_h": float(rate_mae_out) if sum_duration_s else "",
                    "score_total_rate_mae": float(score_total_rate_mae) if sum_duration_s else "",
                }
            )

            for cid, cname in zip(classes, class_names):
                m = _agg_metrics(diffs_in_by_class[cid])
                per_class_rows.append(
                    {
                        "run_id": run.run_id,
                        "model_id": run.model_id,
                        "backend": run.backend,
                        "variant": run.variant,
                        "direction": "IN",
                        "class_id": cid,
                        "class_name": cname,
                        **{f"{k}": v for k, v in m.items()},
                    }
                )
                m = _agg_metrics(diffs_out_by_class[cid])
                per_class_rows.append(
                    {
                        "run_id": run.run_id,
                        "model_id": run.model_id,
                        "backend": run.backend,
                        "variant": run.variant,
                        "direction": "OUT",
                        "class_id": cid,
                        "class_name": cname,
                        **{f"{k}": v for k, v in m.items()},
                    }
                )

        # ranking
        rank_by = getattr(cfg, "rank_by", "video_mae_total")
        key_map = {
            "video_mae_total": "score_total_video_mae",
            "event_wape_total": "score_total_event_wape",
            "rate_mae_total": "score_total_rate_mae",
            "class_wape_total": "score_total_class_wape",
        }
        score_field = key_map.get(rank_by, "score_total_video_mae")

        # if rate-based requested but no durations, fallback
        if rank_by == "rate_mae_total" and not any(r.get("score_total_rate_mae") != "" for r in per_run_rows):
            score_field = "score_total_video_mae"

        ranked = sorted(per_run_rows, key=lambda r: float(r.get(score_field, 0.0) or 0.0))
        for i, r in enumerate(ranked, start=1):
            r["rank"] = i

        dump_json(
            out_root / "benchmark.json",
            {
                "rank_by": rank_by,
                "score_field": score_field,
                "classes": [{"id": cid, "name": cname} for cid, cname in zip(classes, class_names)],
                "ranked_runs": ranked,
                "notes": {
                    "score_total_video_mae": "Avg MAE over (IN_total, OUT_total), each video equal weight.",
                    "score_total_event_wape": "Avg WAPE over (IN_total, OUT_total). WAPE=sum(|err|)/sum(GT).",
                    "score_total_rate_mae": "Avg abs error of passages/hour (IN & OUT). Needs durations.",
                    "score_total_class_wape": "Avg macro-WAPE over classes (IN & OUT). Penalizes wrong class distribution.",
                },
            },
        )

        dump_json(out_root / "metrics.json", {"per_run": per_run_rows, "per_video": per_video_rows, "per_class": per_class_rows})

        per_run_cols = [
            "rank",
            "run_id", "model_id", "backend", "variant",
            score_field,
            "score_total_video_mae",
            "score_total_event_wape",
            "score_total_rate_mae",
            "score_total_class_wape",
            "mae_in_total", "rmse_in_total", "bias_in_total",
            "mae_out_total", "rmse_out_total", "bias_out_total",
            "wape_in_total", "wape_out_total",
            "class_wape_macro_in", "class_wape_macro_out",
            "sum_gt_in_total", "sum_gt_out_total",
            "sum_pred_in_total", "sum_pred_out_total",
            "sum_abs_err_in_total", "sum_abs_err_out_total",
            "sum_duration_s",
            "rate_mae_in_per_h", "rate_mae_out_per_h",
            "mae_in_cls", "rmse_in_cls", "bias_in_cls", "within1_in_cls", "within2_in_cls",
            "mae_out_cls", "rmse_out_cls", "bias_out_cls", "within1_out_cls", "within2_out_cls",
        ]
        _write_csv(out_root / "per_run_metrics.csv", ranked, per_run_cols)

        per_video_cols = [
            "run_id", "model_id", "backend", "variant", "video",
            "duration_s",
            "gt_in_total", "pred_in_total", "abs_err_in_total", "err_in_total",
            "gt_out_total", "pred_out_total", "abs_err_out_total", "err_out_total",
            "gt_in_per_h", "pred_in_per_h",
            "gt_out_per_h", "pred_out_per_h",
            "mae_in_cls", "rmse_in_cls", "bias_in_cls",
            "mae_out_cls", "rmse_out_cls", "bias_out_cls",
        ]
        for cname in class_names:
            per_video_cols += [f"abs_err_in_{cname}", f"err_in_{cname}", f"abs_err_out_{cname}", f"err_out_{cname}"]
        _write_csv(out_root / "per_video_metrics.csv", per_video_rows, per_video_cols)

        per_class_cols = [
            "run_id", "model_id", "backend", "variant",
            "direction", "class_id", "class_name",
            "mae", "rmse", "bias", "within1", "within2",
        ]
        _write_csv(out_root / "per_class_metrics.csv", per_class_rows, per_class_cols)

        if cfg.charts.enabled and len(ranked) >= 1:
            labels = [str(r["model_id"]) for r in ranked]
            scores = [float(r.get(score_field, 0.0) or 0.0) for r in ranked]
            bar_metric(
                charts_dir / f"leaderboard_{score_field}.png",
                f"Leaderboard ({score_field})",
                labels,
                scores,
                ylabel=score_field,
            )

            x = [float(r.get("mae_in_total", 0.0) or 0.0) for r in ranked]
            y = [float(r.get("mae_out_total", 0.0) or 0.0) for r in ranked]
            scatter_xy(
                charts_dir / "scatter_mae_total_in_vs_out.png",
                "MAE total: IN vs OUT",
                x=x,
                y=y,
                labels=labels,
                xlabel="MAE IN total",
                ylabel="MAE OUT total",
            )

            videos = sorted({Path(r["video"]).stem for r in per_video_rows})
            if videos:
                run_labels = [str(r["model_id"]) for r in ranked]
                run_idx = {str(r["run_id"]): i for i, r in enumerate(ranked)}
                vid_idx = {v: j for j, v in enumerate(videos)}

                nan = float("nan")
                mat_in = [[nan for _ in videos] for _ in run_labels]
                mat_out = [[nan for _ in videos] for _ in run_labels]

                for r in per_video_rows:
                    ri = run_idx.get(str(r["run_id"]))
                    vj = vid_idx.get(Path(str(r["video"])).stem)
                    if ri is None or vj is None:
                        continue
                    mat_in[ri][vj] = float(r.get("abs_err_in_total", nan))
                    mat_out[ri][vj] = float(r.get("abs_err_out_total", nan))

                heatmap_matrix(
                    charts_dir / "heatmap_abs_total_error_IN.png",
                    "Abs total error (IN) per video",
                    x_labels=videos,
                    y_labels=run_labels,
                    matrix=mat_in,
                    xlabel="video",
                    ylabel="model",
                    fmt="{:.0f}",
                )
                heatmap_matrix(
                    charts_dir / "heatmap_abs_total_error_OUT.png",
                    "Abs total error (OUT) per video",
                    x_labels=videos,
                    y_labels=run_labels,
                    matrix=mat_out,
                    xlabel="video",
                    ylabel="model",
                    fmt="{:.0f}",
                )

            run_labels = [str(r["model_id"]) for r in ranked]
            class_cols = class_names
            idx_run = {str(r["run_id"]): i for i, r in enumerate(ranked)}
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

        return out_root

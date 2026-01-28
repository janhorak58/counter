from __future__ import annotations

import math
from typing import Dict, List, Optional


def vectorize_counts(d: Dict[int, int], classes: List[int]) -> List[int]:
    """Convert a counts dict into a list aligned with class ids."""
    return [int(d.get(c, 0)) for c in classes]


def diffs(pred_vec: List[int], gt_vec: List[int]) -> List[int]:
    """Return element-wise differences (pred - gt)."""
    return [int(p) - int(g) for p, g in zip(pred_vec, gt_vec)]


def agg_metrics(diffs_: List[float]) -> Dict[str, float]:
    """Aggregate MAE/RMSE/bias and within-1/2 stats for diffs."""
    if not diffs_:
        return {"mae": 0.0, "rmse": 0.0, "bias": 0.0, "within1": 0.0, "within2": 0.0}

    n = float(len(diffs_))
    mae = sum(abs(d) for d in diffs_) / n
    rmse = math.sqrt(sum((d * d) for d in diffs_) / n)
    bias = sum(diffs_) / n
    within1 = sum(1 for d in diffs_ if abs(d) <= 1.0) / n
    within2 = sum(1 for d in diffs_ if abs(d) <= 2.0) / n

    return {
        "mae": float(mae),
        "rmse": float(rmse),
        "bias": float(bias),
        "within1": float(within1),
        "within2": float(within2),
    }


def safe_div(num: float, den: float) -> float:
    """Safe division returning 0.0 when denominator is zero."""
    return float(num / den) if den else 0.0


def duration_s_from_pred_meta(pred_obj: dict) -> Optional[float]:
    """Extract duration in seconds from prediction metadata, if present."""
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


def class_wape_macro(sum_abs_err: Dict[int, float], sum_gt: Dict[int, int]) -> float:
    """Macro average over classes with GT>0: mean(sum(|err|)/sum(GT))."""
    vals: List[float] = []
    for cid, gt in sum_gt.items():
        if gt > 0:
            vals.append(safe_div(float(sum_abs_err.get(cid, 0.0)), float(gt)))
    return float(sum(vals) / float(len(vals))) if vals else 0.0


def class_wape_micro(sum_abs_err: Dict[int, float], sum_gt: Dict[int, int]) -> float:
    """Micro class-aware WAPE: sum_c abs_err[c] / sum_c gt[c]."""
    num = float(sum(float(v) for v in sum_abs_err.values()))
    den = float(sum(int(v) for v in sum_gt.values()))
    return safe_div(num, den)


def class_wape_weighted_macro_gt(sum_abs_err: Dict[int, float], sum_gt: Dict[int, int]) -> float:
    """Weighted macro WAPE with weights w_c = sum_gt[c].

    This reduces algebraically to the micro WAPE, but we keep it explicit
    for reporting terminology.
    """
    num_weighted = 0.0
    den_weights = 0.0
    for cid, gt in sum_gt.items():
        gt_f = float(gt)
        if gt_f <= 0.0:
            continue
        w = gt_f
        per_class_wape = safe_div(float(sum_abs_err.get(cid, 0.0)), gt_f)
        num_weighted += w * per_class_wape
        den_weights += w

    return safe_div(num_weighted, den_weights)

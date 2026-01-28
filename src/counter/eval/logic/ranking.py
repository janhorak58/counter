from __future__ import annotations

from typing import Any, Dict, List, Tuple

from counter.core.schema import EvalConfig


def pick_score_field(cfg: EvalConfig) -> str:
    # keep this mapping in ONE place
    key_map = {
        "video_mae_total": "score_total_video_mae",
        "event_wape_total": "score_total_event_wape",
        "rate_mae_total": "score_total_rate_mae",
        # allow future extension
        "class_wape_total": "score_total_class_wape",
    }
    return key_map.get(getattr(cfg, "rank_by", "video_mae_total"), "score_total_video_mae")


def rank_runs(cfg: EvalConfig, per_run_rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    score_field = pick_score_field(cfg)

    # fallback if rate-based requested but no durations
    if getattr(cfg, "rank_by", "") == "rate_mae_total":
        if not any(r.get("score_total_rate_mae") not in (None, "") for r in per_run_rows):
            score_field = "score_total_video_mae"

    ranked = sorted(per_run_rows, key=lambda r: float(r.get(score_field, 0.0) or 0.0))
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    return score_field, ranked

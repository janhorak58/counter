from __future__ import annotations

from typing import Any, Dict, List, Tuple

from counter.core.schema import EvalConfig


def pick_score_field(cfg: EvalConfig) -> str:
    """Return the score field name to rank by."""
    # Keep this mapping in one place.
    key_map = {
        "video_mae_total": "score_total_video_mae",
        "micro_wape_total": "score_total_micro_wape",
        "event_wape_total": "score_total_micro_wape",
        "rate_mae_total": "score_total_rate_mae",
        "macro_wape_total": "score_total_macro_wape",
        "class_wape_total": "score_total_macro_wape",
    }
    return key_map.get(cfg.rank_by, "score_total_video_mae")


def rank_runs(cfg: EvalConfig, per_run_rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """Sort runs by the chosen score field and add rank indices."""
    score_field = pick_score_field(cfg)
    default_score = "score_total_video_mae"

    # Fallback if rate-based ranking was requested but durations are missing.
    if cfg.rank_by == "rate_mae_total":
        if not any(r.get("score_total_rate_mae") not in (None, "") for r in per_run_rows):
            score_field = default_score

    if not any(r.get(score_field) not in (None, "") for r in per_run_rows):
        score_field = default_score

    ranked = sorted(per_run_rows, key=lambda r: float(r.get(score_field, 0.0) or 0.0))
    for i, r in enumerate(ranked, start=1):
        r["rank"] = i

    return score_field, ranked

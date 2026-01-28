from __future__ import annotations

"""Eval-specific IO helpers.

Keep implementations centralized in counter.core.io to avoid drift.
"""

from counter.core.io.counts import load_counts_json, load_gt_dir_counts

__all__ = ["load_counts_json", "load_gt_dir_counts"]

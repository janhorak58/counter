from __future__ import annotations

from counter.core.pipeline.base import StageContext
from counter.core.io.counts import load_gt_dir_counts
from counter.core.types import CanonicalClass


class LoadGTCounts:
    """Stage that loads ground-truth counts and class metadata."""
    name = "LoadGTCounts"

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg

        gt_map = load_gt_dir_counts(cfg.gt_dir)
        if not gt_map:
            raise FileNotFoundError(f"No GT *.counts.json found in: {cfg.gt_dir}")

        classes = [int(c) for c in CanonicalClass]
        class_names = [c.name for c in CanonicalClass]

        ctx.state["gt_map"] = gt_map
        ctx.state["classes"] = classes
        ctx.state["class_names"] = class_names

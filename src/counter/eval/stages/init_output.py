from __future__ import annotations

from pathlib import Path

from counter.core.pipeline.base import StageContext
from counter.core.utils import ensure_dir


class InitOutput:
    name = "InitOutput"

    def run(self, ctx: StageContext) -> None:
        cfg = ctx.cfg
        out_root = ensure_dir(Path(cfg.out_dir) / f"eval_{cfg.timestamp}")
        charts_dir = ensure_dir(out_root / "charts")

        ctx.state["out_root"] = out_root
        ctx.state["charts_dir"] = charts_dir

        # připravíme sběrné tabulky (stejně jako ve tvém kódu)
        ctx.state["per_run_rows"] = []
        ctx.state["per_video_rows"] = []
        ctx.state["per_class_rows"] = []

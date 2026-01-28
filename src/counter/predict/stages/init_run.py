from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from counter.core.config import load_models_registry
from counter.core.io import ensure_dir
from counter.core.pipeline.log import JsonlLogger


def _ts() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass
class InitRun:
    """Prepare run dirs + logger + load model spec."""

    name: str = "init_run"
    models_yaml: Path | None = None
    debug: bool = False

    def run(self, ctx) -> None:
        cfg = ctx.cfg
        models_yaml = Path(ctx.state.get("models_yaml") or self.models_yaml or "configs/models.yaml")

        registry = load_models_registry(models_yaml)
        if cfg.model_id not in registry.models:
            raise KeyError(
                f"Unknown model_id={cfg.model_id!r}. Not in models registry: {models_yaml}"
            )

        spec = registry.models[cfg.model_id]

        run_id = f"{_ts()}"
        subpath = f"{spec.backend}_{spec.variant}__{cfg.model_id}"
        run_root = ensure_dir(Path(cfg.export.out_dir) / subpath / run_id)
        predict_dir = ensure_dir(run_root / "predict")

        log = JsonlLogger(predict_dir / "predict.log.jsonl")
        ctx.assets["log"] = log

        ctx.state.update(
            {
                "models_yaml": models_yaml,
                "models_registry": registry,
                "model_spec": spec,
                "run_id": run_id,
                "run_root": run_root,
                "predict_dir": predict_dir,
                "debug": bool(ctx.state.get("debug", self.debug)),
            }
        )

        log("run_start", {"run_id": run_id, "model_id": cfg.model_id, "backend": spec.backend, "variant": spec.variant})

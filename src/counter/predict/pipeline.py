from __future__ import annotations

from pathlib import Path

from counter.core.pipeline.base import PipelineRunner, StageContext
from counter.core.schema import PredictConfig

from counter.predict.stages.init_run import InitRun
from counter.predict.stages.build_components import BuildComponents
from counter.predict.stages.predict_videos import PredictVideos
from counter.predict.stages.finalize_run import FinalizeRun


def _noop_log(event: str, payload: dict) -> None:  # pragma: no cover
    return None


class PredictPipeline:

    def __init__(self, *, models_yaml: str | Path = "configs/models.yaml", debug: bool = False):
        self.models_yaml = Path(models_yaml)
        self.debug = bool(debug)

    def run(self, cfg: PredictConfig) -> Path:
        ctx = StageContext(cfg=cfg, state={}, assets={"log": _noop_log})

        stages = [
            InitRun(models_yaml=self.models_yaml, debug=self.debug),
            BuildComponents(),
            PredictVideos(),
            FinalizeRun(),
        ]
        PipelineRunner(stages=stages).run(ctx)
        return Path(ctx.state["run_root"])

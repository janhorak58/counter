from __future__ import annotations

from dataclasses import dataclass

from counter.core.pipeline.base import StageContext
from counter.core.schema import PredictConfig
from counter.predict.counting.counter import TrackCounter
from counter.predict.mapping.factory import make_mapper
from counter.predict.tracking.factory import make_provider
from counter.predict.visual.renderer import FrameRenderer


@dataclass
class BuildComponents:
    """Stage that constructs model provider, mapper, counter, and renderer."""

    name: str = "build_components"

    def run(self, ctx: StageContext) -> None:
        cfg: PredictConfig = ctx.cfg
        spec = ctx.state["model_spec"]
        log = ctx.assets.get("log")

        provider = make_provider(cfg=cfg, spec=spec)
        label_map = provider.get_label_map()
        if label_map:
            log("provider_label_map", {"sample": dict(list(label_map.items())[:10])})
        else:
            log("provider_label_map", {"sample": None})

        mapper = make_mapper(spec=spec, label_map=label_map, log=log)

        # Counter uses mapper.finalize_counts to compute canonical counts.
        counter = TrackCounter(
            line=tuple(cfg.line.coords),
            greyzone_px=float(cfg.greyzone_px),
            finalize_fn=mapper.finalize_counts,
            line_base_resolution=tuple(cfg.line.default_resolution),
        )

        renderer = FrameRenderer(
            line=tuple(cfg.line.coords),
            show_boxes=True,
            show_stats=True,
            show_raw=bool(ctx.state.get("debug", False)),
            show_dropped_raw=bool(ctx.state.get("debug", False)),
        )

        ctx.assets["provider"] = provider
        ctx.assets["mapper"] = mapper
        ctx.assets["counter"] = counter
        ctx.assets["renderer"] = renderer

from __future__ import annotations

from dataclasses import dataclass

from counter.core.pipeline.base import StageContext
from counter.core.schema import PredictConfig
from counter.predict.counting.counter import TrackCounter
from counter.predict.mapping.factory import make_mapper
from counter.predict.tracking.factory import make_provider
from counter.predict.visual.renderer import FrameRenderer

try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None


def _resolve_device(requested: str) -> tuple[str, str | None]:
    device = str(requested or "cpu").strip() or "cpu"

    if not device.startswith("cuda"):
        return device, None

    if torch is None:
        return "cpu", "PyTorch není dostupný s CUDA podporou."

    try:
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        cuda_available = False

    if cuda_available:
        return device, None

    return "cpu", "CUDA není dostupná, používá se CPU."


@dataclass
class BuildComponents:
    """Stage that constructs model provider, mapper, counter, and renderer."""

    name: str = "build_components"

    def run(self, ctx: StageContext) -> None:
        cfg: PredictConfig = ctx.cfg
        spec = ctx.state["model_spec"]
        log = ctx.assets.get("log")

        effective_device, fallback_reason = _resolve_device(cfg.device)
        if fallback_reason is not None:
            log(
                "device_fallback",
                {
                    "requested_device": str(cfg.device),
                    "effective_device": effective_device,
                    "reason": fallback_reason,
                },
            )
        cfg.device = effective_device
        log("device_selected", {"device": effective_device})

        provider = make_provider(cfg=cfg, spec=spec)
        label_map = provider.get_label_map()
        if label_map:
            log("provider_label_map", {"sample": dict(list(label_map.items())[:10])})
        else:
            log("provider_label_map", {"sample": None})

        mapper = make_mapper(spec=spec, label_map=label_map, log=log)

        counter = TrackCounter(
            line=tuple(cfg.line.coords),
            greyzone_px=float(cfg.greyzone_px),
            oscillation_window_frames=int(getattr(cfg, "oscillation_window_frames", 0)),
            trajectory_len=int(getattr(cfg, "trajectory_len", 40)),
            class_vote_window_frames=int(getattr(cfg, "class_vote_window_frames", 30)),
            finalize_fn=mapper.finalize_counts,
            line_base_resolution=tuple(cfg.line.default_resolution),
            log=log,
        )

        renderer = FrameRenderer(
            line=tuple(cfg.line.coords),
            show_boxes=True,
            show_stats=True,
            show_raw=bool(ctx.state.get("debug", False)),
            show_dropped_raw=bool(ctx.state.get("debug", False)),
            show_trajectories=True,
        )

        ctx.assets["provider"] = provider
        ctx.assets["mapper"] = mapper
        ctx.assets["counter"] = counter
        ctx.assets["renderer"] = renderer

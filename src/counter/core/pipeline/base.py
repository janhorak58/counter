from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Protocol


class Stage(Protocol):
    """Protocol for pipeline stages."""
    name: str
    def run(self, ctx: "StageContext") -> None: ...


@dataclass
class StageContext:
    """Shared data passed between pipeline stages."""
    cfg: Any
    state: Dict[str, Any] = field(default_factory=dict)
    assets: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineRunner:
    """Sequential runner for pipeline stages."""
    stages: List[Stage]
    fail_fast: bool = True

    def run(self, ctx: StageContext) -> StageContext:
        log = ctx.assets.get("log")
        for st in self.stages:
            if log:
                log("stage_start", {"stage": st.name})
            try:
                st.run(ctx)
            except Exception as e:
                if log:
                    log("stage_error", {"stage": st.name, "error": repr(e)})
                if self.fail_fast:
                    raise
                ctx.state.setdefault("errors", []).append((st.name, repr(e)))
            if log:
                log("stage_done", {"stage": st.name})
        return ctx

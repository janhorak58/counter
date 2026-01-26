from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict

@dataclass
class RunMeta:
    run_id: str
    created_at: str
    model_id: str
    backend: str
    variant: str
    weights: str
    mapping_policy: str
    thresholds: Dict[str, float]
    tracker: Dict[str, Any]
    line: Dict[str, Any]
    greyzone_px: float
    video: Dict[str, Any]
    extra: Dict[str, Any] | None = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d.setdefault('extra', {})
        return d

@dataclass
class CountsResult:
    video: str
    line_name: str
    in_count: Dict[str, int]
    out_count: Dict[str, int]
    meta: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'video': self.video,
            'line_name': self.line_name,
            'in_count': self.in_count,
            'out_count': self.out_count,
            'meta': self.meta,
        }

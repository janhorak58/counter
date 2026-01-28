from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from counter.core.io import dump_json


def now_iso() -> str:
    """Return current local time as an ISO string (seconds precision)."""
    return datetime.now().isoformat(timespec="seconds")


def build_counts_object(
    *,
    video: str,
    line_name: str,
    in_count: Dict[int, int],
    out_count: Dict[int, int],
    meta: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a counts JSON-compatible dictionary."""
    return {
        "video": video,
        "line_name": line_name,
        "in_count": {str(k): int(v) for k, v in in_count.items()},
        "out_count": {str(k): int(v) for k, v in out_count.items()},
        "meta": meta,
    }


def write_counts_json(path: Path, obj: Dict[str, Any]) -> None:
    """Write counts object as JSON to disk."""
    dump_json(path, obj)

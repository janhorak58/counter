from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_iso() -> str:
    """Return current UTC time as an ISO string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JsonlLogger:
    """Append structured events to a JSONL file and stdout."""
    path: Path

    def __call__(self, event: str, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"t": _now_iso(), "event": event, **payload}
        print(json.dumps(rec, ensure_ascii=False), flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

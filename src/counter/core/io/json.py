from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional


def read_json(path: str | Path) -> Optional[Dict[str, Any]]:
    """Best-effort JSON read. Returns None on any error."""
    p = Path(path)
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def dump_json(path: str | Path, obj: Dict[str, Any], *, indent: int = 2) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    last_err: OSError | None = None
    for _ in range(3):
        try:
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=indent)
            os.replace(tmp, p)
            return p
        except OSError as exc:
            last_err = exc
            if getattr(exc, "errno", None) != 116:
                raise
            time.sleep(0.2)
    if last_err is not None:
        raise last_err
    return p

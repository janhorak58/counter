from __future__ import annotations

import json
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
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return p

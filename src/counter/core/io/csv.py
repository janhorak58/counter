from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


def write_csv(path: str | Path, rows: List[Dict[str, Any]], cols: List[str]) -> Path:
    """Write rows to a CSV file using the provided column order."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [",".join(cols)]
    for r in rows:
        lines.append(",".join(str(r.get(c, "")) for c in cols))

    p.write_text("\n".join(lines), encoding="utf-8")
    return p

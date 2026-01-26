from __future__ import annotations

from pathlib import Path
from typing import Any, Dict
import json

from counter.domain.results import CountsResult

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def dump_json(path: str | Path, obj: Dict[str, Any]):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_counts_json(out_path: str | Path, result: CountsResult):
    dump_json(out_path, result.to_dict())

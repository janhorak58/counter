#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

LABELS = ["person", "skis", "bicycle", "dog"]  # target 4 classes -> keys "0".."3"


def _raw_get(raw: Dict[str, Any], idx: int) -> int:
    # raw dict can have string keys ("0") or int keys (0)
    v = raw.get(str(idx), raw.get(idx, 0))
    try:
        return int(v)
    except Exception:
        return 0


def _mapped_counts(meta: Dict[str, Any], direction: str) -> Optional[List[int]]:
    conv = (meta.get("conversion_debug") or {}).get(direction) or {}
    buckets = conv.get("buckets") or {}
    raw = meta.get(f"raw_{direction}_count") or {}

    out: List[int] = []
    for name in LABELS:
        if name not in buckets:
            return None
        out.append(_raw_get(raw, int(buckets[name])))
    return out


def _direct_counts(meta: Dict[str, Any], direction: str) -> List[int]:
    raw = meta.get(f"raw_{direction}_count") or {}
    return [_raw_get(raw, 0), _raw_get(raw, 1), _raw_get(raw, 2), _raw_get(raw, 3)]


def _pick_counts(meta: Dict[str, Any], direction: str) -> Tuple[List[int], str]:
    mapped = _mapped_counts(meta, direction)
    direct = _direct_counts(meta, direction)

    sum_direct = sum(direct)
    if mapped is None:
        return direct, "direct(no_buckets)"

    sum_mapped = sum(mapped)
    # key heuristic: if mapped gives all zeros but direct has signal, mapping is inconsistent -> fallback
    if sum_mapped == 0 and sum_direct > 0:
        return direct, "direct(fallback)"
    return mapped, "mapped"


def _to_count_obj(vals: List[int]) -> Dict[str, int]:
    return {str(i): int(vals[i]) for i in range(4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "pattern",
        nargs="?",
        default="runs/predict/yolo_tuned_*/predict/vid*.counts.json",
        help="Glob pattern of *.counts.json files",
    )
    ap.add_argument("--dry-run", action="store_true", help="Do not write files")
    ap.add_argument("--no-backup", action="store_true", help="Do not create .bak backups")
    args = ap.parse_args()

    paths = [Path(p) for p in glob.glob(args.pattern)]
    paths = [p for p in paths if p.is_file()]

    if not paths:
        print(f"No files matched: {args.pattern}")
        return 0

    changed = 0
    for p in sorted(paths):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"FAIL {p}  (cannot read json: {e})")
            continue

        meta = data.get("meta") or {}
        in_vals, in_mode = _pick_counts(meta, "in")
        out_vals, out_mode = _pick_counts(meta, "out")

        new_in = _to_count_obj(in_vals)
        new_out = _to_count_obj(out_vals)

        old_in = data.get("in_count") or {}
        old_out = data.get("out_count") or {}

        if old_in == new_in and old_out == new_out:
            print(f"SKIP {p}  (already ok)  in={in_mode} out={out_mode}")
            continue

        data["in_count"] = new_in
        data["out_count"] = new_out

        print(
            f"OK   {p}  in={in_mode} (sum={sum(in_vals)})  out={out_mode} (sum={sum(out_vals)})"
        )

        if not args.dry_run:
            if not args.no_backup:
                shutil.copy2(p, p.with_suffix(p.suffix + ".bak"))
            p.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        changed += 1

    print(f"Done. Updated files: {changed} / {len(paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

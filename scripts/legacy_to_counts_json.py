#!/usr/bin/env python3
"""
Legacy predictions CSV -> new counts.json structure.

Handles:
- canonical 0..3 (tuned YOLO)
- legacy8 0..7 where [4..7] are coco-like buckets (YOLO pretrained)
- COCO category_ids (RF-DETR "tuned" in your legacy outputs)
"""

from __future__ import annotations

import argparse
import ast
import csv
import datetime as dt
import json
import platform
import re
import socket
from pathlib import Path
from typing import Dict, Tuple, Optional


VID_RE = re.compile(r"(vid\d+)", re.IGNORECASE)

# Heuristic mapping into your 4 canonical classes:
# 0 tourist, 1 skier, 2 cyclist, 3 tourist_dog
def to_canonical_from_buckets(person: int, bicycle: int, skis: int, dog: int) -> Dict[str, int]:
    tourist = max(0, int(person) - int(bicycle) - int(skis))
    return {
        "0": tourist,         # tourist
        "1": int(skis),       # skier (heuristic skis -> skier)
        "2": int(bicycle),    # cyclist (heuristic bicycle -> cyclist)
        "3": int(dog),        # tourist_dog (heuristic dog -> tourist_dog)
    }


def parse_counts(s: str) -> Dict[int, int]:
    s = (s or "").strip()
    if not s:
        return {}
    # python dict literal
    try:
        obj = ast.literal_eval(s)
        if isinstance(obj, dict):
            return {int(k): int(v) for k, v in obj.items()}
    except Exception:
        pass
    # json dict
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return {int(k): int(v) for k, v in obj.items()}
    except Exception:
        pass
    raise ValueError(f"Cannot parse counts: {s[:120]}...")


def detect_video_id(filename: str) -> str:
    m = VID_RE.search(filename)
    if not m:
        raise ValueError(f"Cannot detect video id from filename: {filename}")
    return m.group(1).lower()


def backend_variant_from_group(group: str) -> Tuple[str, str]:
    g = group.lower()
    backend = "yolo" if "yolo" in g else ("rfdetr" if "rfdetr" in g else "unknown")
    variant = "tuned" if "tuned" in g else ("pretrained" if "pretrained" in g else "unknown")
    return backend, variant


def infer_style(keys: set[int]) -> str:
    """
    - canonical: only 0..3 (your SSOT already)
    - coco: looks like COCO category_id (1,2,18,35...), usually max key > 10
    - legacy8: max key <= 7
    """
    if keys.issubset({0, 1, 2, 3}):
        return "canonical"
    if (keys & {1, 2, 18, 35}) and (max(keys) > 10):
        return "coco"
    if keys and (max(keys) <= 7):
        return "legacy8"
    return "unknown"


def legacy8_mapping_for_group(group: str) -> Dict[str, int]:
    """
    Your legacy outputs:
    - YOLO pretrained uses: 4=person, 5=skis, 6=bicycle, 7=dog  (verified by vid18 having skis)
    - RFDETR pretrained in your files only uses key 6 heavily -> treat 6 as person.
      We'll assume: 6=person, 4=bicycle, 5=skis, 7=dog (best-effort).
    """
    g = group.lower()
    if "yolo_pretrained" in g:
        return {"person": 4, "skis": 5, "bicycle": 6, "dog": 7}
    if "rfdetr_pretrained" in g:
        return {"person": 6, "skis": 5, "bicycle": 4, "dog": 7}
    # fallback
    return {"person": 4, "skis": 5, "bicycle": 6, "dog": 7}


def convert_counts(group: str, raw: Dict[int, int]) -> Tuple[Dict[str, int], dict]:
    keys = set(raw.keys())
    style = infer_style(keys)

    debug = {"style": style, "raw_keys": sorted(keys)}

    if style == "canonical":
        # ensure all present
        out = {str(k): int(raw.get(k, 0)) for k in [0, 1, 2, 3]}
        return out, debug

    if style == "coco":
        person = raw.get(1, 0)
        bicycle = raw.get(2, 0)
        dog = raw.get(18, 0)
        skis = raw.get(35, 0)
        debug["buckets"] = {"person": 1, "bicycle": 2, "dog": 18, "skis": 35}
        return to_canonical_from_buckets(person, bicycle, skis, dog), debug

    if style == "legacy8":
        m = legacy8_mapping_for_group(group)
        person = raw.get(m["person"], 0)
        bicycle = raw.get(m["bicycle"], 0)
        skis = raw.get(m["skis"], 0)
        dog = raw.get(m["dog"], 0)
        debug["buckets"] = m
        return to_canonical_from_buckets(person, bicycle, skis, dog), debug

    # Unknown: return zeros but keep debug
    return {"0": 0, "1": 0, "2": 0, "3": 0}, debug


def read_legacy_csv(path: Path) -> Tuple[str, Dict[int, int], Dict[int, int]]:
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {path}")
    row0 = rows[0]
    line = (row0.get("line_name") or "Line_1").strip()
    in_d = parse_counts(row0.get("in_count", ""))
    out_d = parse_counts(row0.get("out_count", ""))
    return line, in_d, out_d


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions-dir", type=str, default="predictions/csv")
    ap.add_argument("--out-root", type=str, default="runs/legacy_import")
    ap.add_argument("--force-line", type=str, default=None)
    args = ap.parse_args()

    pred_root = Path(args.predictions_dir)
    out_root = Path(args.out_root)

    if not pred_root.exists():
        print(f"ERROR: {pred_root} does not exist")
        return 2

    created_at = dt.datetime.now().replace(microsecond=0).isoformat()
    hostname = socket.gethostname()

    csv_files = sorted(pred_root.glob("*/*/*.csv"))
    if not csv_files:
        print(f"ERROR: No CSV files found under {pred_root}")
        return 3

    # group by model_id = "<group>/<model>"
    grouped: Dict[str, list[Path]] = {}
    for p in csv_files:
        rel = p.relative_to(pred_root)
        if len(rel.parts) < 3:
            continue
        group = rel.parts[0]
        model = rel.parts[1]
        model_id = f"{group}/{model}"
        grouped.setdefault(model_id, []).append(p)

    for model_id, files in grouped.items():
        group, model = model_id.split("/", 1)
        backend, variant = backend_variant_from_group(group)

        agg_in = {"0": 0, "1": 0, "2": 0, "3": 0}
        agg_out = {"0": 0, "1": 0, "2": 0, "3": 0}

        run_dir = out_root / model_id.replace("/", "__") / "predict"
        print(f"\n== {model_id} -> {run_dir}")

        for f in sorted(files):
            vid = detect_video_id(f.name)
            line_name, in_raw, out_raw = read_legacy_csv(f)
            if args.force_line:
                line_name = args.force_line

            in_can, dbg_in = convert_counts(group, in_raw)
            out_can, dbg_out = convert_counts(group, out_raw)

            for k in agg_in:
                agg_in[k] += in_can.get(k, 0)
                agg_out[k] += out_can.get(k, 0)

            payload = {
                "video": f"{vid}.mp4",
                "line_name": line_name,
                "in_count": in_can,
                "out_count": out_can,
                "meta": {
                    "run_id": f"legacy_import::{model_id}",
                    "created_at": created_at,
                    "model_id": model_id,
                    "backend": backend,
                    "variant": variant,
                    "weights": None,
                    "mapping_policy": "legacy_import_heuristic",
                    "source_csv": str(f.as_posix()),
                    "raw_in_count": in_raw,
                    "raw_out_count": out_raw,
                    "conversion_debug": {"in": dbg_in, "out": dbg_out},
                    "environment": {
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                        "hostname": hostname,
                    },
                },
            }

            out_path = run_dir / f"{vid}.counts.json"
            write_json(out_path, payload)
            print(f"  wrote {out_path.name}  (style in={dbg_in['style']} out={dbg_out['style']})")

        agg_payload = {
            "video": "__aggregate__",
            "line_name": args.force_line or "Line_1",
            "in_count": agg_in,
            "out_count": agg_out,
            "meta": {
                "run_id": f"legacy_import::{model_id}",
                "created_at": created_at,
                "model_id": model_id,
                "backend": backend,
                "variant": variant,
                "weights": None,
                "mapping_policy": "legacy_import_heuristic",
                "aggregate_of": [f"{detect_video_id(p.name)}.mp4" for p in sorted(files)],
                "environment": {
                    "python": platform.python_version(),
                    "platform": platform.platform(),
                    "hostname": hostname,
                },
            },
        }
        write_json(run_dir / "aggregate.counts.json", agg_payload)
        print(f"  wrote aggregate.counts.json")

    print("\nDONE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

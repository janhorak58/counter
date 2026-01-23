from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

from src.eval.evaluator import run_evaluation


def _parse_class_ids(value: str) -> list[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate people counting results.")
    parser.add_argument("--gt-dir", default="data/results/gt")
    parser.add_argument("--pred-dir", default="data/results/predicted")
    parser.add_argument("--output-root", default="eval")
    parser.add_argument("--class-ids", default="0,1,2,3")
    args = parser.parse_args(list(argv) if argv is not None else None)

    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    output_root = Path(args.output_root)
    class_ids = _parse_class_ids(args.class_ids)

    run_evaluation(gt_dir, pred_dir, output_root, class_ids)
    return 0

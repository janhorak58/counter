from __future__ import annotations

import argparse
from pathlib import Path

from counter.core.config import load_eval_config
from counter.eval.pipeline import EvalPipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m counter.eval")
    p.add_argument(
        "--config",
        default="configs/eval.yaml",
        help="Path to eval YAML config (default: configs/eval.yaml)",
    )
    p.add_argument(
        "--predict_run_dir",
        default=None,
        help="Optional: evaluate only one run dir (runs/<run_id>) or a predict dir (runs/<run_id>/predict)",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_eval_config(args.config)
    out = EvalPipeline().run(cfg, predict_run_dir=args.predict_run_dir)
    print(str(out))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse

from counter.core.config import load_predict_config
from counter.predict.pipeline import PredictPipeline


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python -m counter.predict")
    p.add_argument(
        "--config",
        default="configs/predict.yaml",
        help="Path to predict YAML config (default: configs/predict.yaml)",
    )
    p.add_argument(
        "--models",
        default="configs/models.yaml",
        help="Path to models registry YAML (default: configs/models.yaml)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging + draw raw ids/names into overlay.",
    )
    p.add_argument(
        "--probe_frames",
        type=int,
        default=0,
        help="If >0, process only first N frames per video (useful to inspect class IDs quickly).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = load_predict_config(args.config)
    print(cfg)
    if args.debug:
        cfg.debug = True
        
    if args.probe_frames is not None:
        cfg.probe_frames = int(args.probe_frames)

    print(cfg.debug)
    out = PredictPipeline(models_yaml=args.models, debug=cfg.debug).run(cfg)
    print(str(out))


if __name__ == "__main__":
    main()

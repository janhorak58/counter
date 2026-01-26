from __future__ import annotations

import argparse
from pathlib import Path

from counter.config.loader import load_eval_config, load_predict_config
from counter.pipelines.eval import EvalPipeline
from counter.pipelines.predict import PredictPipeline


def cmd_predict(args) -> int:
    cfg = load_predict_config(args.config)

    if args.model_id:
        cfg.model_id = args.model_id

    if args.video:
        p = Path(args.video)
        if p.exists() and p.is_absolute():
            cfg.videos_dir = str(p.parent)
            cfg.videos = [p.name]
        elif p.exists():
            cfg.videos_dir = str(p.parent)
            cfg.videos = [p.name]
        else:
            cfg.videos = [args.video]

    if args.videos_dir:
        cfg.videos_dir = args.videos_dir

    if args.out_dir:
        cfg.export.out_dir = args.out_dir

    if args.save_video:
        cfg.export.save_video = True

    out_dir = PredictPipeline(models_yaml=args.models).run(cfg)
    print(out_dir)
    return 0


def cmd_eval(args) -> int:
    cfg = load_eval_config(args.config)
    out = EvalPipeline().run(cfg)
    print(out)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="counter")
    p.add_argument("--models", default="configs/models.yaml", help="Path to models registry YAML")

    sub = p.add_subparsers(dest="cmd", required=True)

    p_pred = sub.add_parser("predict", help="Run prediction")
    p_pred.add_argument("--config", required=True, help="Path to predict.yaml")
    p_pred.add_argument("--model-id", help="Override model_id from config (e.g. rfdetr_medium_tuned)")
    p_pred.add_argument("--video", help="Override videos list with a single video (path or filename)")
    p_pred.add_argument("--videos-dir", help="Override videos_dir from config")
    p_pred.add_argument("--out-dir", help="Override export.out_dir from config")
    p_pred.add_argument("--save-video", action="store_true", help="Force export.save_video=true")
    p_pred.set_defaults(func=cmd_predict)

    p_eval = sub.add_parser("eval", help="Evaluate predict runs vs GT")
    p_eval.add_argument("--config", required=True, help="Path to eval.yaml")
    p_eval.set_defaults(func=cmd_eval)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

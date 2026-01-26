from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from counter.config.loader import load_eval_config, load_models, load_predict_config
from counter.pipelines.predict import PredictPipeline
from counter.pipelines.eval import EvalPipeline


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_predict_config(args.config)

    # --- overrides, aby ses nemusel hrabat v YAML ---
    if args.model_id:
        cfg.model_id = args.model_id
    if args.videos_dir:
        cfg.videos_dir = args.videos_dir
    if args.device:
        cfg.device = args.device
    if args.out_dir:
        cfg.export.out_dir = args.out_dir
    if args.save_video is not None:
        cfg.export.save_video = bool(args.save_video)

    if args.conf is not None:
        cfg.thresholds.conf = float(args.conf)
    if args.iou is not None:
        cfg.thresholds.iou = float(args.iou)

    # video list: buď explicitně --video, nebo nech YAML
    if args.video:
        cfg.videos = list(args.video)

    run_id = PredictPipeline(models_yaml=args.models).run(cfg)
    print(run_id)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_eval_config(args.config)
    out = EvalPipeline().run(cfg, predict_run_dir=args.predict_run_dir)
    print(str(out))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    reg = load_models(args.models)
    for k, v in reg.models.items():
        w = v.weights or ""
        size = v.rfdetr_size or ""
        print(f"{k:26s} backend={v.backend:5s} variant={v.variant:10s} rfdetr_size={size:6s} weights={w}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="counter")
    p.add_argument("--models", default="configs/models.yaml", help="models registry YAML")

    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("models", help="list models from registry")
    sp.set_defaults(func=cmd_models)

    sp = sub.add_parser("predict", help="run prediction")
    sp.add_argument("--config", required=True, help="predict.yaml")
    sp.add_argument("--model-id", default=None, help="override model_id from YAML")
    sp.add_argument("--videos-dir", default=None, help="override videos_dir from YAML")
    sp.add_argument("--video", action="append", help="one video filename (repeatable). Example: --video vid19.mp4")
    sp.add_argument("--device", default=None, help="cpu | cuda:0 | ...")
    sp.add_argument("--out-dir", default=None, help="runs root (default from YAML)")
    sp.add_argument("--save-video", dest="save_video", action="store_true", help="save mp4 with boxes")
    sp.add_argument("--no-save-video", dest="save_video", action="store_false", help="disable saving video")
    sp.set_defaults(save_video=None)
    sp.add_argument("--conf", type=float, default=None, help="override conf threshold")
    sp.add_argument("--iou", type=float, default=None, help="override iou threshold")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("eval", help="run evaluation")
    sp.add_argument("--config", required=True, help="eval.yaml")
    sp.add_argument("--predict-run-dir", default=None, help="optional single run dir or predict/ dir")
    sp.set_defaults(func=cmd_eval)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

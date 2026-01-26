from __future__ import annotations

import argparse
from typing import List, Optional

from counter.config.loader import load_predict_config, load_eval_config
from counter.domain.model_spec import load_models
from counter.pipelines.eval import EvalPipeline
from counter.pipelines.predict import PredictPipeline


def _comma_list(v: Optional[str]) -> List[str]:
    if not v:
        return []
    return [x.strip() for x in v.split(",") if x.strip()]


def cmd_models(args: argparse.Namespace) -> int:
    models = load_models(args.models)
    for mid, spec in models.items():
        print(
            f"{mid:28s} backend={spec.backend:6s} variant={spec.variant:10s} "
            f"rfdetr_size={spec.rfdetr_size or ''} weights={spec.weights or ''}"
        )
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_predict_config(args.config)

    # overrides
    if args.model_id:
        cfg.model_id = args.model_id
    if args.videos_dir:
        cfg.videos_dir = args.videos_dir
    if args.video:
        cfg.videos = [args.video]
    if args.videos:
        cfg.videos = _comma_list(args.videos)
    if args.out_dir:
        cfg.export.out_dir = args.out_dir
    if args.device:
        cfg.device = args.device
    if args.conf is not None:
        cfg.thresholds.conf = float(args.conf)
    if args.iou is not None:
        cfg.thresholds.iou = float(args.iou)
    if args.save_video is not None:
        cfg.export.save_video = bool(args.save_video)

    run_id = PredictPipeline(models_yaml=args.models).run(cfg)
    print(f"OK: {run_id}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_eval_config(args.config)

    if args.runs_dir:
        cfg.runs_dir = args.runs_dir
    if args.gt_dir:
        cfg.gt_dir = args.gt_dir
    if args.out_dir:
        cfg.out_dir = args.out_dir
    if args.rank_by:
        cfg.rank_by = args.rank_by
    if args.only_completed is not None:
        cfg.only_completed = bool(args.only_completed)

    out = EvalPipeline().run(cfg, predict_run_dir=args.predict_run_dir)
    print(f"OK: {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="counter")
    p.add_argument("--models", default="configs/models.yaml", help="Path to models registry YAML.")
    sub = p.add_subparsers(dest="cmd", required=True)

    spm = sub.add_parser("models", help="List available model IDs from models.yaml")
    spm.set_defaults(func=cmd_models)

    spp = sub.add_parser("predict", help="Run predictions")
    spp.add_argument("--config", default="configs/predict.yaml", help="Predict YAML config.")
    spp.add_argument("--model-id", dest="model_id", help="Override model_id from config.")
    spp.add_argument("--videos-dir", help="Override videos_dir from config.")
    spp.add_argument("--video", help="Run a single video (filename in videos_dir).")
    spp.add_argument("--videos", help="Comma-separated list of videos (filenames).")
    spp.add_argument("--out-dir", help="Override export.out_dir.")
    spp.add_argument("--device", help="cpu / cuda:0 etc.")
    spp.add_argument("--conf", type=float, help="Override detection conf threshold.")
    spp.add_argument("--iou", type=float, help="Override IoU threshold.")
    spp.add_argument("--save-video", dest="save_video", action="store_true", help="Save annotated video output(s).")
    spp.add_argument("--no-save-video", dest="save_video", action="store_false", help="Do not save video.")
    spp.set_defaults(save_video=None)
    spp.set_defaults(func=cmd_predict)

    spe = sub.add_parser("eval", help="Evaluate predict runs against GT")
    spe.add_argument("--config", default="configs/eval.yaml", help="Eval YAML config.")
    spe.add_argument("--predict-run-dir", default=None, help="Evaluate a single run dir or dir with runs.")
    spe.add_argument("--runs-dir", help="Override cfg.runs_dir.")
    spe.add_argument("--gt-dir", help="Override cfg.gt_dir.")
    spe.add_argument("--out-dir", help="Override cfg.out_dir.")
    spe.add_argument("--rank-by", help="Override cfg.rank_by.")
    spe.add_argument("--only-completed", dest="only_completed", action="store_true")
    spe.add_argument("--include-incomplete", dest="only_completed", action="store_false")
    spe.set_defaults(only_completed=None)
    spe.set_defaults(func=cmd_eval)

    return p


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse

from counter.config.loader import load_predict_config, load_eval_config
from counter.pipelines.predict import PredictPipeline
from counter.pipelines.eval import EvalPipeline


def cmd_predict(args: argparse.Namespace) -> int:
    cfg = load_predict_config(args.config)
    pred = PredictPipeline(models_yaml=args.models)
    out_dir = pred.run(cfg)
    print(out_dir)
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = load_eval_config(args.config)
    ev = EvalPipeline()
    out = ev.run(cfg, predict_run_dir=args.predict_run_dir)
    print(out)
    return 0




def main() -> int:
    p = argparse.ArgumentParser(prog='counter')
    p.add_argument('--models', default='configs/models.yaml')
    sub = p.add_subparsers(dest='cmd', required=True)

    sp = sub.add_parser('predict')
    sp.add_argument('--config', default='configs/predict.yaml')
    sp.set_defaults(func=cmd_predict)

    se = sub.add_parser('eval')
    se.add_argument('--config', default='configs/eval.yaml')
    se.add_argument('--predict-run-dir', default=None, help='Optional: path to runs/<id>/predict directory')
    se.set_defaults(func=cmd_eval)

    args = p.parse_args()
    return int(args.func(args))


if __name__ == '__main__':
    raise SystemExit(main())

import argparse
from typing import Iterable, Optional

from src.eval_model.config import load_eval_model_config
from src.eval_model.evaluator import run_model_evaluation


def _apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    overrides = {
        "model_type": args.model_type,
        "model_path": args.model_path,
        "data_yaml": args.data_yaml,
        "dataset_dir": args.dataset_dir,
        "device": args.device,
        "conf": args.conf,
        "iou": args.iou,
        "imgsz": args.imgsz,
        "split": args.split,
        "output_dir": args.output_dir,
        "run_name": args.run_name,
        "rfdetr_box_format": args.rfdetr_box_format,
        "rfdetr_box_normalized": args.rfdetr_box_normalized,
        "visuals": args.visuals,
        "visuals_samples": args.visuals_samples,
        "visuals_iou": args.visuals_iou,
        "coco_eval": args.coco_eval,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    return cfg


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate detection models (YOLO or RF-DETR).")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model-type", choices=["yolo", "rfdetr"])
    parser.add_argument("--model-path")
    parser.add_argument("--data-yaml", help="YOLO data.yaml path (dataset_yolo).")
    parser.add_argument("--dataset-dir", help="COCO dataset directory (dataset_coco).")
    parser.add_argument("--device", default=None)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--rfdetr-box-format", default=None)
    parser.add_argument("--rfdetr-box-normalized", default=None)
    parser.add_argument("--visuals", action="store_true", default=None)
    parser.add_argument("--no-visuals", action="store_false", dest="visuals")
    parser.add_argument("--visuals-samples", type=int, default=None)
    parser.add_argument("--visuals-iou", type=float, default=None)
    parser.add_argument("--coco-eval", action="store_true", default=None)
    parser.add_argument("--no-coco-eval", action="store_false", dest="coco_eval")
    args = parser.parse_args(list(argv) if argv is not None else None)

    cfg = load_eval_model_config(args.config)
    cfg = _apply_overrides(cfg, args)
    run_model_evaluation(cfg)
    return 0

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

from src.models import rfdetr as rfdetr_module
from src.eval_model.coco_eval import evaluate_rfdetr_coco
from src.eval_model.visuals import generate_rfdetr_visuals


@dataclass
class EvalRun:
    output_dir: Path
    model_type: str
    model_path: str
    dataset: str
    split: str


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _normalize_device(device: str) -> str:
    value = str(device).strip().lower()
    if value in {"cpu"}:
        return "cpu"
    if value in {"gpu", "cuda"}:
        return "cuda:0"
    if value.isdigit():
        return f"cuda:{value}"
    return device


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _coerce_metrics(result: Any) -> Dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    if hasattr(result, "results_dict"):
        try:
            return dict(result.results_dict)
        except Exception:
            pass
    if hasattr(result, "metrics"):
        metrics = getattr(result, "metrics")
        if isinstance(metrics, dict):
            return metrics
    if hasattr(result, "__dict__"):
        raw = {
            key: value
            for key, value in result.__dict__.items()
            if isinstance(value, (int, float, str, bool))
        }
        if raw:
            return raw
    return {}


def _extract_yolo_metrics(result: Any) -> Dict[str, Any]:
    metrics = _coerce_metrics(result)
    if metrics:
        return metrics

    box = getattr(result, "box", None)
    if box is not None:
        output: Dict[str, Any] = {}
        for name in ("map", "map50", "map75", "mp", "mr"):
            value = getattr(box, name, None)
            if value is not None:
                output[name] = float(value)
        if output:
            return output
    return {}


def _resolve_yolo_data_yaml(cfg: Dict[str, Any]) -> str:
    data_yaml = str(cfg.get("data_yaml") or "").strip()
    if data_yaml:
        return data_yaml
    return "data/dataset_yolo/data.yaml"


def _resolve_rfdetr_dataset(cfg: Dict[str, Any]) -> Tuple[str, str]:
    dataset_dir = str(cfg.get("dataset_dir") or "data/dataset_coco")
    split = str(cfg.get("split") or "valid")
    if split == "val":
        split = "valid"
    return dataset_dir, split


def _select_eval_output_dir(cfg: Dict[str, Any], model_type: str) -> Path:
    base = Path(cfg.get("output_dir") or "eval/model")
    run_name = cfg.get("run_name")
    if run_name:
        return _ensure_dir(base / str(run_name))
    return _ensure_dir(base / f"{model_type}_{_timestamp()}")


def _save_run_metadata(run: EvalRun) -> None:
    payload = asdict(run)
    _write_json(run.output_dir / "run.json", payload)


def _save_metrics(output_dir: Path, metrics: Dict[str, Any], filename: str) -> None:
    _write_json(output_dir / filename, metrics)


def _run_yolo_eval(cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    from ultralytics import YOLO

    model_path = str(cfg.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("Missing model_path for YOLO evaluation.")

    data_yaml = _resolve_yolo_data_yaml(cfg)
    if not Path(data_yaml).exists():
        raise FileNotFoundError(f"YOLO data.yaml not found: {data_yaml}")
    split = str(cfg.get("split") or "val")
    conf = float(cfg.get("conf", 0.25))
    iou = float(cfg.get("iou", 0.6))
    imgsz = int(cfg.get("imgsz", 640))
    device = _normalize_device(cfg.get("device", "cpu"))

    model = YOLO(model_path)
    result = model.val(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        project=str(output_dir),
        name="yolo_val",
        verbose=True,
    )
    return _extract_yolo_metrics(result)


def _call_with_variants(func: Any, variants: Iterable[Dict[str, Any]]) -> Any:
    last_error: Optional[Exception] = None
    for kwargs in variants:
        try:
            return func(**kwargs)
        except TypeError as exc:
            last_error = exc
    if last_error:
        raise last_error
    return None


def _run_rfdetr_eval(cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    model_path = str(cfg.get("model_path") or "").strip()
    if not model_path:
        raise ValueError("Missing model_path for RF-DETR evaluation.")

    dataset_dir, split = _resolve_rfdetr_dataset(cfg)
    if not Path(dataset_dir).exists():
        raise FileNotFoundError(f"RF-DETR dataset directory not found: {dataset_dir}")
    conf = float(cfg.get("conf", 0.25))
    iou = float(cfg.get("iou", 0.6))
    device = _normalize_device(cfg.get("device", "cpu"))

    model = rfdetr_module.load_rfdetr_model(model_path, device=device)
    methods = ["evaluate", "val", "validate", "test"]

    variants = [
        {
            "dataset_dir": str(dataset_dir),
            "split": split,
            "conf": conf,
            "iou": iou,
            "device": device,
            "output_dir": str(output_dir),
        },
        {
            "dataset_dir": str(dataset_dir),
            "split": split,
            "device": device,
            "output_dir": str(output_dir),
        },
        {
            "dataset_dir": str(dataset_dir),
            "split": split,
        },
        {
            "dataset_dir": str(dataset_dir),
        },
    ]

    result = None
    for name in methods:
        if hasattr(model, name):
            result = _call_with_variants(getattr(model, name), variants)
            break

    if result is None:
        module = rfdetr_module._import_rfdetr_module()
        for name in methods:
            if hasattr(module, name):
                result = _call_with_variants(getattr(module, name), variants)
                break

    return _coerce_metrics(result)


def run_model_evaluation(cfg: Dict[str, Any]) -> EvalRun:
    model_type = str(cfg.get("model_type", "yolo")).lower()
    output_dir = _select_eval_output_dir(cfg, model_type)

    if model_type == "yolo":
        data_yaml = _resolve_yolo_data_yaml(cfg)
        split = str(cfg.get("split") or "val")
        run = EvalRun(
            output_dir=output_dir,
            model_type=model_type,
            model_path=str(cfg.get("model_path") or ""),
            dataset=data_yaml,
            split=split,
        )
        _save_run_metadata(run)
        metrics = _run_yolo_eval(cfg, output_dir)
        _save_metrics(output_dir, metrics, "metrics.json")
        return run

    if model_type == "rfdetr":
        dataset_dir, split = _resolve_rfdetr_dataset(cfg)
        run = EvalRun(
            output_dir=output_dir,
            model_type=model_type,
            model_path=str(cfg.get("model_path") or ""),
            dataset=dataset_dir,
            split=split,
        )
        _save_run_metadata(run)
        metrics = _run_rfdetr_eval(cfg, output_dir)
        _save_metrics(output_dir, metrics, "metrics.json")
        if bool(cfg.get("coco_eval", True)):
            coco_metrics = evaluate_rfdetr_coco(
                model_path=str(cfg.get("model_path") or ""),
                dataset_dir=Path(dataset_dir),
                split=split,
                output_dir=output_dir,
                conf=float(cfg.get("conf", 0.25)),
                iou=float(cfg.get("iou", 0.6)),
                device=_normalize_device(cfg.get("device", "cpu")),
                box_format=str(cfg.get("rfdetr_box_format", "xyxy")),
                box_normalized=str(cfg.get("rfdetr_box_normalized", "auto")),
            )
            _save_metrics(output_dir, coco_metrics, "coco_metrics.json")
        if bool(cfg.get("visuals", True)):
            generate_rfdetr_visuals(
                model_path=str(cfg.get("model_path") or ""),
                dataset_dir=Path(dataset_dir),
                split=split,
                output_dir=output_dir,
                conf=float(cfg.get("conf", 0.25)),
                iou_thresh=float(cfg.get("visuals_iou", 0.5)),
                device=_normalize_device(cfg.get("device", "cpu")),
                box_format=str(cfg.get("rfdetr_box_format", "xyxy")),
                box_normalized=str(cfg.get("rfdetr_box_normalized", "auto")),
                max_samples=int(cfg.get("visuals_samples", 8)),
            )
        return run

    raise ValueError(f"Unsupported model_type: {model_type}")

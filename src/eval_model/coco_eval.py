from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from src.models import rfdetr as rfdetr_module


def _import_pycocotools() -> Any:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:
        raise RuntimeError(
            "pycocotools is required for COCO evaluation. "
            "Install it with: pip install pycocotools"
        ) from exc
    return COCO, COCOeval


def _xyxy_to_xywh(box: np.ndarray) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _per_class_ap(coco_eval: Any) -> Dict[int, Dict[str, float]]:
    precision = coco_eval.eval.get("precision")
    if precision is None:
        return {}

    params = coco_eval.params
    area_idx = params.areaRngLbl.index("all")
    maxdet_idx = len(params.maxDets) - 1
    iou_thrs = params.iouThrs

    def _mean(vals: np.ndarray) -> float:
        vals = vals[vals > -1]
        return float(vals.mean()) if vals.size else float("nan")

    def _iou_index(target: float) -> int:
        return int(np.argmin(np.abs(iou_thrs - target)))

    idx_50 = _iou_index(0.5)
    idx_75 = _iou_index(0.75)

    per_class: Dict[int, Dict[str, float]] = {}
    for k, cat_id in enumerate(coco_eval.params.catIds):
        p_all = precision[:, :, k, area_idx, maxdet_idx]
        p_50 = precision[idx_50, :, k, area_idx, maxdet_idx]
        p_75 = precision[idx_75, :, k, area_idx, maxdet_idx]
        per_class[int(cat_id)] = {
            "ap": _mean(p_all),
            "ap50": _mean(p_50),
            "ap75": _mean(p_75),
        }
    return per_class


def evaluate_rfdetr_coco(
    model_path: str,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    conf: float,
    iou: float,
    device: str,
    box_format: str,
    box_normalized: str,
) -> Dict[str, Any]:
    COCO, COCOeval = _import_pycocotools()

    ann_path = dataset_dir / split / "_annotations.coco.json"
    coco_gt = COCO(str(ann_path))
    image_ids = coco_gt.getImgIds()
    category_ids = set(coco_gt.getCatIds())

    model = rfdetr_module.load_rfdetr_model(model_path, device=device)
    if hasattr(model, "optimize_for_inference"):
        model.optimize_for_inference()
    detections: List[Dict[str, Any]] = []

    for image_id in image_ids:
        info = coco_gt.loadImgs([image_id])[0]
        img_path = dataset_dir / split / info["file_name"]
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        preds = rfdetr_module.predict_rfdetr(
            model,
            image,
            conf=conf,
            iou=iou,
            box_format=box_format,
            box_normalized=box_normalized,
        )
        for box, label, score in preds:
            cat_id = int(label)
            if cat_id not in category_ids:
                continue
            detections.append(
                {
                    "image_id": int(image_id),
                    "category_id": cat_id,
                    "bbox": _xyxy_to_xywh(box),
                    "score": float(score),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    det_path = output_dir / "coco_detections.json"
    det_path.write_text(json.dumps(detections, indent=2), encoding="utf-8")

    coco_dt = coco_gt.loadRes(str(det_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = [
        "ap",
        "ap50",
        "ap75",
        "ap_small",
        "ap_medium",
        "ap_large",
        "ar_1",
        "ar_10",
        "ar_100",
        "ar_small",
        "ar_medium",
        "ar_large",
    ]
    stats = {name: float(val) for name, val in zip(stats_names, coco_eval.stats)}

    per_class = _per_class_ap(coco_eval)
    class_names = {cat["id"]: cat["name"] for cat in coco_gt.loadCats(coco_gt.getCatIds())}
    stats["per_class"] = {
        str(cat_id): {
            "name": class_names.get(cat_id, str(cat_id)),
            **values,
        }
        for cat_id, values in per_class.items()
    }
    return stats

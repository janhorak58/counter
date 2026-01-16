import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.eval.det_eval_utils import (
    confusion_update,
    label_path_from_image,
    list_images,
    load_labels,
    match_predictions,
    resolve_path,
)
from src.eval.utils import CLASS_NAME_MAP, coco_to_dataset_map, dataset_class_ids, load_data_yaml
from src.models.rfdetr import load_rfdetr_model, predict_rfdetr


def _prepare_labels_root(val_path: str, base: str) -> str:
    labels_root = val_path.replace("images", "labels")
    if not os.path.isdir(labels_root):
        labels_root = resolve_path(base, "labels/val")
    return labels_root


def evaluate_rfdetr_mapped(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.5,
    class_map: Dict[int, int] = None,
    box_format: str = "xyxy",
    box_normalized: str = "auto",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    data = load_data_yaml(data_yaml)
    base = data.get("path", "")
    val_path = resolve_path(base, data.get("val", ""))
    images = list_images(val_path)
    if not images:
        return pd.DataFrame(), np.zeros((0, 0), dtype=int), []

    labels_root = _prepare_labels_root(val_path, base)
    if class_map is None:
        class_map = coco_to_dataset_map(data_yaml)

    class_ids = sorted(set(class_map.values()))
    model = load_rfdetr_model(model_path, device=device)

    aggregate = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}
    cm = np.zeros((len(class_ids) + 1, len(class_ids) + 1), dtype=int)

    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_labels = load_labels(label_path_from_image(image_path, labels_root), w, h)

        preds: List[Tuple[int, np.ndarray, float]] = []
        detections = predict_rfdetr(
            model,
            img,
            conf=conf,
            iou=iou,
            box_format=box_format,
            box_normalized=box_normalized,
        )
        for box, cls_id, conf_val in detections:
            if cls_id not in class_map:
                continue
            preds.append((class_map[cls_id], box.astype(np.float32), float(conf_val)))

        stats = match_predictions(preds, gt_labels, iou, class_ids)
        for cid in class_ids:
            aggregate[cid]["tp"] += stats[cid]["tp"]
            aggregate[cid]["fp"] += stats[cid]["fp"]
            aggregate[cid]["fn"] += stats[cid]["fn"]
        confusion_update(preds, gt_labels, iou, class_ids, cm)

    rows = []
    total_tp = total_fp = total_fn = 0
    for cid in class_ids:
        tp = aggregate[cid]["tp"]
        fp = aggregate[cid]["fp"]
        fn = aggregate[cid]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn
        precision = tp / (tp + fp + 1e-9)
        recall = tp / (tp + fn + 1e-9)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        rows.append(
            {
                "class_id": cid,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    precision = total_tp / (total_tp + total_fp + 1e-9)
    recall = total_tp / (total_tp + total_fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    rows.append(
        {
            "class_id": "all",
            "tp": total_tp,
            "fp": total_fp,
            "fn": total_fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )

    labels = [CLASS_NAME_MAP.get(cid, str(cid)) for cid in class_ids] + ["bg"]
    return pd.DataFrame(rows), cm, labels


def evaluate_rfdetr_custom(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.5,
    box_format: str = "xyxy",
    box_normalized: str = "auto",
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    class_map = dataset_class_ids(data_yaml)
    return evaluate_rfdetr_mapped(
        model_path=model_path,
        data_yaml=data_yaml,
        device=device,
        conf=conf,
        iou=iou,
        class_map=class_map,
        box_format=box_format,
        box_normalized=box_normalized,
    )

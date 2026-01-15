import os
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd

from src.eval.utils import CLASS_NAME_MAP, coco_to_dataset_map, dataset_class_ids, load_data_yaml


def evaluate_yolo_standard(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.6,
    split: str = "val",
    project: str = None,
    name: str = None,
) -> Dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(model_path)
    val_kwargs = dict(
        data=data_yaml,
        split=split,
        conf=conf,
        iou=iou,
        device=device,
        verbose=False,
    )
    if project:
        val_kwargs["project"] = project
    if name:
        val_kwargs["name"] = name
    metrics = model.val(**val_kwargs)
    return {
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
    }


def _resolve_path(base: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base, path_value))


def _list_images(val_path: str) -> List[str]:
    if os.path.isfile(val_path) and val_path.lower().endswith(".txt"):
        with open(val_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    if os.path.isdir(val_path):
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        return [
            os.path.join(val_path, f)
            for f in os.listdir(val_path)
            if f.lower().endswith(exts)
        ]
    return []


def _label_path_from_image(image_path: str, labels_root: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    return os.path.join(labels_root, base)


def _xywhn_to_xyxy(x, y, w, h, img_w, img_h) -> Tuple[float, float, float, float]:
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return x1, y1, x2, y2


def _load_labels(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
    if not os.path.exists(label_path):
        return []
    labels = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                continue
            class_id = int(float(parts[0]))
            x, y, w, h = map(float, parts[1:])
            box = np.array(_xywhn_to_xyxy(x, y, w, h, img_w, img_h), dtype=np.float32)
            labels.append((class_id, box))
    return labels


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.array([])
    xA = np.maximum(box[0], boxes[:, 0])
    yA = np.maximum(box[1], boxes[:, 1])
    xB = np.minimum(box[2], boxes[:, 2])
    yB = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = box_area + boxes_area - inter
    return inter / (union + 1e-9)


def _match_predictions(
    preds: List[Tuple[int, np.ndarray, float]],
    gts: List[Tuple[int, np.ndarray]],
    iou_thresh: float,
    class_ids: Iterable[int],
) -> Dict[int, Dict[str, int]]:
    stats: Dict[int, Dict[str, int]] = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}
    for class_id in class_ids:
        gt_boxes = [b for cid, b in gts if cid == class_id]
        pred_boxes = [(b, c) for cid, b, c in preds if cid == class_id]

        if not gt_boxes and not pred_boxes:
            continue

        gt_matched = [False] * len(gt_boxes)
        pred_sorted = sorted(pred_boxes, key=lambda x: x[1], reverse=True)

        for pred_box, _conf in pred_sorted:
            if not gt_boxes:
                stats[class_id]["fp"] += 1
                continue
            ious = _iou(pred_box, np.array(gt_boxes, dtype=np.float32))
            best_idx = int(np.argmax(ious)) if ious.size else -1
            if best_idx >= 0 and ious[best_idx] >= iou_thresh and not gt_matched[best_idx]:
                stats[class_id]["tp"] += 1
                gt_matched[best_idx] = True
            else:
                stats[class_id]["fp"] += 1

        stats[class_id]["fn"] += sum(1 for m in gt_matched if not m)
    return stats


def _confusion_update(
    preds: List[Tuple[int, np.ndarray, float]],
    gts: List[Tuple[int, np.ndarray]],
    iou_thresh: float,
    class_ids: List[int],
    cm: np.ndarray,
) -> None:
    index = {cid: i for i, cid in enumerate(class_ids)}
    bg_idx = len(class_ids)
    matched_gt = set()
    matched_pred = set()

    preds_sorted = sorted(enumerate(preds), key=lambda x: x[1][2], reverse=True)
    for p_idx, (p_class, p_box, _conf) in preds_sorted:
        if not gts:
            cm[bg_idx, index[p_class]] += 1
            continue
        gt_boxes = np.array([b for _, b in gts], dtype=np.float32)
        ious = _iou(p_box, gt_boxes)
        best_idx = int(np.argmax(ious)) if ious.size else -1
        if best_idx >= 0 and ious[best_idx] >= iou_thresh and best_idx not in matched_gt:
            g_class = gts[best_idx][0]
            cm[index[g_class], index[p_class]] += 1
            matched_gt.add(best_idx)
            matched_pred.add(p_idx)
        else:
            cm[bg_idx, index[p_class]] += 1

    for g_idx, (g_class, _box) in enumerate(gts):
        if g_idx not in matched_gt:
            cm[index[g_class], bg_idx] += 1


def evaluate_yolo_mapped(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.5,
    class_map: Dict[int, int] = None,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    from ultralytics import YOLO

    data = load_data_yaml(data_yaml)
    base = data.get("path", "")
    val_path = _resolve_path(base, data.get("val", ""))
    images = _list_images(val_path)
    if not images:
        return pd.DataFrame(), np.zeros((0, 0)), []

    labels_root = val_path.replace("images", "labels")
    if not os.path.isdir(labels_root):
        labels_root = _resolve_path(base, "labels/val")

    if class_map is None:
        class_map = coco_to_dataset_map(data_yaml)

    class_ids = sorted(set(class_map.values()))

    model = YOLO(model_path)

    aggregate = {cid: {"tp": 0, "fp": 0, "fn": 0} for cid in class_ids}
    cm = np.zeros((len(class_ids) + 1, len(class_ids) + 1), dtype=int)

    for image_path in images:
        img = cv2.imread(image_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_labels = _load_labels(_label_path_from_image(image_path, labels_root), w, h)

        result = model.predict(
            source=image_path,
            conf=conf,
            iou=iou,
            device=device,
            verbose=False,
        )[0]

        preds: List[Tuple[int, np.ndarray, float]] = []
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for box, cls_id, conf_val in zip(boxes, classes, confs):
                if cls_id not in class_map:
                    continue
                preds.append((class_map[cls_id], box.astype(np.float32), float(conf_val)))

        stats = _match_predictions(preds, gt_labels, iou, class_ids)
        for cid in class_ids:
            aggregate[cid]["tp"] += stats[cid]["tp"]
            aggregate[cid]["fp"] += stats[cid]["fp"]
            aggregate[cid]["fn"] += stats[cid]["fn"]
        _confusion_update(preds, gt_labels, iou, class_ids, cm)

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


def evaluate_yolo_custom(
    model_path: str,
    data_yaml: str,
    device: str = "cpu",
    conf: float = 0.25,
    iou: float = 0.5,
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    class_map = dataset_class_ids(data_yaml)
    return evaluate_yolo_mapped(
        model_path=model_path,
        data_yaml=data_yaml,
        device=device,
        conf=conf,
        iou=iou,
        class_map=class_map,
    )

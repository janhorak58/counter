import os
from typing import Dict, Iterable, List, Tuple

import numpy as np


def resolve_path(base: str, path_value: str) -> str:
    if os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(base, path_value))


def list_images(val_path: str) -> List[str]:
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


def label_path_from_image(image_path: str, labels_root: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0] + ".txt"
    return os.path.join(labels_root, base)


def xywhn_to_xyxy(x, y, w, h, img_w, img_h) -> Tuple[float, float, float, float]:
    x1 = (x - w / 2) * img_w
    y1 = (y - h / 2) * img_h
    x2 = (x + w / 2) * img_w
    y2 = (y + h / 2) * img_h
    return x1, y1, x2, y2


def load_labels(label_path: str, img_w: int, img_h: int) -> List[Tuple[int, np.ndarray]]:
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
            box = np.array(xywhn_to_xyxy(x, y, w, h, img_w, img_h), dtype=np.float32)
            labels.append((class_id, box))
    return labels


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
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


def match_predictions(
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
            ious = iou(pred_box, np.array(gt_boxes, dtype=np.float32))
            best_idx = int(np.argmax(ious)) if ious.size else -1
            if best_idx >= 0 and ious[best_idx] >= iou_thresh and not gt_matched[best_idx]:
                stats[class_id]["tp"] += 1
                gt_matched[best_idx] = True
            else:
                stats[class_id]["fp"] += 1

        stats[class_id]["fn"] += sum(1 for m in gt_matched if not m)
    return stats


def confusion_update(
    preds: List[Tuple[int, np.ndarray, float]],
    gts: List[Tuple[int, np.ndarray]],
    iou_thresh: float,
    class_ids: List[int],
    cm: np.ndarray,
) -> None:
    index = {cid: i for i, cid in enumerate(class_ids)}
    bg_idx = len(class_ids)
    matched_gt = set()

    preds_sorted = sorted(enumerate(preds), key=lambda x: x[1][2], reverse=True)
    for p_idx, (p_class, p_box, _conf) in preds_sorted:
        if not gts:
            cm[bg_idx, index[p_class]] += 1
            continue
        gt_boxes = np.array([b for _, b in gts], dtype=np.float32)
        ious = iou(p_box, gt_boxes)
        best_idx = int(np.argmax(ious)) if ious.size else -1
        if best_idx >= 0 and ious[best_idx] >= iou_thresh and best_idx not in matched_gt:
            g_class = gts[best_idx][0]
            cm[index[g_class], index[p_class]] += 1
            matched_gt.add(best_idx)
        else:
            cm[bg_idx, index[p_class]] += 1

    for g_idx, (g_class, _box) in enumerate(gts):
        if g_idx not in matched_gt:
            cm[index[g_class], bg_idx] += 1

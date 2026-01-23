from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.models import rfdetr as rfdetr_module


@dataclass
class CocoImage:
    image_id: int
    file_name: str
    width: int
    height: int


@dataclass
class CocoAnn:
    image_id: int
    category_id: int
    bbox: Tuple[float, float, float, float]


def _load_coco(dataset_dir: Path, split: str) -> Tuple[List[CocoImage], List[CocoAnn], Dict[int, str]]:
    ann_path = dataset_dir / split / "_annotations.coco.json"
    payload = json.loads(ann_path.read_text(encoding="utf-8"))

    images = [
        CocoImage(
            image_id=int(img["id"]),
            file_name=str(img["file_name"]),
            width=int(img.get("width", 0)),
            height=int(img.get("height", 0)),
        )
        for img in payload.get("images", [])
    ]
    anns = [
        CocoAnn(
            image_id=int(ann["image_id"]),
            category_id=int(ann["category_id"]),
            bbox=tuple(float(v) for v in ann["bbox"]),
        )
        for ann in payload.get("annotations", [])
    ]
    categories = {int(cat["id"]): str(cat["name"]) for cat in payload.get("categories", [])}
    return images, anns, categories


def _to_xyxy(box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(float(box_a[0]), float(box_b[0]))
    y1 = max(float(box_a[1]), float(box_b[1]))
    x2 = min(float(box_a[2]), float(box_b[2]))
    y2 = min(float(box_a[3]), float(box_b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(0.0, float(box_a[3]) - float(box_a[1]))
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(0.0, float(box_b[3]) - float(box_b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _match_predictions(
    gt_boxes: List[np.ndarray],
    gt_labels: List[int],
    pred_boxes: List[np.ndarray],
    pred_labels: List[int],
    iou_thresh: float,
) -> List[Tuple[int, int]]:
    matches: List[Tuple[int, int, float]] = []
    for gi, gt in enumerate(gt_boxes):
        for pi, pred in enumerate(pred_boxes):
            iou_val = _iou(gt, pred)
            if iou_val >= iou_thresh:
                matches.append((gi, pi, iou_val))

    matches.sort(key=lambda m: m[2], reverse=True)
    matched_gt = set()
    matched_pred = set()
    paired: List[Tuple[int, int]] = []
    for gi, pi, _ in matches:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
        paired.append((gi, pi))
    return paired


def _draw_boxes(
    image: np.ndarray,
    boxes: List[np.ndarray],
    labels: List[int],
    names: Dict[int, str],
    color: Tuple[int, int, int],
    prefix: str,
    scores: List[float] | None = None,
) -> np.ndarray:
    out = image.copy()
    for idx, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = names.get(labels[idx], str(labels[idx]))
        score = ""
        if scores is not None:
            score = f" {scores[idx]:.2f}"
        text = f"{prefix}:{label}{score}"
        cv2.putText(out, text, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return out


def _save_examples_grid(images: List[np.ndarray], output_path: Path, cols: int = 4) -> None:
    if not images:
        return
    rows = int(np.ceil(len(images) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.flat:
        ax.axis("off")
    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        axes[r, c].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _save_confusion_matrix(
    matrix: np.ndarray,
    labels: List[str],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(1.2 * len(labels), 1.1 * len(labels)))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(np.arange(len(labels)), labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground truth")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix[i, j])
            if value > 0:
                ax.text(j, i, str(value), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_rfdetr_visuals(
    model_path: str,
    dataset_dir: Path,
    split: str,
    output_dir: Path,
    conf: float,
    iou_thresh: float,
    device: str,
    box_format: str = "xyxy",
    box_normalized: str = "auto",
    max_samples: int = 8,
) -> None:
    images, anns, categories = _load_coco(dataset_dir, split)
    if not images:
        return

    ann_by_image: Dict[int, List[CocoAnn]] = {}
    for ann in anns:
        ann_by_image.setdefault(ann.image_id, []).append(ann)

    category_ids = sorted(categories.keys())
    label_map = {cid: idx for idx, cid in enumerate(category_ids)}
    label_names = [categories[cid] for cid in category_ids] + ["background"]
    bg_index = len(category_ids)
    confusion = np.zeros((len(label_names), len(label_names)), dtype=np.int32)

    model = rfdetr_module.load_rfdetr_model(model_path, device=device)
    if hasattr(model, "optimize_for_inference"):
        model.optimize_for_inference()

    visual_dir = output_dir / "visuals"
    visual_dir.mkdir(parents=True, exist_ok=True)
    example_images: List[np.ndarray] = []

    sample_images = images[:max_samples]
    for img in sample_images:
        img_path = dataset_dir / split / img.file_name
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue

        gt_ann = ann_by_image.get(img.image_id, [])
        gt_boxes = [np.array(_to_xyxy(ann.bbox), dtype=np.float32) for ann in gt_ann]
        gt_labels = [label_map.get(ann.category_id, -1) for ann in gt_ann]

        detections = rfdetr_module.predict_rfdetr(
            model,
            frame,
            conf=conf,
            iou=0.5,
            box_format=box_format,
            box_normalized=box_normalized,
        )
        pred_boxes = [det[0] for det in detections]
        pred_labels = [label_map.get(det[1], -1) for det in detections]
        pred_scores = [float(det[2]) for det in detections]

        pairs = _match_predictions(gt_boxes, gt_labels, pred_boxes, pred_labels, iou_thresh=iou_thresh)
        matched_gt = {gi for gi, _ in pairs}
        matched_pred = {pi for _, pi in pairs}

        for gi, pi in pairs:
            gt_idx = gt_labels[gi] if gt_labels[gi] >= 0 else bg_index
            pred_idx = pred_labels[pi] if pred_labels[pi] >= 0 else bg_index
            confusion[gt_idx, pred_idx] += 1

        for gi, gt_label in enumerate(gt_labels):
            if gi in matched_gt:
                continue
            gt_idx = gt_label if gt_label >= 0 else bg_index
            confusion[gt_idx, bg_index] += 1

        for pi, pred_label in enumerate(pred_labels):
            if pi in matched_pred:
                continue
            pred_idx = pred_label if pred_label >= 0 else bg_index
            confusion[bg_index, pred_idx] += 1

        gt_names = {label_map[cid]: name for cid, name in categories.items()}
        pred_names = gt_names

        overlay = _draw_boxes(frame, gt_boxes, gt_labels, gt_names, (0, 255, 0), "GT")
        overlay = _draw_boxes(overlay, pred_boxes, pred_labels, pred_names, (0, 0, 255), "PR", pred_scores)
        example_images.append(overlay)

    _save_examples_grid(example_images, visual_dir / "examples.png")
    _save_confusion_matrix(confusion, label_names, visual_dir / "confusion_matrix.png")
    np.savetxt(visual_dir / "confusion_matrix.csv", confusion, fmt="%d", delimiter=",")

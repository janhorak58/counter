import importlib
import os
from typing import Any, Iterable, List, Tuple

import numpy as np


def _import_rfdetr_module() -> Any:
    errors = []
    for name in ("rf_detr", "rfdetr"):
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    details = "; ".join(errors) if errors else "no import attempts"
    raise ImportError(
        "RF-DETR package not found. Install it from https://github.com/roboflow/rf-detr "
        f"and ensure it exposes RFDETR. Details: {details}"
    )


def load_rfdetr_model(model_path: str, device: str = "cpu") -> Any:
    module = _import_rfdetr_module()
    model_cls = getattr(module, "RFDETR", None)
    if model_cls is None:
        raise ImportError("RFDETR class not found in the rf-detr package.")

    model = None
    if model_path:
        if hasattr(model_cls, "from_pretrained"):
            try:
                model = model_cls.from_pretrained(model_path)
            except Exception:
                model = None
        if model is None and hasattr(model_cls, "load_from_checkpoint"):
            try:
                model = model_cls.load_from_checkpoint(model_path)
            except Exception:
                model = None
        if model is None:
            try:
                model = model_cls(model_path)
            except Exception:
                model = None

    if model is None:
        try:
            model = model_cls()
        except Exception as exc:
            raise RuntimeError(
                "Unable to initialize RFDETR. Provide a valid model_path or check the "
                "rf-detr package API."
            ) from exc

    if hasattr(model, "to"):
        try:
            model = model.to(device)
        except Exception:
            pass
    if hasattr(model, "eval"):
        try:
            model.eval()
        except Exception:
            pass
    return model


def _to_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value)


def _normalize_boxes(
    boxes: np.ndarray,
    image_shape: Tuple[int, int],
    box_format: str,
    box_normalized: Any,
) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if boxes.size == 0:
        return boxes

    fmt = (box_format or "xyxy").lower()
    normalized = False
    if isinstance(box_normalized, bool):
        normalized = box_normalized
    elif isinstance(box_normalized, str) and box_normalized.lower() == "auto":
        max_val = float(np.max(boxes))
        normalized = max_val <= 1.5

    if normalized:
        h, w = image_shape
        if fmt in {"xyxy", "xywh", "cxcywh"}:
            scale = np.array([w, h, w, h], dtype=np.float32)
            boxes = boxes * scale

    if fmt == "xywh":
        x, y, w, h = boxes.T
        boxes = np.stack([x, y, x + w, y + h], axis=1)
    elif fmt == "cxcywh":
        cx, cy, w, h = boxes.T
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    return boxes


def _extract_predictions(output: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if isinstance(output, dict):
        boxes = output.get("boxes")
        scores = output.get("scores") or output.get("conf") or output.get("confidence")
        labels = output.get("labels") or output.get("classes") or output.get("class_ids")
        return _to_numpy(boxes), _to_numpy(scores), _to_numpy(labels)

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        if isinstance(output[0], dict):
            return _extract_predictions(output[0])
        if len(output) == 3:
            boxes, scores, labels = output
            return _to_numpy(boxes), _to_numpy(scores), _to_numpy(labels)

    if hasattr(output, "boxes") and hasattr(output, "scores") and hasattr(output, "labels"):
        return _to_numpy(output.boxes), _to_numpy(output.scores), _to_numpy(output.labels)

    raise RuntimeError("Unsupported RF-DETR prediction output format.")


def predict_rfdetr(
    model: Any,
    image: np.ndarray,
    conf: float = 0.25,
    iou: float = 0.5,
    box_format: str = "xyxy",
    box_normalized: Any = "auto",
) -> List[Tuple[np.ndarray, int, float]]:
    if hasattr(model, "predict"):
        try:
            output = model.predict(image, conf=conf, iou=iou)
        except TypeError:
            output = model.predict(image)
    elif callable(model):
        output = model(image)
    else:
        raise RuntimeError("RF-DETR model does not support prediction.")

    boxes, scores, labels = _extract_predictions(output)
    boxes = _normalize_boxes(boxes, image.shape[:2], box_format, box_normalized)

    detections: List[Tuple[np.ndarray, int, float]] = []
    for box, score, label in zip(boxes, scores, labels):
        score_val = float(score) if score is not None else 0.0
        if score_val < conf:
            continue
        detections.append((box.astype(np.float32), int(label), score_val))
    return detections


def train_rfdetr(
    model_path: str,
    data_yaml: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    device: str,
    project: str,
    name: str,
    **kwargs: Any,
) -> None:
    module = _import_rfdetr_module()
    if hasattr(module, "train"):
        module.train(
            model=model_path,
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            workers=workers,
            device=device,
            project=project,
            name=name,
            **kwargs,
        )
        return

    model = load_rfdetr_model(model_path, device=device)
    if hasattr(model, "train"):
        try:
            model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                workers=workers,
                device=device,
                project=project,
                name=name,
                **kwargs,
            )
            return
        except TypeError:
            pass

    raise RuntimeError("RF-DETR training API not found. Update src/models/rfdetr.py with the correct training call.")

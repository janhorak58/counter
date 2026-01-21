import importlib
import os
from typing import Any, Iterable, List, Tuple

import numpy as np


def _import_rfdetr_module() -> Any:
    errors = []
    preferred = os.environ.get("RFDETR_MODULE")
    if preferred:
        try:
            return importlib.import_module(preferred)
        except Exception as exc:
            errors.append(f"{preferred}: {exc}")

    for name in ("rfdetr", "rf_detr"):
        try:
            return importlib.import_module(name)
        except Exception as exc:
            errors.append(f"{name}: {exc}")
    details = "; ".join(errors) if errors else "no import attempts"
    raise ImportError(
        "RF-DETR package not found. Install it from https://github.com/roboflow/rf-detr "
        f"and ensure it exposes RFDETR. Details: {details}"
    )


def _resolve_rfdetr_class(module: Any, model_path: str) -> Any:
    model_cls = getattr(module, "RFDETR", None)
    if model_cls is not None:
        return model_cls

    named_classes = [
        "RFDETRSmall",
        "RFDETRMedium",
        "RFDETRLarge",
        "RFDETRNano",
        "RFDETRBase",
        "RFDETRSegPreview",
    ]
    available = {name: getattr(module, name) for name in named_classes if hasattr(module, name)}

    if model_path:
        direct = getattr(module, model_path, None)
        if direct is not None:
            return direct

        hint = model_path.lower()
        for key, class_name in (
            ("nano", "RFDETRNano"),
            ("small", "RFDETRSmall"),
            ("medium", "RFDETRMedium"),
            ("large", "RFDETRLarge"),
            ("base", "RFDETRBase"),
            ("seg", "RFDETRSegPreview"),
        ):
            if key in hint and class_name in available:
                return available[class_name]

    for class_name in named_classes:
        if class_name in available:
            return available[class_name]
    return None


def load_rfdetr_model(model_path: str, device: str = "cpu") -> Any:
    module = _import_rfdetr_module()
    model_cls = _resolve_rfdetr_class(module, model_path)
    if model_cls is None:
        raise ImportError(
            "RFDETR class not found in the rf-detr package. "
            "Available exports should include RFDETRSmall/Medium/Large/Nano."
        )

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
    def _coerce_label(value: Any) -> int:
        if value is None:
            return -1
        try:
            return int(value)
        except (TypeError, ValueError):
            return -1

    def _coerce_score(value: Any) -> float:
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _xyxy_from_center(cx: float, cy: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2

    def _xyxy_from_xywh(x: float, y: float, w: float, h: float) -> Tuple[float, float, float, float]:
        return x, y, x + w, y + h

    def _extract_from_prediction_list(preds: Iterable[Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes: List[List[float]] = []
        scores: List[float] = []
        labels: List[int] = []

        for pred in preds:
            if pred is None:
                continue
            if isinstance(pred, dict):
                pred_dict = pred
            else:
                pred_dict = {
                    "confidence": getattr(pred, "confidence", None),
                    "score": getattr(pred, "score", None),
                    "probability": getattr(pred, "probability", None),
                    "conf": getattr(pred, "conf", None),
                    "class_id": getattr(pred, "class_id", None),
                    "category_id": getattr(pred, "category_id", None),
                    "class": getattr(pred, "class", None),
                    "label": getattr(pred, "label", None),
                    "name": getattr(pred, "name", None),
                    "bbox": getattr(pred, "bbox", None),
                    "box": getattr(pred, "box", None),
                    "xyxy": getattr(pred, "xyxy", None),
                    "xmin": getattr(pred, "xmin", None),
                    "ymin": getattr(pred, "ymin", None),
                    "xmax": getattr(pred, "xmax", None),
                    "ymax": getattr(pred, "ymax", None),
                    "x1": getattr(pred, "x1", None),
                    "y1": getattr(pred, "y1", None),
                    "x2": getattr(pred, "x2", None),
                    "y2": getattr(pred, "y2", None),
                    "x": getattr(pred, "x", None),
                    "y": getattr(pred, "y", None),
                    "width": getattr(pred, "width", None),
                    "height": getattr(pred, "height", None),
                    "w": getattr(pred, "w", None),
                    "h": getattr(pred, "h", None),
                }

            score = _coerce_score(
                pred_dict.get("confidence")
                or pred_dict.get("score")
                or pred_dict.get("probability")
                or pred_dict.get("conf")
            )
            label = _coerce_label(
                pred_dict.get("class_id")
                or pred_dict.get("category_id")
                or pred_dict.get("class")
                or pred_dict.get("label")
                or pred_dict.get("name")
            )

            box = pred_dict.get("bbox") or pred_dict.get("box") or pred_dict.get("xyxy")
            if box is not None:
                box_vals = list(box)
                if len(box_vals) == 4:
                    x1, y1, x2, y2 = box_vals
                else:
                    continue
            elif all(pred_dict.get(k) is not None for k in ("xmin", "ymin", "xmax", "ymax")):
                x1 = float(pred_dict["xmin"])
                y1 = float(pred_dict["ymin"])
                x2 = float(pred_dict["xmax"])
                y2 = float(pred_dict["ymax"])
            elif all(pred_dict.get(k) is not None for k in ("x1", "y1", "x2", "y2")):
                x1 = float(pred_dict["x1"])
                y1 = float(pred_dict["y1"])
                x2 = float(pred_dict["x2"])
                y2 = float(pred_dict["y2"])
            elif all(pred_dict.get(k) is not None for k in ("x", "y", "width", "height")):
                x1, y1, x2, y2 = _xyxy_from_center(
                    float(pred_dict["x"]), float(pred_dict["y"]), float(pred_dict["width"]), float(pred_dict["height"])
                )
            elif all(pred_dict.get(k) is not None for k in ("x", "y", "w", "h")):
                x1, y1, x2, y2 = _xyxy_from_xywh(
                    float(pred_dict["x"]), float(pred_dict["y"]), float(pred_dict["w"]), float(pred_dict["h"])
                )
            else:
                continue

            boxes.append([x1, y1, x2, y2])
            scores.append(score)
            labels.append(label)

        if not boxes:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)

        return np.array(boxes, dtype=np.float32), np.array(scores, dtype=np.float32), np.array(labels, dtype=np.int32)

    if isinstance(output, dict):
        if "predictions" in output:
            return _extract_from_prediction_list(output["predictions"])
        if "detections" in output:
            return _extract_from_prediction_list(output["detections"])
        if "results" in output:
            return _extract_predictions(output["results"])
        boxes = output.get("boxes")
        scores = output.get("scores") or output.get("conf") or output.get("confidence")
        labels = output.get("labels") or output.get("classes") or output.get("class_ids")
        return _to_numpy(boxes), _to_numpy(scores), _to_numpy(labels)

    if isinstance(output, (list, tuple)):
        if len(output) == 0:
            return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        if isinstance(output[0], dict):
            return _extract_predictions(output[0])
        if hasattr(output[0], "predictions") or hasattr(output[0], "detections") or hasattr(output[0], "xyxy"):
            return _extract_predictions(output[0])
        if len(output) == 3:
            boxes, scores, labels = output
            return _to_numpy(boxes), _to_numpy(scores), _to_numpy(labels)

    if hasattr(output, "boxes") and hasattr(output, "scores") and hasattr(output, "labels"):
        return _to_numpy(output.boxes), _to_numpy(output.scores), _to_numpy(output.labels)
    if hasattr(output, "predictions"):
        return _extract_from_prediction_list(output.predictions)
    if hasattr(output, "detections"):
        return _extract_from_prediction_list(output.detections)
    if hasattr(output, "xyxy"):
        def _first_not_none(*values: Any) -> Any:
            for value in values:
                if value is not None:
                    return value
            return None

        boxes = _to_numpy(output.xyxy)
        scores = _to_numpy(_first_not_none(getattr(output, "confidence", None), getattr(output, "conf", None)))
        labels = _to_numpy(
            _first_not_none(
                getattr(output, "class_id", None),
                getattr(output, "class_ids", None),
                getattr(output, "classes", None),
            )
        )
        return boxes, scores, labels

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

import numpy as np
from typing import Dict, List, Tuple

from ultralytics import YOLO

from src.core.DetectedObject import DetectedObject
from src.models.rfdetr import load_rfdetr_model, predict_rfdetr

from src.utils.device_utils import norm_device

class ObjectTracker:
    """Spravuje všechny sledované objekty"""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.4,
        iou: float = 0.5,
        device: str = "cpu",
        pretrained: bool = False,
        model_type: str = "yolo",
        track_iou_threshold: float = 0.3,
        track_max_lost: int = 15,
        track_match_classes: bool = True,
        rfdetr_box_format: str = "xyxy",
        rfdetr_box_normalized: str = "auto",
    ):
        self.model_type = (model_type or "yolo").lower()
        self.device = norm_device(device)
        self.confidence = confidence
        self.iou = iou
        self.pretrained = pretrained
        self.objects: Dict[int, DetectedObject] = {}
        self._lost_frames: Dict[int, int] = {}
        self._next_id = 1
        self.track_iou_threshold = float(track_iou_threshold)
        self.track_max_lost = int(track_max_lost)
        self.track_match_classes = bool(track_match_classes)
        self.rfdetr_box_format = rfdetr_box_format
        self.rfdetr_box_normalized = rfdetr_box_normalized

        if self.model_type == "yolo":
            self.model = YOLO(model_path)
        elif self.model_type == "rfdetr":
            self.model = load_rfdetr_model(model_path, device=device)
        else:
            raise ValueError(f"Unsupported model_type: {self.model_type}")

    def _map_pretrained_class_id(self, class_id: int) -> int:
        mapping = {0: 4, 30: 5, 1: 6, 16: 7}  # person -> 4, skis -> 5, bicycle -> 6, dog -> 7
        return mapping.get(class_id, -1)

    def _filter_class_id(self, class_id: int) -> int:
        if self.pretrained:
            class_id = self._map_pretrained_class_id(class_id)
        return class_id

    def _iou_matrix(self, boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        if boxes_a.size == 0 or boxes_b.size == 0:
            return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)
        xA = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        yA = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        xB = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        yB = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / (union + 1e-9)

    def _update_yolo(self, frame: np.ndarray) -> Dict[int, DetectedObject]:
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            device=self.device,
            conf=self.confidence,
            iou=self.iou,
            tracker="bytetrack.yaml",
            # tracker="bytetrack_strict.yaml"
        )

        current_ids = set()

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()

            for bbox, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                class_id = self._filter_class_id(int(class_id))
                if class_id == -1:
                    continue

                current_ids.add(track_id)
                if track_id in self.objects:
                    self.objects[track_id].update(bbox, float(conf))
                else:
                    self.objects[track_id] = DetectedObject(
                        track_id, class_id, bbox, float(conf), self.device
                    )

        lost_ids = set(self.objects.keys()) - current_ids
        for lost_id in lost_ids:
            del self.objects[lost_id]

        return {tid: self.objects[tid] for tid in current_ids}

    def _update_rfdetr(self, frame: np.ndarray) -> Dict[int, DetectedObject]:
        detections = predict_rfdetr(
            self.model,
            frame,
            conf=self.confidence,
            iou=self.iou,
            box_format=self.rfdetr_box_format,
            box_normalized=self.rfdetr_box_normalized,
        )

        filtered: List[Tuple[np.ndarray, int, float]] = []
        for box, class_id, conf in detections:
            class_id = self._filter_class_id(int(class_id))
            if class_id == -1:
                continue
            filtered.append((box, class_id, conf))

        if not filtered:
            for track_id in list(self._lost_frames.keys()):
                self._lost_frames[track_id] += 1
                if self._lost_frames[track_id] > self.track_max_lost:
                    self._lost_frames.pop(track_id, None)
                    self.objects.pop(track_id, None)
            return {}

        det_boxes = np.array([d[0] for d in filtered], dtype=np.float32)
        det_classes = [d[1] for d in filtered]
        det_confs = [d[2] for d in filtered]

        track_ids = list(self.objects.keys())
        track_boxes = np.array([self.objects[tid].bbox.cpu().numpy() for tid in track_ids], dtype=np.float32)
        iou_matrix = self._iou_matrix(det_boxes, track_boxes)

        matched_dets = set()
        matched_tracks = set()
        active_ids = set()
        while True:
            if iou_matrix.size == 0:
                break
            det_idx, track_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
            best_iou = iou_matrix[det_idx, track_idx]
            if best_iou < self.track_iou_threshold:
                break
            if det_idx in matched_dets or track_idx in matched_tracks:
                iou_matrix[det_idx, track_idx] = -1
                continue
            if self.track_match_classes:
                if det_classes[det_idx] != self.objects[track_ids[track_idx]].class_id:
                    iou_matrix[det_idx, track_idx] = -1
                    continue

            matched_dets.add(det_idx)
            matched_tracks.add(track_idx)
            track_id = track_ids[track_idx]
            self.objects[track_id].update(det_boxes[det_idx], det_confs[det_idx])
            self._lost_frames[track_id] = 0
            active_ids.add(track_id)
            iou_matrix[det_idx, :] = -1
            iou_matrix[:, track_idx] = -1

        for idx, (box, class_id, conf) in enumerate(filtered):
            if idx in matched_dets:
                continue
            track_id = self._next_id
            self._next_id += 1
            self.objects[track_id] = DetectedObject(track_id, class_id, box, conf, self.device)
            self._lost_frames[track_id] = 0
            active_ids.add(track_id)

        for idx, track_id in enumerate(track_ids):
            if idx in matched_tracks:
                continue
            self._lost_frames[track_id] = self._lost_frames.get(track_id, 0) + 1
            if self._lost_frames[track_id] > self.track_max_lost:
                self._lost_frames.pop(track_id, None)
                self.objects.pop(track_id, None)

        return {tid: self.objects[tid] for tid in active_ids}

    def update(self, frame: np.ndarray) -> Dict[int, DetectedObject]:
        """Zpracuje frame a vrA­tA- aktualizovanAc objekty"""
        if self.model_type == "yolo":
            return self._update_yolo(frame)
        return self._update_rfdetr(frame)

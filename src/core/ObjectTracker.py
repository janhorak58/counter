import numpy as np
from typing import Dict
from ultralytics import YOLO
from src.core.DetectedObject import DetectedObject


class ObjectTracker:
    """Spravuje všechny sledované objekty"""
    
    def __init__(self, model_path: str, confidence: float = 0.4, iou: float = 0.5, device: str = 'cpu', pretrained: bool = False):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.pretrained = pretrained
        self.objects: Dict[int, DetectedObject] = {}

    
    def update(self, frame: np.ndarray) -> Dict[int, DetectedObject]:
        """Zpracuje frame a vrátí aktualizované objekty"""

        def map_pretrained_class_id(class_id: int) -> int:
            """Mapuje ID třídy z předtrénovaného modelu na naše interní ID"""
            mapping = {0: 4, 30:5, 1: 6, 16: 7 }  # person -> 4, skis -> 5, bicycle -> 6, dog -> 7
            return mapping.get(class_id, -1)  # -1 pro neznámé třídy
        results = self.model.track(frame, persist=True, verbose=False, device=self.device, conf=self.confidence, iou=self.iou, tracker="bytetrack.yaml")

        
        current_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for bbox, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
                if self.pretrained:
                    # Mapování ID třídy pro předtrénovaný model
                    class_id = map_pretrained_class_id(class_id)
                if class_id == -1:
                    continue  # Přeskočit neznámé třídy
                
                current_ids.add(track_id)
                if track_id in self.objects:
                    # Aktualizace existujícího objektu
                    self.objects[track_id].update(bbox, conf)
                else:
                    # Nový objekt
                    self.objects[track_id] = DetectedObject(
                        track_id, class_id, bbox, conf, self.device
                    )
        
        # Odstranění ztracených objektů
        lost_ids = set(self.objects.keys()) - current_ids
        for lost_id in lost_ids:
            del self.objects[lost_id]
        
        return {tid: self.objects[tid] for tid in current_ids}
import numpy as np
from typing import Dict
from ultralytics import YOLO
from src.core.DetectedObject import DetectedObject


class ObjectTracker:
    """Spravuje všechny sledované objekty"""
    
    def __init__(self, model_path: str, confidence: float = 0.4, iou: float = 0.5, device: str = 'cpu'):
        self.model = YOLO(model_path)
        self.device = device
        self.confidence = confidence
        self.iou = iou
        self.objects: Dict[int, DetectedObject] = {}

    
    def update(self, frame: np.ndarray) -> Dict[int, DetectedObject]:
        """Zpracuje frame a vrátí aktualizované objekty"""
        results = self.model.track(frame, persist=True, verbose=False, device=self.device, conf=self.confidence, iou=self.iou)
        
        current_ids = set()
        
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confidences = results[0].boxes.conf.cpu().numpy()
            
            for bbox, track_id, class_id, conf in zip(boxes, track_ids, class_ids, confidences):
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
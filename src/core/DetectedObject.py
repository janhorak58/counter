import cv2
import numpy as np
import torch
from typing import Tuple, List, Optional


class DetectedObject:
    """Reprezentuje detekovaný a sledovaný objekt"""
    MAX_FRAME_HISTORY = 30  # Počet snímků pro historii pozic
    class_names = {0: 'tourist', 1: 'skier', 2: 'cyclist', 3: 'tourist_dog'}
    colors = {0: (0, 255, 0), 1: (0, 0, 255), 2: (255, 0, 0), 3: (255, 255, 0)}
    
    def __init__(self, obj_id: int, class_id: int, bbox, confidence: float, device='cpu'):
        self.id = obj_id
        self.class_id = class_id
        self.class_name = self.class_names.get(class_id, 'unknown')
        self.color = self.colors.get(class_id, (255, 255, 255))
        self.confidence = confidence
        self.device = device
        
        self.bbox = torch.tensor(bbox, dtype=torch.float32).to(device)
        self.centroid = self._compute_centroid()
        
        self.initial_side: Optional[int] = None
        self.positions: List[Tuple[int, int]] = []
        self.counted: bool = False 
        self.counted_direction: Optional[str] = None  # 'in' nebo 'out' pro zobrazení
        
   
    def _compute_centroid(self) -> torch.Tensor:
        """Vypočítá střed spodní hrany bboxu"""
        x1, y1, x2, y2 = self.bbox
        cx = (x1 + x2) / 2
        cy = y2  # Spodní střed
        return torch.tensor([cx, cy], dtype=torch.float32).to(self.device)
    
    def get_centroid_int(self) -> Tuple[int, int]:
        """Vrátí centroid jako tuple intů pro OpenCV"""
        c = self.centroid.cpu().numpy()
        return (int(c[0]), int(c[1]))
    
    def update(self, bbox, confidence: float):
        """Aktualizuje pozici objektu"""
        self.bbox = torch.tensor(bbox, dtype=torch.float32).to(self.device)
        self.centroid = self._compute_centroid()
        self.confidence = confidence
        
        # Uložení pozice do historie
        self.positions.append(self.get_centroid_int())
        if len(self.positions) > self.MAX_FRAME_HISTORY:
            self.positions.pop(0)
    
    def draw(self, frame: np.ndarray, show_trajectory: bool = True):
        """Vykreslí bbox, label a trajektorii"""
        x1, y1, x2, y2 = map(int, self.bbox.cpu().numpy())
        center = self.get_centroid_int()
        
        # Bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), self.color, 2)
        
        # Label
        status = ""
        if self.counted_direction == 1:
            status = " [IN]"
        elif self.counted_direction == -1:
            status = " [OUT]"
        
        label = f"{self.class_name} #{self.id}{status}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)
        
        # Trajektorie
        if show_trajectory and len(self.positions) > 1:
            for i in range(1, len(self.positions)):
                cv2.line(frame, self.positions[i-1], self.positions[i], self.color, 2)
        
        # Centroid
        cv2.circle(frame, center, 4, self.color, -1)

        # Confidence
        conf_label = f"{self.confidence:.2f}"
        cv2.putText(frame, conf_label, (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color, 2)


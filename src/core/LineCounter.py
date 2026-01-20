# models/LineCounter.py
from typing import Tuple, Dict, Optional
from collections import defaultdict
import torch
import cv2
import numpy as np

from src.core.DetectedObject import DetectedObject
from src.utils.device_utils import norm_device



class ObjectState:
    """Stav sledovaného objektu vůči jedné čáře"""
    def __init__(self):
        self.initial_side: Optional[int] = None
        self.counted: bool = False
        self.last_side: Optional[int] = None
        self.counted_direction: Optional[str] = None

class LineCounter:
    """Počítá průchody objektů přes definovanou čáru"""

    # Barvy pro různé čáry
    line_colors = [(0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0), (255, 128, 0)]

    def __init__(self, line_start: Tuple[int, int], line_end: Tuple[int, int],
                 min_distance: float = 20.0, name: str = "Line", device='cpu'):

        device = norm_device(device)
        self.A = torch.tensor(line_start, dtype=torch.float32).to(device)
        self.B = torch.tensor(line_end, dtype=torch.float32).to(device)
        self.line_vec = self.B - self.A
        
        self.min_distance = min_distance
        
        # Počítadla
        self.counts_in: Dict[int, int] = defaultdict(int)
        self.counts_out: Dict[int, int] = defaultdict(int)
        
        # Stav objektů PRO TUTO ČÁRU (každá čára má vlastní stav)
        self.object_states: Dict[int, ObjectState] = {}
        # {obj_id: {'initial_side': int, 'counted': bool}}
    
    def get_signed_distance(self, point: torch.Tensor) -> float:
        """Vrátí znaménkovou vzdálenost bodu od čáry"""
        AP = point - self.A
        cross = self.line_vec[0] * AP[1] - self.line_vec[1] * AP[0]
        line_len = torch.norm(self.line_vec)
        if line_len == 0:
            return 0.0
        return (cross / line_len).item()
    
    def get_side(self, centroid: torch.Tensor) -> int:
        """Určí stranu čáry"""
        signed_dist = self.get_signed_distance(centroid)
        
        if abs(signed_dist) < self.min_distance:
            return 0
        elif signed_dist > 0:
            return 1
        else:
            return -1
    
    def check_crossing(self, obj: DetectedObject) -> Optional[str]:
        """Zkontroluje pr?chod objektu p?es TUTO ??ru"""
        obj_id = obj.id
        
        # Inicializace stavu pro nov? objekt
        if obj_id not in self.object_states:
            self.object_states[obj_id] = ObjectState()
        
        state : ObjectState = self.object_states[obj_id]
        
        current_side = self.get_side(obj.centroid)
        
        if state.initial_side is None:
            if current_side != 0:
                state.initial_side = current_side
                state.last_side = current_side
            return None
        
        if current_side == 0:
            return None
        
        if state.last_side is None:
            state.last_side = current_side
            return None
        
        if current_side != state.last_side:
            if not state.counted:
                if state.initial_side == -1 and current_side == 1:
                    direction = 'in'
                else:
                    direction = 'out'
                
                state.counted = True
                state.counted_direction = direction
                obj.counted_direction = current_side
                
                if direction == 'in':
                    self.counts_in[obj.class_id] += 1
                else:
                    self.counts_out[obj.class_id] += 1
                state.last_side = current_side
                return direction
            
            if current_side == state.initial_side and state.counted_direction:
                if state.counted_direction == 'in':
                    self.counts_in[obj.class_id] = max(0, self.counts_in[obj.class_id] - 1)
                else:
                    self.counts_out[obj.class_id] = max(0, self.counts_out[obj.class_id] - 1)
                state.counted = False
                state.counted_direction = None
                obj.counted_direction = None
            state.last_side = current_side
            return None
        
        return None
    
    def get_total_in(self) -> int:
        return sum(self.counts_in.values())
    
    def get_total_out(self) -> int:
        return sum(self.counts_out.values())
    
    def draw(self, frame: np.ndarray, color: Tuple[int, int, int] = None):
        """Vykreslí čáru"""
        if color is None:
            color = (0, 255, 255)
        
        A = tuple(map(int, self.A.cpu().numpy()))
        B = tuple(map(int, self.B.cpu().numpy()))
        
        cv2.line(frame, A, B, color, 3)
        
        # Název čáry
        mid_x = (A[0] + B[0]) // 2
        mid_y = (A[1] + B[1]) // 2
        cv2.putText(frame, self.name, (mid_x - 30, mid_y - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, "OUT ->", (mid_x + 10, mid_y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, "<- IN", (mid_x - 70, mid_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    def draw_counts(self, frame: np.ndarray, x: int = 10, y: int = 30):
        """Vykreslí počty pro tuto čáru"""
        cv2.putText(frame, f"=== {self.name} ===", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for class_id, name in DetectedObject.class_names.items():
            y += 22
            in_count = self.counts_in[class_id]
            out_count = self.counts_out[class_id]
            color = DetectedObject.colors.get(class_id, (255, 255, 255))
            text = f"{name}: IN={in_count} OUT={out_count}"
            cv2.putText(frame, text, (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        
        y += 25
        cv2.putText(frame, f"Total: IN={self.get_total_in()} OUT={self.get_total_out()}",
                    (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

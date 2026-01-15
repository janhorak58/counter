import cv2
import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from src.core.ObjectTracker import ObjectTracker
from src.core.LineCounter import LineCounter
from src.core.DetectedObject import DetectedObject

class Counter:
    """Hlavní API pro počítání průchodů - podporuje více čar"""
    
    def __init__(self, model_path: str, lines: List[Dict], 
                 min_distance: float = 20.0, device: str = 'cpu', confidence: float = 0.4, iou: float = 0.5, pretrained=False):
        """
        Args:
            model_path: Cesta k YOLO modelu
            lines: Seznam čar, každá jako dict:
                   {'start': (x1, y1), 'end': (x2, y2), 'name': 'Line 1'}
            min_distance: Minimální vzdálenost od čáry
            device: 'cpu' nebo 'cuda'
        """
        self.tracker = ObjectTracker(model_path, confidence, iou, device, pretrained=pretrained)
        self.device = device
        self.min_distance = min_distance
        self.frame_num = 0
        self.pretrained = pretrained
        
        # Vytvoření LineCounterů pro každou čáru
        self.line_counters: Dict[str, LineCounter] = {}
        for line in lines:
            name = line.get('name', f"Line_{len(self.line_counters)}")
            self.line_counters[name] = LineCounter(
                line_start=line['start'],
                line_end=line['end'],
                min_distance=min_distance,
                name=name,
                device=device
            )
    
    def add_line(self, start: Tuple[int, int], end: Tuple[int, int], name: str = None):
        """Přidá novou čáru za běhu"""
        if name is None:
            name = f"Line_{len(self.line_counters)}"
        
        self.line_counters[name] = LineCounter(
            line_start=start,
            line_end=end,
            min_distance=self.min_distance,
            name=name,
            device=self.device
        )
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, List[Tuple[DetectedObject, Optional[str]]]]:
        """
        Zpracuje jeden frame.
        Vrací: dict s výsledky pro každou čáru
               {'Line_1': [(obj, 'in'), (obj, None)], 'Line_2': [...]}
        """
        self.frame_num += 1
        
        # Aktualizace trackeru
        active_objects = self.tracker.update(frame)
        
        results = {}
        for line_name, line_counter in self.line_counters.items():
            line_results = []
            for obj in active_objects.values():
                crossing = line_counter.check_crossing(obj)
                line_results.append((obj, crossing))
                
                if crossing:
                    print(f"Frame {self.frame_num}: {obj.class_name} (ID: {obj.id}) -> {crossing.upper()} @ {line_name}")
            
            results[line_name] = line_results
        
        return results
    
    def draw(self, frame: np.ndarray, show_trajectory: bool = True):
        """Vykreslí vše do framu"""
        # Všechny čáry
        for line_counter in self.line_counters.values():
            line_counter.draw(frame)
        
        # Panel s počty pro všechny čáry
        y_offset = 30
        for line_name, line_counter in self.line_counters.items():
            line_counter.draw_counts(frame, x=10, y=y_offset)
            y_offset += 150  # Posun pro další čáru
        
        # Objekty (jen jednou)
        for obj in self.tracker.objects.values():
            if len(obj.positions) > 0:
                obj.draw(frame, show_trajectory)
    
    def get_counts(self) -> Dict[str, dict]:
        """Vrátí počty pro všechny čáry"""
        return {
            name: {
                'in': dict(lc.counts_in),
                'out': dict(lc.counts_out),
                'total_in': lc.get_total_in(),
                'total_out': lc.get_total_out()
            }
            for name, lc in self.line_counters.items()
        }
    
    @staticmethod
    def select_lines_interactive(frame: np.ndarray, num_lines: int = 1) -> List[Dict]:
        """Interaktivní výběr více čar"""
        lines = []
        
        for i in range(num_lines):
            line_coords = []
            drawing = False
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal drawing, line_coords
                if event == cv2.EVENT_LBUTTONDOWN:
                    drawing = True
                    line_coords = [(x, y)]
                elif event == cv2.EVENT_MOUSEMOVE and drawing:
                    if len(line_coords) == 2:
                        line_coords[1] = (x, y)
                    else:
                        line_coords.append((x, y))
                elif event == cv2.EVENT_LBUTTONUP:
                    drawing = False
                    if len(line_coords) == 1:
                        line_coords.append((x, y))
                    else:
                        line_coords[1] = (x, y)
            
            window_name = f"Draw line {i+1}/{num_lines} - press 'q' when done"
            cv2.namedWindow(window_name)
            cv2.setMouseCallback(window_name, mouse_callback)
            
            print(f"Nakreslete čáru {i+1}/{num_lines} a stiskněte 'q'")
            
            while True:
                temp_frame = frame.copy()
                
                # Vykresli předchozí čáry
                colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
                for j, prev_line in enumerate(lines):
                    color = colors[j % len(colors)]
                    cv2.line(temp_frame, prev_line['start'], prev_line['end'], color, 2)
                    cv2.putText(temp_frame, prev_line['name'], prev_line['start'],
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Vykresli aktuální čáru
                if len(line_coords) >= 2:
                    color = colors[i % len(colors)]
                    cv2.line(temp_frame, line_coords[0], line_coords[1], color, 2)
                elif len(line_coords) == 1:
                    cv2.circle(temp_frame, line_coords[0], 5, (0, 255, 0), -1)
                
                cv2.imshow(window_name, temp_frame)
                if cv2.waitKey(1) & 0xFF == ord('q') and len(line_coords) == 2:
                    break
            
            cv2.destroyWindow(window_name)
            
            lines.append({
                'start': tuple(line_coords[0]),
                'end': tuple(line_coords[1]),
                'name': f"Line_{i+1}"
            })
        
        return lines
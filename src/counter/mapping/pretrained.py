from __future__ import annotations

from typing import Dict, Optional, Tuple

from counter.domain.types import Detection, CanonicalClass
from counter.mapping.base import MappingPolicy

PERSON = 100
BICYCLE = 101
SKIS = 102
DOG = 103

class CocoBaselineMapping(MappingPolicy):
    def __init__(self, coco_ids: Dict[str, int]):
        self.person_id = int(coco_ids.get('person', 0))
        self.bicycle_id = int(coco_ids.get('bicycle', 1))
        self.dog_id = int(coco_ids.get('dog', 16))
        self.skis_id = int(coco_ids.get('skis', 30))

    def map_detection(self, det: Detection) -> Optional[int]:
        cid = int(det.raw_class_id)
        if cid == self.person_id:
            return PERSON
        if cid == self.bicycle_id:
            return BICYCLE
        if cid == self.skis_id:
            return SKIS
        if cid == self.dog_id:
            return DOG
        return None

    def finalize_counts(self, in_counts: Dict[int, int], out_counts: Dict[int, int]) -> Tuple[Dict[int, int], Dict[int, int]]:
        def conv(src: Dict[int, int]) -> Dict[int, int]:
            person = int(src.get(PERSON, 0))
            bicycle = int(src.get(BICYCLE, 0))
            skis = int(src.get(SKIS, 0))
            dog = int(src.get(DOG, 0))
            tourist = max(0, person - bicycle - skis)
            return {
                int(CanonicalClass.TOURIST): tourist,
                int(CanonicalClass.SKIER): skis,
                int(CanonicalClass.CYCLIST): bicycle,
                int(CanonicalClass.TOURIST_DOG): dog,
            }
        return conv(in_counts), conv(out_counts)

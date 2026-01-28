from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from counter.core.types import CanonicalClass
from counter.predict.mapping.base import MappingPolicy
from counter.core.schema import MappingCfg, ModelSpecCfg

# Intermediate class IDs for COCO-style mapping.
PERSON = 100
BICYCLE = 101
SKIS = 102
DOG = 103


INTERMEDIATE_NAMES: Dict[int, str] = {
    PERSON: "person",
    BICYCLE: "bicycle",
    SKIS: "skis",
    DOG: "dog",
}


@dataclass(frozen=True)
class CocoBaselineMapping(MappingPolicy):
    """Map COCO ids to intermediates, then compute canonical counts.

    Important: we count line-crossings separately for PERSON/BICYCLE/SKIS/DOG and
    approximate:
      cyclist = bicycle
      skier = skis
      tourist = max(person - bicycle - skis, 0)
      tourist_dog = dog

    This is a pragmatic baseline; better association (person+bicycle pairs) can be
    added later.
    """

    mapping: Optional[MappingCfg] = None

    def map_raw(self, *, raw_class_id: int, raw_class_name: str) -> Optional[int]:

        rid = int(raw_class_id)
        if rid == int(self.mapping.tourist if self.mapping else -1):
            return PERSON
        if rid == int(self.mapping.cyclist if self.mapping else -1):
            return BICYCLE
        if rid == int(self.mapping.skier if self.mapping else -1):
            return SKIS
        if rid == int(self.mapping.tourist_dog if self.mapping else -1):
            return DOG
        return None

    def finalize_counts(
        self,
        in_counts: Dict[int, int],
        out_counts: Dict[int, int],
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        def _mk(counts: Dict[int, int]) -> Dict[int, int]:
            person = int(counts.get(PERSON, 0))
            bicycle = int(counts.get(BICYCLE, 0))
            skis = int(counts.get(SKIS, 0))
            dog = int(counts.get(DOG, 0))

            tourist = max(person - bicycle - skis, 0)
            skier = skis
            cyclist = bicycle
            tourist_dog = dog

            return {
                int(CanonicalClass.TOURIST): int(tourist),
                int(CanonicalClass.SKIER): int(skier),
                int(CanonicalClass.CYCLIST): int(cyclist),
                int(CanonicalClass.TOURIST_DOG): int(tourist_dog),
            }

        return _mk(in_counts), _mk(out_counts)

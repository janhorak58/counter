from counter.mapping.pretrained import CocoBaselineMapping, PERSON, BICYCLE, SKIS, DOG
from counter.domain.types import CanonicalClass

def test_coco_baseline_finalize_counts():
    m = CocoBaselineMapping({'person':0,'bicycle':1,'dog':16,'skis':30})
    raw_in = {PERSON: 10, BICYCLE: 3, SKIS: 2, DOG: 1}
    raw_out = {PERSON: 5, BICYCLE: 7, SKIS: 0, DOG: 0}
    fin_in, fin_out = m.finalize_counts(raw_in, raw_out)
    assert fin_in[int(CanonicalClass.CYCLIST)] == 3
    assert fin_in[int(CanonicalClass.SKIER)] == 2
    assert fin_in[int(CanonicalClass.TOURIST_DOG)] == 1
    assert fin_in[int(CanonicalClass.TOURIST)] == 5
    assert fin_out[int(CanonicalClass.TOURIST)] == 0

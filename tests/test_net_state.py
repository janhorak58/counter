from counter.counting.geometry import Line
from counter.counting.net_state import NetStateCounter
from counter.domain.types import Track

def test_net_state_counts_in_and_unwinds_on_return():
    line = Line((0,0),(10,0), name='Line_1')
    c = NetStateCounter(line, greyzone_px=1.0)

    t_out = Track(track_id=1, bbox=(0,-10,10,-5), score=0.9, mapped_class_id=0)
    t_in  = Track(track_id=1, bbox=(0,  5,10,10), score=0.9, mapped_class_id=0)
    t_out2= Track(track_id=1, bbox=(0,-10,10,-5), score=0.9, mapped_class_id=0)

    c.observe([t_out])
    c.observe([t_in])
    c.observe([t_out2])

    in_counts, out_counts = c.finalize_raw_counts()
    assert in_counts == {}
    assert out_counts == {}

def test_net_state_counts_in_stays_after_disappearance():
    line = Line((0,0),(10,0), name='Line_1')
    c = NetStateCounter(line, greyzone_px=1.0)

    t_out = Track(track_id=1, bbox=(0,-10,10,-5), score=0.9, mapped_class_id=0)
    t_in  = Track(track_id=1, bbox=(0,  5,10,10), score=0.9, mapped_class_id=0)

    c.observe([t_out])
    c.observe([t_in])

    in_counts, out_counts = c.finalize_raw_counts()
    assert in_counts.get(0) == 1

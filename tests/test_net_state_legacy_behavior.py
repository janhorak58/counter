from counter.counting.geometry import Line
from counter.counting.net_state import NetStateCounter
from counter.domain.types import Track

def test_net_state_counts_increment_and_decrement():
    # Horizontal line from (0,0) to (10,0). Signed distance >0 is 'IN' (Side.IN=1)
    line = Line(start=(0, 0), end=(10, 0), name="Line_1")
    c = NetStateCounter(line=line, greyzone_px=0.0)

    tid = 1
    # Start below line (OUT), then cross above (IN), then cross back to OUT => final should be zero.
    # bbox bottom_center: (x_mid, y2)
    out_bbox = (4.0, -10.0, 6.0, -1.0)   # bottom y2=-1 => below line? actually line y=0; y2=-1 -> OUT
    in_bbox  = (4.0,  1.0, 6.0, 10.0)    # y2=10 -> IN

    c.observe([Track(track_id=tid, bbox=out_bbox, score=0.9, mapped_class_id=0)])
    c.observe([Track(track_id=tid, bbox=in_bbox,  score=0.9, mapped_class_id=0)])

    raw_in, raw_out = c.finalize_raw_counts()
    assert raw_in.get(0, 0) + raw_out.get(0, 0) == 1  # counted somewhere

    # Cross back
    c.observe([Track(track_id=tid, bbox=out_bbox, score=0.9, mapped_class_id=0)])
    raw_in2, raw_out2 = c.finalize_raw_counts()
    assert raw_in2.get(0, 0) == 0
    assert raw_out2.get(0, 0) == 0

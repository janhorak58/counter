from counter.counting.geometry import Line, signed_distance, side_of, bottom_center
from counter.domain.types import Side

def test_signed_distance_symmetry():
    line = Line((0,0),(10,0))
    assert signed_distance((0,  10), line) > 0
    assert signed_distance((0, -10), line) < 0

def test_greyzone_unknown():
    line = Line((0,0),(10,0))
    assert side_of((5, 0.5), line, greyzone_px=2.0) == Side.UNKNOWN
    assert side_of((5, 3.0), line, greyzone_px=2.0) == Side.IN
    assert side_of((5,-3.0), line, greyzone_px=2.0) == Side.OUT

def test_bottom_center():
    assert bottom_center((0,0,10,20)) == (5.0, 20)

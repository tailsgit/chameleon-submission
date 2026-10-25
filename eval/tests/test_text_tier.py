"""engine.select._text_kept (CHAMELEON_TEXT_TIER=1): text may not fall under half its readable size."""
from engine.select import _text_kept


def _design(size, h=1000):
    return {"canvas": {"width": 1000, "height": h},
            "elements": [{"id": "t", "type": "text", "font_size": size}, {"id": "i", "type": "image"}]}


def test_text_kept_floor_is_half_the_readable_size_capped_by_source():
    src = _design(40)
    assert _text_kept(src, {"width": 1000, "height": 1000}, _design(15))      # floor max(12, 15) = 15 -> needs 7.5
    assert not _text_kept(src, {"width": 1000, "height": 1000}, _design(7))
    small = _design(4)                                                      # source fine print 4px -> needs 2
    assert _text_kept(small, {"width": 1000, "height": 1000}, _design(2))
    assert not _text_kept(small, {"width": 1000, "height": 1000}, _design(1))

"""engine.preserve: the source layout, changed as little as possible."""

import pytest

from engine.analyze import analyze
from engine.fit import Sizer, realize
from engine.preserve import preserve_plans
from eval.check import check
from eval.score import relations_kept

IMG = "solstice_bottle.png"


def _design():
    img = lambda i, x, y, w, h: {"id": i, "type": "image", "x": x, "y": y, "width": w, "height": h, "asset_id": IMG, "fit": "contain"}
    txt = lambda i, x, y, w, h, c, fs: {"id": i, "type": "text", "x": x, "y": y, "width": w, "height": h, "content": c,
                                        "font_family": "Solstice Sans", "font_size": fs, "color": "rgba(0,0,0,1)"}
    els = [img("p1", 80, 80, 400, 300), img("p2", 360, 240, 400, 300), img("p3", 160, 420, 400, 300),
           txt("h", 600, 120, 420, 200, "A collage of our summer", 56), txt("b", 600, 900, 420, 80, "See more at the shop", 28)]
    for z, e in enumerate(els):
        e["z_index"] = z
    return {"canvas": {"width": 1080, "height": 1080}, "elements": els}


@pytest.mark.parametrize("target", [{"width": 1080, "height": 1350}, {"width": 1200, "height": 628}])
def test_preserve_keeps_the_collage_and_passes(target):
    src = _design()
    a = analyze(src)
    plans = preserve_plans(a, Sizer(a, target["width"], target["height"]), target["width"], target["height"])
    assert plans
    for p in plans:
        out = realize(a, p, target["width"], target["height"])
        assert check(src, target, out) == []
        o = {e["id"]: e for e in out["elements"]}
        k = o["p1"]["width"] / 400  # the three photos: one scale, same offsets
        assert abs(o["p2"]["x"] - o["p1"]["x"] - 280 * k) < 1.5 and abs(o["p3"]["y"] - o["p1"]["y"] - 340 * k) < 1.5
        assert relations_kept(src, target, out)[0] == 1.0


def test_preserve_is_deterministic():
    src = _design()
    a = analyze(src)
    runs = [[(s.uid, s.x, s.y, s.w) for p in preserve_plans(a, Sizer(a, 1200, 628), 1200, 628) for s in p.slots] for _ in range(2)]
    assert runs[0] == runs[1]

"""Extra candidates from the chosen layout: accents pinned to their corners, the content block grown."""

from engine.analyze import analyze
from engine.finish import accents, fill, pin
from eval.check import check
from eval.tests.test_structure import design, image, text


def test_corner_accent_is_pinned_and_content_fills_the_canvas():
    src = design(1000, 1000, [image("photo", 300, 300, 400, 400), image("flourish", 900, 0, 100, 100),
                              text("t", 300, 720, 400, 60, "Caption", fs=40)])
    a = analyze(src)
    assert accents(a) == ["flourish"]
    target = {"width": 2000, "height": 1000}
    out = {"canvas": target, "elements": [dict(e) for e in src["elements"]]}  # the source, unmoved, on a wider canvas
    pinned = {e["id"]: e for e in pin(a, out, ["flourish"], 2000, 1000)["elements"]}
    assert (pinned["flourish"]["x"] + pinned["flourish"]["width"], pinned["flourish"]["y"]) == (2000, 0)  # top-right corner
    grown = fill(a, out, ["flourish"], 2000, 1000)
    g = {e["id"]: e for e in grown["elements"]}
    assert g["photo"]["width"] > 400 and abs(g["photo"]["x"] + g["photo"]["width"] / 2 - 1000) < 1  # bigger, centred
    assert g["t"]["font_size"] > 40 and g["flourish"] == out["elements"][1]  # text grows with it; the accent stays
    assert check(src, target, grown) == []

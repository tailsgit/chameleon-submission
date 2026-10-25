"""text_layout / text overflow must agree with what the renderer actually draws.

Each case renders one text element alone (black on white, with a margin so
overflow stays visible) and compares the ink bounding box with the predicted
layout and with check()'s overflow verdict."""

import json
from pathlib import Path

import numpy as np
import pytest

from eval.check import check, text_layout
from eval.render import render_image

REPO = Path(__file__).resolve().parents[2]
MARGIN = 150
PX = 3  # ink vs layout tolerance: glyph side bearings and anti-aliasing


def _source_texts():
    for name in ["level1_warmup", "level2_core", "level3_advanced"]:
        task = json.loads((REPO / "input" / f"chameleon_sample_{name}.json").read_text())
        for e in task["source"]["elements"]:
            if e["type"] == "text":
                yield pytest.param(e, id=f"{name}:{e['id']}")


def _t(content, w, h, fs, **kw):
    return dict(id="t", type="text", x=0, y=0, width=w, height=h, content=content,
                font_family=kw.pop("font_family", "Solstice Sans"), font_size=fs, color="rgba(0,0,0,1)", **kw)


LOREM = "Small-batch botanical infusions blended with wild honey and blood orange zest"
SYNTHETIC = [
    pytest.param(_t(LOREM, 300, 200, 20.7), id="fractional-size"),
    pytest.param(_t(LOREM, 260, 300, 18, text_align="right"), id="right-multiline"),
    pytest.param(_t("SUNSET SALE TODAY", 400, 40, 20, letter_spacing=4), id="letter-spacing"),
    pytest.param(_t(LOREM, 300, 200, 22, font_style="italic", font_family="Solstice Serif"), id="italic-serif"),
    pytest.param(_t(LOREM, 300, 200, 20, style_runs=[{"start": 0, "end": 11, "font_weight": "bold"}]), id="bold-run"),
    pytest.param(_t(LOREM, 300, 200, 20, line_height=1.6), id="line-height"),
    pytest.param(_t("Supercalifragilistic", 80, 60, 24, text_align="center"), id="overflow-width-center"),
    pytest.param(_t("Supercalifragilistic", 80, 60, 24), id="overflow-width-left"),
    pytest.param(_t(LOREM, 200, 40, 20), id="overflow-height"),
    pytest.param(_t(LOREM, 900, 40, 9.9), id="tiny"),
]


def ink_box(el):
    e = dict(el, x=MARGIN, y=MARGIN, rotation=0, opacity=1, color="rgba(0,0,0,1)",
             style_runs=[{k: v for k, v in r.items() if k != "color"} for r in el.get("style_runs", [])])
    w, h = int(el["width"] + 2 * MARGIN), int(3 * el["height"] + 2 * MARGIN)
    assert max(w, h) <= 2048  # keep the render at 1:1 scale
    ink = np.asarray(render_image({"canvas": {"width": w, "height": h}, "elements": [e]}).convert("L")) < 160
    ys, xs = np.nonzero(ink)
    rows = np.flatnonzero(ink.any(axis=1))
    line_count = 1 + int(np.sum(np.diff(rows) > 1))
    return (xs.min() - MARGIN, ys.min() - MARGIN, xs.max() + 1 - MARGIN, ys.max() + 1 - MARGIN), line_count


@pytest.mark.parametrize("el", [*_source_texts(), *SYNTHETIC])
def test_layout_matches_render(el):
    lay = text_layout(el)
    (x0, y0, x1, y1), ink_lines = ink_box(el)
    assert abs(x0 - lay["left"]) <= PX and abs(x1 - lay["right"]) <= PX, (lay, (x0, x1))
    assert y0 >= -1 and y1 <= lay["height"] + 1, (lay, (y0, y1))
    assert y1 >= lay["height"] - 0.4 * el["font_size"]  # ink reaches the last line
    assert ink_lines == lay["lines"]

    # check()'s overflow verdict must match the ink wherever the case is clear-cut
    rules = {v["rule"] for v in check({"elements": [el], "canvas": {"width": 1, "height": 1}},
                                      {"width": 1, "height": 1}, {"canvas": {"width": 1, "height": 1}, "elements": [el]})}
    if abs(lay["height"] - el["height"]) > 0.4 * el["font_size"]:
        assert ("text_overflow_height" in rules) == (y1 > el["height"] + 1)
    if min(abs(lay["left"]), abs(lay["right"] - el["width"])) > PX:
        assert ("text_overflow_width" in rules) == (x0 < -PX or x1 > el["width"] + PX)

import copy
import json
from pathlib import Path

import pytest

from eval.baseline import stretch as resize  # the known-bad uniform stretch
from eval.score import PENALTIES, score

REPO = Path(__file__).resolve().parents[2]
SAMPLES = ["level1_warmup", "level2_core", "level3_advanced"]
GEOMETRIC = [k for k, (_f, _w, render) in PENALTIES.items() if not render]


def load(name):
    return json.loads((REPO / "input" / f"chameleon_sample_{name}.json").read_text())


def el(design, eid):
    return next(e for e in design["elements"] if e["id"] == eid)


@pytest.mark.parametrize("name", SAMPLES)
def test_identity_is_perfect_except_target_relative_space(name):
    src = load(name)["source"]
    r = score(src, src["canvas"], copy.deepcopy(src))
    assert set(r["breakdown"]) == set(GEOMETRIC)
    assert all(b["penalty"] == 0 for k, b in r["breakdown"].items() if k != "space")
    # space is judged against the canvas, not the source: the sparse L1 layout pays, the full ones do not
    assert (r["breakdown"]["space"]["penalty"] > 0.1) == (name == "level1_warmup")


def test_empty_space_is_judged_against_the_target():
    src = load("level1_warmup")["source"]
    wide = {"width": 2160, "height": 1080}
    huddled = copy.deepcopy(src)  # the source as is, on the left half of a 2:1 canvas
    huddled["canvas"] = wide
    el(huddled, "bg").update(width=2160)
    centred = copy.deepcopy(huddled)
    for e in centred["elements"][1:]:
        e["x"] += 540
    own = score(src, src["canvas"], src)["breakdown"]
    s_h, s_c = (score(src, wide, o)["breakdown"] for o in (huddled, centred))
    assert s_c["space"]["penalty"] > own["space"]["penalty"] + 0.1  # same content, twice the canvas
    assert s_h["balance"]["penalty"] > 0.3 and s_c["balance"]["penalty"] == 0


def test_hole_between_halves_is_empty_space():
    src = load("level1_warmup")["source"]
    t = {"width": 2160, "height": 1080}
    def split(text_x):  # logo on the left, text block at text_x
        out = copy.deepcopy(src)
        out["canvas"] = t
        el(out, "bg").update(width=2160)
        el(out, "logo").update(x=100, y=490)
        el(out, "headline").update(x=text_x, y=420)
        el(out, "cta").update(x=text_x, y=620)
        return score(src, t, out)["breakdown"]["space"]["penalty"]
    assert split(1260) > split(520) + 0.1


def test_baseline_problems_are_penalised():
    b = {}
    for name in SAMPLES:
        t = load(name)
        b[name] = score(t["source"], t["target_canvas"], resize(t["source"], t["target_canvas"]))
    # L2: text shrinks to 6-8px on a 4:1 banner (stretching does fill the canvas, so space is fine)
    assert b["level2_core"]["breakdown"]["small_text"]["penalty"] > 0.2
    # L3: every font collapses to 1-2px
    assert b["level3_advanced"]["breakdown"]["small_text"]["penalty"] > 0.8
    assert max(b["level2_core"]["total"], b["level3_advanced"]["total"]) < 95


def _mutated(mutate):
    src = load("level2_core")["source"]
    out = copy.deepcopy(src)
    mutate(out)
    return score(src, src["canvas"], out)["breakdown"]


def test_new_overlap_penalised_but_designed_layering_is_not():
    assert _mutated(lambda o: el(o, "headline").update(y=500))["overlap"]["penalty"] > 0.3
    # the source already layers the bottle over deco_circle_1 and text over the CTA pill
    assert _mutated(lambda o: None)["overlap"]["penalty"] == 0


def test_headline_must_stay_largest():
    assert _mutated(lambda o: el(o, "subhead").update(font_size=70))["headline"]["penalty"] > 0


def test_headline_dominance_is_visual_not_nominal():
    # 14px bold serif over a 12px regular sans: nominally bigger, visually close to parity
    def squash(o):
        el(o, "headline").update(font_size=14)
        el(o, "subhead").update(font_size=12)
        el(o, "badge_label").update(font_size=11)
    assert _mutated(squash)["headline"]["penalty"] > 0.3
    assert _mutated(lambda o: el(o, "headline").update(font_size=44))["headline"]["penalty"] == 0


def test_near_miss_alignment():
    assert _mutated(lambda o: el(o, "subhead").update(x=148))["alignment"]["penalty"] > 0


def test_reading_order():
    def swap(o):
        el(o, "logo").update(y=950)
        for i in ("cta_bg", "cta_label"):
            el(o, i)["y"] -= 890
    assert _mutated(swap)["reading_order"]["penalty"] > 0


def test_side_by_side_reading_down_each_column_is_fine():
    src = load("level1_warmup")["source"]
    t = {"width": 2160, "height": 1080}
    out = copy.deepcopy(src)  # logo on the left half, headline then CTA stacked on the right half
    out["canvas"] = t
    el(out, "bg").update(width=2160)
    el(out, "logo").update(x=380, y=490)
    el(out, "headline").update(x=1220, y=420)
    el(out, "cta").update(x=1220, y=620)
    assert score(src, t, out)["breakdown"]["reading_order"]["penalty"] == 0


def test_floating_elements_must_keep_their_corner():
    src = load("level3_advanced")["source"]
    def ribbon_at(o, x, y):
        for i in ("ribbon_bg", "ribbon_label"):
            el(o, i)["x"] += x - 740
            el(o, i)["y"] += y - 30
        return score(src, src["canvas"], o)["breakdown"]["floating"]
    assert ribbon_at(copy.deepcopy(src), 740, 30)["penalty"] == 0
    moved = ribbon_at(copy.deepcopy(src), 300, 400)
    assert moved["penalty"] > 0.4 and "ribbon" in moved["detail"]


def test_contrast_only_when_asked_and_detects_dark_text():
    src = load("level1_warmup")["source"]
    assert "contrast" not in score(src, src["canvas"], src)["breakdown"]
    ok = score(src, src["canvas"], src, contrast=True)["breakdown"]["contrast"]["penalty"]
    dark = copy.deepcopy(src)
    el(dark, "headline")["color"] = "rgba(50,30,45,1)"
    bad = score(src, src["canvas"], dark, contrast=True)["breakdown"]["contrast"]
    assert ok == 0 and bad["penalty"] > 0.4 and bad["detail"].startswith("headline")  # mean over 2 texts


def test_contrast_the_source_already_had_is_not_penalised():
    src = copy.deepcopy(load("level1_warmup")["source"])
    el(src, "cta")["color"] = "rgba(70,40,60,1)"  # the designer's own low-contrast choice
    assert score(src, src["canvas"], src, contrast=True)["breakdown"]["contrast"]["penalty"] == 0
    worse = copy.deepcopy(src)
    el(worse, "cta")["color"] = "rgba(55,35,50,1)"
    assert score(src, src["canvas"], worse, contrast=True)["breakdown"]["contrast"]["penalty"] > 0


def test_emphasis_is_saturated_and_off_palette():
    from eval.score import emphasis
    sal = emphasis(load("level2_core")["source"])
    assert {"cta_bg", "badge_bg"} <= set(sal)  # orange on a dark plum palette
    assert not {"headline", "subhead", "cta_label", "bg"} & set(sal)  # white text is contrast, not colour emphasis


def test_emphasised_unit_off_focus_is_penalised():
    def strand(o):  # the orange CTA pushed to the left edge, no longer last in the reading path
        for i in ("cta_bg", "cta_label"):
            el(o, i)["x"] -= 320
            el(o, i)["y"] -= 200
    assert _mutated(lambda o: None)["emphasis"]["penalty"] == 0
    assert _mutated(strand)["emphasis"]["penalty"] > 0.5


def test_attached_unit_must_stay_on_its_host_and_tilt_the_right_way():
    src = copy.deepcopy(load("level3_advanced")["source"])
    for i in ("ribbon_bg", "ribbon_label"):  # put the ribbon across the bottle's top-right corner
        el(src, i)["x"] += 400 - 740
        el(src, i)["y"] += 170 - 30
    def floating(mutate):
        out = copy.deepcopy(src)
        mutate(out)
        return score(src, src["canvas"], out)["breakdown"]["floating"]
    def move(dx, dy, mirror=False):
        def f(o):
            for i in ("ribbon_bg", "ribbon_label"):
                el(o, i)["x"] += dx
                el(o, i)["y"] += dy
                if mirror:
                    el(o, i)["rotation"] *= -1
        return f
    def with_bottle(f):
        def g(o):
            for i in ("product",):
                el(o, i)["x"] -= 300
            f(o)
        return g
    assert floating(lambda o: None)["penalty"] == 0
    assert floating(with_bottle(move(-300, 0)))["penalty"] == 0  # moved together with the bottle
    assert floating(move(0, 600))["penalty"] >= 0.5  # left the bottle behind (mean with the watermark, which stayed)
    left = floating(move(-250, 0))                                 # now on the bottle's left side, same tilt
    assert "tilts the wrong way" in left["detail"]
    assert "tilts" not in floating(move(-250, 0, mirror=True))["detail"]


def test_sibling_that_differs_is_the_emphasised_one():
    from eval.score import emphasis, siblings
    src = load("level3_advanced")["source"]
    assert [[u[0] for u in g] for g in siblings(src)] == [["badge1_bg", "badge2_bg", "badge3_bg"]]
    sal = emphasis(src)
    assert "badge2_bg" in sal  # the dark badge among orange ones, though dark is not a palette accent
    assert not {"badge1_bg", "badge3_bg"} & set(sal)
    assert {"cta_bg", "ribbon_bg"} <= set(sal)  # not in a group: the palette rule as before


def test_split_sibling_row_is_penalised():
    src = load("level3_advanced")["source"]
    def siblings_pen(mutate):
        out = copy.deepcopy(src)
        mutate(out)
        return score(src, src["canvas"], out)["breakdown"]["siblings"]["penalty"]
    def drop_one(o):  # badge2 moved down on its own
        for i in ("badge2_bg", "badge2_label"):
            el(o, i)["y"] += 200
    def shrink_one(o):
        el(o, "badge3_bg").update(width=130, height=25)
    assert siblings_pen(lambda o: None) == 0
    assert siblings_pen(drop_one) == 0.5 and siblings_pen(shrink_one) == 0.5


def test_shrunken_hero_and_headline_cost_scale():
    assert _mutated(lambda o: None)["scale"]["penalty"] == 0
    def shrink(o):  # the bottle at half size
        el(o, "product").update(width=200, height=200)
    assert _mutated(shrink)["scale"]["penalty"] > 0.3


def test_restacked_siblings_cost_less_than_split_ones():
    src = load("level3_advanced")["source"]
    def pen(mutate):
        out = copy.deepcopy(src)
        mutate(out)
        return score(src, src["canvas"], out)["breakdown"]["siblings"]["penalty"]
    def stack(o):  # badges 2 and 3 under badge 1, same size, centred
        for n, i in enumerate(("badge2", "badge3"), 1):
            for part in ("_bg", "_label"):
                el(o, i + part).update(x=el(o, i + part)["x"] - (310 if n == 1 else 670) + 20, y=el(o, i + part)["y"] + 70 * n)
    def scatter(o):
        for part in ("_bg", "_label"):
            el(o, "badge3" + part)["y"] -= 900
    assert pen(stack) == 0.25 and pen(scatter) == 0.5


def test_text_leaving_its_button_is_penalised():
    assert _mutated(lambda o: None)["holders"]["penalty"] == 0
    def move_label(o):  # the CTA label leaves its pill
        el(o, "cta_label")["y"] -= 300
    pen = _mutated(move_label)["holders"]
    assert pen["penalty"] > 0.2 and "cta_label off cta_bg" in pen["detail"]


def test_groups_and_fill_catch_a_split_pair_and_shrunk_content():
    t = lambda i, x, y, w, h, c, fs, **kw: {"id": i, "type": "text", "x": x, "y": y, "width": w, "height": h, "content": c,
                                          "font_family": "Solstice Sans", "font_size": fs, "color": "rgba(0,0,0,1)", **kw}
    src = {"canvas": {"width": 1000, "height": 1000}, "elements": [
        t("h", 100, 100, 800, 120, "A big headline here", 90), t("lab", 100, 500, 300, 40, "Price:", 28),
        t("val", 100, 550, 300, 60, "$20 per box", 44, font_weight="bold")]}
    canvas = src["canvas"]
    assert score(src, canvas, src)["breakdown"]["groups"]["penalty"] == 0
    split = copy.deepcopy(src)
    split["elements"][2].update(x=600, y=850)  # the value walks away from its label
    assert score(src, canvas, split)["breakdown"]["groups"]["penalty"] == 1
    small = copy.deepcopy(src)
    for e in small["elements"]:  # everything squeezed into the top-left quarter
        e.update(x=e["x"] / 3, y=e["y"] / 3, width=e["width"] / 3, height=e["height"] / 3, font_size=e["font_size"] // 3)
    assert score(src, canvas, small)["breakdown"]["fill"]["penalty"] > 0.9

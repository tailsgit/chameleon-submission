import copy
import json
from pathlib import Path

import pytest

from eval.baseline import stretch as resize  # the known-bad uniform stretch
from eval.check import check

REPO = Path(__file__).resolve().parents[2]
SAMPLES = ["level1_warmup", "level2_core", "level3_advanced"]


def load(name):
    return json.loads((REPO / "input" / f"chameleon_sample_{name}.json").read_text())


def pairs(violations):
    return {(v["id"], v["rule"]) for v in violations}


# What the uniform-stretch baseline gets wrong, per our PNG review.
BASELINE_EXPECTED = {
    "level1_warmup": {("logo", "image_aspect")},
    "level2_core": {("deco_circle_1", "off_canvas"), ("logo", "image_aspect"), ("product", "image_aspect")},
    "level3_advanced": {
        ("logo", "image_aspect"), ("product", "image_aspect"), ("holiday_watermark", "image_aspect"),
        # the -12deg ribbon squashed to 3.6px tall becomes a long diagonal slash poking above the canvas
        ("ribbon_bg", "off_canvas_rotated"), ("ribbon_label", "off_canvas_rotated"),
    },
}


@pytest.mark.parametrize("name", SAMPLES)
def test_baseline_outputs(name):
    task = load(name)
    out = resize(task["source"], task["target_canvas"])
    assert pairs(check(task["source"], task["target_canvas"], out)) == BASELINE_EXPECTED[name]


SOURCE_EXPECTED = {  # designed bleeds that already exist in the sources
    "level1_warmup": set(),
    "level2_core": {("deco_circle_1", "off_canvas")},
    "level3_advanced": {("ribbon_bg", "off_canvas_rotated")},
}


@pytest.mark.parametrize("name", SAMPLES)
def test_source_against_itself(name):
    src = load(name)["source"]
    assert pairs(check(src, src["canvas"], copy.deepcopy(src))) == SOURCE_EXPECTED[name]


def el(design, eid):
    return next(e for e in design["elements"] if e["id"] == eid)


def _set(eid, **kw):
    return lambda o: el(o, eid).update(kw)


def _del(eid, field):
    return lambda o: el(o, eid).pop(field)


MUTATIONS = {  # rule: (mutation of the identity output, rules it may also trigger)
    "duplicate_id": (lambda o: o["elements"].append(copy.deepcopy(el(o, "logo"))), set()),
    "missing_id": (lambda o: o["elements"].remove(el(o, "logo")), set()),
    "unexpected_id": (lambda o: o["elements"].append(dict(copy.deepcopy(el(o, "badge_bg")), id="invented")), set()),
    "missing_field": (_del("headline", "font_size"), set()),
    "schema": (_set("headline", font_weight="heavy"), set()),
    "canvas_mismatch": (lambda o: o["canvas"].update(width=1200), {"background_not_full_canvas"}),
    "content_changed": (_set("headline", content="Chase Light"), set()),
    "font_family_changed": (_set("headline", font_family="Solstice Sans"), set()),
    "immutable_changed": (_set("product", asset_id="solstice_logo_lockup.png"), set()),
    "fit_changed": (_set("product", fit="fill"), set()),
    "hidden_element": (_set("product", opacity=0), set()),
    "off_canvas": (_set("logo", x=-50), set()),
    "off_canvas_rotated": (_set("logo", x=0, y=0, rotation=30), set()),
    "image_aspect": (_set("product", width=560), set()),
    "background_not_full_canvas": (_set("bg", width=900), set()),
    "text_overflow_height": (_set("subhead", height=20), set()),
    "text_overflow_width": (_set("headline", width=100), {"text_overflow_height"}),
}


@pytest.mark.parametrize("rule", MUTATIONS)
def test_each_rule_fires(rule):
    src = load("level2_core")["source"]
    out = copy.deepcopy(src)
    mutate, also = MUTATIONS[rule]
    mutate(out)
    found = pairs(check(src, src["canvas"], out)) - SOURCE_EXPECTED["level2_core"]
    rules = {r for _id, r in found}
    assert rule in rules
    assert rules <= {rule} | also, found


def test_schema_extra_top_level_key():
    src = load("level1_warmup")["source"]
    out = dict(copy.deepcopy(src), task_id="x")
    assert ("schema" in {v["rule"] for v in check(src, src["canvas"], out)})


def test_garbage_output_does_not_crash():
    src = load("level1_warmup")["source"]
    assert check(src, src["canvas"], {"canvas": {}, "elements": [None, {"id": "bg"}]})
    assert check(src, src["canvas"], [])

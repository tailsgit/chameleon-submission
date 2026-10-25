"""engine.retrieve: the assignment it relies on, and retrieved plans that pass the checks."""

from itertools import permutations

import numpy as np
import pytest

import engine.retrieve as retrieve
from engine.analyze import analyze
from engine.fit import realize
from engine.recipes import Sizer
from engine.retrieve import assign
from eval.check import check


@pytest.mark.parametrize("seed", range(20))
def test_assignment_is_optimal(seed):
    rng = np.random.default_rng(seed)
    n, m = int(rng.integers(1, 6)), int(rng.integers(1, 7))
    n = min(n, m)
    cost = rng.random((n, m)) * 10
    pairs = assign(cost)
    assert sorted(i for i, _ in pairs) == list(range(n)) and len({j for _, j in pairs}) == n
    best = min(sum(cost[i, p[i]] for i in range(n)) for p in permutations(range(m), n))
    assert sum(cost[i, j] for i, j in pairs) == pytest.approx(best)


def test_retrieved_plans_map_by_role_and_pass(monkeypatch):
    src = {"canvas": {"width": 1080, "height": 1080}, "elements": [
        {"id": "photo", "type": "image", "x": 0, "y": 0, "width": 1080, "height": 600, "asset_id": "solstice_bottle.png", "fit": "contain", "z_index": 0},
        {"id": "title", "type": "text", "x": 80, "y": 680, "width": 920, "height": 120, "content": "A big title", "font_family": "Solstice Sans",
         "font_size": 80, "color": "rgba(0,0,0,1)", "z_index": 1},
        {"id": "note", "type": "text", "x": 80, "y": 900, "width": 920, "height": 60, "content": "Some smaller words below it", "font_family": "Solstice Sans",
         "font_size": 30, "color": "rgba(0,0,0,1)", "z_index": 1}]}
    a = analyze(src)
    roles = {u.id: u.role for u in a.units.values()}
    # one wide design: photo on the left half, headline top right, body text below it
    lib = [{"aspect": 1.9, "roles": {"hero": 1, "headline": 1, "body": 1}, "units": [
        {"role": "hero", "kind": "image", "box": [0, 0, 0.5, 1]},
        {"role": "headline", "kind": "text", "box": [0.55, 0.1, 0.95, 0.4]},
        {"role": "body", "kind": "text", "box": [0.55, 0.5, 0.95, 0.8]}]}]
    monkeypatch.setattr(retrieve, "library", lambda: lib)
    target = {"width": 1920, "height": 1000}
    plans = retrieve.retrieved(a, Sizer(a, 1920, 1000), 1920, 1000)
    assert len(plans) == 1 and roles["photo"] == "hero" and roles["title"] == "headline"
    out = {e["id"]: e for e in realize(a, plans[0], 1920, 1000)["elements"]}
    assert out["photo"]["x"] + out["photo"]["width"] <= 960 + 1  # the photo went into the left slot
    assert out["title"]["x"] >= 0.55 * 1920 - 1 and out["title"]["y"] < out["note"]["y"]
    assert check(src, target, {"canvas": {"width": 1920, "height": 1000}, "elements": list(out.values())}) == []
    assert retrieve.retrieved(a, Sizer(a, 1000, 1920), 1000, 1920) == []  # a tall target: no design that shape


def _sample(level="level3_advanced"):
    import json
    from pathlib import Path
    return json.loads((Path(__file__).resolve().parents[2] / "input" / f"chameleon_sample_{level}.json").read_text())["source"]


def test_missing_library_falls_back_to_recipes(monkeypatch, tmp_path, caplog):
    import logging
    from engine.resize import resize
    monkeypatch.setattr(retrieve, "LIBRARY", tmp_path / "no_such_library.json")
    retrieve.library.cache_clear()
    try:
        assert retrieve.library() == []
        src, target = _sample(), {"width": 1920, "height": 1080}
        with caplog.at_level(logging.INFO, logger="engine"):
            out = resize(src, target)
        assert check(src, target, out) == []
        winners = [r.getMessage() for r in caplog.records if r.getMessage().startswith("winner: ")]
        assert winners and winners[0] != "winner: retrieved"
        assert not any(r.levelno >= logging.ERROR for r in caplog.records)  # nothing failed on the way
    finally:
        retrieve.library.cache_clear()  # the next test reads the real library again


@pytest.mark.parametrize("budget", [0.0, 0.5])
def test_output_does_not_depend_on_a_time_budget(monkeypatch, budget):
    """Work is bounded by counts (engine.candidate_budget), not elapsed time: BUDGET_S changes nothing."""
    import engine.resize
    src, target = _sample(), {"width": 728, "height": 90}
    default = engine.resize.resize(src, target)
    monkeypatch.setattr(engine.resize, "BUDGET_S", budget)
    out = engine.resize.resize(src, target)
    assert check(src, target, out) == [] and out == default


def test_switch_turns_retrieval_off(monkeypatch):
    a = analyze(_sample())
    sizer = Sizer(a, 1920, 1080)
    assert retrieve.retrieved(a, sizer, 1920, 1080)  # on by default: the library has wide layouts
    monkeypatch.setenv("CHAMELEON_RETRIEVAL", "0")
    assert retrieve.retrieved(a, sizer, 1920, 1080) == []


def test_text_newly_on_a_busy_image_is_rejected(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    import eval.score as score_mod
    from engine.select import select
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (128, 128, 3), dtype=np.uint8)).save(tmp_path / "busy.png")
    monkeypatch.setenv("CHAMELEON_ASSETS", str(tmp_path))
    score_mod._asset_grey.cache_clear()
    src = {"canvas": {"width": 1000, "height": 500}, "elements": [
        {"id": "photo", "type": "image", "x": 0, "y": 0, "width": 500, "height": 500, "asset_id": "busy.png", "fit": "cover", "z_index": 0},
        {"id": "title", "type": "text", "x": 560, "y": 200, "width": 400, "height": 100, "content": "Title", "font_family": "Solstice Sans",
         "font_size": 60, "color": "rgba(0,0,0,1)", "z_index": 1}]}
    on_photo = {"canvas": src["canvas"], "elements": [dict(src["elements"][0]), dict(src["elements"][1], x=60)]}
    assert score_mod.busy_text_over_image(src, src["canvas"], on_photo) == [("title", "photo")]
    assert score_mod.busy_text_over_image(src, src["canvas"], src) == []
    assert select(src, src["canvas"], [on_photo, src]) is not on_photo
    assert select(src, src["canvas"], [on_photo]) is on_photo  # if that is all there is, it is still returned
    score_mod._asset_grey.cache_clear()


def _panel_design():
    """White text on a dark panel that covers the left 35% (a compact panel) of a white design."""
    return {"canvas": {"width": 1000, "height": 1000}, "elements": [
        {"id": "panel", "type": "shape", "x": 0, "y": 0, "width": 350, "height": 1000, "shape_type": "rectangle",
         "fill_color": "rgba(20,20,20,1)", "z_index": 0},
        {"id": "title", "type": "text", "x": 20, "y": 400, "width": 310, "height": 100, "content": "Summer", "font_family": "Solstice Sans",
         "font_size": 72, "color": "rgba(255,255,255,1)", "z_index": 1}]}


def test_text_off_its_backdrop_and_lost_contrast_are_detected():
    from eval.score import contrast_drops, text_off_backdrop
    src = _panel_design()
    assert text_off_backdrop(src, src["canvas"], src) == [] and contrast_drops(src, src["canvas"], src) == []
    off = {"canvas": src["canvas"], "elements": [src["elements"][0], dict(src["elements"][1], x=560)]}  # white on white
    assert [t for t, _b, _s in text_off_backdrop(src, src["canvas"], off)] == ["title"]
    assert [t for t, _a, _b in contrast_drops(src, src["canvas"], off)] == ["title"]


def test_select_prefers_legible_and_falls_back_to_best_contrast():
    from engine.select import select
    src = _panel_design()
    half = {"canvas": src["canvas"], "elements": [src["elements"][0], dict(src["elements"][1], x=200)]}  # half on: off its panel
    off = {"canvas": src["canvas"], "elements": [src["elements"][0], dict(src["elements"][1], x=560)]}
    assert select(src, src["canvas"], [off, half, src]) is src
    assert select(src, src["canvas"], [off, half]) is half  # none legible: the one that kept more contrast

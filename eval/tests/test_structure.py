"""Structure the engine must recognise on unseen designs: turned backgrounds, rule lines,
text on image cards, grids of repeated cells, and a big hero photo."""

import pytest

import engine.select
from engine.analyze import analyze
from engine.resize import resize
from eval.check import check

IMG = "solstice_bottle.png"  # any existing asset; contain keeps its box aspect


@pytest.fixture(autouse=True)
def fewer_renders(monkeypatch):
    monkeypatch.setattr(engine.select, "RENDER_TOP", 1)


def text(id, x, y, w, h, content="Caption text", fs=24, **kw):
    return {"id": id, "type": "text", "x": x, "y": y, "width": w, "height": h, "content": content,
            "font_family": "Solstice Sans", "font_size": fs, "color": "rgba(0,0,0,1)", **kw}


def image(id, x, y, w, h, **kw):
    return {"id": id, "type": "image", "x": x, "y": y, "width": w, "height": h, "asset_id": IMG, "fit": "contain", **kw}


def design(w, h, els):
    for z, e in enumerate(els):
        e["z_index"] = z
    return {"canvas": {"width": w, "height": h}, "elements": els}


@pytest.mark.parametrize("turn", [180, 90])
def test_turned_background_is_recognised_and_placed(turn):
    bg = image("bg", 0, 0, 1080, 1080, rotation=turn, fit="cover")
    src = design(1080, 1080, [bg, text("t", 100, 400, 880, 120, "A headline over a turned photo", fs=60)])
    assert analyze(src).background == ["bg"]
    for target in ({"width": 1080, "height": 1080}, {"width": 1920, "height": 1080}):
        out = resize(src, target)
        assert check(src, target, out) == []
        b = next(e for e in out["elements"] if e["id"] == "bg")
        if turn == 180 or target["width"] == target["height"]:
            assert (b["x"], b["y"], b["width"], b["height"]) == (0, 0, target["width"], target["height"])
        else:  # a quarter turn fills only the centred square the box may occupy
            assert b["width"] == b["height"] == min(target.values())


def test_rule_lines_stay_visible_and_on_their_own():
    src = design(1080, 1080, [
        image("card", 200, 200, 680, 400),
        text("t", 240, 250, 600, 80, "Title on the card", fs=48),
        image("line", 240, 340, 600, 2),  # a divider on the card
        text("u", 240, 700, 600, 60, "Below the card", fs=36),
    ])
    a = analyze(src)
    assert "line" in [m for d in a.decorations for m in d.members]
    out = resize(src, {"width": 300, "height": 250})
    line = next(e for e in out["elements"] if e["id"] == "line")
    assert min(line["width"], line["height"]) >= 1  # 600x2 scaled to fit 300px would be 1px; never less
    assert check(src, {"width": 300, "height": 250}, out) == []


def test_text_on_an_image_card_is_one_rigid_unit():
    src = design(1080, 1080, [
        image("bubble", 300, 300, 400, 200),
        text("say", 340, 340, 320, 100, "Hello there", fs=40),
        text("other", 100, 800, 880, 100, "Elsewhere", fs=40),
    ])
    u = analyze(src).units
    assert any({"bubble", "say"} <= set(x.members) for x in u.values())
    assert not any("other" in x.members and len(x.members) > 1 for x in u.values())


def _grid(rows=2, cols=3):
    els = []
    for r in range(rows):
        for c in range(cols):
            x, y = 60 + c * 340, 300 + r * 380
            els += [image(f"p{r}{c}", x, y, 300, 220), text(f"c{r}{c}", x, y + 230, 300, 40, f"Item {r}{c}", fs=24)]
    return design(1080, 1080, [text("title", 60, 60, 960, 120, "Our menu", fs=80), *els])


def test_grid_of_cells_is_one_unit_and_reflows():
    src = _grid()
    grids = [u for u in analyze(src).units.values() if u.cols]
    assert len(grids) == 1 and grids[0].cols == 3 and len(grids[0].parts) == 6
    assert all(len(m) == 2 for m, _b in grids[0].parts)  # each cell: its photo and its caption
    out = {e["id"]: e for e in resize(src, {"width": 1080, "height": 1920})["elements"]}
    cols = {round(out[f"p{r}{c}"]["x"]) for r in range(2) for c in range(3)}
    assert len(cols) < 3  # a tall canvas: fewer columns
    for r in range(2):
        for c in range(3):  # captions stay under their own photo
            p, t = out[f"p{r}{c}"], out[f"c{r}{c}"]
            assert abs((p["x"] + p["width"] / 2) - (t["x"] + t["width"] / 2)) < 2 and t["y"] > p["y"]


def test_big_hero_keeps_its_share_on_a_wide_canvas():
    src = design(1080, 1350, [
        image("photo", 0, 0, 1080, 700),
        text("h", 80, 760, 920, 200, "A big headline under the photo", fs=72),
        text("b", 80, 1000, 920, 160, "Some body text that explains the offer in a sentence or two.", fs=32),
    ])
    target = {"width": 1584, "height": 396}
    out = resize(src, target)
    p = next(e for e in out["elements"] if e["id"] == "photo")
    assert check(src, target, out) == []
    assert p["height"] >= 0.9 * 396  # it keeps the full height of the banner instead of shrinking with the text


def test_a_big_flat_card_holds_its_text_but_a_photo_does_not(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    Image.new("RGBA", (64, 64), (240, 230, 200, 255)).save(tmp_path / "card.png")
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (64, 64, 3), dtype=np.uint8)).save(tmp_path / "photo.png")
    monkeypatch.setenv("CHAMELEON_ASSETS", str(tmp_path))
    def on(asset):  # a 600x600 image (31% of the canvas) with a title on it
        src = design(1080, 1080, [{**image("img", 240, 240, 600, 600), "asset_id": asset}, text("t", 300, 400, 480, 100, "On it", fs=60)])
        return any({"img", "t"} <= set(u.members) for u in analyze(src).units.values())
    assert on("card.png") and not on("photo.png")


def test_a_tall_stack_of_texts_flows_into_columns_on_a_wide_banner():
    src = design(500, 2000, [text(f"t{i}", 50, 40 + i * 120, 400, 60, f"Line number {i}", fs=40) for i in range(16)])
    target = {"width": 1584, "height": 396}
    out = resize(src, target)
    assert check(src, target, out) == []
    els = {e["id"]: e for e in out["elements"]}
    col = {i: els[f"t{i}"]["x"] // 400 for i in range(16)}  # columns are ~528px apart on this banner
    assert 2 <= len(set(col.values())) <= 3  # side by side in 2 or 3 columns
    assert all(els[f"t{i}"]["font_size"] >= 14 for i in range(16))  # instead of all shrinking to fit one column
    assert sorted(range(16), key=lambda i: (col[i], els[f"t{i}"]["y"])) == list(range(16))  # down each column, then right


def test_an_oversized_panel_fits_around_its_text():
    panel = {"id": "panel", "type": "shape", "x": 60, "y": 60, "width": 960, "height": 960,
             "shape_type": "rectangle", "fill_color": "rgba(255,255,255,1)"}
    src = design(1080, 1080, [panel, text("h", 140, 140, 800, 120, "A short title", fs=80)])
    assert "panel" in analyze(src).panels
    target = {"width": 1920, "height": 1080}
    res = resize(src, target)
    assert check(src, target, res) == []
    out = {e["id"]: e for e in res["elements"]}
    p, h = out["panel"], out["h"]
    assert p["x"] <= h["x"] and p["y"] <= h["y"]  # still behind its text...
    assert p["width"] > 1.5 * p["height"]  # ...but no longer a mostly empty square: wide, like its one line


def test_near_full_bleed_photo_and_colour_tiles_fill_the_target():
    def rect(id, x, y, w, h, fill):
        return {"id": id, "type": "shape", "shape_type": "rectangle", "x": x, "y": y, "width": w, "height": h, "fill_color": fill}
    photo = image("photo", 20, 20, 1040, 1040, fit="cover")  # a framed full-bleed photo: about 93% of the canvas...
    src = design(1080, 1080, [rect("left", 0, 0, 540, 1080, "rgba(200,0,0,1)"), rect("right", 540, 0, 540, 1080, "rgba(0,0,200,1)"),
                              photo, text("t", 100, 400, 880, 120, "A headline", fs=60)])
    assert set(analyze(src).background) == {"left", "right", "photo"}  # ...under 95%: one more tile on the colour tiles
    target = {"width": 1920, "height": 600}
    out = {e["id"]: e for e in resize(src, target)["elements"]}
    assert (out["left"]["x"], out["left"]["width"], out["right"]["x"] + out["right"]["width"]) == (0, 960, 1920)
    assert round(out["photo"]["x"]) == round(20 * 1920 / 1080) and round(out["photo"]["height"]) == round(1040 * 600 / 1080)
    src["elements"] = [photo, src["elements"][-1]]
    assert analyze(src).background == []
    photo.update(x=10, y=10, width=1060, height=1060)  # about 96%: a background on its own
    assert analyze(src).background == ["photo"]
    assert check(src, target, resize(src, target)) == []


def test_coverage_penalty_sees_bare_canvas():
    from eval.score import coverage, score
    bg = {"id": "bg", "type": "shape", "shape_type": "rectangle", "x": 0, "y": 0, "width": 1000, "height": 1000, "fill_color": "rgba(0,0,0,1)"}
    src = design(1000, 1000, [bg])
    half = design(1000, 1000, [dict(bg, width=500)])
    assert coverage(str(src).replace("'", '"')) == 1.0 and abs(coverage(str(half).replace("'", '"')) - 0.5) < 0.03
    assert score(src, src["canvas"], half)["breakdown"]["coverage"]["penalty"] == 1.0


def test_label_and_value_move_as_one_unit():
    src = design(1080, 1080, [
        text("h", 100, 100, 880, 160, "A big headline here", fs=96),
        text("lab", 100, 600, 300, 40, "Price:", fs=28),
        text("val", 100, 650, 300, 60, "$20 per box", fs=44, font_weight="bold"),
        text("far", 600, 950, 400, 40, "Fine print far away", fs=20),
    ])
    unit_of = lambda a, i: next(uid for uid, u in a.units.items() if i in u.members)
    tight, plain = analyze(src), analyze(src, tight=False)
    assert unit_of(tight, "lab") == unit_of(tight, "val") != unit_of(tight, "h")
    assert unit_of(plain, "lab") != unit_of(plain, "val")
    target = {"width": 1920, "height": 1080}
    assert check(src, target, resize(src, target)) == []


def test_alternate_splits_the_plans_between_analyses():
    from engine.resize import _alternate
    assert list(_alternate([iter("abcd"), iter("ABCDEF")])) == list("aBcDEF")
    assert list(_alternate([iter("ab")])) == list("ab")

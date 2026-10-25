"""Convert Crello (v4, Hugging Face parquet shards) designs into Chameleon v0.9 designs.

    python -m eval.crello_convert                 # convert every shard, validate, write the report
    python -m eval.crello_convert --compare 20    # also render 20 next to Crello's own previews

Reads   $CRELLO_DATA      (default ../data/crello: the shards matching $CRELLO_SHARDS, default validation-*.parquet)
Writes  $CRELLO_CHAMELEON (default ../data/crello_chameleon), outside the repo:
          designs/<id>.json   {"source": design, "meta": {...}}
          assets/<asset_id>   one PNG per image element: Crello's own crop of it
          previews/<id>.png   Crello's rendering of the whole design
          compare/            our render | Crello's preview, side by side (--compare)
          report.json         counts, skip reasons, font usage
Crello images and text never go into the repo or the Docker image: render and
score them with CHAMELEON_ASSETS=<out>/assets (eval.render.assets_dir).

Conversion
- Coordinates are already canvas pixels; `angle` is degrees clockwise -> rotation.
- Element order is the stacking order -> z_index.
- TextElement -> text. Text colour, weight and style are per character in Crello:
  the most common value becomes the element's, the rest become style_runs.
  `capitalize` is applied to the content (the schema has no text transform).
- ColoredBackground -> rectangle shape of its colour. SvgElement, ImageElement and
  SvgMaskElement -> image, using Crello's raster crop of that element (it has the
  box's aspect ratio, so it draws exactly as in Crello). Fit "cover" when it fills
  the canvas (a background may be cropped), else "contain", like the samples' logos
  and products: the checker then holds its aspect ratio.
- A crop that is one flat colour over its whole box (opaque or evenly translucent; colour std under
  FLAT_STD) becomes a rectangle shape of that colour, as an Adobe design would draw a colour panel.
  Photos, textures, gradients and shaped cut-outs stay images. Counted in meta.flat_to_shape.
- Rotations within 0.5 degrees of a multiple of 90 are editor noise and snap to it
  (0.01 -> 0, 89.9 -> 90).
- An unrotated image that bleeds off the canvas is cut to the visible part (its
  pixels too), so the source itself is on the canvas. Elements entirely off the
  canvas, empty texts and images without pixels are dropped.
- Fonts: Crello's ~260 families aren't available. Each maps to the closest of the
  two we have, "Solstice Serif" or "Solstice Sans" (eval/crello_fonts.json). When
  the substitute is wider, the font size is reduced (to 50% at most) until the text
  wraps to Crello's own line count within its box (meta.font_refit).

Skipped designs (counted in the report by reason): fewer than 3 or more than 40
elements after conversion, no text, text our two fonts can't draw (Cyrillic, CJK,
emoji...: it would render as boxes), a conversion error, or a schema violation.
"""

import argparse
import io
import json
import os
import traceback
from collections import Counter
from pathlib import Path

import jsonschema
import pyarrow.parquet as pq
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CRELLO_DATA") or REPO.parent / "data" / "crello")
OUT = Path(os.environ.get("CRELLO_CHAMELEON") or REPO.parent / "data" / "crello_chameleon")
FONTS = REPO / "eval" / "crello_fonts.json"
SCHEMA = json.loads((REPO / "chameleon_design_schema_v0.9.json").read_text())
MIN_ELEMENTS, MAX_ELEMENTS = 3, 40
COLUMNS = ["id", "length", "format", "canvas_width", "canvas_height", "type", "left", "top", "width", "height", "text_line",
           "angle", "opacity", "color", "image", "text", "font", "font_size", "font_bold", "font_italic",
           "text_color", "text_align", "capitalize", "line_height", "letter_spacing"]

# Serif and slab-serif families among Crello's fonts; everything else (sans, script,
# handwriting, display, mono) is closer to the sans than to the serif.
SERIF = {
    "Abril Fatface", "Alfa Slab One", "Alice", "Arapey", "Bellefair", "Bentham", "Berkshire Swash", "Copse",
    "Cormorant Infant", "Crete Round", "David Libre", "Droid Serif", "Dukomdesign Constantine", "Eb Garamond",
    "Elsie", "Elsie Swash Caps", "Fauna One", "Forum", "Frank Ruhl Libre", "Gabriela", "Gfs Didot",
    "Glass Antiqua", "Gravitas One", "Im Fell Dw Pica Sc", "Italiana", "Jacques Francois Shadow", "Josefin Slab",
    "Junge", "Kelly Slab", "Kreon", "Kumar One", "Kumar One Outline", "Kurale", "Libre Baskerville", "Limelight",
    "Marcellus Sc", "Mate", "Merriweather", "Noticia Text", "Old Standard Tt", "Oranienbaum", "Patua One",
    "Playfair Display", "Podkova", "Port Lligat Slab", "Quattrocento", "Radley", "Rokkitt", "Ruslan Display", "Rye",
    "Scope One", "Source Serif Pro", "Stint Ultra Expanded", "Suez One", "Superclarendon", "Tinos", "Vast Shadow",
    "Vollkorn", "Yeseva One",
}
TYPES = ["SvgElement", "TextElement", "ImageElement", "ColoredBackground", "SvgMaskElement"]
ID_PREFIX = {"SvgElement": "svg", "TextElement": "text", "ImageElement": "image", "ColoredBackground": "bg",
             "SvgMaskElement": "mask"}
ALIGN = ["left", "left", "center", "right"]  # Crello's "" (unset) is left


def font_names(pf):
    meta = json.loads(pf.schema_arrow.metadata[b"huggingface"])["info"]["features"]
    return meta["font"]["feature"]["names"], meta["format"]["names"]


def _coverage():
    import skia
    faces = {"Solstice Serif": "SolsticeSerif-Regular.ttf", "Solstice Sans": "SolsticeSans-Regular.ttf"}
    return {fam: skia.Font(skia.Typeface.MakeFromFile(str(REPO / "brand_kit" / "fonts" / f)), 12) for fam, f in faces.items()}


_FONTS = None


def missing_glyphs(text, fam):
    """Characters the mapped font can't draw (they would render as boxes)."""
    global _FONTS
    _FONTS = _FONTS or _coverage()
    chars = sorted({c for c in text if not c.isspace()})
    return [c for c, g in zip(chars, _FONTS[fam].textToGlyphs("".join(chars))) if g == 0]


def family(crello_font):
    return "Solstice Serif" if crello_font in SERIF else "Solstice Sans"


def convert(row, fonts, formats):
    """One parquet row -> (design, meta, {asset_id: PIL image}). Raises on anything unconvertible."""
    cw, ch = int(row["canvas_width"]), int(row["canvas_height"])
    els, images, used, dropped, refit, flat = [], {}, Counter(), Counter(), [], Counter()
    for i in range(row["length"]):
        kind = TYPES[row["type"][i]]
        x, y, w, h = (float(row[k][i]) for k in ("left", "top", "width", "height"))
        rot = float(row["angle"][i] or 0.0)
        rot = (rot + 180) % 360 - 180
        if abs(rot - 90 * round(rot / 90)) < 0.5:  # editor noise (0.01 deg, 89.9 deg), not a design choice
            rot = float((90 * round(rot / 90) + 180) % 360 - 180)
        if w <= 0 or h <= 0 or x >= cw or y >= ch or x + w <= 0 or y + h <= 0:
            dropped["off_canvas_or_empty"] += 1
            continue
        el = {"id": f"{ID_PREFIX[kind]}{i}", "x": x, "y": y, "width": w, "height": h, "z_index": i}
        if rot:
            el["rotation"] = rot
        if row["opacity"][i] is not None and row["opacity"][i] < 1:
            el["opacity"] = max(0.0, float(row["opacity"][i]))
        if kind == "TextElement":
            text = row["text"][i] or ""
            if not text.strip():
                dropped["empty_text"] += 1
                continue
            if row["capitalize"][i]:
                text = text.upper()
            name = fonts[row["font"][i]]
            used[name] += 1
            el.update(type="text", content=text, font_family=family(name), font_size=max(1.0, float(row["font_size"][i])),
                      text_align=ALIGN[row["text_align"][i]], line_height=float(row["line_height"][i] or 1.2),
                      letter_spacing=float(row["letter_spacing"][i] or 0.0))
            _text_styles(el, row["text_color"][i], row["font_bold"][i], row["font_italic"][i])
            lines = max(row["text_line"][i] or [0]) + 1
            f = _fit_font(el, lines)
            if f < 1:
                refit.append(round(f, 3))
        elif kind == "ColoredBackground" and row["color"][i]:
            el.update(type="shape", shape_type="rectangle", fill_color=_rgba(row["color"][i][0]))
        else:
            raw = row["image"][i]["bytes"] if row["image"][i] else None
            if not raw:
                dropped["no_pixels"] += 1
                continue
            img = Image.open(io.BytesIO(raw)).convert("RGBA")
            if not rot:
                img, box = _clip(img, (x, y, w, h), cw, ch)
                if img is None:
                    dropped["off_canvas_or_empty"] += 1
                    continue
                el.update(x=box[0], y=box[1], width=box[2], height=box[3])
            fill = _flat_fill(img)
            if fill:  # a flat colour panel: a designer's rectangle, free to stretch
                el.update(type="shape", shape_type="rectangle", fill_color=fill)
                flat[kind] += 1
                els.append(el)
                continue
            asset_id = f"crello_{row['id']}_{i}.png"
            images[asset_id] = img
            from eval.check import rotated_bbox
            d = rotated_bbox(el) if rot else (el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])
            fills = d[0] <= 0.01 * cw and d[1] <= 0.01 * ch and d[2] >= 0.99 * cw and d[3] >= 0.99 * ch  # as drawn
            el.update(type="image", asset_id=asset_id, fit="cover" if fills else "contain")
        els.append(el)
    design = {"canvas": {"width": cw, "height": ch}, "elements": els}
    meta = {"id": row["id"], "format": formats[row["format"]], "elements": len(els), "aspect": round(cw / ch, 4),
            "fonts": dict(used), "dropped": dict(dropped), "font_refit": refit, "flat_to_shape": dict(flat)}
    return design, meta, images


def _fit_font(el, lines, floor=0.5):
    """The substitute font is often wider than Crello's (a condensed display face -> our sans):
    shrink the font, 2% at a time down to `floor`, until the text wraps to no more lines than
    in Crello and fits its box's width, as it did there. Returns the factor applied."""
    from eval.check import text_layout
    fs0 = el["font_size"]
    f = 1.0
    while f > floor:
        lay = text_layout(el)
        if lay["lines"] <= lines and lay["right"] - lay["left"] <= el["width"] + 0.5:
            break
        f -= 0.02
        el["font_size"] = max(1.0, round(fs0 * f, 2))
    return el["font_size"] / fs0


FLAT_STD = 4.0        # RGB std (0-255) under which a crop is one flat colour
FLAT_COVER = 0.98     # ...and at least this share of its pixels carry the same alpha (a full rectangle)


def _flat_fill(img):
    """'rgba(r,g,b,a)' when the crop is a rectangle of one colour (opaque or evenly translucent), else None.
    Photos, textures, gradients and shaped cut-outs (circles, blobs, rounded cards) stay images."""
    import numpy as np
    px = np.asarray(img.resize((min(64, img.width), min(64, img.height))), dtype=float).reshape(-1, 4)
    a = px[:, 3]
    if a.max() < 8 or np.mean(np.abs(a - np.median(a)) <= 8) < FLAT_COVER:
        return None
    rgb = px[a >= 8, :3]
    if rgb.std(axis=0).max() > FLAT_STD:
        return None
    r, g, b = (int(round(v)) for v in rgb.mean(axis=0))
    return f"rgba({r},{g},{b},{round(float(np.median(a)) / 255, 3)})"


def _rgba(c):
    return c.replace(" ", "")


def _text_styles(el, colors, bold, italic):
    """Most common colour / weight / style per character -> the element's; differing runs -> style_runs."""
    n = len(el["content"])
    def per_char(xs, default):  # Crello's per-character lists can be shorter than the text
        xs = list(xs or [default])[:n]
        return xs + [xs[-1]] * (n - len(xs))
    chars = list(zip(per_char([_rgba(c) for c in colors or []], "rgba(0,0,0,1)"), per_char(bold, False), per_char(italic, False)))
    base = tuple(Counter(c[k] for c in chars).most_common(1)[0][0] for k in range(3))
    el.update(color=base[0], font_weight="bold" if base[1] else "regular", font_style="italic" if base[2] else "normal")
    runs, start = [], 0
    for k in range(1, n + 1):
        if k == n or chars[k] != chars[start]:
            c = chars[start]
            if c != base:
                run = {"start": start, "end": k}
                if c[0] != base[0]:
                    run["color"] = c[0]
                if c[1] != base[1]:
                    run["font_weight"] = "bold" if c[1] else "regular"
                if c[2] != base[2]:
                    run["font_style"] = "italic" if c[2] else "normal"
                runs.append(run)
            start = k
    if runs:
        el["style_runs"] = runs


def _clip(img, box, cw, ch):
    """Cut an image (and its box) to the part inside the canvas."""
    x, y, w, h = box
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, cw), min(y + h, ch)
    if (x0, y0, x1, y1) == (x, y, x + w, y + h):
        return img, box
    sx, sy = img.width / w, img.height / h
    crop = (round((x0 - x) * sx), round((y0 - y) * sy), round((x1 - x) * sx), round((y1 - y) * sy))
    if crop[2] - crop[0] < 1 or crop[3] - crop[1] < 1:
        return None, None
    # the box is the exact visible part; the crop is the nearest whole pixels (aspect off by < 1 px)
    return img.crop(crop), (x0, y0, x1 - x0, y1 - y0)


def run(compare=0):
    (OUT / "designs").mkdir(parents=True, exist_ok=True)
    (OUT / "assets").mkdir(exist_ok=True)
    (OUT / "previews").mkdir(exist_ok=True)
    skipped, fonts_used, kept = Counter(), Counter(), []
    font_list = None
    for shard in sorted(DATA.glob(os.environ.get("CRELLO_SHARDS") or "validation-*.parquet")):
        pf = pq.ParquetFile(shard)
        fonts, formats = font_names(pf)
        font_list = fonts
        for g in range(pf.num_row_groups):
            for row in pf.read_row_group(g, columns=COLUMNS + ["preview"]).to_pylist():
                try:
                    design, meta, images = convert(row, fonts, formats)
                except Exception:
                    skipped["conversion_error"] += 1
                    print(f"{row['id']}: {traceback.format_exc(limit=1)}")
                    continue
                n = len(design["elements"])
                texts = [e for e in design["elements"] if e["type"] == "text"]
                reason = ("too_few_elements" if n < MIN_ELEMENTS else "too_many_elements" if n > MAX_ELEMENTS
                          else "no_text" if not texts
                          else "unsupported_characters" if any(missing_glyphs(e["content"], e["font_family"]) for e in texts)
                          else None)
                if reason is None:
                    try:
                        jsonschema.validate(design, SCHEMA)
                    except jsonschema.ValidationError as e:
                        reason = "schema_violation"
                        print(f"{row['id']}: {e.message}")
                if reason:
                    skipped[reason] += 1
                    continue
                for asset_id, img in images.items():
                    img.save(OUT / "assets" / asset_id)
                (OUT / "previews" / f"{row['id']}.png").write_bytes(row["preview"]["bytes"])
                (OUT / "designs" / f"{row['id']}.json").write_text(json.dumps({"source": design, "meta": meta}))
                fonts_used.update(meta["fonts"])
                kept.append(meta)
    mapping = {f: family(f) for f in font_list if f}
    FONTS.write_text(json.dumps({"_note": "Crello font family -> the closest available Chameleon font (serif or sans). "
                                          "Written by eval/crello_convert.py.", **mapping}, indent=1) + "\n")
    refits = [f for m in kept for f in m["font_refit"]]
    n_texts = sum(m["fonts"][f] for m in kept for f in m["fonts"])
    report = {"converted": len(kept), "skipped": dict(skipped), "skipped_total": sum(skipped.values()),
              "font_refit": {"texts": n_texts, "shrunk": len(refits),
                             "median_factor": sorted(refits)[len(refits) // 2] if refits else 1.0,
                             "at_floor": sum(f <= 0.51 for f in refits)},
              "dropped_elements": dict(sum((Counter(m["dropped"]) for m in kept), Counter())),
              "flat_to_shape": {"elements": dict(sum((Counter(m["flat_to_shape"]) for m in kept), Counter())),
                                "designs": sum(bool(m["flat_to_shape"]) for m in kept)},
              "formats": dict(Counter(m["format"] for m in kept).most_common()),
              "fonts_used": {f: {"elements": n, "mapped_to": family(f)} for f, n in fonts_used.most_common()}}
    (OUT / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: report[k] for k in ("converted", "skipped_total", "skipped", "dropped_elements", "font_refit", "flat_to_shape")}, indent=1))
    if compare:
        compare_sheet(kept, compare)
    return report


def compare_sheet(kept, n, seed=0):
    """Our render of n converted designs next to Crello's preview, one PNG each plus a sheet.
    Designs with flat panels turned into shapes come first: they are the ones the conversion changes most."""
    import random
    os.environ["CHAMELEON_ASSETS"] = str(OUT / "assets")
    from eval.render import render_image
    (OUT / "compare").mkdir(exist_ok=True)
    pairs = []
    flat = [m for m in kept if m["flat_to_shape"]]
    for m in random.Random(seed).sample(flat if len(flat) >= n else kept, min(n, len(kept))):
        design = json.loads((OUT / "designs" / f"{m['id']}.json").read_text())["source"]
        ours = render_image(design)
        theirs = Image.open(OUT / "previews" / f"{m['id']}.png").convert("RGB")
        h = 400
        a = ours.resize((max(1, round(ours.width * h / ours.height)), h))
        b = theirs.resize((max(1, round(theirs.width * h / theirs.height)), h))
        pair = Image.new("RGB", (a.width + b.width + 10, h), "white")
        pair.paste(a, (0, 0))
        pair.paste(b, (a.width + 10, 0))
        pair.save(OUT / "compare" / f"{m['id']}.png")
        pairs.append(pair)
    cols, cell = 4, 820
    sheet = Image.new("RGB", (cols * cell, ((len(pairs) + cols - 1) // cols) * 420), "white")
    for k, p in enumerate(pairs):
        p.thumbnail((cell - 20, 400))
        sheet.paste(p, ((k % cols) * cell, (k // cols) * 420))
    sheet.save(OUT / "compare" / "_sheet.png")
    print(f"compare: {OUT / 'compare'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", type=int, default=0, help="render N designs next to Crello's previews")
    run(ap.parse_args().compare)

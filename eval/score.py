"""Geometric quality score for resize() outputs (no rendering, except the
optional contrast check).

`score(source, target_canvas, output, contrast=False)` returns
    {"total": 0..100 (higher is better),
     "breakdown": {name: {"penalty": 0..1, "weight": w, "detail": str}}}
with total = 100 * (1 - sum(w * penalty) / sum(w)).

Each penalty is one function below, registered in PENALTIES with its weight,
so they can be tuned independently. All geometry uses what is actually drawn,
not the raw box: text = its laid-out lines (eval.check.text_layout, which
matches the renderer), contain images = the letterboxed asset area.

Roles are decided from the SOURCE and carried to the output by id:
  background  = element that fills the source canvas (excluded everywhere)
  decoration  = ungrouped shape with effective alpha < 0.5 (excluded everywhere)
  unit        = a group (all members merged) or an ungrouped foreground element
"""

import json
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from PIL import Image

from eval.check import _font_manager, rotated_bbox, text_layout
from eval.render import assets_dir
from renderer.converter.colorutil import parse_rgba

# --- tunables -----------------------------------------------------------------
MIN_TEXT_PX_ABS = 12         # never demand less than this (the samples' own fine print is 12px)
MIN_TEXT_REL_HEIGHT = 0.015  # min readable font size as a fraction of canvas height
DECORATION_ALPHA = 0.5
EMPTY_CELL_COVERAGE = 0.02   # a ninth of the canvas with less foreground than this is "empty"
EMPTY_NINTHS_FREE = 4        # this many empty ninths of the target are fine (margins, breathing room)
MIN_EXTENT = 0.6             # content should span at least this fraction of the target on each axis
HEADLINE_LEAD = 1.4          # the headline should look at least this much bigger than any other text...
                             # ...unless it led by less in the source
FLOAT_MAX_AREA = 0.1         # a corner-anchored unit smaller than this fraction of the canvas is floating
FLOAT_CORNER = 0.12          # "anchored" = within this fraction of the canvas side from two adjacent edges
FLOAT_DRIFT_FULL = 0.3       # drift from its corner of 30% of the canvas side = full penalty
ATTACH_MIN = 0.15            # a floating unit overlapping a bigger unit by this share of its area is attached to it
TILT_FLIP = 0.5              # a rotated unit whose tilt no longer points the way it did, relative to its anchor
ACCENT_MIN_CHROMA = 30       # CIELAB chroma: neutrals (white, black, greys) never count as colour emphasis
ACCENT_MIN_DELTA = 35        # CIELAB distance from every main-palette colour
MAIN_SHARE = 0.2             # a colour covering this share of the painted area is part of the main palette
SAME_COLOUR = 15             # CIELAB distance under which two colours are one palette entry
FOCUS_OFF_ZONE = 0.25        # emphasised unit's centre this far from its column's centre (x column width) = full penalty
ROW_GAP = 0.1                # side-by-side units closer than this (x canvas width) form one row block
FOCUS_OFF_CANVAS = 0.35      # ...or, alone in its column, this far from the canvas centre (x canvas width)
FOCUS_FREE = 0.05            # off-centre by less than this is centred
SCALE_FREE, SCALE_FULL = 1.0, 0.4  # headline/hero at this share of the size the target affords: no / full penalty
BACKDROP_ON = 0.95           # a text this much inside one element (a panel, card, shape or image) sits on it...
BACKDROP_MAX = 0.4           # ...if that element is compact (under this share of the canvas): a card, label or panel.
                             # Text on a large photo or illustration may leave it; contrast_drops still guards it
BACKDROP_KEEP = 0.9          # ...and a candidate where it is less than this much on it is rejected
CONTRAST_GRID = (32, 8)      # points sampled under a text's box (across, down)
CONTRAST_LOW = 25            # a text's contrast: this percentile of the WCAG ratios at those points
CONTRAST_DROP = 0.5          # rejected when it falls below this share of the source's contrast...
CONTRAST_OK = 3.0            # ...and below WCAG's large-text minimum (never asked to beat the source)
LARGE_SHARE = 0.1            # an element covering this share of the source canvas is large...
SHARE_FLOOR = 0.5            # ...and no candidate may leave it less than this share of the share it can keep
DRIFT_FULL = math.log(4)     # an element at 4x (or 1/4) the share it can keep: full share drift
REL_BIG = 0.4                # elements covering this share of the canvas are backdrops: no relationships
REL_OVERLAP = 0.2            # overlapping by this share of the smaller one's area is an overlap pair (designed
                             # stacking; a few percent is incidental, e.g. a subhead's box touching a button)
REL_TIGHT, REL_LOOSE = 0.35, 0.7  # close neighbours: their gap under this many times their gap to anything else
REL_BLOCK_MAX = 0.5          # a proximity cluster may not grow past this share of the canvas
REL_LOOSEN_ASPECT = 1.5     # from this aspect change on, something overlapping a picture may be moved off it
REL_KEPT = 0.2               # a pair is kept when its relative offset moved by at most this (x pair size)
TIGHT_IMAGE = 0.05            # an image under this share of the canvas next to a text is one thing with it
LABEL_WORDS, LABEL_CHARS = 4, 32  # a label: at most this many words or characters, or ending in ':'
OVER_FREE, OVER_FULL = 0.05, 0.5  # share of a text newly on top of an image: no / full penalty
BUSY_COVER = 0.2             # a text newly this much on top of an image...
BUSY = 0.02                  # ...whose pixels under it vary this much (mean |gradient|, 0-1 grey) is rejected outright
                             # (text designers put on images: median 0.000, 75th percentile 0.008; the headline
                             # retrieval put on bamboo 0.035, on a photo 0.057)
KEY_FREE, KEY_FULL = 0.85, 0.5  # headline/main image at this share of the scale the target affords: no / full penalty
HEADLINE_WRAPS = tuple(1 - 0.05 * i for i in range(17))  # widths tried, as shares of the one-line width
KEY_SQUEEZE_FREE, KEY_SQUEEZE_FULL = 0.6, 0.25  # a cropped main photo's box aspect vs the source's (smaller/larger)
HERO_MIN_AREA = 0.05         # the largest foreground image is the hero when it covers this share of the source
HOLDER_MAX_AREA = 0.5       # a card/bubble/label/button holding text covers less than this share of the canvas
HOLDER_MAX_ASPECT = 4.0      # ...and, if an image, is at most this elongated (a long bar is a backdrop, not a card)
SIBLING_HEIGHT = 1.25        # sibling units' heights within this ratio of each other
SIBLING_SCALE = 0.1          # siblings in the output scaled within 10% of each other
SIBLING_RESTACK = 0.25       # kept together but no longer a row: worth it only when the row would be unreadable
SIBLING_SPREAD = 2.0         # their union box at most this times their summed area (or the source's, if looser)
FILL_FREE, FILL_FULL = 0.6, 0.25  # content box's share of the target vs of the source: no / full penalty
GAP_FREE, GAP_FULL = 0.1, 0.3  # an empty band between content: free up to 10% of the side, full penalty at 30%
COVER_GRID = 48               # points per side sampled for canvas coverage
COVER_SRC = 0.2              # a source painted over at least this share of its canvas (a panel, a half-bleed photo)...
COVER_FULL = 0.15            # ...and this share of the target newly showing bare canvas = full penalty
BALANCE_FULL = 0.25          # content centre this much further off-centre than in the source = full penalty
NEAR_MISS_REL = 0.02         # edges closer than 2% of the canvas side, but not aligned, are a near miss
ALIGNED_PX = 1.0
CONTRAST_MIN, CONTRAST_MIN_LARGE = 4.5, 3.0  # WCAG AA


@dataclass
class Ctx:
    source: dict
    output: dict
    src: dict          # id -> source element
    out: dict          # id -> output element
    src_box: dict      # id -> drawn box in the source (computed once: text layout is the slow part)
    out_box: dict      # id -> drawn box in the output
    units: list        # list of lists of ids (foreground only)
    sw: float
    sh: float
    tw: float
    th: float
    anchors: dict = None  # tuple(unit) -> None | ("corner", "top-left"...) | ("host", host unit)


# --- penalties ---------------------------------------------------------------

def overlap_penalty(c: Ctx):
    """New overlaps between foreground units. For each pair, overlap ratio =
    intersection / smaller unit's area. Only the increase over the same
    pair's ratio in the source counts, so designed layering (text on a
    shape, product over a circle) is free. Penalty = sum of increases, capped at 1."""
    total, worst = 0.0, []
    for a, b in combinations(c.units, 2):
        r_src = _overlap_ratio(_unit_box(c.src_box, a), _unit_box(c.src_box, b))
        r_out = _overlap_ratio(_unit_box(c.out_box, a), _unit_box(c.out_box, b))
        if r_out - r_src > 0.01:
            total += r_out - r_src
            worst.append(f"{_name(a)}~{_name(b)} +{r_out - r_src:.2f}")
    return min(1.0, total), ", ".join(worst)


def holders_penalty(c: Ctx):
    """Text that sat on a card, a speech bubble, a label or a button (a shape, or a compact
    image: at most HOLDER_MAX_ASPECT elongated, under HOLDER_MAX_AREA of the canvas) must stay
    on it. Per (text, holder) pair: how much of the text left it (inside >= 90% is fine).
    Penalty = mean over pairs."""
    pairs = _holders(c)
    if not pairs:
        return 0.0, ""
    pens, off = [], []
    for t, h in pairs:
        p = min(1.0, max(0.0, (0.9 - _inside(c.out_box[t], c.out_box[h])) / 0.9))
        pens.append(p)
        if p > 0.2:
            off.append(f"{t} off {h}")
    return sum(pens) / len(pens), ", ".join(off)


def _holders(c):
    pairs = []
    for t, tb in c.src_box.items():
        if c.src[t]["type"] != "text":
            continue
        best = None
        for h, hb in c.src_box.items():
            e = c.src[h]
            if h == t or e["type"] == "text" or _area(hb) >= HOLDER_MAX_AREA * c.sw * c.sh or _area(hb) <= _area(tb):
                continue
            w, ht = hb[2] - hb[0], hb[3] - hb[1]
            if e["type"] == "image" and max(w, ht) > HOLDER_MAX_ASPECT * max(min(w, ht), 1e-6):
                continue
            if _inside(tb, hb) >= 0.9 and (best is None or _area(hb) < _area(c.src_box[best])):
                best = h
        if best:
            pairs.append((t, best))
    return pairs


def _inside(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3]))) / max(_area(a), 1e-6)


def small_text_penalty(c: Ctx):
    """Text rendered smaller than max(MIN_TEXT_PX_ABS, MIN_TEXT_REL_HEIGHT *
    canvas height), but never demanding more than the element's source size
    (fine print stays fine print). Uses int(font_size), the size the renderer
    actually draws. Penalty = mean relative shortfall over text elements."""
    floor = max(MIN_TEXT_PX_ABS, MIN_TEXT_REL_HEIGHT * c.th)
    shortfalls, small = [], []
    for eid, el in c.out.items():
        if el["type"] != "text":
            continue
        need = min(floor, c.src[eid]["font_size"])
        drawn = int(el["font_size"])
        shortfalls.append(max(0.0, (need - drawn) / need))
        if drawn < need:
            small.append(f"{eid} {drawn}px<{need:.0f}")
    return (sum(shortfalls) / len(shortfalls) if shortfalls else 0.0), ", ".join(small)


def headline_penalty(c: Ctx):
    """The source's visually biggest text must stay clearly the biggest.
    Visual size = x-height of the drawn font x sqrt(stroke weight / 400), so a
    14px regular serif no longer "beats" a 13px bold sans. Lead = headline's
    visual size / the largest other text's. Needed lead = min(HEADLINE_LEAD,
    source lead); penalty = shortfall from it, 0 at the needed lead, 1 at parity or below."""
    texts = [i for i, e in c.src.items() if e["type"] == "text" and i in c.out]
    if len(texts) < 2:
        return 0.0, ""
    vs = {i: _visual_size(c.src[i]) for i in texts}
    head = max(texts, key=lambda i: (vs[i], c.src[i]["width"] * c.src[i]["height"]))
    need = min(HEADLINE_LEAD, vs[head] / max(max(v for i, v in vs.items() if i != head), 1e-6))
    if need <= 1:
        return 0.0, ""
    vo = {i: _visual_size(c.out[i]) for i in texts}
    other = max((i for i in texts if i != head), key=vo.get)
    lead = vo[head] / max(vo[other], 1e-6)
    p = min(1.0, max(0.0, (need - lead) / (need - 1)))
    return p, (f"headline {head} leads {other} by {lead:.2f}x (needs {need:.2f}x)" if p else "")


def scale_penalty(c: Ctx):
    """The headline and the hero image should keep their size: each one's output
    scale over the scale the target affords it, which is the area ratio's square
    root (the scale that keeps the design's share of the canvas), or less when the
    unit itself would not fit the target at that scale. A ratio of SCALE_FREE or more
    costs nothing, SCALE_FULL or less costs 1. Mean over the two."""
    area = math.sqrt(c.tw * c.th / (c.sw * c.sh))
    texts = [i for i, e in c.src.items() if e["type"] == "text" and i in c.out]
    imgs = [u for u in c.units if len(u) == 1 and c.src[u[0]]["type"] == "image"
            and _area(c.src_box[u[0]]) >= HERO_MIN_AREA * c.sw * c.sh]
    pens, notes = [], []
    for i in [max(texts, key=lambda i: _visual_size(c.src[i]))] if texts else []:
        e = c.src[i]
        r = c.out[i]["font_size"] / e["font_size"] / min(area, c.tw / e["width"], c.th / max(e["height"], 1e-6))
        pens.append((i, r))
    for u in [max(imgs, key=lambda u: _area(c.src_box[u[0]]))] if imgs else []:
        sb, ob = c.src_box[u[0]], c.out_box[u[0]]
        w, h = sb[2] - sb[0], sb[3] - sb[1]
        r = max((ob[2] - ob[0]) / max(w, 1e-6), (ob[3] - ob[1]) / max(h, 1e-6)) / min(area, c.tw / w, c.th / h)
        pens.append((u[0], r))
    out = []
    for i, r in pens:
        p = min(1.0, max(0.0, (SCALE_FREE - r) / (SCALE_FREE - SCALE_FULL)))
        out.append(p)
        if p:
            notes.append(f"{i} at {r:.2f} of its size")
    return (sum(out) / len(out) if out else 0.0), ", ".join(notes)


def text_over_image_penalty(c: Ctx):
    """Text (alone or in a group) newly placed on top of a non-background image breaks a
    relationship the design did not have: per text, the share of its drawn box covered by
    images in the output, minus the same share in the source (each image counted separately,
    capped at 1). OVER_FREE or less is free, OVER_FULL or more costs 1. Penalty = the worst text."""
    worst, note = 0.0, ""
    for t, more, img in _new_text_over_image(c):
        p = min(1.0, max(0.0, (more - OVER_FREE) / (OVER_FULL - OVER_FREE)))
        if p > worst:
            worst, note = p, f"{t} newly {more:.0%} on {img}"
    return worst, note


def _new_text_over_image(c):
    """[(text id, newly covered share of its drawn box, the image covering most of it)]."""
    imgs = [i for i, e in c.src.items() if e["type"] == "image" and i in c.out and not _is_background(e, c.sw, c.sh)]
    out = []
    for t, e in c.src.items():
        if e["type"] != "text" or t not in c.out:
            continue
        sa, oa = max(_area(c.src_box[t]), 1e-9), max(_area(c.out_box[t]), 1e-9)
        incs = {i: _inter(c.out_box[t], c.out_box[i]) / oa - _inter(c.src_box[t], c.src_box[i]) / sa for i in imgs}
        more = min(1.0, sum(v for v in incs.values() if v > 0))
        if more > 0:
            out.append((t, more, max(incs, key=incs.get)))
    return out


def busy_text_over_image(source, target_canvas, output):
    """[(text id, image id)] where a text newly covers BUSY_COVER of itself with one image and that
    image is busy under it: the engine rejects such a candidate (a headline on a photo or plant it
    was never on). Rotation is ignored when mapping the text onto the image's pixels."""
    c = _context(source, target_canvas, output)
    els = {e["id"]: e for e in output["elements"]}
    bad = []
    for t, _more, _img in _new_text_over_image(c):
        for i, e in c.src.items():
            if e["type"] != "image" or i not in c.out or _is_background(e, c.sw, c.sh):
                continue
            ob = c.out_box[t]
            inc = _inter(ob, c.out_box[i]) / max(_area(ob), 1e-9) - _inter(c.src_box[t], c.src_box[i]) / max(_area(c.src_box[t]), 1e-9)
            if inc >= BUSY_COVER and _busyness(els[i], ob) > BUSY:
                bad.append((t, i))
    return bad


@lru_cache(maxsize=128)
def _asset_grey(asset_id):
    """(grey 0-1, alpha 0-1, scale) of an asset at most 256 px on its long side; None if missing."""
    try:
        with Image.open(assets_dir() / asset_id) as im:
            im = im.convert("RGBA")
            k = min(1.0, 256 / max(im.size))
            small = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))))
    except (OSError, TypeError):
        return None
    a = np.asarray(small, dtype=np.float32) / 255
    return a[..., :3].mean(axis=2), a[..., 3], k


def _busyness(el, region):
    """Mean |gradient| of the image's pixels under `region` (canvas coords), with transparent pixels
    counting as flat, mapped through the renderer's fit (contain letterboxes, cover centre-crops)."""
    g = _asset_grey(el.get("asset_id"))
    if g is None:
        return 0.0
    grey, alpha, k = g
    ih, iw = grey.shape
    x, y, w, h = el["x"], el["y"], el["width"], el["height"]
    fit = el.get("fit", "contain")
    if fit in ("contain", "cover"):
        s = (min if fit == "contain" else max)(w / iw, h / ih)
        sx = sy = s
    else:
        sx, sy = w / iw, h / ih
    ox, oy = x + (w - iw * sx) / 2, y + (h - ih * sy) / 2
    c0, c1 = int(max(0, (region[0] - ox) / sx)), int(min(iw, math.ceil((region[2] - ox) / sx)))
    r0, r1 = int(max(0, (region[1] - oy) / sy)), int(min(ih, math.ceil((region[3] - oy) / sy)))
    if c1 - c0 < 2 or r1 - r0 < 2:
        return 0.0
    v = grey[r0:r1, c0:c1] * alpha[r0:r1, c0:c1]
    return float((np.abs(np.diff(v, axis=1))[:-1] + np.abs(np.diff(v, axis=0))[:, :-1]).mean())


def _inter(a, b):
    return _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def text_off_backdrop(source, target_canvas, output):
    """[(text id, backdrop id, share of the text on it)] for texts that sat on an element in the source
    (>= BACKDROP_ON inside the smallest non-text, non-background element under it) and are now less than
    BACKDROP_KEEP on it: white text off its dark panel, a label off its card. Any generator; the engine
    rejects such candidates."""
    out = {e["id"]: e for e in output["elements"]}
    bad = []
    for t, b in _backdrops(json.dumps(source, sort_keys=True)):
        if t in out and b in out:
            tb = _drawn_box(out[t])
            share = _inter(tb, _drawn_box(out[b])) / max(_area(tb), 1e-9)
            if share < BACKDROP_KEEP:
                bad.append((t, b, share))
    return bad


@lru_cache(maxsize=64)
def _backdrops(source_json):
    """((text id, the element it sits on), ...) in the source."""
    source = json.loads(source_json)
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    order = {e["id"]: (e.get("z_index", 0), n) for n, e in enumerate(source["elements"])}
    pairs = []
    for t in source["elements"]:
        if t["type"] != "text":
            continue
        tb = _drawn_box(t)
        under = [e for e in source["elements"] if e["type"] != "text" and order[e["id"]] < order[t["id"]]
                 and not _is_background(e, sw, sh) and _inter(tb, _drawn_box(e)) >= BACKDROP_ON * max(_area(tb), 1e-9)]
        under = [e for e in under if _area(_drawn_box(e)) < BACKDROP_MAX * sw * sh]
        if under:
            pairs.append((t["id"], min(under, key=lambda e: _area(_drawn_box(e)))["id"]))
    return tuple(pairs)


def contrast_drops(source, target_canvas, output):
    """[(text id, source contrast, output contrast)] for texts whose contrast against what lies under
    them fell below CONTRAST_DROP of the source's and below CONTRAST_OK. Contrast is estimated without
    rendering: the layers under the text's box (canvas, then every shape and image below it, images
    sampled from their asset through their fit) are composited at CONTRAST_GRID points, and the text's
    colour is compared with each (WCAG ratio); the CONTRAST_LOW percentile is its contrast."""
    src = _text_contrasts(json.dumps(source, sort_keys=True))
    out = _text_contrasts(json.dumps(output, sort_keys=True))
    return [(t, src[t], out[t]) for t in src if t in out and out[t] < CONTRAST_DROP * src[t] and out[t] < CONTRAST_OK]


def contrast_kept(source, target_canvas, output):
    """Mean over texts of min(1, output contrast / source contrast), for ranking when every candidate
    loses some (the worst text alone ties too often: one text at 1:1 in each)."""
    src = _text_contrasts(json.dumps(source, sort_keys=True))
    out = _text_contrasts(json.dumps(output, sort_keys=True))
    kept = [min(1.0, out[t] / src[t]) for t in src if t in out]
    return sum(kept) / len(kept) if kept else 1.0


def contrast_worst(source, target_canvas, output):
    """The least any text kept of its source contrast (min over texts of min(1, output / source))."""
    src = _text_contrasts(json.dumps(source, sort_keys=True))
    out = _text_contrasts(json.dumps(output, sort_keys=True))
    return min((min(1.0, out[t] / src[t]) for t in src if t in out), default=1.0)


@lru_cache(maxsize=256)
def _text_contrasts(design_json):
    design = json.loads(design_json)
    els = sorted(enumerate(design["elements"]), key=lambda t: (t[1].get("z_index", 0), t[0]))  # paint order
    res = {}
    for n, (_k, t) in enumerate(els):
        if t["type"] != "text" or t.get("id") in res:
            continue
        x0, y0, x1, y1 = _drawn_box(t)
        if x1 <= x0 or y1 <= y0:
            continue
        gx, gy = np.meshgrid(np.linspace(x0, x1, CONTRAST_GRID[0]), np.linspace(y0, y1, CONTRAST_GRID[1]))
        px, py = gx.ravel(), gy.ravel()
        col = np.ones((px.size, 3))  # uncovered canvas renders white
        for _k2, e in els[:n]:
            if e["type"] == "text" or not _inter((x0, y0, x1, y1), (e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"])):
                continue  # shortcut: rotation is ignored here; a tilted panel is sampled as its unrotated box
            rgb, a = _layer(e, px, py)
            if rgb is not None:
                col = col * (1 - a[:, None]) + rgb * a[:, None]
        r, g, b, a = parse_rgba(t.get("color"))
        fg = a * np.array([r, g, b]) / 255 + (1 - a) * col
        lf, lb = _rel_luminance(fg), _rel_luminance(col)
        res[t["id"]] = float(np.percentile((np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05), CONTRAST_LOW))
    return res


def _layer(e, px, py):
    """(rgb 0-1 per point, alpha per point) of one shape or image at canvas points, or (None, None)."""
    x, y, w, h = e["x"], e["y"], e["width"], e["height"]
    if w <= 0 or h <= 0:
        return None, None
    op = e.get("opacity", 1)
    if e["type"] == "shape":
        if e.get("shape_type") == "line":
            return None, None
        r, g, b, fa = parse_rgba(e.get("fill_color"), default=(200, 200, 200, 1.0))
        if e.get("shape_type") == "ellipse":
            inside = ((px - x - w / 2) / (w / 2)) ** 2 + ((py - y - h / 2) / (h / 2)) ** 2 <= 1
        else:
            inside = (px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)
        return np.tile(np.array([r, g, b]) / 255, (px.size, 1)), inside * fa * op
    if e["type"] != "image":
        return None, None
    img = _asset_rgba(e.get("asset_id"))
    if img is None:
        return None, None
    ih, iw = img.shape[:2]
    fit = e.get("fit", "contain")
    if fit in ("contain", "cover"):
        sx = sy = (min if fit == "contain" else max)(w / iw, h / ih)
    else:
        sx, sy = w / iw, h / ih
    ox, oy = x + (w - iw * sx) / 2, y + (h - ih * sy) / 2
    cx, cy = np.floor((px - ox) / sx).astype(int), np.floor((py - oy) / sy).astype(int)
    inside = (cx >= 0) & (cx < iw) & (cy >= 0) & (cy < ih) & (px >= x) & (px <= x + w) & (py >= y) & (py <= y + h)
    pix = img[np.clip(cy, 0, ih - 1), np.clip(cx, 0, iw - 1)]
    return pix[:, :3], pix[:, 3] * inside * op


@lru_cache(maxsize=128)
def _asset_rgba(asset_id):
    """RGBA 0-1 of an asset at most 256 px on its long side (scaled back by the caller's fit), or None."""
    try:
        with Image.open(assets_dir() / asset_id) as im:
            im = im.convert("RGBA")
            w, h = im.size
            k = min(1.0, 256 / max(w, h))
            # the small copy keeps the original's proportions; callers map with its own size
            return np.asarray(im.resize((max(1, round(w * k)), max(1, round(h * k)))), dtype=np.float32) / 255
    except (OSError, TypeError):
        return None


def share_drift_penalty(c: Ctx):
    """Each element's share of the canvas should not fall below its share of the source: per element,
    ln(the share it can keep / output share) / DRIFT_FULL, 0 when it grew, capped at 1, where the
    share it can keep is its source share, or less for an image whose aspect can't fill that much
    of the target (text re-wraps, so it can). Penalty = mean weighted by source share: large
    elements count most, small ones barely. Growth is free: penalising it as well agreed less with
    the hand labels (pooled 0.849 vs 0.867), which favour an enlarged headline or product."""
    rows = _shares(c)
    if not rows:
        return 0.0, ""
    total = sum(src for _i, src, _out, _keep in rows)
    drift = [(i, src, min(1.0, max(0.0, math.log(keep / max(out, 1e-9))) / DRIFT_FULL)) for i, src, out, keep in rows]
    worst = max(drift, key=lambda t: t[1] * t[2])
    return sum(src * d for _i, src, d in drift) / total, f"{worst[0]} drifted {worst[2]:.2f}" if worst[2] > 0.05 else ""


def _shares(c):
    """[(id, source share, output share, share it can keep)] for foreground elements."""
    rows = []
    for i, e in c.src.items():
        if i not in c.out_box or _is_background(e, c.sw, c.sh) or _is_decoration(e):
            continue
        sb, ob = c.src_box[i], c.out_box[i]
        src = _area(sb) / (c.sw * c.sh)
        if src <= 0:
            continue
        keep = src
        if e["type"] != "text":
            r = (sb[2] - sb[0]) / max(sb[3] - sb[1], 1e-9)
            keep = min(src, c.tw / (r * c.th) if r >= c.tw / c.th else r * c.th / c.tw)
        rows.append((i, src, _area(ob) / (c.tw * c.th), max(keep, 1e-9)))
    return rows


def large_share_kept(source, target_canvas, output):
    """Lowest (output share / share it can keep) over large elements (>= LARGE_SHARE of the source), or
    1.0 when there are none. The engine drops candidates under SHARE_FLOOR."""
    c = _context(source, target_canvas, output)
    return min((out / keep for _i, src, out, keep in _shares(c) if src >= LARGE_SHARE), default=1.0)


def largest_share_ratio(source, target_canvas, output):
    """Output share / source share of the largest foreground element, for reports."""
    c = _context(source, target_canvas, output)
    rows = _shares(c)
    return max(rows, key=lambda t: t[1])[2] / max(rows, key=lambda t: t[1])[1] if rows else 1.0


def relationships_penalty(c: Ctx):
    """Relationships the source design has between elements (see relations()) should survive:
    a pair survives when one element sits where it sat relative to the other, scaled with them
    (offset error <= REL_KEPT of the pair's size) and, for an overlap pair, they still overlap.
    Penalty = strength-weighted share of pairs broken."""
    rels = relations(c.source)
    if abs(math.log((c.tw / c.th) / (c.sw / c.sh))) >= math.log(REL_LOOSEN_ASPECT):  # a callout may leave its picture
        rels = [r for r in rels if not _on_picture(c, r)]
    if not rels:
        return 0.0, ""
    broken = [r for r in rels if not _kept(c, r)]
    total = sum(r["strength"] for r in rels)
    return sum(r["strength"] for r in broken) / total, ", ".join(f"{r['kind']} {r['a']}~{r['b']}" for r in broken[:5])


def groups_penalty(c: Ctx):
    """Tight groups the eye reads as one thing must stay together, whatever else moves (tight_relations).
    Same test as relationships_penalty (_kept), only these pairs, weighted far higher. Penalty = share broken."""
    rels = tight_relations(c.source)
    if not rels:
        return 0.0, ""
    broken = [r for r in rels if not _kept(c, r)]
    return len(broken) / len(rels), ", ".join(f"{r['kind']} {r['a']}~{r['b']}" for r in broken[:5])


def tight_relations(source):
    """The relationships (relations()) that make one thing to the eye: a label and what it labels (label + value,
    icon + text), a text drawn over another text (a shadow copy, an outline), a text and its rule line (an
    underline), two texts that are close neighbours (strength-1 near pair), and a small image (under
    TIGHT_IMAGE of the canvas: a logo mark, an icon) right next to a text. engine.analyze makes them rigid."""
    from engine.analyze import _is_rule_line  # engine.analyze imports this module
    src = {e["id"]: e for e in source["elements"]}
    small = TIGHT_IMAGE * source["canvas"]["width"] * source["canvas"]["height"]
    out = []
    for r in relations(source):
        a, b = src[r["a"]], src[r["b"]]
        types = {a["type"], b["type"]}
        near = r["kind"] == "near" and r["strength"] == 1
        if (r["kind"] == "label" or types == {"text"} and (r["kind"] == "overlap" or near)
                or "text" in types and (_is_rule_line(a) or _is_rule_line(b))
                or near and types == {"text", "image"} and min(_area(_drawn_box(e)) for e in (a, b) if e["type"] == "image") < small):
            out.append(r)
    return out


def related_order_penalty(c: Ctx):
    """Texts the source relates (a label, close neighbours: any relationship but an overlap) must still read in
    their source order, row by row or column by column as in reading_order_penalty. Works per element, so it
    also sees "shower Baby" inside one unit. Penalty = share of those pairs flipped under both readings."""
    pairs = [(r["a"], r["b"]) for r in relations(c.source) if r["kind"] != "overlap"
             and c.src[r["a"]]["type"] == c.src[r["b"]]["type"] == "text" and r["a"] in c.out_box and r["b"] in c.out_box]
    if not pairs:
        return 0.0, ""
    first = lambda boxes, a, b, cols=False: _reading_order(boxes, [[a], [b]], columns=cols)[0][0]
    flipped = [(a, b) for a, b in pairs
               if first(c.src_box, a, b) not in (first(c.out_box, a, b), first(c.out_box, a, b, True))]
    return len(flipped) / len(pairs), ", ".join(f"{a}|{b}" for a, b in flipped[:5])


def _on_picture(c, r):
    """An overlap pair whose larger member is an image: a callout over a photo, not text on its own shape."""
    big = max((r["a"], r["b"]), key=lambda i: _area(_drawn_box(c.src[i])))
    return r["kind"] == "overlap" and c.src[big]["type"] == "image"


def relations_kept(source, target_canvas, output):
    """(strength-weighted share of the source's relationships kept, number of pairs) for reports."""
    rels = relations(source)
    if not rels:
        return 1.0, 0
    c = _context(source, target_canvas, output)
    return sum(r["strength"] for r in rels if _kept(c, r)) / sum(r["strength"] for r in rels), len(rels)


def _kept(c, r):
    a, b = r["a"], r["b"]
    if a not in c.out_box or b not in c.out_box:
        return False
    sa, sb, oa, ob = c.src_box[a], c.src_box[b], c.out_box[a], c.out_box[b]
    if r["kind"] == "overlap" and _inter(oa, ob) <= 0:
        return False
    s = math.sqrt(max(_area(oa) + _area(ob), 1e-9) / max(_area(sa) + _area(sb), 1e-9))
    centre = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
    (sax, say), (sbx, sby), (oax, oay), (obx, oby) = centre(sa), centre(sb), centre(oa), centre(ob)
    err = math.hypot((obx - oax) - s * (sbx - sax), (oby - oay) - s * (sby - say))
    u = _union_box([sa, sb])
    return err <= REL_KEPT * s * math.hypot(u[2] - u[0], u[3] - u[1])


@lru_cache(maxsize=64)
def _relations_cached(source_json):
    source = json.loads(source_json)
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    els = [e for e in source["elements"] if not _is_background(e, sw, sh) and not _is_decoration(e)]
    box = {e["id"]: _drawn_box(e) for e in els}
    els = [e for e in els if 0 < _area(box[e["id"]]) < REL_BIG * sw * sh]
    rels, seen = [], set()

    def add(a, b, kind, strength):
        key = tuple(sorted((a, b)))
        if key not in seen:
            seen.add(key)
            rels.append({"a": a, "b": b, "kind": kind, "strength": strength})

    for x, y in combinations(els, 2):  # overlap pairs: intentional stacking (a collage, text on a card)
        if _inter(box[x["id"]], box[y["id"]]) >= REL_OVERLAP * min(_area(box[x["id"]]), _area(box[y["id"]])):
            add(x["id"], y["id"], "overlap", 1.0)
    head = key_elements(source)[0]
    for lab in els:  # label pairs: a short heading or "Label:" right above or beside what it labels
        if lab["type"] != "text" or lab["id"] == head or not _is_label(lab["content"]):
            continue
        lb, lh = box[lab["id"]], lab["font_size"] * lab.get("line_height", 1.2)
        best = None
        for e in els:
            if e is lab or e["type"] == "text" and _style(e) == _style(lab) and not lab["content"].strip().endswith(":"):
                continue  # a label looks different from what it labels (a list of alike lines is not label + content)
            b = box[e["id"]]
            below = b[1] - lb[3]
            right = b[0] - lb[2]
            if -0.5 * lh <= below <= 2 * lh and min(lb[2], b[2]) - max(lb[0], b[0]) > 0.5 * min(lb[2] - lb[0], b[2] - b[0]):
                d = below
            elif -0.5 * lh <= right <= 3 * lh and min(lb[3], b[3]) - max(lb[1], b[1]) > 0.5 * min(lb[3] - lb[1], b[3] - b[1]):
                d = right
            else:
                continue
            if best is None or d < best[0]:
                best = (d, e["id"])
        if best:
            add(lab["id"], best[1], "label", 1.0)
    for icon in els:  # an icon labelling the text right of it (a phone icon and its number)
        ib = box[icon["id"]]
        if icon["type"] == "text" or _area(ib) > 0.01 * sw * sh:
            continue
        near = [(e["id"], box[e["id"]][0] - ib[2]) for e in els if e["type"] == "text"
                and min(ib[3], box[e["id"]][3]) - max(ib[1], box[e["id"]][1]) > 0.5 * min(ib[3] - ib[1], box[e["id"]][3] - box[e["id"]][1])
                and 0 <= box[e["id"]][0] - ib[2] <= 1.5 * (ib[3] - ib[1])]
        if near:
            add(icon["id"], min(near, key=lambda t: t[1])[0], "label", 1.0)
    # close neighbours: a pair much closer to each other than either is to anything else (its surroundings)
    ids = [e["id"] for e in els]
    gaps = {(x, y): _gap(box[x], box[y]) for x, y in combinations(ids, 2)}
    gap = lambda x, y: gaps.get((x, y), gaps.get((y, x)))
    parent = {i: i for i in ids}
    span = dict(box)

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for (x, y), g in sorted(gaps.items(), key=lambda kv: kv[1]):
        around = min([gap(x, o) for o in ids if o not in (x, y)] + [gap(y, o) for o in ids if o not in (x, y)], default=math.inf)
        if g > REL_LOOSE * around:
            continue
        rx, ry = find(x), find(y)
        if rx == ry:
            continue
        u = _union_box([span[rx], span[ry]])
        if _area(u) > REL_BLOCK_MAX * sw * sh:
            continue
        parent[rx], span[ry] = ry, u
        add(x, y, "near", 1.0 if g <= REL_TIGHT * around else 0.5)
    return tuple(json.dumps(r, sort_keys=True) for r in rels)


def relations(source):
    """The source's relationships between non-background elements, as [{a, b, kind, strength}]:
      overlap  the two overlap (REL_OVERLAP of the smaller): stacking the designer chose (strength 1)
      label    a short text (<= LABEL_WORDS words or LABEL_CHARS characters, or ending in ':'; not the
               headline, a web address, an email or a bare number), directly above (within 2 lines) or
               beside (within 3 lines) the nearest text (in a different style, unless the label ends
               in ':'), icon or image it labels; or a small icon
               (< 1% of the canvas) with text right beside it (strength 1)
      near     neighbours much closer to each other than to their surroundings: their gap is within
               REL_LOOSE of the smallest gap from either of them to any other element (within REL_TIGHT:
               strength 1, else 0.5); joined by single linkage, never past REL_BLOCK_MAX of the canvas.
    Backdrops (>= REL_BIG of the canvas) and translucent decorations take part in none.
    Shared with engine.analyze, which turns them into blocks that move as one."""
    rels = _relations_cached(json.dumps(source, sort_keys=True))
    return [json.loads(r) for r in rels]


def _style(el):
    return (el["font_family"], el.get("font_weight"), round(el["font_size"]), el.get("color"))


def _is_label(text):
    """Short words that head something; not a web address, an email or a bare number/price."""
    t = text.strip()
    if "@" in t or "www" in t.lower() or re.search(r"\w\.(com|net|org|[a-z]{2})\b", t.lower()):
        return False
    if not re.search(r"[A-Za-z]{2}", t) and not re.fullmatch(r"\d{1,2}[.)]?", t):  # keep list markers like "1." / "02"
        return False
    return t.endswith(":") or len(t.split()) <= LABEL_WORDS and len(t) <= LABEL_CHARS


def _gap(a, b):
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return math.hypot(dx, dy)


def _union_box(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def key_size_penalty(c: Ctx):
    """The most important content must keep its share of the canvas: the headline (the
    biggest-looking text) and the main image (the largest non-background image, alone or
    inside a group). Each one's output scale (square root of its drawn area ratio) over the
    scale the target affords it (the area ratio's square root, or less when an image at its
    own aspect cannot fit that, or less when the headline re-wrapped at its best cannot). KEY_FREE or more is free, KEY_FULL or less costs 1, steeply
    in between. A cover-cropped main image whose box was cut to a sliver (aspect changed by
    more than KEY_SQUEEZE_FREE) costs too. Penalty = the worst of these: either one
    shrinking is a failure, and decorations are not measured at all."""
    area = math.sqrt(c.tw * c.th / (c.sw * c.sh))
    pens = []
    h, m = key_elements(c.source)
    if h is not None and h in c.out:
        best = min(area, _headline_fit(json.dumps(c.src[h], sort_keys=True), c.tw, c.th))
        r = math.sqrt(_area(c.out_box[h]) / max(_area(c.src_box[h]), 1e-9)) / best
        pens.append((_steep(KEY_FREE, KEY_FULL, r), f"headline {h} at {r:.2f} of its share"))
    if m is not None and m in c.out:
        sb, ob = c.src_box[m], c.out_box[m]
        w, h = sb[2] - sb[0], sb[3] - sb[1]
        r = math.sqrt(_area(ob) / max(w * h, 1e-9)) / min(area, c.tw / max(w, 1e-9), c.th / max(h, 1e-9))
        pens.append((_steep(KEY_FREE, KEY_FULL, r), f"main image {m} at {r:.2f} of its share"))
        q = (ob[2] - ob[0]) / max(ob[3] - ob[1], 1e-9) / (w / max(h, 1e-9))
        q = min(q, 1 / max(q, 1e-9))
        pens.append((_steep(KEY_SQUEEZE_FREE, KEY_SQUEEZE_FULL, q), f"main image {m} squeezed to {q:.2f} of its aspect"))
    worst = max(pens, default=(0.0, ""))
    return worst[0], worst[1] if worst[0] > 0 else ""


@lru_cache(maxsize=256)
def _headline_fit(el_json, tw, th):
    """The largest scale at which a text, re-wrapped at its best width, still fits the canvas: what the
    target's shape lets it keep, when that is less than its share of the area (a long headline on a
    skyscraper). Line breaks tried at HEADLINE_WRAPS widths, from one line down to a fifth of it."""
    el = json.loads(el_json)
    one = text_layout(dict(el, width=1e6))
    full = one["right"] - one["left"]
    best = 0.0
    for f in HEADLINE_WRAPS:
        lay = text_layout(dict(el, width=max(1.0, full * f)))
        w = max(lay["right"] - lay["left"], 1.0) if lay["lines"] else 1.0
        best = max(best, min(tw / w, th / max(lay["height"], 1.0)))
    return best


def key_elements(source):
    """(headline id, main image id) of a design, either None: the biggest-looking text, and the
    largest non-background image covering at least HERO_MIN_AREA of the canvas (alone or in a
    group). Shared with engine.resize, which guards retrieved layouts with the same two."""
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    els = source["elements"]
    texts = [e for e in els if e["type"] == "text"]
    head = max(texts, key=_visual_size)["id"] if texts else None
    imgs = [e for e in els if e["type"] == "image" and not _is_background(e, sw, sh)]
    main = max(imgs, key=lambda e: _area(_drawn_box(e)), default=None)
    return head, main["id"] if main and _area(_drawn_box(main)) >= HERO_MIN_AREA * sw * sh else None


def _steep(free, full, r):
    """0 at r >= free, 1 at r <= full, rising fast in between (half way already costs 0.75)."""
    t = min(1.0, max(0.0, (free - r) / (free - full)))
    return t * (2 - t)


def space_penalty(c: Ctx):
    """Unused target canvas, regardless of how sparse the source was:
    - extent (half): the content's bounding box should span MIN_EXTENT of the
      target on each axis (mean shortfall);
    - gap (quarter): the widest empty band running right across the content
      (a hole between a split layout's halves), beyond GAP_FREE of that side;
    - empty ninths (quarter): ninths of the target with no content, beyond EMPTY_NINTHS_FREE."""
    box = _content_box(c.out_box, c.units)
    if box is None:
        return 0.0, ""
    fx, fy = (box[2] - box[0]) / c.tw, (box[3] - box[1]) / c.th
    p_ext = (max(0.0, MIN_EXTENT - fx) + max(0.0, MIN_EXTENT - fy)) / (2 * MIN_EXTENT)
    boxes = [_unit_box(c.out_box, u) for u in c.units]
    gap = max(_widest_gap([(b[0], b[2]) for b in boxes]) / c.tw, _widest_gap([(b[1], b[3]) for b in boxes]) / c.th)
    p_gap = min(1.0, max(0.0, gap - GAP_FREE) / (GAP_FULL - GAP_FREE))
    _cov, empty = _coverage(c.out_box, c.units, c.tw, c.th)
    p_empty = max(0, empty - EMPTY_NINTHS_FREE) / (9 - EMPTY_NINTHS_FREE)
    return (0.5 * p_ext + 0.25 * p_gap + 0.25 * p_empty,
            f"content spans {fx:.0%} x {fy:.0%} of the canvas, widest inner gap {gap:.0%}, empty ninths {empty}")


def fill_penalty(c: Ctx):
    """Content shrunk into part of the canvas: the content box's share of the target canvas against its share of
    the source's (both clipped to the canvas); free down to FILL_FREE of it, full penalty at FILL_FULL (_steep)."""
    bs, bo = _content_box(c.src_box, c.units), _content_box(c.out_box, c.units)
    if bs is None or bo is None:
        return 0.0, ""
    share = lambda b, w, h: _area((max(b[0], 0), max(b[1], 0), min(b[2], w), min(b[3], h))) / (w * h)
    s, o = share(bs, c.sw, c.sh), share(bo, c.tw, c.th)
    return _steep(FILL_FREE, FILL_FULL, o / max(s, 1e-9)), f"content box {s:.0%} of the source canvas, {o:.0%} of the target"


def balance_penalty(c: Ctx):
    """Content huddled to one side: how much further the content box's centre
    sits from the canvas centre (as a fraction of each side, worst axis) than
    it did in the source. BALANCE_FULL more = full penalty."""
    bs, bo = _content_box(c.src_box, c.units), _content_box(c.out_box, c.units)
    if bs is None or bo is None:
        return 0.0, ""
    def off(b, w, h):
        return max(abs((b[0] + b[2]) / 2 / w - 0.5), abs((b[1] + b[3]) / 2 / h - 0.5))
    o_s, o_o = off(bs, c.sw, c.sh), off(bo, c.tw, c.th)
    return min(1.0, max(0.0, o_o - o_s) / BALANCE_FULL), f"centre offset {o_s:.0%}->{o_o:.0%}"


def floating_penalty(c: Ctx):
    """Floating elements (badges, ribbons, stamps) belong where they were
    anchored (see _float_anchor): either to another unit they partly cover (a
    ribbon across a card) or to a canvas corner.
    - Attached: drift = change of its centre's position inside the host's box,
      as fractions of the host's sides.
    - Corner: drift = change in its gaps to that corner's two edges, as
      fractions of the canvas sides.
    Per unit: drift / FLOAT_DRIFT_FULL, plus TILT_FLIP when a rotated unit's
    tilt no longer points the way it did relative to its anchor (it moved to
    the mirrored side without its angle being mirrored, or the reverse).
    Penalty = mean over floating units, each capped at 1."""
    pens, notes = [], []
    for u in c.units:
        anchor = c.anchors[tuple(u)]
        if anchor is None:
            continue
        kind, ref = anchor
        if kind == "host":
            gs, go = _rel_pos(c.src_box, u, ref), _rel_pos(c.out_box, u, ref)
            where = f"its place on {_name(ref)}"
        else:
            gs = _corner_gaps(_unit_box(c.src_box, u), ref, c.sw, c.sh)
            go = _corner_gaps(_unit_box(c.out_box, u), ref, c.tw, c.th)
            where = ref
        d = abs(go[0] - gs[0]) + abs(go[1] - gs[1])
        p = d / FLOAT_DRIFT_FULL
        if d > 0.05:
            notes.append(f"{_name(u)} drifted {d:.0%} from {where}")
        if _tilt_wrong(c, u, anchor):
            p += TILT_FLIP
            notes.append(f"{_name(u)} tilts the wrong way")
        pens.append(min(1.0, p))
    return (sum(pens) / len(pens) if pens else 0.0), ", ".join(notes)


def emphasis_penalty(c: Ctx):
    """Units that stand out by colour (see `emphasis`) belong in a focal
    position: the end of the reading path (last in row or column order), or
    the centre of their column (the units stacked above or below them). A
    row of side-by-side units (a row of badges) is judged as one block. Alone
    in its column, a unit is judged against the canvas centre instead.
    Floating units are judged by floating_penalty. Penalty = off-focus amount,
    averaged with each unit's emphasis strength as weight."""
    sal = emphasis(c.source)
    units = [u for u in c.units if c.anchors[tuple(u)] is None]
    if not units:
        return 0.0, ""
    last = {tuple(_reading_order(c.out_box, units)[-1]), tuple(_reading_order(c.out_box, units, columns=True)[-1])}
    boxes = {tuple(u): _unit_box(c.out_box, u) for u in units}
    num = den = 0.0
    off = []
    for u in units:
        w = max(sal.get(i, 0.0) for i in u)
        if w == 0 or tuple(u) in last:
            continue
        row = _row_block(boxes, tuple(u), ROW_GAP * c.tw)
        b = _content_box(boxes, [[k] for k in row])
        cx = (b[0] + b[2]) / 2
        col = [boxes[k] for k in boxes if k not in row and min(b[2], boxes[k][2]) > max(b[0], boxes[k][0])]
        if col:
            z = _content_box({0: b, **{k + 1: v for k, v in enumerate(col)}}, [[k] for k in range(len(col) + 1)])
            rel, full = abs(cx - (z[0] + z[2]) / 2) / max(z[2] - z[0], 1e-6), FOCUS_OFF_ZONE
        else:
            rel, full = abs(cx - c.tw / 2) / c.tw, FOCUS_OFF_CANVAS
        p = min(1.0, max(0.0, rel - FOCUS_FREE) / (full - FOCUS_FREE))
        num, den = num + w * p, den + w
        if p > 0.2:
            off.append(f"{_name(u)} off-focus {p:.2f}")
    return (num / den if den else 0.0), ", ".join(off)


def siblings_penalty(c: Ctx):
    """A row of sibling units (see `siblings`: a row of badges or buttons) must stay
    together: all at one scale (within SIBLING_SCALE), and one compact block (their
    union box at most SIBLING_SPREAD times their summed area: a row, or a stack or
    grid when the row would not stay readable) with nothing else in it. Half the
    penalty for each failure; a compact stack or grid costs SIBLING_RESTACK, less than
    the small-text penalty an unreadable row would cost. Mean over groups."""
    groups = siblings(c.source)
    if not groups:
        return 0.0, ""
    pens, notes = [], []
    for g in groups:
        boxes = {tuple(u): _unit_box(c.out_box, u) for u in g}
        scales = [(boxes[tuple(u)][2] - boxes[tuple(u)][0]) / max(_unit_box(c.src_box, u)[2] - _unit_box(c.src_box, u)[0], 1e-6)
                  for u in g]
        uneven = max(scales) > (1 + SIBLING_SCALE) * min(scales)
        span = _content_box(boxes, [[k] for k in boxes])
        src_span = _content_box({tuple(u): _unit_box(c.src_box, u) for u in g}, [[tuple(u)] for u in g])
        spread = _area(span) / max(sum(_area(b) for b in boxes.values()), 1e-6)
        src_spread = _area(src_span) / max(sum(_area(_unit_box(c.src_box, u)) for u in g), 1e-6)
        between = any(_overlap_ratio(_unit_box(c.out_box, u), span) > 0.5 for u in c.units if tuple(u) not in boxes)
        split = between or spread > max(SIBLING_SPREAD, src_spread * 1.1)
        restacked = not split and len(_row_block(boxes, tuple(g[0]), math.inf)) < len(g)
        p = 0.5 * uneven + 0.5 * split + SIBLING_RESTACK * restacked
        pens.append(p)
        if p:
            notes.append(f"{'+'.join(_name(u) for u in g)} " + " and ".join(
                w for w, bad in (("uneven", uneven), ("split", split), ("no longer a row", restacked)) if bad))
    return sum(pens) / len(pens), ", ".join(notes)


def alignment_penalty(c: Ctx):
    """Half: near-miss alignments, i.e. a pair of units whose left/centre/right
    (or top/middle/bottom) differ by more than ALIGNED_PX but less than
    NEAR_MISS_REL of the canvas side, with no exact alignment on that axis;
    near misses added over the source, per unit, capped at 1. Half: lost
    alignments, i.e. pairs aligned in the source (a row of badges, a centred
    stack) that share no edge or centre on either axis in the output; as a
    share of the source's aligned pairs. A pair that went from stacked to side by
    side, or back, was re-laid out on purpose and is not lost (a row may become a
    column). Rotated units are left out of near misses."""
    # a rotated unit's box edges are not lines anyone sees
    flat = [u for u in c.units if not any(c.src[i].get("rotation", 0) for i in u)]
    n_src = _near_misses(c.src_box, flat, c.sw, c.sh)
    n_out = _near_misses(c.out_box, flat, c.tw, c.th)
    p_near = min(1.0, max(0, n_out - n_src) / max(1, len(c.units)))
    aligned = [(a, b) for a, b in combinations(c.units, 2)
               if _aligned(_unit_box(c.src_box, a), _unit_box(c.src_box, b))]
    # stacked in the source and side by side now (or the reverse): the layout was transposed on purpose
    lost = [(a, b) for a, b in aligned if not _aligned(_unit_box(c.out_box, a), _unit_box(c.out_box, b))
            and _beside(_unit_box(c.src_box, a), _unit_box(c.src_box, b)) == _beside(_unit_box(c.out_box, a), _unit_box(c.out_box, b))]
    p_lost = len(lost) / len(aligned) if aligned else 0.0
    return 0.5 * p_near + 0.5 * p_lost, (f"near misses {n_src}->{n_out}, lost alignments {len(lost)}/{len(aligned)}"
                                         + "".join(f", {_name(a)}|{_name(b)}" for a, b in lost[:4]))


def reading_order_penalty(c: Ctx):
    """Source reading order = rows top to bottom (a unit joins a row when its
    vertical centre falls inside the row), left to right within a row. The
    output may be read either that way or column by column (a unit joins a
    column when its horizontal centre falls inside it; columns left to right,
    top to bottom within one), so a side-by-side layout that reads correctly
    down each column is not punished. Floating units are judged by
    floating_penalty instead. Penalty = fraction of unit pairs whose order
    flipped vs the source (Kendall), under the better of the two readings."""
    units = [u for u in c.units if c.anchors[tuple(u)] is None]
    src_rank = {tuple(u): i for i, u in enumerate(_reading_order(c.src_box, units))}
    best = None
    for order in (_reading_order(c.out_box, units), _reading_order(c.out_box, units, columns=True)):
        pairs = list(combinations(order, 2))
        if not pairs:
            return 0.0, ""
        flipped = [(a, b) for a, b in pairs if src_rank[tuple(a)] > src_rank[tuple(b)]]
        if best is None or len(flipped) < len(best[0]):
            best = flipped, len(pairs)
    flipped, n = best
    return len(flipped) / n, ", ".join(f"{_name(a)} now before {_name(b)}" for a, b in flipped[:5])


def contrast_penalty(c: Ctx):
    """(Optional, renders twice.) Render the design without its text, then for
    each text element compare its colour (alpha-composited) with the pixels
    under its drawn lines. Contrast = WCAG ratio against the 10th-percentile
    worst pixel. Only contrast the layout lost counts: the bar is 4.5 (3.0 for
    >=24px or bold >=19px), or the element's contrast in the source if that
    was already lower (white on the brand's orange badge is the designer's
    choice, not ours). Penalty per element = shortfall below the bar, scaled to 0..1; total = mean."""
    src_ratio = _contrast_ratios(json.dumps(c.source, sort_keys=True))
    pens, low = [], []
    for eid, worst in _contrast_ratios(json.dumps(c.output, sort_keys=True)).items():
        el = c.out[eid]
        fs = int(el["font_size"])
        need = CONTRAST_MIN_LARGE if fs >= 24 or (el.get("font_weight") == "bold" and fs >= 19) else CONTRAST_MIN
        need = min(need, src_ratio.get(eid, need))
        pens.append(max(0.0, min(1.0, (need - worst) / max(need - 1, 1e-6))))
        if worst < need:
            low.append(f"{eid} {worst:.1f}:1 (source {src_ratio.get(eid, float('nan')):.1f}:1)")
    return (sum(pens) / len(pens) if pens else 0.0), ", ".join(low)


def coverage_penalty(c: Ctx):
    """Bare canvas (renders white) newly showing through, when the source painted at least COVER_SRC of its canvas:
    share of the source canvas covered minus share of the target covered, COVER_FULL = full penalty."""
    s, o = coverage(json.dumps(c.source, sort_keys=True)), coverage(json.dumps(c.output, sort_keys=True))
    if s < COVER_SRC:
        return 0.0, f"source {s:.0%} covered"
    return min(1.0, max(0.0, s - o) / COVER_FULL), f"covered: source {s:.0%}, output {o:.0%}"


@lru_cache(maxsize=256)
def coverage(design_json):
    """Share of the canvas painted at least half opaque by shapes and images (texts don't cover)."""
    design = json.loads(design_json)
    w, h = design["canvas"]["width"], design["canvas"]["height"]
    gx, gy = np.meshgrid((np.arange(COVER_GRID) + 0.5) * w / COVER_GRID, (np.arange(COVER_GRID) + 0.5) * h / COVER_GRID)
    px, py = gx.ravel(), gy.ravel()
    bare = np.ones(px.size)
    for e in design["elements"]:  # shortcut: rotation is ignored, as in _text_contrasts
        if e["type"] != "text":
            _rgb, a = _layer(e, px, py)
            if a is not None:
                bare *= 1 - a
    return float(np.mean(bare <= 0.5))


def legibility_penalty(c: Ctx):
    """contrast_penalty without rendering, so every score sees white text on a white background: the
    same bars and shortfall, with each text's contrast estimated from the layers under its box
    (_text_contrasts, as for contrast_drops)."""
    src = _text_contrasts(json.dumps(c.source, sort_keys=True))
    pens, low = [], []
    for eid, worst in _text_contrasts(json.dumps(c.output, sort_keys=True)).items():
        if eid not in c.out:
            continue
        el = c.out[eid]
        fs = int(el["font_size"])
        need = CONTRAST_MIN_LARGE if fs >= 24 or (el.get("font_weight") == "bold" and fs >= 19) else CONTRAST_MIN
        need = min(need, src.get(eid, need))
        pens.append(max(0.0, min(1.0, (need - worst) / max(need - 1, 1e-6))))
        if worst < need:
            low.append(f"{eid} ~{worst:.1f}:1 (source ~{src.get(eid, float('nan')):.1f}:1)")
    return (sum(pens) / len(pens) if pens else 0.0), ", ".join(low)


@lru_cache(maxsize=8)  # the engine scores many candidates against one source: render it once
def _contrast_ratios(design_json):
    """{text id: 10th-percentile WCAG ratio of its colour against what is drawn under its lines}."""
    from eval.render import render_image, render_scale

    design = json.loads(design_json)
    no_text = {"canvas": design["canvas"], "elements": [e for e in design["elements"] if e["type"] != "text"]}
    img = np.asarray(render_image(no_text), dtype=np.float64) / 255
    s = render_scale(design)
    lum = _rel_luminance(img)
    out = {}
    for el in design["elements"]:
        if el["type"] != "text" or el.get("id") in out:
            continue
        x0, y0, x1, y1 = (int(round(v * s)) for v in _drawn_box(el))
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, img.shape[1]), min(y1, img.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        r, g, b, a = parse_rgba(el.get("color"))
        bg = img[y0:y1, x0:x1].reshape(-1, 3)
        fg = a * np.array([r, g, b]) / 255 + (1 - a) * bg
        lf, lb = _rel_luminance(fg), lum[y0:y1, x0:x1].reshape(-1)
        ratio = (np.maximum(lf, lb) + 0.05) / (np.minimum(lf, lb) + 0.05)
        out[el["id"]] = float(np.percentile(ratio, 10))
    return out


PENALTIES = {  # name: (function, weight, needs_render). Fitted to all hand labels (samples, crello-dev,
    # crello-retrieval-review): python -m eval.calibrate --fit --suite samples,crello-dev,crello-retrieval-review
    # key_size is pinned at 2.0 (the fit leaves it at the floor because the sample labels pull the other way):
    # shrinking the headline or main image must cost far more than anything else. emphasis is kept by request.
    "overlap": (overlap_penalty, 0.5, False),
    "holders": (holders_penalty, 0.5, False),
    "small_text": (small_text_penalty, 1.16, False),
    "headline": (headline_penalty, 0.83, False),
    "scale": (scale_penalty, 0.68, False),
    "key_size": (key_size_penalty, 2.0, False),
    "text_over_image": (text_over_image_penalty, 3.0, False),  # pinned like key_size: free fit 0.63
    "relationships": (relationships_penalty, 0.5, False),
    "groups": (groups_penalty, 2.0, False),  # pinned like key_size: what reads as one thing must not fall apart
    "related_order": (related_order_penalty, 1.0, False),
    "fill": (fill_penalty, 1.15, False),  # as space
    "share_drift": (share_drift_penalty, 1.0, False),  # all 135 labels: pooled 0.864 -> 0.867, crello-dev 0.876 -> 0.938  # labels barely move with it: pooled 0.857 -> 0.858, samples -0.014
    "space": (space_penalty, 1.15, False),
    "balance": (balance_penalty, 1.38, False),
    "floating": (floating_penalty, 0.5, False),
    "emphasis": (emphasis_penalty, 0.95, False),
    "siblings": (siblings_penalty, 0.5, False),
    "alignment": (alignment_penalty, 0.68, False),
    "reading_order": (reading_order_penalty, 0.63, False),
    "legibility": (legibility_penalty, 2.0, False),  # 153 labels: pooled 0.866 (off) -> 0.869 (best of 0.5-4); the fit leaves it at the floor (few labelled outputs lose contrast)
    "coverage": (coverage_penalty, 3.0, False),  # pinned like text_over_image: white space the source never had
    "contrast": (contrast_penalty, 0.5, True),
}


def score(source: dict, target_canvas: dict, output: dict, contrast: bool = False) -> dict:
    c = _context(source, target_canvas, output)
    breakdown = {}
    for name, (fn, weight, needs_render) in PENALTIES.items():
        if needs_render and not contrast:
            continue
        p, detail = fn(c)
        breakdown[name] = {"penalty": round(p, 4), "weight": weight, "detail": detail}
    wsum = sum(b["weight"] for b in breakdown.values())
    total = 100 * (1 - sum(b["weight"] * b["penalty"] for b in breakdown.values()) / wsum)
    return {"total": round(total, 2), "breakdown": breakdown}


# --- helpers -----------------------------------------------------------------

def _context(source, target_canvas, output):
    src = {e["id"]: e for e in source["elements"]}
    out = {}
    for e in output["elements"]:  # first occurrence wins; check.py reports duplicates
        if e.get("id") in src and e["id"] not in out:
            out[e["id"]] = e
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    fg = [i for i in out if not _is_background(src[i], sw, sh) and not _is_decoration(src[i])]
    groups: dict = {}
    for i in fg:
        groups.setdefault(src[i].get("group_id") or ("__" + i), []).append(i)
    units = list(groups.values())
    c = Ctx(source, output, src, out, {i: _drawn_box(src[i]) for u in units for i in u},
            {i: _drawn_box(out[i]) for u in units for i in u}, units, sw, sh,
            target_canvas["width"], target_canvas["height"])
    c.anchors = {tuple(u): _float_anchor(c, u) for u in units}
    return c


def _is_background(el, cw, ch):
    return el["x"] <= 1 and el["y"] <= 1 and el["x"] + el["width"] >= cw - 1 and el["y"] + el["height"] >= ch - 1


def _is_decoration(el):
    if el["type"] != "shape" or el.get("group_id"):
        return False
    return el.get("opacity", 1) * parse_rgba(el.get("fill_color"))[3] < DECORATION_ALPHA


@lru_cache(maxsize=64)
def _asset_size(asset_id):
    try:
        with Image.open(assets_dir() / asset_id) as im:
            return im.size
    except (OSError, TypeError):
        return None  # missing asset: renderer draws a placeholder filling the box


def _drawn_box(el):
    """Absolute (x0, y0, x1, y1) of what the renderer actually paints."""
    if el.get("rotation", 0):
        return rotated_bbox(el)
    x, y, w, h = el["x"], el["y"], el["width"], el["height"]
    if el["type"] == "text":
        lay = text_layout(el)
        if not lay["lines"]:
            return (x, y, x, y)
        return (x + lay["left"], y, x + lay["right"], y + lay["height"])
    if el["type"] == "image" and el.get("fit", "contain") == "contain" and w > 0 and h > 0:
        size = _asset_size(el.get("asset_id"))
        if size:
            k = min(w / size[0], h / size[1])
            nw, nh = size[0] * k, size[1] * k
            return (x + (w - nw) / 2, y + (h - nh) / 2, x + (w + nw) / 2, y + (h + nh) / 2)
    return (x, y, x + w, y + h)


def _unit_box(boxes_by_id, unit):
    boxes = [boxes_by_id[i] for i in unit]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _overlap_ratio(a, b):
    inter = _area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))
    smaller = min(_area(a), _area(b))
    return inter / smaller if smaller > 0 else 0.0


def _coverage(boxes, units, cw, ch, res=192):
    k = res / max(cw, ch)
    gw, gh = max(3, round(cw * k)), max(3, round(ch * k))
    mask = np.zeros((gh, gw), dtype=bool)
    for u in units:
        for i in u:
            x0, y0, x1, y1 = boxes[i]
            mask[max(0, int(y0 * k)):max(0, math.ceil(y1 * k)), max(0, int(x0 * k)):max(0, math.ceil(x1 * k))] = True
    empty = sum(
        mask[r * gh // 3:(r + 1) * gh // 3, q * gw // 3:(q + 1) * gw // 3].mean() < EMPTY_CELL_COVERAGE
        for r in range(3) for q in range(3)
    )
    return float(mask.mean()), int(empty)


def _near_misses(boxes_by_id, units, cw, ch):
    boxes = [_unit_box(boxes_by_id, u) for u in units]
    n = 0
    for a, b in combinations(boxes, 2):
        for axis, dim in ((0, cw), (1, ch)):
            lo, hi = axis, axis + 2
            ds = [abs(a[lo] - b[lo]), abs((a[lo] + a[hi]) - (b[lo] + b[hi])) / 2, abs(a[hi] - b[hi])]
            if min(ds) > ALIGNED_PX and any(d < NEAR_MISS_REL * dim for d in ds):
                n += 1
    return n


def _reading_order(boxes_by_id, units, columns=False):
    """Rows top to bottom, left to right within a row; or with columns=True,
    the same with the axes swapped."""
    lo, hi = (0, 2) if columns else (1, 3)
    items = sorted(((_unit_box(boxes_by_id, u), u) for u in units), key=lambda t: t[0][lo])
    rows = []
    for box, u in items:
        mid = (box[lo] + box[hi]) / 2
        if rows and rows[-1]["top"] <= mid <= rows[-1]["bottom"]:
            rows[-1]["items"].append((box, u))
            rows[-1]["bottom"] = max(rows[-1]["bottom"], box[hi])
        else:
            rows.append({"top": box[lo], "bottom": box[hi], "items": [(box, u)]})
    return [u for row in rows for _box, u in sorted(row["items"], key=lambda t: t[0][1 - lo])]


def _beside(a, b):
    """Side by side (overlapping vertically) rather than stacked."""
    return min(a[3], b[3]) - max(a[1], b[1]) > 0 and min(a[2], b[2]) - max(a[0], b[0]) <= 0


def _aligned(a, b):
    """Share a left, centre or right edge, or a top, middle or bottom edge (within ALIGNED_PX)."""
    return any(abs(a[lo] - b[lo]) <= ALIGNED_PX or abs(a[hi] - b[hi]) <= ALIGNED_PX
               or abs((a[lo] + a[hi]) - (b[lo] + b[hi])) / 2 <= ALIGNED_PX for lo, hi in ((0, 2), (1, 3)))


def _widest_gap(spans):
    """Widest stretch between the first start and the last end that no span covers."""
    spans = sorted(spans)
    gap, reach = 0.0, spans[0][1] if spans else 0.0
    for lo, hi in spans[1:]:
        gap = max(gap, lo - reach)
        reach = max(reach, hi)
    return gap


def _content_box(boxes_by_id, units):
    boxes = [_unit_box(boxes_by_id, u) for u in units]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _float_anchor(c, unit):
    """What a floating unit is pinned to in the source, or None if it isn't floating.
    Small units only (< FLOAT_MAX_AREA of the canvas). Attached: it covers
    ATTACH_MIN..90% of a bigger unit (more is plain layering) or covers any of
    it while rotated -> ("host", that unit). Otherwise rotated, or within
    FLOAT_CORNER of two adjacent edges -> ("corner", "top-left" etc.)."""
    b = _unit_box(c.src_box, unit)
    if _area(b) >= FLOAT_MAX_AREA * c.sw * c.sh:
        return None
    rotated = any(c.src[i].get("rotation", 0) for i in unit)
    best, share = None, 0.0
    for v in c.units:
        vb = _unit_box(c.src_box, v)
        if v is unit or _area(vb) <= _area(b):
            continue
        r = _area((max(b[0], vb[0]), max(b[1], vb[1]), min(b[2], vb[2]), min(b[3], vb[3]))) / max(_area(b), 1e-9)
        if r > share:
            best, share = v, r
    if best is not None and share >= ATTACH_MIN and (share < 0.9 or rotated):
        return ("host", best)
    left, right = b[0] <= FLOAT_CORNER * c.sw, b[2] >= (1 - FLOAT_CORNER) * c.sw
    top, bottom = b[1] <= FLOAT_CORNER * c.sh, b[3] >= (1 - FLOAT_CORNER) * c.sh
    if left == right or top == bottom:  # spans the canvas, or not at an edge: only a rotated unit floats
        if not rotated:
            return None
        cx, cy = (b[0] + b[2]) / 2 / c.sw, (b[1] + b[3]) / 2 / c.sh
        left, top = cx < 0.5, cy < 0.5
    return ("corner", ("top" if top else "bottom") + "-" + ("left" if left else "right"))


def _row_block(boxes, key, max_gap):
    """`key` plus every unit reachable through side-by-side neighbours: vertical
    centre inside the other's span, not overlapping horizontally, gap under max_gap."""
    block, todo = {key}, [key]
    while todo:
        a = boxes[todo.pop()]
        for k, b in boxes.items():
            if k in block:
                continue
            same_row = a[1] <= (b[1] + b[3]) / 2 <= a[3] or b[1] <= (a[1] + a[3]) / 2 <= b[3]
            gap = max(a[0], b[0]) - min(a[2], b[2])  # negative = they overlap horizontally: stacked, not side by side
            if same_row and -ALIGNED_PX <= gap < max_gap:
                block.add(k)
                todo.append(k)
    return block


def _rel_pos(boxes, unit, host):
    """Centre of `unit` as a fraction of `host`'s box."""
    b, h = _unit_box(boxes, unit), _unit_box(boxes, host)
    return (((b[0] + b[2]) / 2 - h[0]) / max(h[2] - h[0], 1e-6), ((b[1] + b[3]) / 2 - h[1]) / max(h[3] - h[1], 1e-6))


def _side(b, ref):
    """Which side of the point `ref` the centre of box b is on: (+-1, +-1)."""
    return (1 if (b[0] + b[2]) / 2 >= ref[0] else -1, 1 if (b[1] + b[3]) / 2 >= ref[1] else -1)


def _tilt_wrong(c, unit, anchor):
    """A rotated unit should keep its source angle, negated exactly when it moved to the
    mirrored side of its anchor on one axis (mirroring on both is a half-turn: same angle)."""
    rot = [i for i in unit if c.src[i].get("rotation", 0)]
    if not rot:
        return False
    def centre(boxes, w, h):
        if anchor[0] == "host":
            hb = _unit_box(boxes, anchor[1])
            return ((hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2)
        return (w / 2, h / 2)
    ss = _side(_unit_box(c.src_box, unit), centre(c.src_box, c.sw, c.sh))
    so = _side(_unit_box(c.out_box, unit), centre(c.out_box, c.tw, c.th))
    flip = ss[0] * so[0] * ss[1] * so[1]
    i = rot[0]
    return math.copysign(1, c.out[i].get("rotation", 0) or 0.0) != math.copysign(1, c.src[i]["rotation"]) * flip


@lru_cache(maxsize=32)
def _emphasis_cached(source_json):
    source = json.loads(source_json)
    source_els = {e["id"]: e for e in source["elements"]}
    cw, ch = source["canvas"]["width"], source["canvas"]["height"]
    painted = []  # (lab colour, area)
    own = {}      # element id -> lab colour
    for el in source["elements"]:
        b = _drawn_box(el)
        area = _area((max(b[0], 0), max(b[1], 0), min(b[2], cw), min(b[3], ch)))
        if el["type"] == "image":
            rgb = _asset_mean_rgb(el.get("asset_id"))
            if rgb is not None:
                painted.append((_lab(rgb), area))
            continue
        r, g, bl, a = parse_rgba(el.get("color") if el["type"] == "text" else el.get("fill_color"))
        a *= el.get("opacity", 1)
        if a < DECORATION_ALPHA:
            continue
        lab = _lab((r, g, bl))
        own[el["id"]] = lab
        painted.append((lab, area * (0.3 if el["type"] == "text" else 1.0) * a))  # text: roughly its ink
    palette = []  # [lab, area], greedy clusters
    for lab, area in sorted(painted, key=lambda t: -t[1]):
        hit = next((p for p in palette if _delta(p[0], lab) < SAME_COLOUR), None)
        if hit:
            hit[1] += area
        else:
            palette.append([lab, area])
    total = sum(p[1] for p in palette) or 1.0
    main = [p[0] for p in palette if p[1] / total >= MAIN_SHARE]
    out, grouped = {}, set()
    for g in (g for g in _siblings_cached(source_json) if len(g) >= 3):
        cols = []  # each sibling's colour: its largest opaque member's
        for u in g:
            painted_ids = [i for i in u if i in own]
            cols.append(own[max(painted_ids, key=lambda i: _area(_drawn_box(source_els[i])))] if painted_ids else None)
        if None in cols:
            continue
        grouped.update(i for u in g for i in u)
        for n, lab in enumerate(cols):  # the odd one out, if the rest agree
            rest = cols[:n] + cols[n + 1:]
            d = min(_delta(lab, r) for r in rest)
            if d >= ACCENT_MIN_DELTA and all(_delta(a, b) < SAME_COLOUR for a, b in combinations(rest, 2)):
                for i in g[n]:
                    out[i] = round(min(1.0, d / 70), 3)
        if not any(i in out for u in g for i in u):
            grouped.difference_update(i for u in g for i in u)  # uniform group: judged by the palette like the rest
    for i, lab in own.items():
        if i in grouped:
            continue
        chroma = math.hypot(lab[1], lab[2])
        share = sum(p[1] for p in palette if _delta(p[0], lab) < SAME_COLOUR) / total
        d = min((_delta(lab, m) for m in main), default=0.0)
        if chroma >= ACCENT_MIN_CHROMA and d >= ACCENT_MIN_DELTA and share < MAIN_SHARE:
            out[i] = round(min(1.0, d / 70), 3)
    return out


def emphasis(source):
    """{element id: 0..1} for elements whose own colour (text colour, shape fill)
    stands out from the design's main palette: saturated (CIELAB chroma >=
    ACCENT_MIN_CHROMA), at least ACCENT_MIN_DELTA from every colour that covers
    MAIN_SHARE of the painted area, and not itself that common. Images count
    towards the palette (by their mean colour) but are never emphasised by colour.
    Strength = distance from the nearest main colour / 70, capped at 1.
    Members of a sibling group (see `siblings`) are judged against each other instead:
    the one sibling whose colour differs by ACCENT_MIN_DELTA from the others, which
    agree among themselves, is emphasised, whether bright, dark or light, and the
    others are not. A group whose colours all agree falls back to the palette rule."""
    return _emphasis_cached(json.dumps(source, sort_keys=True))


def siblings(source):
    """Rows of similar units (a row of badges or buttons): groups of >= 2 source units
    that each hold a shape or image, have the same member types, heights within
    SIBLING_HEIGHT of each other, and sit side by side in one row with gaps under
    the average width and no other unit between them. Lists of units (lists of ids).
    Emphasis by sibling contrast needs >= 3, so that the others can agree."""
    return _siblings_cached(json.dumps(source, sort_keys=True))


@lru_cache(maxsize=32)
def _siblings_cached(source_json):
    source = json.loads(source_json)
    src = {e["id"]: e for e in source["elements"]}
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    by_group = {}
    for i, e in src.items():
        if not _is_background(e, sw, sh) and not _is_decoration(e):
            by_group.setdefault(e.get("group_id") or ("__" + i), []).append(i)
    units = list(by_group.values())
    boxes = {tuple(u): _unit_box({i: _drawn_box(src[i]) for i in u}, u) for u in units}

    def sig(u):
        return tuple(sorted(src[i]["type"] for i in u))

    alike = [u for u in units if {"shape", "image"} & set(sig(u)) and not any(src[i].get("rotation", 0) for i in u)]
    groups, seen = [], set()
    for u in sorted(alike, key=lambda u: boxes[tuple(u)][0]):
        if tuple(u) in seen:
            continue
        row = [u]
        for v in sorted(alike, key=lambda v: boxes[tuple(v)][0]):
            a, b = boxes[tuple(row[-1])], boxes[tuple(v)]
            hs = [boxes[tuple(w)][3] - boxes[tuple(w)][1] for w in (*row, v)]
            gap = b[0] - a[2]
            mean_w = sum(boxes[tuple(w)][2] - boxes[tuple(w)][0] for w in (*row, v)) / (len(row) + 1)
            if (tuple(v) in seen or v is u or sig(v) != sig(u) or not 0 <= gap < mean_w
                    or max(hs) > SIBLING_HEIGHT * min(hs) or not a[1] <= (b[1] + b[3]) / 2 <= a[3]):
                continue
            between = (a[2], min(a[1], b[1]), b[0], max(a[3], b[3]))
            # something else in the gap breaks the row (a big unit the row sits on does not)
            if any(_overlap_ratio(boxes[tuple(w)], between) > 0.5 for w in units if w is not v and w is not row[-1]
                   and _area(boxes[tuple(w)]) < 4 * _area(between)):
                continue
            row.append(v)
        if len(row) >= 2:
            groups.append(row)
            seen.update(tuple(w) for w in row)
    return groups


@lru_cache(maxsize=64)
def _asset_mean_rgb(asset_id):
    try:
        with Image.open(assets_dir() / asset_id) as im:
            im = im.convert("RGBA").resize((32, 32))
            px = np.asarray(im, dtype=np.float64)
    except (OSError, TypeError, ValueError):
        return None
    a = px[..., 3:] / 255
    if a.sum() == 0:
        return None
    return tuple(float(v) for v in (px[..., :3] * a).sum((0, 1)) / a.sum())


def _lab(rgb):
    """sRGB 0..255 -> CIELAB (D65)."""
    c = np.asarray(rgb, dtype=np.float64) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    x, y, z = (np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]]) @ c
               / np.array([0.95047, 1.0, 1.08883]))
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    return (116 * f(y) - 16, 500 * (f(x) - f(y)), 200 * (f(y) - f(z)))


def _delta(a, b):
    return math.dist(a, b)


def _corner_gaps(b, corner, w, h):
    gx = b[0] / w if corner.endswith("left") else (w - b[2]) / w
    gy = b[1] / h if corner.startswith("top") else (h - b[3]) / h
    return gx, gy


@lru_cache(maxsize=256)
def _font_metrics(family, bold, size):
    import skia
    style = "bold" if bold else "regular"
    tf = skia.Typeface.MakeFromData(_font_manager().lookup(family, style, style))
    return skia.Font(tf, size).getMetrics().fXHeight, tf.fontStyle().weight()


def _visual_size(el):
    """How big a text looks: x-height at the size the renderer draws, times sqrt(weight / 400)."""
    xh, weight = _font_metrics(el["font_family"], el.get("font_weight") == "bold", int(el["font_size"]))
    return xh * math.sqrt(max(weight, 100) / 400)


def _rel_luminance(rgb):
    lin = np.where(rgb <= 0.03928, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    return lin[..., 0] * 0.2126 + lin[..., 1] * 0.7152 + lin[..., 2] * 0.0722


def _name(unit):
    return unit[0] if len(unit) == 1 else "[" + "+".join(unit) + "]"

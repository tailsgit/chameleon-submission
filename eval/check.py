"""Hard-rule checker for resize() outputs.

`check(source, target_canvas, output)` returns a list of violations, each a
dict `{"id": element id or None, "rule": str, "details": str}`. An empty list
means every rule we can check passed. Rules (see RULES.md and FAQ.md):

  schema               output fails chameleon_design_schema_v0.9.json (types, enums, ranges, extra top-level keys)
  missing_field        a required field is absent (schema `required`, incl. per-type)
  canvas_mismatch      output canvas != target_canvas
  missing_id / unexpected_id / duplicate_id
                       every source id must appear exactly once
  content_changed      text `content` differs from source
  font_family_changed  text `font_family` differs from source
  immutable_changed    type / asset_id / shape_type / style_runs differ (asset swap = disqualifying)
  fit_changed          image `fit` differs (e.g. contain -> fill distorts)
  hidden_element       element shrunk below 1px or set to opacity 0 (a "removed" element in disguise)
  off_canvas           unrotated box outside the canvas (same 0.5px tolerance as engine/main.py)
  off_canvas_rotated   box is inside, but its rotated bounding box is not
  image_aspect         non-cover image box aspect differs from source box aspect by > 5%
  background_not_full_canvas
                       a cover image that filled the source canvas doesn't fill the output (our assumption;
                       not asked of one turned 90/270 degrees on a non-square canvas, which can't without
                       its box going off the canvas)
  text_overflow_height / text_overflow_width
                       wrapped text, laid out exactly as the renderer draws it, exceeds its box
"""

import json
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

REPO = Path(__file__).resolve().parents[1]
FONTS_DIR = REPO / "brand_kit" / "fonts"
ASPECT_TOLERANCE = 0.05
CANVAS_TOLERANCE = 0.5  # px, matches engine/main.py's self-check
TEXT_TOLERANCE = 1.0  # px of overflow allowed before flagging
IMMUTABLE_FIELDS = ("type", "asset_id", "shape_type", "style_runs")

_VALIDATOR = Draft202012Validator(json.loads((REPO / "chameleon_design_schema_v0.9.json").read_text()))


def check(source: dict, target_canvas: dict, output: Any) -> list[dict]:
    violations: list[dict] = []

    def add(eid, rule, details):
        violations.append({"id": eid, "rule": rule, "details": details})

    for err in _VALIDATOR.iter_errors(output):
        path = list(err.absolute_path)
        add(_id_at(output, path), "missing_field" if err.validator == "required" else "schema",
            f"{'/'.join(map(str, path)) or '(root)'}: {err.message}")

    if not isinstance(output, dict) or not isinstance(output.get("elements"), list):
        return violations
    elements = [el for el in output["elements"] if isinstance(el, dict)]

    canvas = output.get("canvas") or {}
    cw, ch = canvas.get("width"), canvas.get("height")
    if (cw, ch) != (target_canvas["width"], target_canvas["height"]):
        add(None, "canvas_mismatch", f"canvas {cw}x{ch}, target {target_canvas['width']}x{target_canvas['height']}")

    src_by_id = {el["id"]: el for el in source["elements"]}
    counts = Counter(el.get("id") for el in elements)
    for eid in src_by_id.keys() - counts.keys():
        add(eid, "missing_id", "source element missing from output")
    for eid in counts.keys() - src_by_id.keys():
        add(eid, "unexpected_id", "element not in source")
    for eid, n in counts.items():
        if n > 1 and eid in src_by_id:
            add(eid, "duplicate_id", f"appears {n} times")

    seen = set()
    for el in elements:
        eid = el.get("id")
        if eid not in src_by_id or eid in seen or not _has_numeric_box(el):
            continue
        seen.add(eid)
        src = src_by_id[eid]

        for field, rule in (("content", "content_changed"), ("font_family", "font_family_changed")):
            if src.get(field) != el.get(field):
                add(eid, rule, f"{src.get(field)!r} -> {el.get(field)!r}")
        for field in IMMUTABLE_FIELDS:
            if src.get(field) != el.get(field):
                add(eid, "immutable_changed", f"{field}: {src.get(field)!r} -> {el.get(field)!r}")
        if src.get("type") == "image" and src.get("fit", "contain") != el.get("fit", "contain"):
            add(eid, "fit_changed", f"{src.get('fit', 'contain')} -> {el.get('fit', 'contain')}")

        if _is_hidden(el) and not _is_hidden(src):
            add(eid, "hidden_element", f"{el['width']:.2f}x{el['height']:.2f}, opacity {el.get('opacity', 1)}")

        if isinstance(cw, (int, float)) and isinstance(ch, (int, float)):
            box = (el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])
            rbox = rotated_bbox(el)
            if _outside(box, cw, ch):
                add(eid, "off_canvas", f"box {_fmt(box)} outside {cw}x{ch}")
            elif _outside(rbox, cw, ch):
                add(eid, "off_canvas_rotated", f"rotated bbox {_fmt(rbox)} outside {cw}x{ch}")

            # a background turned 90/270 degrees can't fill a non-square canvas with its box on it
            # (official off-canvas rule): only the centred square is possible, so that is all we ask
            quarter = round((el.get("rotation", 0) or 0) / 90) % 2 == 1 and cw != ch
            if (el.get("type") == "image" and el.get("fit", "contain") == "cover" and not quarter
                    and _fills(src, source["canvas"]["width"], source["canvas"]["height"])
                    and not _fills(el, cw, ch)):
                add(eid, "background_not_full_canvas", f"box {_fmt(box)} on {cw}x{ch}")

        if el.get("type") == "image" and el.get("fit", "contain") != "cover" and not _is_hidden(el) and not _is_hidden(src):
            ratio = (el["width"] / el["height"]) / (src["width"] / src["height"])
            if abs(ratio - 1) > ASPECT_TOLERANCE:
                add(eid, "image_aspect", f"aspect {ratio - 1:+.1%} vs source box")

        if el.get("type") == "text" and _text_measurable(el):
            layout = text_layout(el)
            if layout["height"] > el["height"] + TEXT_TOLERANCE:
                add(eid, "text_overflow_height",
                    f"{layout['lines']} line(s) need {layout['height']:.1f}px, box is {el['height']:.1f}px")
            if layout["left"] < -TEXT_TOLERANCE or layout["right"] > el["width"] + TEXT_TOLERANCE:
                add(eid, "text_overflow_width",
                    f"widest line spans {layout['left']:.1f}..{layout['right']:.1f}px, box is {el['width']:.1f}px")

    return violations


def text_layout(el: dict) -> dict:
    """Lay out a text element exactly as the official renderer does and
    return its extent in element-local px (unrotated, origin = box top-left):
    {"lines", "left", "right", "height", "line_boxes": [(x0, y0, x1, y1), ...]}.

    Wrapping comes from renderer/converter (`_append_text`: greedy, spaces
    only, float font size, no letter spacing). Line placement comes from
    cr_renderer.text_utils.render_text: line i's top at i * font_size *
    line_height, glyphs at int(font_size), horizontal offset from
    get_textalign_offset. Height is the last line's ascent-to-descent box,
    i.e. what the font reserves, not the ink.
    """
    from cr_renderer import text_utils
    from cr_renderer.schema import TextElement

    fm = _font_manager()
    cols = {k: [] for k in ("text", "font_size", "font", "line_height", "text_align", "capitalize",
                            "letter_spacing", "font_bold", "font_italic", "text_color", "text_line")}
    from renderer.converter.convert import _append_text
    _append_text(cols, el, fm)
    te = TextElement(  # same construction as cr_renderer.renderer._render_to_surface
        uuid="", type="textElement", width=float(el["width"]), height=float(el["height"]),
        text=cols["text"][0], fontSize=cols["font_size"][0], font=cols["font"][0],
        lineHeight=cols["line_height"][0], textAlign=cols["text_align"][0],
        capitalize=cols["capitalize"][0], letterSpacing=cols["letter_spacing"][0],
        boldMap=text_utils.generate_map(cols["font_bold"][0]),
        italicMap=text_utils.generate_map(cols["font_italic"][0]),
        colorMap=text_utils.generate_map(cols["text_color"][0]),
        lineMap=text_utils.generate_map(cols["text_line"][0]),
    )
    boxes = []
    if te.text:
        for i, line in enumerate(text_utils.make_text_spans(te)):
            y = i * te.fontSize * (te.lineHeight or 1.0)
            blobs = [text_utils.make_text_blob(te, span, fm) for span in line]
            x = text_utils.get_textalign_offset(te.textAlign, te.width, blobs)
            depth = max(b.metrics.fDescent - b.metrics.fAscent for b in blobs)
            boxes.append((x, y, x + sum(b.width for b in blobs), y + depth))
    return {
        "lines": len(boxes),
        "left": min((b[0] for b in boxes), default=0.0),
        "right": max((b[2] for b in boxes), default=0.0),
        "height": max((b[3] for b in boxes), default=0.0),
        "line_boxes": boxes,
    }


def rotated_bbox(el: dict) -> tuple[float, float, float, float]:
    """Axis-aligned bounds of the box rotated about its centre."""
    r = math.radians(el.get("rotation", 0) or 0)
    c, s = abs(math.cos(r)), abs(math.sin(r))
    w, h = el["width"], el["height"]
    cx, cy = el["x"] + w / 2, el["y"] + h / 2
    hw, hh = (w * c + h * s) / 2, (w * s + h * c) / 2
    return (cx - hw, cy - hh, cx + hw, cy + hh)


@lru_cache(maxsize=1)
def _font_manager():
    from renderer.converter.convert import _get_renderer
    return _get_renderer(str(FONTS_DIR)).font_manager


def _id_at(output, path):
    if len(path) >= 2 and path[0] == "elements":
        try:
            return output["elements"][path[1]].get("id")
        except (AttributeError, IndexError, KeyError, TypeError):
            pass
    return None


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _has_numeric_box(el):
    return all(_num(el.get(k)) for k in ("x", "y", "width", "height"))


def _text_measurable(el):
    return isinstance(el.get("content"), str) and _num(el.get("font_size")) and el["font_size"] >= 1 \
        and isinstance(el.get("font_family"), str) and el["width"] > 0 and el["height"] > 0


def _is_hidden(el):
    return el["width"] < 1 or el["height"] < 1 or el.get("opacity", 1) == 0


def _outside(box, cw, ch):
    t = CANVAS_TOLERANCE
    return box[0] < -t or box[1] < -t or box[2] > cw + t or box[3] > ch + t


def _fills(el, cw, ch):
    t = 1.0
    return el["x"] <= t and el["y"] <= t and el["x"] + el["width"] >= cw - t and el["y"] + el["height"] >= ch - t


def _fmt(box):
    return "(" + ", ".join(f"{v:.1f}" for v in box) + ")"

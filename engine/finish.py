"""Extra candidates made from the chosen layout, so a big shape change uses its canvas:

  pinned  accents that sat at the source's edges or corners (a script flourish, a corner logo)
          are pinned to the same edges or corners of the new canvas, framing the layout
          instead of sitting in the middle with the content.
  filled  the content block (everything but the background, decorations and pinned accents)
          scaled up as one, so its groups and alignments stay as they are, and centred in
          the free canvas.

They compete with the chosen layout (engine.select); they never replace it.
"""

import copy
import math

from engine.analyze import full_box
from engine.recipes import margin

EDGE_PIN = 0.05     # within this share of a canvas side from an edge: pinned to that edge
ACCENT_MAX = 0.15   # an accent covers less than this share of the source canvas
FILL_MAX = 1.6      # the content block grows at most this much
FILL_MIN = 1.05     # ...and a smaller gain is not worth a candidate


def variants(a, output, tw, th):
    """[pinned, filled, pinned then filled], leaving out those that change nothing."""
    acc = accents(a)
    pinned = pin(a, output, acc, tw, th) if acc else None
    out = [pinned, fill(a, output, acc, tw, th), fill(a, pinned, acc, tw, th) if pinned else None]
    return [o for o in out if o is not None]


def accents(a):
    """Ids of lone, unrotated images (not the hero) under ACCENT_MAX of the canvas that touch a canvas edge."""
    hero = a.by_role("hero")
    ids = []
    for u in a.units.values():
        el = a.els[u.members[0]]
        if (len(u.members) == 1 and el["type"] == "image" and not el.get("rotation", 0) and u is not hero
                and el["width"] * el["height"] < ACCENT_MAX * a.sw * a.sh and _edges(el, a.sw, a.sh) != (None, None)):
            ids.append(el["id"])
    return ids


def _edges(el, sw, sh):
    """('left' | 'right' | None, 'top' | 'bottom' | None): the canvas edges the element sits at."""
    x0, y0, x1, y1 = el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"]
    h = "left" if x0 <= EDGE_PIN * sw else "right" if x1 >= (1 - EDGE_PIN) * sw else None
    v = "top" if y0 <= EDGE_PIN * sh else "bottom" if y1 >= (1 - EDGE_PIN) * sh else None
    return h, v


def pin(a, output, ids, tw, th):
    out = copy.deepcopy(output)
    els = {e["id"]: e for e in out["elements"]}
    for i in ids:
        src, el = a.els[i], els[i]
        s = el["width"] / max(src["width"], 1e-6)  # its gap to the edge scales with it
        h, v = _edges(src, a.sw, a.sh)
        w, hh = el["width"], el["height"]
        if h:
            el["x"] = max(0.0, src["x"]) * s if h == "left" else tw - w - max(0.0, a.sw - src["x"] - src["width"]) * s
        if v:
            el["y"] = max(0.0, src["y"]) * s if v == "top" else th - hh - max(0.0, a.sh - src["y"] - src["height"]) * s
        el["x"], el["y"] = min(max(0.0, el["x"]), tw - w), min(max(0.0, el["y"]), th - hh)
    return out if out != output else None


def fill(a, output, exclude, tw, th):
    skip = set(a.background) | set(exclude) | {m for d in a.decorations for m in d.members}
    block = [e for e in output["elements"] if e["id"] not in skip]
    if not block:
        return None
    boxes = [full_box(e) for e in block]
    b = (min(x[0] for x in boxes), min(x[1] for x in boxes), max(x[2] for x in boxes), max(x[3] for x in boxes))
    m = margin(tw, th)
    f = min(FILL_MAX, (tw - 2 * m) / max(b[2] - b[0], 1e-6), (th - 2 * m) / max(b[3] - b[1], 1e-6))
    if f < FILL_MIN:
        return None
    out = copy.deepcopy(output)
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    for e in out["elements"]:
        if e["id"] in skip:
            continue
        ex, ey = e["x"] + e["width"] / 2, e["y"] + e["height"] / 2  # scale about the block's centre, then centre it
        e["width"], e["height"] = e["width"] * f, e["height"] * f
        e["x"], e["y"] = tw / 2 + (ex - cx) * f - e["width"] / 2, th / 2 + (ey - cy) * f - e["height"] / 2
        if e["type"] == "text":
            e["font_size"] = max(1, math.floor(e["font_size"] * f))  # floored: the scaled box still holds it
    return out

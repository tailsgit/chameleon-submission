"""Choose the winning candidate, and the safety nets behind it.

1. Drop duplicate candidates and any with a hard-rule violation (eval.check), then keep the best tier
   available of: no text both off the element it sat on and short of its contrast; no text newly on a
   busy image; no large element far below its share of the canvas (see _safest).
2. Rank the rest by the geometric score (eval.score, ~1 ms each), render the top
   RENDER_TOP (fewer once the deadline has passed) and pick the best full score including contrast. Rendering is
   expensive under amd64 emulation, so only those few get rendered. A `preferred`
   candidate (the source scaled uniformly, when the aspect ratio barely changes)
   wins unless another beats it by more than PREFER_MARGIN points.
3. If nothing passes: repair every candidate (pull into the canvas, fix image
   proportions, shrink text until it fits) plus a uniform-scale fallback, and return
   the one with the fewest violations, then the best score.
"""

import copy
import json
import logging
import math
import os

from engine.analyze import full_box
from engine.clock import now
from eval.check import check, text_layout
from eval.score import MIN_TEXT_PX_ABS, MIN_TEXT_REL_HEIGHT, SHARE_FLOOR, busy_text_over_image, contrast_drops, contrast_kept, contrast_worst, large_share_kept, score, text_off_backdrop

RENDER_TOP = 10
REPAIR_SPACED_LAYOUTS = 300  # repair(): text layouts tried per letter-spaced text that fits at no size (count, not time)
PREFER_MARGIN = 2.0  # points out of 100: "clearly better" than the preferred candidate
log = logging.getLogger(__name__)


def select(source, target, outputs, preferred=None, deadline=None, *, render_limit=None):
    pref_key = json.dumps(preferred, sort_keys=True) if preferred is not None else None
    bonus = lambda o: PREFER_MARGIN if pref_key and json.dumps(o, sort_keys=True) == pref_key else 0.0
    seen, unique = set(), []
    for o in ([preferred] if preferred is not None else []) + outputs:
        key = json.dumps(o, sort_keys=True)
        if key not in seen:
            seen.add(key)
            unique.append(o)

    passing = [(o, score(source, target, o)["total"] + bonus(o)) for o in unique if _passes(source, target, o)]
    passing = _safest(source, target, passing)
    if passing:
        top = sorted(passing, key=lambda t: -t[1])[:RENDER_TOP if render_limit is None else min(RENDER_TOP, render_limit)]
        best, rendered = top[0][0], []
        try:
            for o, _s in top:  # best geometric first; past the deadline, only what is already rendered counts
                if deadline is not None and now() > deadline:
                    log.info("time budget reached after %d contrast renders", len(rendered))
                    break
                rendered.append((score(source, target, o, contrast=True)["total"] + bonus(o), o))
        except Exception:  # the renderer is unavailable: fall back to the geometric ranking
            log.exception("contrast scoring failed; using geometric ranking")
            return best
        return max(rendered, key=lambda t: t[0])[1] if rendered else best

    log.warning("no candidate passed check(); repairing %d candidates", len(unique))
    repaired = [repair(source, target, o) for o in unique] + [fallback(source, target)]
    def badness(o):
        try:
            return len(check(source, target, o)), -score(source, target, o)["total"]
        except Exception:
            return math.inf, 0
    return min(repaired, key=badness)


def _safest(source, target, passing):
    """The candidates in the best tier available, judged together (one filter after another could leave
    only candidates the next one rejects): first, no text both ends up mostly off the card, label or panel it
    sat on (eval.score.text_off_backdrop) and loses its contrast (contrast_drops, estimated without
    rendering); then no
    text newly on a busy image (busy_text_over_image); then no large element far below its share
    (large_share_kept). When no candidate is legible, the one whose worst text kept the most of its contrast
    remains (then by the contrast kept on average, then the score)."""
    if not passing:
        return passing
    tiers = [(_tier(source, target, o), o, s) for o, s in passing]
    best = max(t for t, _o, _s in tiers)
    kept = [(o, s) for t, o, s in tiers if t == best]
    if not best[0]:  # nothing is legible everywhere: the best worst-case contrast wins, whatever else scores
        rank = lambda t: (contrast_worst(source, target, t[1]), contrast_kept(source, target, t[1]), t[2])
        _t, o, s = max(tiers, key=rank)
        log.info("no legible candidate of %d: chose the best contrast", len(tiers))
        return [(o, s)]
    names = ("text off its backdrop or contrast lost", "text newly on a busy image", "a large element far below its share",
             "text far below its readable size")
    for k, name in enumerate(names[:len(best)]):
        n = sum(1 for t, _o, _s in tiers if t[:k] == best[:k] and t[k] < best[k])
        if n:
            log.info("rejected %d candidates with %s", n, name)
    return kept


def _tier(source, target, o):
    """(legible, no text newly on a busy image, large elements keep their share): see _safest."""
    off = {t for t, _b, _s in text_off_backdrop(source, target, o)}
    legible = not off or not off & {t for t, _s, _o in contrast_drops(source, target, o)}
    tier = legible, not busy_text_over_image(source, target, o), large_share_kept(source, target, o) >= SHARE_FLOOR
    return tier + (_text_kept(source, target, o),) if os.environ.get("CHAMELEON_TEXT_TIER") == "1" else tier


TEXT_TIER_SHARE = 0.5  # CHAMELEON_TEXT_TIER=1: no text drawn under this share of the size small_text_penalty asks for


def _text_kept(source, target, o):
    """Every text drawn at >= TEXT_TIER_SHARE x min(readable floor, its source size), the floor of eval.score.small_text_penalty."""
    floor = max(MIN_TEXT_PX_ABS, MIN_TEXT_REL_HEIGHT * target["height"])
    src = {e["id"]: e for e in source["elements"]}
    return all(int(e["font_size"]) >= TEXT_TIER_SHARE * min(floor, src[e["id"]]["font_size"])
               for e in o["elements"] if e["type"] == "text" and e["id"] in src)


def better_variant(source, target, best, extra):
    """The variant (engine.finish) to use instead of the chosen layout, else the chosen layout: it must pass
    check(), be in no worse a tier (_tier) and score higher. A safer tier alone doesn't win here: the chosen
    layout already won its tier against every candidate."""
    tb, sb = _tier(source, target, best), score(source, target, best)["total"]
    ok = [(score(source, target, v)["total"], k, v) for k, v in enumerate(extra)
          if _passes(source, target, v) and _tier(source, target, v) >= tb]
    s, _k, v = max(ok, default=(sb, 0, best))
    return v if s > sb else best


def _passes(source, target, output):
    try:
        return not check(source, target, output)
    except Exception:
        log.exception("check() failed on a candidate; discarding it")
        return False


def fallback(source, target):
    """Uniform scale of the whole design (nothing distorts, wrapping is unchanged),
    centred, background full canvas, then repaired."""
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    tw, th = target["width"], target["height"]
    s = min(tw / sw, th / sh)
    ox, oy = (tw - sw * s) / 2, (th - sh * s) / 2
    out = {"canvas": {"width": tw, "height": th}, "elements": copy.deepcopy(source["elements"])}
    for el in out["elements"]:
        el.update(x=ox + el["x"] * s, y=oy + el["y"] * s, width=el["width"] * s, height=el["height"] * s)
        if el["type"] == "text":
            el["font_size"] = max(1, math.floor(el["font_size"] * s))
    try:
        return repair(source, target, out)
    except Exception:  # the plain uniform fit is still in proportion and on the canvas: the last resort
        log.exception("repair failed; using the unrepaired uniform fit")
        return out


def repair(source, target, output):
    """Best-effort fix of every hard rule we can fix locally. Never raises."""
    tw, th = target["width"], target["height"]
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    src = {e["id"]: e for e in source["elements"]}
    out = copy.deepcopy(output)
    out["canvas"] = {"width": tw, "height": th}
    for el in out["elements"]:
        s = src.get(el.get("id"))
        if s is None:
            continue
        try:
            _repair_element(el, s, sw, sh, tw, th)
        except Exception:
            log.exception("repair failed for %s", el.get("id"))
    return out


def _repair_element(el, s, sw, sh, tw, th):
    fills_src = s["x"] <= 1 and s["y"] <= 1 and s["x"] + s["width"] >= sw - 1 and s["y"] + s["height"] >= sh - 1
    if fills_src and el["type"] in ("image", "shape") and not el.get("rotation", 0):
        if el["type"] == "shape" or el.get("fit", "contain") == "cover":
            el.update(x=0, y=0, width=tw, height=th)
            return
    el["width"], el["height"] = max(1.0, el["width"]), max(1.0, el["height"])
    if el["type"] == "image" and el.get("fit", "contain") != "cover" and s["width"] > 0 and s["height"] > 0:
        ar = s["width"] / s["height"]
        cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
        w = min(el["width"], el["height"] * ar)
        el.update(width=w, height=w / ar)
        el.update(x=cx - el["width"] / 2, y=cy - el["height"] / 2)

    b = full_box(el)
    bw, bh = b[2] - b[0], b[3] - b[1]
    if bw > tw or bh > th:
        f = 0.999 * min(tw / bw, th / bh)
        cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
        el.update(width=el["width"] * f, height=el["height"] * f)
        el.update(x=cx - el["width"] / 2, y=cy - el["height"] / 2)
        if el["type"] == "text":
            el["font_size"] = max(1, math.floor(el["font_size"] * f))
        b = full_box(el)
    dx = -b[0] if b[0] < 0 else (tw - b[2] if b[2] > tw else 0)
    dy = -b[1] if b[1] < 0 else (th - b[3] if b[3] > th else 0)
    el.update(x=el["x"] + dx, y=el["y"] + dy)

    if el["type"] == "text":
        def fits(fs, w, h):
            lay = text_layout(dict(el, font_size=fs, width=w))
            return lay["height"] <= h + 0.5, lay["left"] >= -0.5 and lay["right"] <= w + 0.5, lay["lines"]

        fs = int(el["font_size"])
        while fs > 1 and not all(fits(fs, el["width"], el["height"])[:2]):
            fs -= 1
        el["font_size"] = max(1, fs)
        w = el["width"] if all(fits(el["font_size"], el["width"], el["height"])[:2]) else None
        if w is None and el.get("letter_spacing"):
            # Letter-spaced text that fits at no size: the converter wraps without the spacing the renderer adds, so
            # a narrower box (earlier line breaks) can leave room for it (as fit.Sizer.fit_width), with the box free to
            # grow to the canvas height. Sizes from the source's down, 3% narrower each try; at most BUDGET layouts.
            budget = REPAIR_SPACED_LAYOUTS
            h_max = tw if round((el.get("rotation", 0) or 0) / 90) % 2 else th  # a quarter-turned box's height runs across
            words = len(el["content"].split())
            for f2 in range(int(s["font_size"]), 0, -1):
                w2 = el["width"]
                while budget > 0 and w2 >= 1:
                    budget -= 1
                    tall_ok, wide_ok, lines = fits(f2, w2, h_max)
                    if not tall_ok or (lines >= words and not wide_ok):
                        break  # a narrower box only adds lines; or every word is on its own line already
                    if wide_ok:
                        w = w2
                        break
                    w2 *= 0.97
                if w is not None or budget <= 0:
                    break
            if w is not None:
                el["font_size"] = f2
                el["height"] = min(h_max, text_layout(dict(el, width=w))["height"] + 1)
        if w is not None:  # narrowed about its centre (unchanged when it already fits), kept on the canvas
            el.update(x=el["x"] + (el["width"] - w) / 2, width=w)
            b = full_box(el)
            el.update(x=el["x"] + (-b[0] if b[0] < 0 else min(0.0, tw - b[2])),
                      y=el["y"] + (-b[1] if b[1] < 0 else min(0.0, th - b[3])))
        else:  # still too big at this size: grow the box, keeping it on the canvas
            lay = text_layout(el)
            w = min(tw, max(el["width"], lay["right"] - min(0.0, lay["left"]) + 1))
            el.update(width=w, x=min(max(0.0, el["x"] + min(0.0, lay["left"])), tw - w))
            h = min(th, text_layout(el)["height"] + 1)
            el.update(height=h, y=min(el["y"], th - h))
            b = full_box(el)  # a rotated box can outgrow the canvas: shrink it about its centre (the text may still
            if b[2] - b[0] > tw or b[3] - b[1] > th:  # overflow, but nothing goes off the canvas), then pull it back in
                f = 0.999 * min(tw / (b[2] - b[0]), th / (b[3] - b[1]))
                cx, cy = el["x"] + el["width"] / 2, el["y"] + el["height"] / 2
                el.update(width=el["width"] * f, height=el["height"] * f)
                el.update(x=cx - el["width"] / 2, y=cy - el["height"] / 2)
                b = full_box(el)
            el.update(x=el["x"] + (-b[0] if b[0] < 0 else min(0.0, tw - b[2])),
                      y=el["y"] + (-b[1] if b[1] < 0 else min(0.0, th - b[3])))

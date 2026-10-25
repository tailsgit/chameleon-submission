"""Retrieved layouts: candidate plans copied from real designs of a similar shape.

The library (engine/data/layout_library.json, built by eval/build_library.py) holds the
geometry and roles of real designs, as our analyzer sees them. For a task:

1. keep the library designs whose aspect ratio is within ASPECT_TOL of the target's;
2. rank them by how closely their composition (units per role) matches ours, keep TOP_K;
3. map our layout units into each design's slots (its units, scaled to the target canvas)
   with an optimal assignment (Hungarian) on role, kind and size. A unit whose best slot
   costs more than LEFTOVER is left for free space (engine.fit places it like any floating
   unit), and unused slots are ignored;
4. size each mapped unit inside its slot with the same Sizer the recipes use (text: the
   largest font whose wrapped lines fit the slot; images and groups: one uniform scale,
   sibling rows and grids re-flowed), so engine.fit applies the same rules to these plans.

Switch: CHAMELEON_RETRIEVAL=0 in the environment turns retrieval off (recipes only).
"""

import json
import logging
import math
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

from engine.clock import now

log = logging.getLogger(__name__)
LIBRARY = Path(__file__).parent / "data" / "layout_library.json"
ASPECT_TOL = 0.2   # |ln(aspect) - ln(target aspect)|: within ~22% of the target's shape
TOP_K = 30
LEFTOVER = 2.5     # a unit whose best slot costs more than this goes into free space instead
FAMILY = {"headline": "text", "subhead": "text", "body": "text", "fineprint": "text", "cta": "cta",
          "hero": "image", "logo": "image", "badge": "image", "decoration": "image"}
BISECT_STEPS = 14
CAP, IMG_CAP = 1.2, 1.4  # never larger than the recipes allow: k_cap(1.2), and 1.4x that for images


@lru_cache(maxsize=1)
def library():
    try:
        return json.loads(LIBRARY.read_text())
    except (OSError, ValueError):
        return []


def retrieved(a, sizer, tw, th, deadline=None, *, limit=None, spread=False):
    """Up to TOP_K plans, one per retrieved design, best composition match first; none
    more once `deadline` (now()) has passed. None when switched off (see module doc)."""
    if os.environ.get("CHAMELEON_RETRIEVAL", "1") == "0":
        return []
    from engine.recipes import Plan, margin  # recipes imports nothing from here; avoids a cycle at import time

    want = math.log(tw / th)
    ours = _roles(a.units.values())
    near = [d for d in library() if abs(math.log(d["aspect"]) - want) <= ASPECT_TOL]
    near.sort(key=lambda d: (sum(abs(ours.get(r, 0) - d["roles"].get(r, 0)) for r in set(ours) | set(d["roles"])),
                             abs(math.log(d["aspect"]) - want)))
    flow = [u for u in a.units.values() if not u.floating and not u.host]
    plans = []
    ranked = list(enumerate(near[:TOP_K]))
    if spread and limit is not None and 0 < limit < len(ranked):
        # Fixed rank strata do not shift when fewer than TOP_K entries are near.
        # Include the best match, then each stratum's far edge; fill vacant tail
        # strata from the head. Seven representatives span five-rank bands.
        stride = math.ceil(TOP_K / (limit - 1)) if limit > 1 else TOP_K
        positions = {0} | {min(i * stride - 1, len(ranked) - 1) for i in range(1, limit)}
        for i in range(len(ranked)):
            if len(positions) >= limit:
                break
            positions.add(i)
        ranked = [ranked[i] for i in sorted(positions)]
    elif limit is not None:
        ranked = ranked[:max(0, limit)]
    for rank, d in ranked:
        if deadline is not None and now() > deadline:
            log.info("time budget reached after %d retrieved layouts", rank)
            break
        slots, used = [], []
        if flow:
            cost = np.array([[_cost(a, sizer, u, s, tw, th) for s in d["units"]] for u in flow])
            for i, j in assign(np.hstack([cost, np.full((len(flow), len(flow)), LEFTOVER)])):
                if j < len(d["units"]):
                    slots.append(_place(a, sizer, flow[i], d["units"][j]["box"], tw, th))
                    used.append(flow[i].id)
        floating = [uid for uid in a.units if uid not in used]
        scales = [s.scale if s.fs is None else s.fs / a.els[a.units[s.uid].members[0]]["font_size"] for s in slots]
        k = float(np.median(scales)) if scales else math.sqrt(tw * th / (a.sw * a.sh))
        plans.append(Plan("retrieved", {"rank": rank}, slots, floating, k, margin(tw, th), sizer))
    return plans


def _roles(units):
    out = {}
    for u in units:
        out[u.role] = out.get(u.role, 0) + 1
    return out


def _cost(a, sizer, u, slot, tw, th):
    """Role (same, same family, other), kind (text into a text slot), size (area share)
    and, for units that keep their aspect, how much of the slot they could fill. Text, or a
    group with text, that can't fit a slot at a readable size never goes there (it is left
    for free space)."""
    x0, y0, x1, y1 = slot["box"]
    if u.kind == "text":
        eid = u.members[0]
        if not _text_fits(sizer, eid, sizer.fs_min(eid), (x1 - x0) * tw, (y1 - y0) * th):
            return 2 * LEFTOVER
    elif sizer.read_scale(u) and sizer.unit_extent(u, 1e6, max(1.0, (x1 - x0) * tw), max(1.0, (y1 - y0) * th)).scale < sizer.read_scale(u) - 1e-9:
        return 2 * LEFTOVER  # a group whose text would be unreadable in this slot
    role = 0.0 if u.role == slot["role"] else 0.6 if FAMILY.get(u.role, u.role) == FAMILY.get(slot["role"], slot["role"]) else 1.5
    kind = 0.0 if (u.kind == "text") == (slot["kind"] == "text") else 1.0
    ua, sa = max(u.w * u.h / (a.sw * a.sh), 1e-6), max((x1 - x0) * (y1 - y0), 1e-6)
    size = 0.5 * abs(math.log(ua / sa))
    shape = 0.0 if u.kind == "text" else 0.5 * abs(math.log(max(u.w, 1e-6) / max(u.h, 1e-6) / max((x1 - x0) * tw, 1e-6) * max((y1 - y0) * th, 1e-6)))
    return role + kind + size + shape


def _place(a, sizer, u, box, tw, th):
    from engine.recipes import Slot, k_cap

    x0, y0 = box[0] * tw, box[1] * th
    W, H = max(1.0, (box[2] - box[0]) * tw), max(1.0, (box[3] - box[1]) * th)
    if u.kind == "text":
        eid = u.members[0]
        src_fs = a.els[eid]["font_size"]
        lo, hi = 0.0, k_cap(a, tw, th, CAP)  # largest scale whose lines fit the slot
        for _ in range(BISECT_STEPS):
            mid = (lo + hi) / 2
            fs = max(sizer.fs_min(eid), math.floor(src_fs * mid + 1e-9))
            lo, hi = (mid, hi) if _text_fits(sizer, eid, fs, W, H) else (lo, mid)
        e = sizer.text_extent(eid, lo, W)
        align = a.els[eid].get("text_align", "left")
        x = x0 + (W - e.w) / 2 if align == "center" else x0 + W - e.w if align == "right" else x0 - e.inset
        x, y = _clamp(x, 0, tw - e.w), _clamp(y0 + (H - e.h) / 2, 0, th - e.h)
        return Slot(u.id, x, y, e.w, e.h, fs=e.fs, slack=e.slack)
    e = sizer.unit_extent(u, IMG_CAP * k_cap(a, tw, th, CAP), W, H)  # as large as fits the slot, up to the cap
    x, y = _clamp(x0 + (W - e.w) / 2, 0, tw - e.w), _clamp(y0 + (H - e.h) / 2, 0, th - e.h)
    return Slot(u.id, x, y, e.w, e.h, scale=e.scale, cols=e.cols)


def _text_fits(sizer, eid, fs, W, H):
    """Text at font size fs wraps inside a W x H slot (one layout; letter-spaced text that
    overflows the full width searches narrower boxes, as engine.fit does)."""
    W = max(1.0, W)
    if sizer.layout(eid, fs, W)["height"] > H:  # narrower boxes only wrap to more lines
        return False
    w = sizer.fit_width(eid, fs, W)
    return w is not None and sizer.layout(eid, fs, w)["height"] <= H


def _clamp(v, lo, hi):
    return max(lo, min(v, hi)) if hi >= lo else lo


def assign(cost):
    """Minimum-cost assignment of every row to a distinct column (rows <= columns), as
    [(row, col)]: the Hungarian algorithm with potentials, O(rows^2 x cols)."""
    n, m = cost.shape
    u, v = np.zeros(n + 1), np.zeros(m + 1)
    p, way = np.zeros(m + 1, dtype=int), np.zeros(m + 1, dtype=int)  # p[j]: row (1-based) on column j
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minv, used = np.full(m + 1, np.inf), np.zeros(m + 1, dtype=bool)
        while p[j0]:
            used[j0] = True
            i0 = p[j0]
            free = ~used[1:]
            cur = cost[i0 - 1] - u[i0] - v[1:]
            better = free & (cur < minv[1:])
            minv[1:][better], way[1:][better] = cur[better], j0
            j1 = 1 + int(np.argmin(np.where(free, minv[1:], np.inf)))
            delta = minv[j1]
            u[p[used]] += delta
            v[used] -= delta
            minv[1:][free] -= delta
            j0 = j1
        while j0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
    return sorted((int(p[j]) - 1, j - 1) for j in range(1, m + 1) if p[j])

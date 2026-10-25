"""Preserve-first candidates: the source layout, changed as little as possible to fit the target.

The source is laid on the target canvas in normalised coordinates, then one small integer program
(OR-Tools CP-SAT) moves and resizes its items as little as possible:

  items       units joined by the source's strong relationships (eval.score.relations: overlaps,
              labels, tight neighbours) move as one item, at one scale, insides and stacking as
              designed; other units are items of their own; attached badges follow their host
              (engine.fit) as before
  variables   per item: x, y (px) and a scale s (thousandths); width and height follow s, so image
              proportions are exact
  hard        inside the canvas; text at least readable; a large item (>= LARGE_SHARE of the source)
              keeps at least LARGE_KEEP of the scale that keeps its share of the canvas; no new overlap
              between items that did not overlap in the source
  objective   move (|x - x*| + |y - y*|) and resize (|s - s*|, shrinking costs SHRINK_COST times more)
              as little as possible, each weighted by the item's share of the source; change the
              source's left/right or above/below order between items (ORDER), or break an alignment
              they shared (ALIGN), only when it pays

Variants (target sizes s* and positions x*, y*): "share" keeps each item's share of the canvas,
centres at their normalised positions; "fit" scales the whole design uniformly to fit; "edges"
keeps the share and keeps items that touched a canvas edge at that edge; "between" is halfway
between share and fit. These suit moderate shape changes; recipes and retrieval cover extreme ones.
The solver runs single-threaded with a deterministic time limit, so a task gives the same layout
on every machine.
"""

import logging
import math

from ortools.sat.python import cp_model

from engine.clock import now
from eval.score import relations

log = logging.getLogger(__name__)
LARGE_SHARE = 0.1    # an item covering this share of the source canvas is large...
LARGE_KEEP = 0.7     # ...and keeps at least this share of the scale that preserves its share of the canvas
SHRINK_COST = 3      # shrinking costs this many times more than growing
ORDER = 2000         # cost of swapping two items' left/right or above/below order (in px moved)
ALIGN = 2            # cost per px of drifting apart from an edge two items shared in the source
RES = 100            # costs are integers: px x RES
CLUSTER_MAX = 0.6    # a cluster of units never grows past this share of the source canvas
SOLVE_TIME = 0.5     # deterministic seconds per variant
VARIANTS = ("share", "fit", "edges", "between")


def preserve_plans(a, sizer, tw, th, deadline=None):
    """One plan per variant that the solver can satisfy."""
    from engine.recipes import Plan

    items = _items(a)
    if not items:
        return []
    plans = []
    for v in VARIANTS:
        if deadline is not None and now() > deadline:
            log.info("time budget reached after %d preserve variants", len(plans))
            break
        sol = _solve(a, sizer, items, tw, th, v)
        if sol is not None:
            slots = [s for item, (x, y, k) in zip(items, sol) for s in _slots(a, item, x, y, k)]
            ks = sorted(k for _x, _y, k in sol)
            plans.append(Plan("preserve", {"variant": v}, slots, [u for u, w in a.units.items() if w.host], ks[len(ks) // 2], 0.0, sizer))
    return plans


def _items(a):
    """[(unit ids, source box)]: units joined by strong relationships, bottom-to-top stacking kept."""
    units = {uid: u for uid, u in a.units.items() if not u.host}
    owner = {m: uid for uid, u in units.items() for m in u.members}
    parent = {uid: uid for uid in units}
    span = {uid: u.box for uid, u in units.items()}

    def find(i):
        while parent[i] != i:
            i = parent[i]
        return i
    for r in relations(a.source):
        if r["strength"] < 1.0 or r["a"] not in owner or r["b"] not in owner:
            continue
        x, y = find(owner[r["a"]]), find(owner[r["b"]])
        if x == y:
            continue
        box = _union([span[x], span[y]])
        if (box[2] - box[0]) * (box[3] - box[1]) > CLUSTER_MAX * a.sw * a.sh:
            continue
        parent[x], span[y] = y, box
    groups = {}
    for uid in units:
        groups.setdefault(find(uid), []).append(uid)
    return [(uids, _union([units[u].box for u in uids])) for uids in groups.values()]


def _solve(a, sizer, items, tw, th, variant):
    sw, sh = a.sw, a.sh
    share_s = math.sqrt(tw * th / (sw * sh))      # scale that keeps a share of the canvas
    fit_s = min(tw / sw, th / sh)                 # scale that fits the whole design
    m = cp_model.CpModel()
    X, Y, S, W, H, cost = [], [], [], [], [], []
    for uids, (x0, y0, x1, y1) in items:
        w0, h0 = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
        share = w0 * h0 / (sw * sh)
        fits = min(tw / w0, th / h0)              # largest scale that fits the canvas at all
        keep = min(share_s, fits)
        want = {"share": keep, "fit": fit_s, "edges": keep, "between": math.sqrt(keep * fit_s)}[variant]
        want = min(want, fits)
        lo = max(sizer.read_scale(_as_unit(a, uids)), max(sizer.floor_scale(a.units[u]) for u in uids))
        if share >= LARGE_SHARE:
            lo = max(lo, LARGE_KEEP * keep)
        if lo > fits:
            return None                            # can't stay readable / large and fit: not for this variant
        s = m.NewIntVar(math.ceil(lo * 1000), max(math.ceil(lo * 1000), math.floor(fits * 1000)), "")
        w, h = m.NewIntVar(0, tw, ""), m.NewIntVar(0, th, "")
        m.Add(1000 * w >= s * round(w0) - 500)
        m.Add(1000 * w <= s * round(w0) + 500)
        m.Add(1000 * h >= s * round(h0) - 500)
        m.Add(1000 * h <= s * round(h0) + 500)
        x, y = m.NewIntVar(0, tw, ""), m.NewIntVar(0, th, "")
        m.Add(x + w <= tw)
        m.Add(y + h <= th)
        # target position: the item's centre at its normalised source centre, or at the edge it touched
        cx, cy = (x0 + x1) / 2 / sw * tw - w0 * want / 2, (y0 + y1) / 2 / sh * th - h0 * want / 2
        if variant == "edges":
            cx = x0 * want if x0 <= 0.12 * sw else tw - (sw - x1) * want - w0 * want if x1 >= 0.88 * sw else cx
            cy = y0 * want if y0 <= 0.12 * sh else th - (sh - y1) * want - h0 * want if y1 >= 0.88 * sh else cy
        weight = 1 + round(20 * share)
        for var, target, hi in ((x, cx, tw), (y, cy, th)):
            d = m.NewIntVar(0, hi, "")
            m.AddAbsEquality(d, var - round(target))
            cost.append(RES * weight * d)
        grow, shrink = m.NewIntVar(0, 10 ** 6, ""), m.NewIntVar(0, 10 ** 6, "")
        m.Add(s - round(want * 1000) == grow - shrink)
        px = max(1, round(RES * weight * max(w0, h0) / 1000))  # a thousandth of scale moves its longer side w0/1000 px
        cost.append(px * grow)
        cost.append(px * SHRINK_COST * shrink)
        X.append(x), Y.append(y), S.append(s), W.append(w), H.append(h)
    n = len(items)
    for i in range(n):
        bi = items[i][1]
        for j in range(i + 1, n):
            bj = items[j][1]
            if min(bi[2], bj[2]) - max(bi[0], bj[0]) > 0 and min(bi[3], bj[3]) - max(bi[1], bj[1]) > 0:
                continue                           # overlapped in the source: may overlap
            sides = [m.NewBoolVar("") for _ in range(4)]   # i left of j, right of j, above j, below j
            m.Add(X[i] + W[i] <= X[j]).OnlyEnforceIf(sides[0])
            m.Add(X[j] + W[j] <= X[i]).OnlyEnforceIf(sides[1])
            m.Add(Y[i] + H[i] <= Y[j]).OnlyEnforceIf(sides[2])
            m.Add(Y[j] + H[j] <= Y[i]).OnlyEnforceIf(sides[3])
            m.AddBoolOr(sides)
            was = [bi[2] <= bj[0], bj[2] <= bi[0], bi[3] <= bj[1], bj[3] <= bi[1]]
            kept = [sides[k] for k in range(4) if was[k]]
            if kept:                               # the source's order between them, unless it pays to change it
                broken = m.NewBoolVar("")
                m.AddBoolOr(kept + [broken])
                cost.append(RES * ORDER * broken)
            for lo_i, hi_i, lo_j, hi_j, V, D in ((bi[0], bi[2], bj[0], bj[2], X, W), (bi[1], bi[3], bj[1], bj[3], Y, H)):
                if abs(lo_i - lo_j) <= 2:          # a shared left (top) edge stays shared
                    d = m.NewIntVar(0, max(tw, th), "")
                    m.AddAbsEquality(d, V[i] - V[j])
                    cost.append(RES * ALIGN * d)
                elif abs(hi_i - hi_j) <= 2:        # ...or right (bottom) edge
                    d = m.NewIntVar(0, max(tw, th), "")
                    m.AddAbsEquality(d, V[i] + D[i] - V[j] - D[j])
                    cost.append(RES * ALIGN * d)
    m.Minimize(sum(cost))
    solver = cp_model.CpSolver()
    solver.parameters.num_workers = 1
    solver.parameters.random_seed = 0
    solver.parameters.max_deterministic_time = SOLVE_TIME
    if solver.Solve(m) not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    return [(solver.Value(X[i]), solver.Value(Y[i]), solver.Value(S[i]) / 1000) for i in range(n)]


def _slots(a, item, x, y, k):
    """A slot per unit of the item: its source offset in the item, scaled by k."""
    from engine.recipes import Slot

    uids, (x0, y0, _x1, _y1) = item
    out = []
    for uid in uids:
        u = a.units[uid]
        ux, uy, w, h = x + (u.box[0] - x0) * k, y + (u.box[1] - y0) * k, u.w * k, u.h * k
        if u.kind == "text":
            el = a.els[u.members[0]]
            fs = max(1, math.floor(el["font_size"] * k + 1e-9))
            out.append(Slot(uid, ux, uy, w, h, fs=fs, slack=fs * el.get("line_height", 1.2)))
        else:
            out.append(Slot(uid, ux, uy, w, h, scale=k, cols=(u.cols or len(u.parts)) if u.parts else None))
    return out


def _as_unit(a, uids):
    """A stand-in unit with all the items' members, for the Sizer's readable-scale check."""
    from engine.analyze import Unit
    return Unit(id="+".join(uids), members=[m for u in uids for m in a.units[u].members], kind="rigid", box=(0, 0, 1, 1))


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))

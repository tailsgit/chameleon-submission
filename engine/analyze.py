"""Understand a source design before re-laying it out.

Everything here works from geometry, element types and text content only, with no
knowledge of particular designs or ids. Steps:

1. Background: elements whose drawing covers the whole canvas (within 1% of each side),
   unrotated or turned by a multiple of 90 degrees, at the bottom of the stacking order.
2. Rigid units (union-find; each unit then moves and scales as one piece):
   - elements sharing a group_id
   - text whose drawn lines sit >= 90% inside a lower shape that covers < 25% of
     the canvas (a button or badge), or a lower image that covers < 50% (a card,
     a speech bubble, a label); the tightest such holder wins
   - a non-text element >= 90% inside such a holder (an icon on a card)
   - a decoration whose area is >= 80% inside a content unit (tight layering)
   - the pairs of eval.score.tight_relations (label + value, shadow copy, text + underline, logo + name),
     while the joined unit covers <= TIGHT_MAX_AREA of the canvas; the headline and main image only take
     a shadow copy (a text over the headline). analyze(source, tight=False) skips this step
3. Decorations: ungrouped shapes that contain nothing and are faint (effective
   alpha < 0.5), lines, or bleed off the canvas, and thin rule lines of any type
   (<= 4px thick, 10x longer). They are not laid out, only re-anchored afterwards.
4. Roles, each with a confidence (rules in `_assign_roles`):
   headline / subhead / body / fineprint / cta / logo / hero / badge / decoration.
5. Importance = role base x (0.8 + 0.2 x confidence) + up to 0.1 for relative area,
   + up to EMPHASIS_BOOST for a unit whose colour stands out from the palette
   (eval.score.emphasis: saturated and far from the main colours).
   Floating units that partly cover a bigger unit (a ribbon across a card) are
   attached to it: `host` and their centre's position inside it (`rel`).
   A row of sibling units (a row of badges or buttons, eval.score.siblings) is merged
   into one rigid unit, so it is never split. Every unit records which side of its anchor (host centre, else canvas centre)
   it sits on (`side`), so a rotated unit moved to the mirrored side can mirror its tilt.
6. Relationships: containment pairs, shared alignment edges/centres, canvas-edge
   anchoring, reading-order rows and size order.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

from eval.check import rotated_bbox, text_layout
from eval.score import ATTACH_MIN, emphasis, key_elements, siblings, tight_relations
from renderer.converter.colorutil import parse_rgba

ACTION_WORDS = {
    "shop", "buy", "get", "order", "learn", "discover", "explore", "try", "start", "join", "sign", "subscribe",
    "download", "book", "register", "call", "visit", "save", "claim", "apply", "donate", "watch", "read", "see",
    "find", "contact", "reserve", "grab", "upgrade", "install", "play", "listen", "enroll", "rsvp",
}
ROLE_BASE = {"headline": 1.0, "hero": 0.9, "cta": 0.85, "logo": 0.75, "subhead": 0.6, "body": 0.45,
             "badge": 0.35, "fineprint": 0.2, "decoration": 0.1}
PRIMARY_ROLES = ("headline", "cta", "logo", "hero")
FILL_TOLERANCE = 0.01
NEAR_BLEED = 0.95       # a photo or opaque shape covering this share of the canvas is a background (fills the target)
TILE_MIN = 0.1          # opaque shapes and photos of at least this share that together cover NEAR_BLEED of the
                        # canvas are a background layer, each stretched with the canvas (colour panels side by side)
EDGE_ZONE = 0.12        # a box edge within 12% of a canvas side counts as anchored to it
CONTAINER_MAX_AREA = {"shape": 0.25, "image": 0.2}  # a bigger image holds text only if flat (CARD_MAX_AREA)
DECORATION_ALPHA = 0.5  # same threshold as eval.score
EMPHASIS_BOOST = 0.15
CARD_MAX_AREA = 0.5     # ...and up to this share of the canvas when it is flat (a card, not a photo)
FLAT_STD = 25.0         # an image whose opaque pixels vary less than this (RGB std) is flat
TEXT_ON_HOLDER = 0.8   # text this much on a card, blob or button drawn under it belongs to it (a tilted headline
                       # on a brushstroke pokes out); other elements need 0.9
TIGHT_MAX_AREA = 0.25  # a tight group joins only while the result covers at most this share of the canvas
LABEL_MAX_AREA = 0.05  # ...unless it is smaller than this share of the canvas
CARD_MAX_ASPECT = 4.0   # an image holds text as a card only when at most this elongated
GRID_SIZE = 1.15        # grid cells' anchors: width and height within this ratio of each other
THIN_PX = 4.0           # a non-text element this thin (and 10x longer) is a rule line: a decoration


@dataclass
class Unit:
    id: str
    members: list            # element ids, bottom-to-top
    kind: str                # "text" | "image" | "rigid"
    box: tuple               # source union of (rotated) boxes, x0 y0 x1 y1
    drawn: tuple = ()        # source union of what is drawn (text = its lines)
    role: str = "body"
    confidence: float = 0.5
    importance: float = 0.5
    edges: tuple = ()        # subset of ("left", "right", "top", "bottom")
    floating: bool = False   # placed into leftover space after the main layout
    rotated: bool = False
    emphasis: float = 0.0    # 0..1, colour stands out from the design's palette
    host: str | None = None  # unit this floating unit is attached to
    rel: tuple = ()          # its drawn centre as a fraction of the host's drawn box
    side: tuple = (1, 1)     # side of its anchor it sits on (x, y), each +-1
    parts: list = field(default_factory=list)  # a sibling row or grid: (members, source box) of each part, reading order
    cols: int = 0            # a grid: its source column count (a sibling row: 0, i.e. all parts in one row)

    @property
    def w(self):
        return self.box[2] - self.box[0]

    @property
    def h(self):
        return self.box[3] - self.box[1]

    @property
    def corner(self):
        horiz = "left" if "left" in self.edges else "right" if "right" in self.edges else None
        vert = "top" if "top" in self.edges else "bottom" if "bottom" in self.edges else None
        return f"{vert}-{horiz}" if horiz and vert else None


@dataclass
class Analysis:
    source: dict
    sw: float
    sh: float
    els: dict                # id -> source element
    background: list         # element ids
    decorations: list        # Units
    units: dict              # uid -> content Unit
    rows: list               # reading-order rows of non-floating unit ids
    relations: dict = field(default_factory=dict)
    text_align: str = "center"
    panels: dict = field(default_factory=dict)  # panel id -> ids of what sits on it (see _panels)

    def by_role(self, role):
        return next((u for u in self.units.values() if u.role == role), None)

    def anchor(self, u):
        """Source centre of a unit as a fraction of the canvas."""
        return ((u.box[0] + u.box[2]) / 2 / self.sw, (u.box[1] + u.box[3]) / 2 / self.sh)


def element_box(el):
    """What the renderer draws: rotated bounds, laid-out lines for text."""
    if el.get("rotation", 0):
        return rotated_bbox(el)
    if el["type"] == "text":
        lay = text_layout(el)
        if lay["lines"]:
            return (el["x"] + lay["left"], el["y"], el["x"] + lay["right"], el["y"] + lay["height"])
    return (el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])


def full_box(el):
    """What has to stay on the canvas: the box itself and, when rotated, its
    rotated bounds too (check() tests both, as the official self-check tests the box)."""
    box = (el["x"], el["y"], el["x"] + el["width"], el["y"] + el["height"])
    return _union([box, rotated_bbox(el)]) if el.get("rotation", 0) else box


def analyze(source, tight=True):
    els = {e["id"]: e for e in source["elements"]}
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    z = {i: (e.get("z_index", 0), n) for n, (i, e) in enumerate(els.items())}

    background = _background(els, z, sw, sh)
    rest = [i for i in els if i not in background]
    panels = _panels(els, z, rest, sw, sh)
    rest = [i for i in rest if i not in panels]

    parent = {i: i for i in rest}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(a, b):
        parent[find(a)] = find(b)

    by_group = {}
    for i in rest:
        if els[i].get("group_id"):
            by_group.setdefault(els[i]["group_id"], []).append(i)
    for members in by_group.values():
        for m in members[1:]:
            union(members[0], m)

    containment = []
    holders = [i for i in rest if _holder(els[i], sw, sh)]
    holders.sort(key=lambda i: _area(full_box(els[i])))  # the tightest holder wins
    for i in rest:
        if _is_rule_line(els[i]):
            continue  # a divider stays a decoration: inside a rigid group it would shrink out of sight
        inner = element_box(els[i])
        for s in holders:
            if (s != i and z[s] < z[i] and _area(full_box(els[s])) > _area(inner)
                    and _inside(inner, full_box(els[s])) >= (TEXT_ON_HOLDER if els[i]["type"] == "text" else 0.9)):
                containment.append((s, i))
                union(s, i)
                break

    # a photo in a frame: an image inside another (smaller than a background), whichever is on top
    frames = [i for i in holders if els[i]["type"] == "image"]
    for i in rest:
        if els[i]["type"] == "image" and not _is_rule_line(els[i]):
            for f in frames:
                if f != i and _area(full_box(els[f])) > _area(full_box(els[i])) and _inside(full_box(els[i]), full_box(els[f])) >= 0.9:
                    union(f, i)
                    break

    # tight groups (eval.score.tight_relations: label + value, a shadow copy, text + underline, logo + name) move as one
    # (the headline and main image keep their own unit, so they can still re-wrap and grow; a shadow copy still joins)
    keys = set(key_elements(source))
    for r in tight_relations(source) if tight else ():
        a, b = r["a"], r["b"]
        if {a, b} & keys and not (r["kind"] == "overlap" and els[a]["type"] == els[b]["type"] == "text"):
            continue
        if a in parent and b in parent and find(a) != find(b):
            joined = [full_box(els[i]) for i in rest if find(i) in (find(a), find(b))]
            if _area(_union(joined)) <= TIGHT_MAX_AREA * sw * sh:
                union(a, b)

    clusters = {}
    for i in rest:
        clusters.setdefault(find(i), []).append(i)

    decorations, units = [], {}
    for members in clusters.values():
        members.sort(key=lambda m: z[m])
        if len(members) == 1 and _is_decoration(els[members[0]], containment, sw, sh):
            decorations.append(_make_unit(members[0], members, els))
        else:
            uid = next((els[m]["group_id"] for m in members if els[m].get("group_id")), members[0]) if len(members) > 1 else members[0]
            units[uid] = _make_unit(uid, members, els)

    # tightly layered decorations travel with the unit they sit on
    for d in list(decorations):
        if all(_is_rule_line(els[m]) for m in d.members):
            continue  # dividers are re-anchored on their own, so they can keep a visible thickness
        host = max(units.values(), key=lambda u: _inside(d.box, u.box), default=None)
        if host and _inside(d.box, host.box) >= 0.8 and _area(d.box) < _area(host.box):
            host.members = sorted(host.members + d.members, key=lambda m: z[m])
            host.box, host.drawn = _union([host.box, d.box]), _union([host.drawn, d.drawn])
            host.kind = "rigid"
            decorations.remove(d)

    for u in [*units.values(), *decorations]:
        u.edges = _edges(u.box, sw, sh)
    for d in decorations:
        d.role, d.confidence, d.importance = "decoration", 0.8, ROLE_BASE["decoration"]

    _assign_roles(units, els, sw, sh)
    sal = emphasis(source)
    for u in units.values():
        area_bonus = 0.1 * min(1.0, 4 * _area(u.box) / (sw * sh))
        u.emphasis = max(sal.get(m, 0.0) for m in u.members)
        u.importance = round(ROLE_BASE[u.role] * (0.8 + 0.2 * u.confidence) + area_bonus + EMPHASIS_BOOST * u.emphasis, 4)
        # small badges pinned to a corner, rotated stickers and loose graphics are placed after the main layout
        u.floating = u.role in ("badge", "decoration") and (u.corner is not None or u.rotated) or u.role == "decoration"
    _merge_grids(units, els, z, sw, sh)
    _merge_siblings(units, source, z, sw, sh)
    _attach(units)
    for u in [*units.values(), *decorations]:
        h = units[u.host].drawn if u.host else (0, 0, sw, sh)
        u.side = side_of(u.drawn, h)

    rows = reading_rows({uid: u.drawn for uid, u in units.items() if not u.floating})
    aligns = Counter()  # dominant text alignment, weighted by characters
    for e in els.values():
        if e["type"] == "text" and e["id"] not in background:
            aligns[e.get("text_align", "left")] += len(e["content"])

    return Analysis(
        source=source, sw=sw, sh=sh, els=els, background=background, decorations=decorations, units=units,
        rows=rows, text_align=aligns.most_common(1)[0][0] if aligns else "center", panels=panels,
        relations={
            "containment": containment,
            "alignment": _alignments(units),
            "anchors": {uid: u.edges for uid, u in units.items()},
            "reading_order": [uid for row in rows for uid in row],
            "size_order": sorted(units, key=lambda uid: -_area(units[uid].box)),
        },
    )


def _edges(b, sw, sh):
    return tuple(e for e, near in (("left", b[0] <= EDGE_ZONE * sw), ("right", b[2] >= (1 - EDGE_ZONE) * sw),
                                   ("top", b[1] <= EDGE_ZONE * sh), ("bottom", b[3] >= (1 - EDGE_ZONE) * sh)) if near)


def _merge_siblings(units, source, z, sw, sh):
    """A row of sibling units (eval.score.siblings: a row of badges or buttons) becomes one
    rigid unit, so it is scaled, moved or demoted as one piece and never split up. When the row
    would be too wide to stay readable it may rearrange into a stack or grid (engine.fit), but
    its members always stay together, at one scale."""
    owner = {m: uid for uid, u in units.items() for m in u.members}
    for g in siblings(source):
        uids = list(dict.fromkeys(owner.get(i) for u in g for i in u))
        if None in uids or len(uids) < 2 or any(x not in units or units[x].floating for x in uids):
            continue
        parts = [units.pop(x) for x in uids]
        lead = max(parts, key=lambda u: u.importance)
        box = _union([u.box for u in parts])
        # a sibling centred on the design's axis (the middle badge under a centred headline) keeps
        # that axis: pad the row's box to be symmetric about it, so centring the row centres it
        axes = [(v.drawn[0] + v.drawn[2]) / 2 for v in units.values() if not v.floating]
        mid = next((c for c in ((u.drawn[0] + u.drawn[2]) / 2 for u in parts) if any(abs(c - x) <= 2 for x in axes)), None)
        if mid is not None:
            half = max(mid - box[0], box[2] - mid)
            box = (mid - half, box[1], mid + half, box[3])
        units["+".join(uids)] = Unit(
            id="+".join(uids), members=sorted((m for u in parts for m in u.members), key=lambda m: z[m]), kind="rigid",
            box=box, drawn=_union([u.drawn for u in parts]), role=lead.role, confidence=lead.confidence,
            importance=lead.importance, edges=_edges(box, sw, sh), emphasis=max(u.emphasis for u in parts),
            parts=[(u.members, u.box) for u in sorted(parts, key=lambda u: u.box[0])])


def _merge_grids(units, els, z, sw, sh):
    """A grid of repeated cells (storyboard panels, menu items, a product grid) becomes one unit
    whose cells move together and can re-flow to another column count (engine.fit arrangements).
    Anchors: >= 4 non-floating image units (or units holding an image) of similar size (w and h
    within GRID_SIZE of each other) on a lattice of >= 2 rows and >= 2 columns, at most one cell
    empty per row. The other units whose centre falls in a cell's band join that cell: bands either
    start at their anchor (captions follow the photo) or end at it (captions lead), on each axis,
    whichever gives the most even cells. A cell is then one rigid piece."""
    cand = [u for u in units.values() if not u.floating and any(els[m]["type"] == "image" for m in u.members)]
    used = set()
    for seed in sorted(cand, key=lambda u: -_area(u.box)):
        if seed.id in used or seed.id not in units:
            continue
        alike = [u for u in cand if u.id not in used and u.id in units and _similar(u.box, seed.box)]
        if len(alike) < 4 or any(_overlap(a.box, b.box) for n, a in enumerate(alike) for b in alike[n + 1:]):
            continue
        cols = _bands([u.box[0] for u in alike], seed.w)
        rows = _bands([u.box[1] for u in alike], seed.h)
        if len(cols) < 2 or len(rows) < 2 or len(alike) < len(rows) * len(cols) - len(rows) + 1:
            continue
        cell_of = {u.id: (_band_index(rows, u.box[1]), _band_index(cols, u.box[0])) for u in alike}
        if len(set(cell_of.values())) < len(alike):
            continue
        others = [u for u in units.values() if u.id not in cell_of and not u.floating]
        occupied = set(cell_of.values())
        best = None
        for after_x in (True, False):
            for after_y in (True, False):
                xb = _cell_bands(cols, alike, 0, after_x)
                yb = _cell_bands(rows, alike, 1, after_y)
                got = {}
                for o in others:
                    cx, cy = (o.box[0] + o.box[2]) / 2, (o.box[1] + o.box[3]) / 2
                    c, r = _which(xb, cx), _which(yb, cy)
                    if (r, c) in occupied and _area(o.box) < 2 * _area(seed.box):
                        got.setdefault((r, c), []).append(o)
                counts = [len(got.get(k, [])) for k in occupied]
                spread = max(counts) - min(counts)
                key = (spread, -sum(counts), not after_x, not after_y)
                if best is None or key < best[0]:
                    best = (key, got)
        got = best[1]
        cells = []
        for a in sorted(alike, key=lambda u: cell_of[u.id]):
            members = [*a.members, *(m for o in got.get(cell_of[a.id], []) for m in o.members)]
            parts = [a, *got.get(cell_of[a.id], [])]
            cells.append((sorted(members, key=lambda m: z[m]), _union([u.box for u in parts]), parts))
        if any(_overlap(a[1], b[1]) for n, a in enumerate(cells) for b in cells[n + 1:]):
            continue  # satellites made cells overlap: not a clean grid
        parts_all = [u for _m, _b, ps in cells for u in ps]
        for u in parts_all:
            units.pop(u.id)
            used.add(u.id)
        lead = max(parts_all, key=lambda u: u.importance)
        box = _union([b for _m, b, _p in cells])
        gid = "grid:" + "+".join(a.id for a in sorted(alike, key=lambda u: cell_of[u.id]))
        units[gid] = Unit(id=gid, members=sorted((m for ms, _b, _p in cells for m in ms), key=lambda m: z[m]),
                          kind="rigid", box=box, drawn=_union([u.drawn for u in parts_all]), role=lead.role,
                          confidence=lead.confidence, importance=lead.importance, edges=_edges(box, sw, sh),
                          emphasis=max(u.emphasis for u in parts_all), parts=[(ms, b) for ms, b, _p in cells],
                          cols=len(cols))


def _similar(a, b):
    return all(max(x, y) <= GRID_SIZE * min(x, y) for x, y in ((a[2] - a[0], b[2] - b[0]), (a[3] - a[1], b[3] - b[1])))


def _overlap(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 1 and min(a[3], b[3]) - max(a[1], b[1]) > 1


def _bands(starts, size):
    """Cluster start coordinates (within 25% of the cell size) -> sorted band starts."""
    out = []
    for v in sorted(starts):
        if out and v - out[-1][-1] <= 0.25 * size:
            out[-1].append(v)
        else:
            out.append([v])
    return [sum(b) / len(b) for b in out]


def _band_index(bands, v):
    return min(range(len(bands)), key=lambda k: abs(bands[k] - v))


def _cell_bands(starts, anchors, axis, after):
    """Cell spans along one axis: from each anchor's start to the next one's (after=True),
    or from the previous anchor's end to each one's end (after=False)."""
    size = sum(u.box[axis + 2] - u.box[axis] for u in anchors) / len(anchors)
    pitch = (starts[-1] - starts[0]) / (len(starts) - 1)
    if after:
        return [(s, starts[k + 1] if k + 1 < len(starts) else s + pitch) for k, s in enumerate(starts)]
    ends = [s + size for s in starts]
    return [(ends[k - 1] if k else e - pitch, e) for k, e in enumerate(ends)]


def _which(spans, v):
    return next((k for k, (lo, hi) in enumerate(spans) if lo <= v < hi), None)


def side_of(b, ref):
    """Which side of box ref's centre the centre of box b is on: (+-1, +-1)."""
    return (1 if b[0] + b[2] >= ref[0] + ref[2] else -1, 1 if b[1] + b[3] >= ref[1] + ref[3] else -1)


def _attach(units):
    """A badge or decoration that partly covers a bigger non-floating unit (ATTACH_MIN
    to 90% of its area, or any share when rotated) travels with that unit."""
    for u in units.values():
        if u.role not in ("badge", "decoration"):
            continue
        best, share = None, 0.0
        for v in units.values():
            if v is u or v.floating or _area(v.drawn) <= _area(u.drawn):
                continue
            r = _inside(u.drawn, v.drawn)
            if r > share:
                best, share = v, r
        if best is not None and share >= ATTACH_MIN and (share < 0.9 or u.rotated):
            hb, b = best.drawn, u.drawn
            u.floating, u.host = True, best.id
            u.rel = (((b[0] + b[2]) / 2 - hb[0]) / max(hb[2] - hb[0], 1e-6),
                     ((b[1] + b[3]) / 2 - hb[1]) / max(hb[3] - hb[1], 1e-6))


def reading_rows(boxes):
    """Group boxes into rows (vertical overlap >= half the smaller height), top to
    bottom, left to right within a row. Returns lists of keys."""
    rows = []
    for key, b in sorted(boxes.items(), key=lambda kv: kv[1][1]):
        h = b[3] - b[1]
        if rows:
            top, bottom = rows[-1]["span"]
            overlap = min(bottom, b[3]) - max(top, b[1])
            if overlap >= 0.5 * max(1e-6, min(h, bottom - top)):
                rows[-1]["items"].append((key, b))
                rows[-1]["span"] = (min(top, b[1]), max(bottom, b[3]))
                continue
        rows.append({"span": (b[1], b[3]), "items": [(key, b)]})
    return [[k for k, _b in sorted(r["items"], key=lambda kb: kb[1][0])] for r in rows]


def quarter_turns(el):
    """0..3 when the element is turned by a multiple of 90 degrees (within 0.5), else None."""
    r = el.get("rotation", 0) or 0
    q = round(r / 90)
    return q % 4 if abs(r - 90 * q) <= 0.5 else None


def _background(els, z, sw, sh):
    tol_x, tol_y = FILL_TOLERANCE * sw, FILL_TOLERANCE * sh
    fills = set()
    for i, e in els.items():  # what is drawn covers the canvas: a quarter- or half-turned photo counts too
        if e["type"] in ("image", "shape") and quarter_turns(e) is not None:
            b = rotated_bbox(e) if e.get("rotation", 0) else (e["x"], e["y"], e["x"] + e["width"], e["y"] + e["height"])
            if b[0] <= tol_x and b[1] <= tol_y and b[2] >= sw - tol_x and b[3] >= sh - tol_y:
                fills.add(i)
    # stretchable only: a contain image must keep its proportions, so it can't follow the canvas
    stretch = {i: e for i, e in els.items() if i not in fills and not e.get("rotation", 0) and _alpha(e) >= DECORATION_ALPHA
               and (e["type"] == "shape" and e.get("shape_type", "rectangle") == "rectangle"
                    or e["type"] == "image" and e.get("fit") == "cover")}
    for i, e in stretch.items():
        if _canvas_share([e], sw, sh) >= NEAR_BLEED:
            fills.add(i)
    tiles = [e for i, e in stretch.items() if i not in fills and _canvas_share([e], sw, sh) >= TILE_MIN]
    if len(tiles) > 1 and _canvas_share(tiles, sw, sh) >= NEAR_BLEED:
        fills.update(e["id"] for e in tiles)
    bg = []
    for i in sorted(els, key=lambda i: z[i]):  # consecutive fills from the bottom of the stack
        if i not in fills:
            break
        bg.append(i)
    return bg


def _canvas_share(els, sw, sh, n=40):
    """Share of the canvas inside the union of the elements' boxes (sampled on an n x n grid)."""
    pts = [((k + 0.5) * sw / n, (m + 0.5) * sh / n) for k in range(n) for m in range(n)]
    return sum(any(e["x"] <= x <= e["x"] + e["width"] and e["y"] <= y <= e["y"] + e["height"] for e in els)
               for x, y in pts) / len(pts)


def is_tile(el, sw, sh):
    """A background that doesn't fill the source canvas (a tile or a near-full-bleed photo)."""
    return _canvas_share([el], sw, sh) < NEAR_BLEED


def _is_decoration(el, containment, sw, sh):
    if el["type"] == "text" or el.get("group_id") or any(c == el["id"] for c, _ in containment):
        return False
    if _is_rule_line(el):
        return True
    if el["type"] != "shape":
        return False
    return _alpha(el) < DECORATION_ALPHA or el.get("shape_type") == "line" or _bleeds(full_box(el), sw, sh)


def _is_rule_line(el):
    """A thin line (a divider or underline, often drawn as an image): at most THIN_PX thick, 10x longer."""
    thin, long = sorted((el["width"], el["height"]))
    return thin <= THIN_PX and long >= 10 * thin


def _holder(el, sw, sh):
    """A shape or an image (a card, a speech bubble, a label) that holds what sits on it as one rigid group."""
    b = full_box(el)
    return (el["type"] in ("shape", "image") and _alpha(el) >= DECORATION_ALPHA
            and (_area(b) < CONTAINER_MAX_AREA[el["type"]] * sw * sh
                 # a bigger image holds text only when it is a flat card or bubble, not a photo
                 or el["type"] == "image" and _area(b) < CARD_MAX_AREA * sw * sh and _flat(el))
            and not _bleeds(b, sw, sh) and not _is_rule_line(el)
            # an image card is compact; a long bar (a table row, a band) is a backdrop the text reflows over
            and (el["type"] == "shape" or max(el["width"], el["height"]) <= CARD_MAX_ASPECT * min(el["width"], el["height"])
                 or _area(b) < LABEL_MAX_AREA * sw * sh))  # a small ribbon or label bar still holds its text


def _panels(els, z, rest, sw, sh):
    """Flat cards and panels too big to be a holder, with text on them: {id: ids on it}.
    Kept as one rigid group they would keep their source proportions and end up mostly empty, so
    they are left out of the layout and fitted around wherever their content goes (engine.fit.place_panel)."""
    panels = {}
    for i in rest:
        e = els[i]
        b = full_box(e)
        if (e["type"] not in ("shape", "image") or e.get("rotation", 0) or _alpha(e) < DECORATION_ALPHA
                or _bleeds(b, sw, sh) or _is_rule_line(e) or _holder(e, sw, sh) or e["type"] == "image" and not _flat(e)
                or max(e["width"], e["height"]) > CARD_MAX_ASPECT * min(e["width"], e["height"])):  # a long bar is a backdrop
            continue
        on = [j for j in rest if j != i and z[j] > z[i] and _inside(element_box(els[j]), b) >= 0.9]
        if any(els[j]["type"] == "text" for j in on):  # a panel holds text; an illustration with a cloud on it is not one
            panels[i] = on
    return panels


def _flat(el):
    """Mostly one colour (a card, a bubble, a label), as opposed to a photo or an illustration."""
    import numpy as np
    from PIL import Image

    from eval.render import assets_dir
    try:
        with Image.open(assets_dir() / el["asset_id"]) as im:
            px = np.asarray(im.convert("RGBA").resize((48, 48)), dtype=float).reshape(-1, 4)
    except (OSError, KeyError, TypeError, ValueError):
        return False
    px = px[px[:, 3] > 128, :3]
    return len(px) > 0 and float(px.std(axis=0).mean()) < FLAT_STD


def _alpha(el):
    return el.get("opacity", 1) * parse_rgba(el.get("fill_color"), default=(0, 0, 0, 1.0))[3]


def _bleeds(b, sw, sh):
    return b[0] < -0.5 or b[1] < -0.5 or b[2] > sw + 0.5 or b[3] > sh + 0.5


def _make_unit(uid, members, els):
    boxes = [full_box(els[m]) for m in members]
    rotated = any(els[m].get("rotation", 0) for m in members)
    if len(members) == 1 and not rotated and els[members[0]]["type"] in ("text", "image"):
        kind = els[members[0]]["type"]
    else:
        kind = "rigid"
    return Unit(id=uid, members=members, kind=kind, box=_union(boxes), drawn=_union([element_box(els[m]) for m in members]),
                rotated=rotated)


def _union(boxes):
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _assign_roles(units, els, sw, sh):
    """Role rules, in priority order:
    headline  unit holding the largest-font text (ties: larger box); confidence grows
              with how much larger it is than the runner-up
    cta       shape+text unit whose text has an action verb (0.9); otherwise the lowest,
              widest shape+text unit if it's in the bottom 40% (0.6); or a short (<= 5 words)
              standalone text starting with an action verb (0.7)
    badge     other shape+text units (rotated ones 0.8), and small images that aren't the logo
    hero      largest image covering >= 5% of the canvas (confidence by size)
    logo      topmost remaining image under 8% of the canvas that is wide (aspect >= 1.3)
              or touches the top/bottom zone
    fineprint text with font <= 0.6 x median, or smallest font + >= 60 chars + bottom zone
    subhead   largest remaining text with >= 15 chars
    body      any other text
    decoration shapes or graphics with no text (they float into leftover space)
    """
    texts = [(u, els[i]) for u in units.values() for i in u.members if els[i]["type"] == "text"]
    if texts:
        ranked = sorted(texts, key=lambda t: (t[1]["font_size"], t[1]["width"] * t[1]["height"]), reverse=True)
        head = ranked[0]
        runner = next((t[1]["font_size"] for t in ranked[1:] if t[0] is not head[0]), None)
        head[0].role = "headline"
        head[0].confidence = 1.0 if runner is None else min(1.0, 0.5 + (head[1]["font_size"] / runner - 1))

    def unit_text(u):
        return " ".join(els[i]["content"] for i in u.members if els[i]["type"] == "text")

    def has_action(text):
        words = re.findall(r"[a-z]+", text.lower())
        return any(w in ACTION_WORDS for w in words[:3])

    containers = [u for u in units.values() if u.role != "headline" and u.kind == "rigid"
                  and any(els[i]["type"] == "shape" for i in u.members) and any(els[i]["type"] == "text" for i in u.members)]
    cta = next((u for u in containers if not u.rotated and has_action(unit_text(u))), None)
    if cta:
        cta.role, cta.confidence = "cta", 0.9
    else:
        plain = [u for u in containers if not u.rotated and u.box[1] >= 0.6 * sh]
        if plain:
            cta = max(plain, key=lambda u: (u.box[3], u.w))
            cta.role, cta.confidence = "cta", 0.6
    for u in containers:
        if u is not cta:
            u.role, u.confidence = "badge", 0.8 if u.rotated else 0.6

    # an image, or images layered as one (a photo in a frame)
    images = sorted((u for u in units.values() if all(els[i]["type"] == "image" for i in u.members)), key=lambda u: -_area(u.box))
    hero = images[0] if images and _area(images[0].box) >= 0.05 * sw * sh else None
    if hero:
        hero.role, hero.confidence = "hero", min(1.0, 0.5 + _area(hero.box) / (sw * sh))
    small = sorted((u for u in images if u is not hero and _area(u.box) < 0.08 * sw * sh), key=lambda u: u.box[1])
    logo = next((u for u in small if u.w / max(u.h, 1e-6) >= 1.3 or "top" in u.edges or "bottom" in u.edges), None)
    if logo:
        logo.role, logo.confidence = "logo", 0.8 if logo.w / max(logo.h, 1e-6) >= 2 else 0.6
    for u in images:
        if u.role == "body":  # untouched default
            u.role, u.confidence = "badge", 0.5

    rest = [(u, els[u.members[0]]) for u in units.values() if u.role == "body" and u.kind == "text"]
    sizes = sorted(e["font_size"] for _u, e in texts) if texts else [0]
    median = sizes[len(sizes) // 2]
    for u, e in rest:
        words = e["content"].split()
        if len(words) <= 5 and has_action(e["content"]) and not cta:
            u.role, u.confidence, cta = "cta", 0.7, u
        elif e["font_size"] <= 0.6 * median or (e["font_size"] == sizes[0] and len(e["content"]) >= 60 and u.box[3] >= 0.85 * sh):
            u.role, u.confidence = "fineprint", 0.7
    rest = sorted((t for t in rest if t[0].role == "body"), key=lambda t: -t[1]["font_size"])
    if rest and len(rest[0][1]["content"]) >= 15:
        rest[0][0].role, rest[0][0].confidence = "subhead", 0.6

    for u in units.values():
        if u.role == "body" and not any(els[i]["type"] == "text" for i in u.members):
            u.role, u.confidence = "decoration", 0.5


def _alignments(units, tol=2.0):
    out = []
    items = [(uid, u.drawn) for uid, u in units.items() if not u.floating]
    for n, (a, ba) in enumerate(items):
        for b, bb in items[n + 1:]:
            for name, va, vb in (("left", ba[0], bb[0]), ("right", ba[2], bb[2]), ("center_x", ba[0] + ba[2], bb[0] + bb[2]),
                                 ("top", ba[1], bb[1]), ("bottom", ba[3], bb[3]), ("center_y", ba[1] + ba[3], bb[1] + bb[3])):
                if abs(va - vb) <= (2 * tol if name.startswith("center") else tol):
                    out.append((a, b, name))
    return out


def _area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inside(inner, outer):
    """Fraction of `inner`'s area that lies inside `outer`."""
    a = _area(inner)
    if a <= 0:
        return 0.0
    return _area((max(inner[0], outer[0]), max(inner[1], outer[1]), min(inner[2], outer[2]), min(inner[3], outer[3]))) / a

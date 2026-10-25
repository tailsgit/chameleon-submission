"""Pick the Crello test set from the converted designs (eval/crello_convert.py).

    python -m eval.crello_testset      # writes eval/crello_testset.json (ids and sizes only, no Crello content)

About 300 designs, spread over format, element count and aspect ratio: designs
are bucketed by (format, element-count band, aspect band) and drawn round-robin
over the buckets in a seeded order, so rare kinds are not swamped by common ones.
Each design gets 3 different random catalogue sizes (never its own size).
Designs, not cases, are split: two thirds "dev" (look at it, tune on it), one
third "holdout" (only run it to check for overfitting; never tune on it).
"""

import json
import random
from collections import defaultdict

from eval.crello_convert import OUT, REPO

N_DESIGNS, SIZES_PER_DESIGN, SEED = 300, 3, 2026
TESTSET = REPO / "eval" / "crello_testset.json"
CATALOGUE = REPO / "eval" / "canvas_catalogue.json"
STRUCTURE = REPO / "eval" / "crello_structure.json"
QUICK = REPO / "eval" / "crello_quick.json"
QUICK_N = 100
ASPECT_CHANGE = (0.1, 0.5, 1.2)          # |ln(target aspect / source aspect)|: same | moderate | large | extreme
ELEMENT_BANDS = (3, 7, 13, 21, 41)       # [3,7) [7,13) [13,21) [21,41)
ASPECT_BANDS = (0.7, 0.95, 1.05, 1.6)    # tall | portrait | square | landscape | wide


def band(x, edges):
    return next((k for k, e in enumerate(edges) if x < e), len(edges))


def build():
    rng = random.Random(SEED)
    metas = sorted((json.loads(p.read_text())["meta"] for p in (OUT / "designs").glob("*.json")), key=lambda m: m["id"])
    buckets = defaultdict(list)
    for m in metas:
        buckets[(m["format"], band(m["elements"], ELEMENT_BANDS), band(m["aspect"], ASPECT_BANDS))].append(m)
    keys = sorted(buckets)
    rng.shuffle(keys)
    for k in keys:
        rng.shuffle(buckets[k])
    picked = []
    while len(picked) < N_DESIGNS and any(buckets.values()):
        for k in keys:
            if buckets[k] and len(picked) < N_DESIGNS:
                picked.append(buckets[k].pop())
    catalogue = json.loads(CATALOGUE.read_text())
    rng.shuffle(picked)
    designs = []
    for n, m in enumerate(picked):
        src = json.loads((OUT / "designs" / f"{m['id']}.json").read_text())["source"]["canvas"]
        sizes = [f for f in catalogue if (f["width"], f["height"]) != (src["width"], src["height"])]
        designs.append({"id": m["id"], "split": "holdout" if n % 3 == 2 else "dev", "format": m["format"],
                        "elements": m["elements"], "aspect": m["aspect"],
                        "targets": [{"name": f["name"], "width": f["width"], "height": f["height"]}
                                    for f in rng.sample(sizes, SIZES_PER_DESIGN)]})
    TESTSET.write_text(json.dumps({"_note": __doc__.strip().splitlines()[0], "seed": SEED, "designs": designs}, indent=1) + "\n")
    for split in ("dev", "holdout"):
        ds = [d for d in designs if d["split"] == split]
        print(f"{split}: {len(ds)} designs, {sum(len(d['targets']) for d in ds)} cases, "
              f"{len({d['format'] for d in ds})} formats")
    return designs


def cases(split):
    """[(case name, design id, target {name, width, height})] for one split: dev, holdout, or
    structure (the dev designs in STRUCTURE, with their dev targets)."""
    designs = json.loads(TESTSET.read_text())["designs"]
    if split == "quick":
        keep = set(json.loads(QUICK.read_text())["cases"])
        return [c for c in cases("dev") if c[0] in keep]
    if split == "structure":
        keep = set(json.loads(STRUCTURE.read_text())["designs"])
        designs = [d for d in designs if d["id"] in keep]
        split = "dev"
    return [(f"crello_{d['id']}__{_slug(t['name'])}", d["id"], t) for d in designs if d["split"] == split for t in d["targets"]]


def build_quick(n=QUICK_N):
    """About n crello-dev cases for iterating on a fix, stratified by aspect change (ASPECT_CHANGE) and
    element-count band (ELEMENT_BANDS): each stratum gets its share of n (at least 2 when it has any),
    drawn with a fixed seed. Case names only. Run with the samples: python -m eval.bench --suite quick"""
    import math
    designs = {d["id"]: d for d in json.loads(TESTSET.read_text())["designs"]}
    strata = defaultdict(list)
    for case, did, t in cases("dev"):
        d = designs[did]
        change = abs(math.log(t["width"] / t["height"] / d["aspect"]))
        strata[(band(change, ASPECT_CHANGE), band(d["elements"], ELEMENT_BANDS))].append(case)
    total = sum(len(v) for v in strata.values())
    rng = random.Random(SEED)
    chosen = []
    for key in sorted(strata):
        pool = sorted(strata[key])
        k = min(len(pool), max(2, round(n * len(pool) / total)))
        chosen += rng.sample(pool, k)
    QUICK.write_text(json.dumps({"_note": "crello-dev cases for quick iteration; see build_quick", "cases": sorted(chosen),
                                 "strata": {f"aspect{a}_elements{e}": len(v) for (a, e), v in sorted(strata.items())}}, indent=1))
    print(f"{len(chosen)} cases from {total} in {len(strata)} strata")


def build_structure(n=40):
    """Pick about n crello-dev designs with the structure the engine must keep (eval.score.relations):
      panel    text sitting >= 90% inside a shape or image it overlaps (text on a panel or card)
      label    a label pair (a heading or "Label:" and what it labels)
      collage  >= 3 images joined by overlaps
    Designs with more of these patterns first, then balanced across patterns; ids and patterns only."""
    import os
    from collections import Counter
    from eval.crello_convert import OUT
    os.environ.setdefault("CHAMELEON_ASSETS", str(OUT / "assets"))
    from eval.score import _drawn_box, _inter, _area, relations
    found = {}
    for did in sorted({d for _c, d, _t in cases("dev")}):
        src = json.loads((OUT / "designs" / f"{did}.json").read_text())["source"]
        els = {e["id"]: e for e in src["elements"]}
        rels = relations(src)
        pats = set()
        for r in rels:
            a, b = els[r["a"]], els[r["b"]]
            if r["kind"] == "label" and a["type"] == "text":
                pats.add("label")
            if r["kind"] == "overlap":
                t, h = (a, b) if a["type"] == "text" else (b, a)
                if t["type"] == "text" and h["type"] != "text" and _inter(_drawn_box(t), _drawn_box(h)) >= 0.9 * _area(_drawn_box(t)):
                    pats.add("panel")
        imgs = {r[k] for r in rels if r["kind"] == "overlap" and els[r["a"]]["type"] == els[r["b"]]["type"] == "image" for k in ("a", "b")}
        parent = {i: i for i in imgs}
        def find(i):
            while parent[i] != i:
                i = parent[i]
            return i
        for r in rels:
            if r["kind"] == "overlap" and r["a"] in imgs and r["b"] in imgs:
                parent[find(r["a"])] = find(r["b"])
        if imgs and max(Counter(find(i) for i in imgs).values()) >= 3:
            pats.add("collage")
        if pats:
            found[did] = sorted(pats)
    chosen, count = [], Counter()
    for did, pats in sorted(found.items(), key=lambda kv: -len(kv[1])):
        if len(chosen) < n and (len(pats) > 1 or min(count[p] for p in pats) < n // 3):
            chosen.append(did)
            count.update(pats)
    STRUCTURE.write_text(json.dumps({"_note": "crello-dev designs with structure to keep; see build_structure",
                                     "designs": chosen, "patterns": {d: found[d] for d in chosen}}, indent=1))
    print(f"{len(chosen)} designs ({len(chosen) * 3} cases); patterns {dict(count)}; candidates {len(found)} of 200")


def _slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


if __name__ == "__main__":
    build()

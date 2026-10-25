"""Random synthetic designs: the engine must never crash or break a hard rule on them.

    python -m eval.synthetic                 # 300 designs (seeds 40..339; seeds 0..39 are in the test suite)
    python -m eval.synthetic --n 1000 --start 0

Prints how many outputs pass check(), the crashes, and each failure's violations.
"""

import argparse
import copy
import random
import traceback
from collections import Counter

WORDS = ("summer sale botanical light golden hour shop now limited edition crafted fresh new flavor "
         "discover the collection exclusive offer today only free shipping on all orders").split()
FONTS = ("Solstice Serif", "Solstice Sans")
ASSETS = ("solstice_bg_gradient.png", "solstice_logo_lockup.png", "solstice_bottle.png", "does_not_exist.png")
TARGETS = [(1080, 1080), (1080, 1920), (1600, 400), (728, 90), (300, 250), (160, 600), (1920, 1080), (600, 200)]



def random_design(rng):
    sw, sh = rng.choice([(1080, 1080), (1080, 1350), (1200, 628), (1920, 1080), (800, 2000)])
    els = []
    if rng.random() < 0.8:
        els.append({"id": "bg", "type": "image", "x": 0, "y": 0, "width": sw, "height": sh, "z_index": 0,
                    "asset_id": rng.choice(ASSETS), "fit": "cover"})
    for n in range(rng.randint(1, 12)):
        w, h = rng.uniform(20, sw * 0.8), rng.uniform(10, sh * 0.4)
        el = {"id": f"e{n}", "x": rng.uniform(-0.1 * sw, sw - w * 0.5), "y": rng.uniform(-0.1 * sh, sh - h * 0.5),
              "width": w, "height": h, "z_index": rng.randint(1, 6)}
        kind = rng.choice(["text", "text", "image", "shape"])
        if kind == "text":
            fs = rng.choice([10, 14, 18, 24, 36, 48, 72])
            el.update(type="text", content=" ".join(rng.choice(WORDS) for _ in range(rng.randint(1, 25))).title(),
                      font_family=rng.choice(FONTS), font_size=fs, color="rgba(255,255,255,1)",
                      text_align=rng.choice(["left", "center", "right"]), font_weight=rng.choice(["regular", "bold"]),
                      height=max(h, fs * 1.3), letter_spacing=rng.choice([0, 0, 2]))
        elif kind == "image":
            el.update(type="image", asset_id=rng.choice(ASSETS), fit=rng.choice(["contain", "contain", "cover"]))
        else:
            el.update(type="shape", shape_type=rng.choice(["rectangle", "ellipse", "line"]),
                      fill_color=f"rgba(232,106,58,{rng.choice([1, 0.3])})", corner_radius=rng.choice([0, 20]))
        if rng.random() < 0.15:
            el["rotation"] = rng.choice([-30, -12, 8, 45, 90])
        if rng.random() < 0.25:
            el["group_id"] = f"g{rng.randint(0, 2)}"
        if rng.random() < 0.2:
            el["opacity"] = rng.choice([0.2, 0.6])
        els.append(el)
    return {"canvas": {"width": sw, "height": sh}, "elements": els}



def run(n=300, start=40):
    from engine.resize import resize
    from eval.check import check
    clean, crashes, fails = 0, [], []
    for seed in range(start, start + n):
        rng = random.Random(seed)
        src = random_design(rng)
        target = dict(zip(("width", "height"), rng.choice(TARGETS)))
        try:
            v = check(src, target, resize(copy.deepcopy(src), target))
        except Exception:
            crashes.append((seed, traceback.format_exc(limit=1)))
            continue
        if v:
            fails.append((seed, v))
        else:
            clean += 1
    print(f"{clean}/{n} clean, {len(fails)} with violations, {len(crashes)} crashes")
    print("violations by rule:", dict(Counter(x["rule"] for _s, v in fails for x in v)))
    for seed, v in fails:
        print(f"  seed {seed}: " + "; ".join(f"{x['rule']}:{x['id']}" for x in v))
    for seed, tb in crashes:
        print(f"  seed {seed} crashed: {tb}")
    return clean, fails, crashes


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=int, default=40)
    a = ap.parse_args()
    run(a.n, a.start)

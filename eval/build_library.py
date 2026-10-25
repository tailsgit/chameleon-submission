"""Build the layout library engine.retrieve searches: one fingerprint per converted Crello design
that is in neither crello-dev nor crello-holdout.

A fingerprint is geometry and roles only, as our own analyzer sees the design:
  aspect   canvas width / height
  units    one per layout unit (a text, an image, or a group that moves as one), bottom to top:
           role, kind (text | image | rigid) and box (x0, y0, x1, y1) as fractions of the canvas
  roles    composition summary: how many units have each role
No text, no pixels, no asset or Crello ids; the list is ordered by aspect, not by source.

    python -m eval.build_library            # -> engine/data/layout_library.json
"""

import json
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor

from eval.crello_convert import OUT

os.environ.setdefault("CHAMELEON_ASSETS", str(OUT / "assets"))  # flat-card detection reads image pixels

from engine.analyze import analyze  # noqa: E402
from engine.retrieve import LIBRARY  # noqa: E402
from eval.crello_testset import TESTSET  # noqa: E402


def fingerprint(path):
    source = json.loads(path.read_text())["source"]
    sw, sh = source["canvas"]["width"], source["canvas"]["height"]
    try:
        a = analyze(source)
    except Exception:  # a design our analyzer can't read is simply left out
        return None
    units = [{"role": u.role, "kind": u.kind,
              "box": [round(max(0.0, min(1.0, v)), 4) for v in (u.box[0] / sw, u.box[1] / sh, u.box[2] / sw, u.box[3] / sh)]}
             for u in sorted(a.units.values(), key=lambda u: min(a.els[m].get("z_index", 0) for m in u.members))]
    if not units:
        return None
    return {"aspect": round(sw / sh, 4), "units": units, "roles": dict(Counter(u["role"] for u in units))}


def main():
    held = {d["id"] for d in json.loads(TESTSET.read_text())["designs"]}
    paths = sorted(p for p in (OUT / "designs").glob("*.json") if p.stem not in held)
    with ProcessPoolExecutor() as ex:
        prints = [f for f in ex.map(fingerprint, paths, chunksize=8) if f]
    prints.sort(key=lambda f: (f["aspect"], len(f["units"])))
    LIBRARY.parent.mkdir(parents=True, exist_ok=True)
    LIBRARY.write_text(json.dumps(prints, separators=(",", ":")))
    print(f"{len(prints)} layouts from {len(paths)} designs ({len(held)} test designs excluded), "
          f"{sum(len(f['units']) for f in prints)} units, {LIBRARY.stat().st_size / 1e6:.2f} MB -> {LIBRARY}")


if __name__ == "__main__":
    main()

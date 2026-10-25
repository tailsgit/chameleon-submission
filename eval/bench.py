"""Benchmark engine.resize over a suite of (design, target size) cases.

    python -m eval.bench                              # the 3 samples x every catalogue size
    python -m eval.bench --render                     # also render each case + a contact sheet
    python -m eval.bench --contrast                   # also the render-based contrast penalty
    python -m eval.bench --suite crello-dev --render --jobs 8
    python -m eval.bench --suite quick --jobs 8       # while iterating: ~100 crello-dev cases, then the samples
    python -m eval.bench --suite crello-holdout       # only to check for overfitting: never tune on it

Suites: samples (eval/results/), crello-quick, crello-dev, crello-structure and crello-holdout (eval/results/<suite>/;
see eval/crello_testset.py; needs the converted designs from eval/crello_convert.py,
whose assets are found through CHAMELEON_ASSETS, set here).

Writes <results>/summary.md, cases.csv, each output as outputs/<case>.json (the review
server snapshots it into a label) and, with --render, renders/*.png (each case, plus
source__<sample>.png and the starter's naive stretch as stretch__<case>.png, for the
review page) + contact_sheet.png (samples only). Crello outputs and renders contain
Crello content; keep every result out of the submission (docs/SUBMISSION.md).
"""

import argparse
import copy
import csv
import json
import logging
import os
import statistics
import subprocess
import time
import traceback
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from engine.resize import resize
from eval.baseline import stretch
from eval.check import check, text_layout
from eval.score import PENALTIES, score

REPO = Path(__file__).resolve().parents[1]
CATALOGUE = REPO / "eval" / "canvas_catalogue.json"
SAMPLES = sorted((REPO / "input").glob("chameleon_sample_*.json"))
SUITES = ("samples", "crello-dev", "crello-holdout", "crello-structure", "crello-quick")
CELL, THUMB = 230, 200
WORST_LISTED = 30


def slug(s):
    return "".join(c if c.isalnum() else "_" for c in s.lower()).strip("_")


def load_suite(suite):
    """([(case, sample, format name, target, source)], results folder)."""
    if suite == "samples":
        formats = json.loads(CATALOGUE.read_text())
        tasks = [(p.stem.replace("chameleon_sample_", ""), json.loads(p.read_text())) for p in SAMPLES]
        return [(f"{sample}__{slug(f['name'])}", sample, f["name"], {"width": f["width"], "height": f["height"]},
                 task["source"]) for sample, task in tasks for f in formats], REPO / "eval" / "results"
    from eval.crello_convert import OUT
    from eval.crello_testset import cases
    os.environ["CHAMELEON_ASSETS"] = str(OUT / "assets")  # worker processes inherit it
    sources = {}
    out = []
    for case, did, t in cases(suite.split("-", 1)[1]):
        if did not in sources:
            sources[did] = json.loads((OUT / "designs" / f"{did}.json").read_text())["source"]
        out.append((case, f"crello_{did}", t["name"], {"width": t["width"], "height": t["height"]}, sources[did]))
    return out, REPO / "eval" / "results" / suite


def run(suite="samples", render=False, contrast=False, jobs=1, out_dir=None):
    commit = _commit()  # before writing results, which would make the tree look dirty
    cases, default_dir = load_suite(suite)
    out_dir = out_dir or default_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "outputs").mkdir(exist_ok=True)
    if render:
        (out_dir / "renders").mkdir(exist_ok=True)
    _warm_up(render)
    if render:
        from eval.render import render_image
        seen = {}
        for _c, sample, _f, _t, source in cases:  # the original designs, for side-by-side review
            seen.setdefault(sample, source)
        for sample, source in seen.items():
            render_image(source).save(out_dir / "renders" / f"source__{sample}.png")

    work = [(c, render, contrast, str(out_dir)) for c in cases]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_warm_up, initargs=(render,)) as pool:
            rows = list(pool.map(_one, work, chunksize=4))
    else:
        rows = [_one(w) for w in work]

    _write_csv(rows, out_dir / "cases.csv")
    summary = _summary(rows, suite, render, contrast, commit, jobs)
    (out_dir / "summary.md").write_text(summary)
    if render and suite == "samples":
        formats = [f["name"] for f in json.loads(CATALOGUE.read_text())]
        thumbs = [(r["sample"], r["format"], Image.open(out_dir / "renders" / f"{r['case']}.png"), r)
                  for r in rows if (out_dir / "renders" / f"{r['case']}.png").is_file()]
        _contact_sheet(thumbs, sorted({r["sample"] for r in rows}), formats, out_dir / "contact_sheet.png")
    print(summary)
    return rows


class _Count(logging.Handler):
    def __init__(self):
        super().__init__(logging.INFO)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _one(job):
    (case, sample, fmt, target, source), render, contrast, out_dir = job
    out_dir = Path(out_dir)
    row = {"case": case, "sample": sample, "format": fmt, "width": target["width"], "height": target["height"]}
    log = _Count()
    logging.getLogger("engine").addHandler(log)
    logging.getLogger("engine").setLevel(logging.INFO)  # for the "winner: <recipe>" line
    try:
        t = time.perf_counter()
        output = resize(copy.deepcopy(source), target)
        row["resize_ms"] = (time.perf_counter() - t) * 1000
    except Exception:
        row.update(passed=False, violations=1, rules="crash", total=0.0, error=traceback.format_exc(limit=1))
        return row
    finally:
        logging.getLogger("engine").removeHandler(log)
    msgs = [r.getMessage() for r in log.records]
    row["winner"] = next((m.split(": ", 1)[1] for m in msgs if m.startswith("winner: ")), "")
    row["budget_hit"] = int(any(m.startswith("time budget reached") for m in msgs))
    row["illegible_rejected"] = sum(int(m.split()[1]) for m in msgs if m.startswith("rejected") and "backdrop or contrast" in m)
    row["no_legible"] = int(any(m.startswith("no legible candidate") for m in msgs))
    row["guard_dropped"] = sum(int(m.split()[2]) for m in msgs if m.startswith("guard dropped"))
    row["fallback"] = sum("fallback" in m for m in msgs)
    row["candidate_errors"] = sum("failed" in m and "fallback" not in m for m in msgs)
    errs = [f"{r.exc_info[0].__name__}: {r.exc_info[1]}"[:200] for r in log.records if r.exc_info]
    if errs:
        row["error"] = errs[0]

    (out_dir / "outputs" / f"{case}.json").write_text(json.dumps(output, indent=1))
    t = time.perf_counter()
    v = check(source, target, output)
    row["check_ms"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    s = score(source, target, output, contrast=contrast)
    row["score_ms"] = (time.perf_counter() - t) * 1000
    row.update(passed=not v, violations=len(v), rules=";".join(f"{x['rule']}:{x['id']}" for x in v),
               total=s["total"], **{k: b["penalty"] for k, b in s["breakdown"].items()})
    if render:
        from eval.render import render_image
        t = time.perf_counter()
        render_image(output).save(out_dir / "renders" / f"{case}.png")
        row["render_ms"] = (time.perf_counter() - t) * 1000
        render_image(stretch(source, target)).save(out_dir / "renders" / f"stretch__{case}.png")
    return row


def _warm_up(render=False):
    """Font loading and first-call imports take ~1s; keep them out of the per-case timings."""
    text_layout({"id": "w", "type": "text", "x": 0, "y": 0, "width": 200, "height": 40, "content": "Warm up",
                 "font_family": "Solstice Sans", "font_size": 20, "color": "rgba(0,0,0,1)"})
    if render:
        from eval.render import render_image
        render_image({"canvas": {"width": 10, "height": 10},
                      "elements": [{"id": "w", "type": "shape", "x": 0, "y": 0, "width": 10, "height": 10,
                                    "shape_type": "rectangle", "fill_color": "rgba(0,0,0,1)"}]})


def _write_csv(rows, path):
    cols = ["case", "sample", "format", "width", "height", "passed", "violations", "total",
            *PENALTIES, "rules", "winner", "budget_hit", "guard_dropped", "illegible_rejected", "no_legible", "fallback", "candidate_errors", "resize_ms", "check_ms", "score_ms", "render_ms", "error"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4g}" if isinstance(v, float) else v) for k, v in r.items()})


def _commit():
    try:
        return subprocess.run(["git", "describe", "--always", "--dirty"], cwd=REPO, capture_output=True,
                              text=True).stdout.strip() or "?"
    except OSError:
        return "?"


def _summary(rows, suite, render, contrast, commit, jobs):
    def mean(xs):
        xs = [x for x in xs if x is not None]
        return statistics.mean(xs) if xs else float("nan")

    def ms(key):
        xs = [r[key] for r in rows if key in r]
        if not xs:
            return "n/a"
        xs.sort()
        return (f"mean {statistics.mean(xs):.0f} ms, median {statistics.median(xs):.0f} ms, "
                f"p95 {xs[int(0.95 * (len(xs) - 1))]:.0f} ms, max {xs[-1]:.0f} ms")

    passed = sum(r["passed"] for r in rows)
    by_rule = Counter()
    cases_by_rule = Counter()
    for r in rows:
        rules = [x.split(":")[0] for x in r["rules"].split(";") if x]
        by_rule.update(rules)
        cases_by_rule.update(set(rules))
    penalties = [k for k in PENALTIES if any(k in r for r in rows)]
    samples = sorted({r["sample"] for r in rows})
    crashes = [r for r in rows if r["rules"] == "crash"]
    fallbacks = [r for r in rows if r.get("fallback")]
    cand = [r for r in rows if r.get("candidate_errors")]

    out = [f"# Benchmark summary: {suite}\n",
           f"Engine commit `{commit}`, {len(rows)} cases ({len(samples)} designs), "
           f"render={'yes' if render else 'no'}, contrast={'yes' if contrast else 'no'}, jobs={jobs}.\n",
           f"- **Cases passing every hard rule: {passed}/{len(rows)}** ({100 * passed / len(rows):.1f}%)",
           f"- **Average score: {mean(r['total'] for r in rows if r['rules'] != 'crash'):.2f}** / 100",
           f"- Crashes (resize raised): {len(crashes)}",
           f"- Engine fell back to a plain uniform scale: {len(fallbacks)} cases",
           f"- Cases where some layout candidate raised (others used): {len(cand)}\n",
           "## Violations by rule\n", "| rule | violations | cases |", "|---|---|---|"]
    out += [f"| {rule} | {n} | {cases_by_rule[rule]} |" for rule, n in by_rule.most_common()] or ["| (none) | 0 | 0 |"]
    errors = Counter(r["error"].strip().splitlines()[-1][:120] for r in rows if r.get("error"))
    if errors:
        out += ["\n## Exceptions (first per case)\n", "| exception | cases |", "|---|---|"]
        out += [f"| `{e}` | {n} |" for e, n in errors.most_common(10)]
    out += ["\n## Average penalty (0 = perfect, 1 = worst)\n", "| " + " | ".join(penalties) + " |",
            "|" + "---|" * len(penalties),
            "| " + " | ".join(f"{mean(r.get(k) for r in rows):.3f}" for k in penalties) + " |"]
    if suite == "samples":
        formats = json.loads(CATALOGUE.read_text())
        out += ["\n## Score by format\n", "| format | size | " + " | ".join(samples) + " | passed |",
                "|---|---|" + "---|" * (len(samples) + 1)]
        for f in formats:
            fr = [r for r in rows if r["format"] == f["name"]]
            cells = [f"{next(r['total'] for r in fr if r['sample'] == s):.1f}" for s in samples]
            out.append(f"| {f['name']} | {f['width']}x{f['height']} | " + " | ".join(cells) + f" | {sum(r['passed'] for r in fr)}/{len(fr)} |")
    else:
        out += ["\n## By target format\n", "| target | size | cases | passed | mean score |", "|---|---|---|---|---|"]
        for fmt in sorted({r["format"] for r in rows}):
            fr = [r for r in rows if r["format"] == fmt]
            out.append(f"| {fmt} | {fr[0]['width']}x{fr[0]['height']} | {len(fr)} | {sum(r['passed'] for r in fr)} | "
                       f"{mean(r['total'] for r in fr):.1f} |")
        worst = sorted(rows, key=lambda r: (r["passed"], r["total"]))[:WORST_LISTED]
        out += [f"\n## {WORST_LISTED} worst cases (failures first)\n", "| case | score | violations |", "|---|---|---|"]
        out += [f"| {r['case']} | {r['total']:.1f} | {r['rules'][:80]} |" for r in worst]
    out += ["\n## Timings (per case, this machine" + (f", {jobs} worker processes in parallel" if jobs > 1 else "") + ")\n",
            f"- resize: {ms('resize_ms')}", f"- check: {ms('check_ms')}",
            f"- score: {ms('score_ms')}" + (" (incl. contrast render)" if contrast else ""), f"- render: {ms('render_ms')}"]
    return "\n".join(out) + "\n"


def _contact_sheet(thumbs, samples, formats, path):
    label_h = 28
    sheet = Image.new("RGB", (CELL * len(formats) + 110, (CELL + label_h) * len(samples) + 10), "white")
    d = ImageDraw.Draw(sheet)
    for sample, fmt, img, row in thumbs:
        r, c = samples.index(sample), formats.index(fmt)
        x0, y0 = 110 + c * CELL, 5 + r * (CELL + label_h)
        t = img.copy()
        t.thumbnail((THUMB, THUMB))
        sheet.paste(t, (x0 + (CELL - t.width) // 2, y0 + (THUMB - t.height) // 2))
        d.rectangle([x0 + (CELL - t.width) // 2 - 1, y0 + (THUMB - t.height) // 2 - 1,
                     x0 + (CELL + t.width) // 2, y0 + (THUMB + t.height) // 2], outline="#999")
        mark = "PASS" if row["passed"] else f"{row['violations']} viol"
        d.text((x0 + 4, y0 + THUMB + 3), f"{fmt[:24]}", fill="black")
        d.text((x0 + 4, y0 + THUMB + 15), f"{row['total']:.1f}  {mark}", fill="#2F855A" if row["passed"] else "#C53030")
        if c == 0:
            d.text((5, y0 + THUMB // 2), sample, fill="black")
    sheet.save(path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--render", action="store_true", help="render every case to PNG + contact sheet")
    ap.add_argument("--contrast", action="store_true", help="include the render-based contrast penalty")
    ap.add_argument("--suite", choices=(*SUITES, "quick"), default="samples",
                    help="quick = crello-quick (about 100 dev cases) then the samples: the suite to iterate on")
    ap.add_argument("--jobs", type=int, default=1, help="worker processes (timings then include contention)")
    args = ap.parse_args()
    os.environ["CHAMELEON_CPU_CLOCK"] = "1"  # budget in CPU time: parallel cases are not cut short by the load (engine.clock)
    for suite in (("crello-quick", "samples") if args.suite == "quick" else (args.suite,)):
        t = time.perf_counter()
        run(suite=suite, render=args.render, contrast=args.contrast, jobs=args.jobs)
        rows = list(csv.DictReader((load_suite(suite)[1] / "cases.csv").open()))
        print(f"{suite}: {sum(r['passed'] == 'True' for r in rows)}/{len(rows)} pass, average "
              f"{sum(float(r['total'] or 0) for r in rows) / len(rows):.2f}, {time.perf_counter() - t:.0f} s")

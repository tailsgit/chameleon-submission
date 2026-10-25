"""Stored outputs of earlier engine versions, so they are never regenerated just to compare against.

    python -m eval.snapshots save v3 --suite crello-dev [--from DIR]   # store DIR/<suite>/outputs (default eval/results)
    python -m eval.snapshots compare v3 --suite crello-dev             # this run (eval/results/<suite>) vs the snapshot
    python -m eval.snapshots list

A snapshot is the outputs of one suite, kept in $CHAMELEON_CACHE (default: cache/ next to the data/ folder
that holds the converted Crello designs, so every checkout and worktree shares it; Crello outputs are
Crello content and stay out of git). Its scores are computed once per scorer and
stored next to the outputs, keyed by a hash of eval/score.py and eval/check.py: comparing after a
scorer change rescores the snapshot once (about a minute for crello-dev), never re-runs the engine.
`compare` reports pass counts, averages, cases better/worse and the cases whose output changed most.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _cache():
    from eval.crello_convert import OUT  # <data>/crello_chameleon
    return Path(os.environ.get("CHAMELEON_CACHE") or OUT.parent.parent / "cache") / "snapshots"


CACHE = _cache()
SAMPLES_DIR = REPO / "eval" / "results"


def results_dir(suite, root=None):
    root = Path(root) if root else SAMPLES_DIR
    return root if suite == "samples" else root / suite


def scorer_hash():
    h = hashlib.sha1()
    for f in ("score.py", "check.py"):
        h.update((REPO / "eval" / f).read_bytes())
    return h.hexdigest()[:10]


def save(name, suite, root=None):
    """Store a run's outputs. crello-quick and crello-structure are subsets of crello-dev: without a run of
    their own, their cases are taken from the crello-dev run."""
    src = results_dir(suite, root)
    if not (src / "outputs").is_dir() and suite in ("crello-quick", "crello-structure"):
        src = results_dir("crello-dev", root)
    dest = CACHE / name / suite
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "outputs").mkdir(parents=True)
    if suite == "samples":
        keep = None
    else:
        from eval.crello_testset import cases
        keep = {c for c, _d, _t in cases(suite.split("-", 1)[1])}
    for f in (src / "outputs").glob("*.json"):
        if keep is None or f.stem in keep:
            shutil.copy(f, dest / "outputs" / f.name)
    summary = (src / "summary.md").read_text() if (src / "summary.md").is_file() else ""
    commit = next((ln.split("`")[1] for ln in summary.splitlines() if "Engine commit `" in ln), "?")
    (dest / "meta.json").write_text(json.dumps({"engine": commit, "suite": suite, "cases": len(list((dest / "outputs").glob("*.json")))}))
    print(f"saved {name}/{suite}: {len(list((dest / 'outputs').glob('*.json')))} outputs (engine {commit}) -> {dest}")


def scores(name, suite):
    """{case: row} of the snapshot under the current scorer (computed once per scorer, then read back)."""
    d = CACHE / name / suite
    path = d / f"scores_{scorer_hash()}.csv"
    if not path.is_file():
        rows = rescore(d / "outputs", suite)
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return {r["case"]: r for r in csv.DictReader(path.open())}


def rescore(outputs, suite):
    from eval.bench import load_suite
    if suite == "samples":
        os.environ.pop("CHAMELEON_ASSETS", None)  # the samples' assets are brand_kit's
    cases, _dir = load_suite(suite)  # sets CHAMELEON_ASSETS for Crello suites
    from eval.check import check
    from eval.score import largest_share_ratio, relations_kept, score
    rows = []
    for case, sample, fmt, target, source in cases:
        o = json.loads((Path(outputs) / f"{case}.json").read_text())
        kept, pairs = relations_kept(source, target, o)
        rows.append({"case": case, "sample": sample, "format": fmt, "passed": not check(source, target, o),
                     "total": score(source, target, o)["total"], "rel_kept": kept, "rel_pairs": pairs,
                     "largest_share": largest_share_ratio(source, target, o)})
    return rows


def compare(name, suite, root=None, top=15):
    """The current run of `suite` (its cases.csv, scored by the current scorer) against snapshot `name`."""
    old = scores(name, suite)
    d = results_dir(suite, root)
    new = {r["case"]: r for r in csv.DictReader((d / "cases.csv").open())}
    both = [c for c in new if c in old]
    mean = lambda rows: sum(float(r["total"] or 0) for r in rows) / len(rows)
    delta = {c: float(new[c]["total"] or 0) - float(old[c]["total"]) for c in both}
    changed = [c for c in both if json.loads((d / "outputs" / f"{c}.json").read_text())
               != json.loads((CACHE / name / suite / "outputs" / f"{c}.json").read_text())]
    print(f"{suite}: {name} {sum(old[c]['passed'] == 'True' for c in both)}/{len(both)} pass, {mean([old[c] for c in both]):.2f}"
          f"  ->  now {sum(new[c]['passed'] == 'True' for c in both)}/{len(both)} pass, {mean([new[c] for c in both]):.2f}")
    print(f"  outputs changed {len(changed)}; better {sum(delta[c] > 0.01 for c in both)}, worse {sum(delta[c] < -0.01 for c in both)}")
    for c in sorted(changed, key=lambda c: delta[c])[:top]:
        print(f"  {delta[c]:+6.2f}  {c}")
    return {c: delta[c] for c in changed}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("save", "compare", "list"))
    ap.add_argument("name", nargs="?")
    ap.add_argument("--suite", default="crello-dev")
    ap.add_argument("--from", dest="root", help="results root holding <suite>/outputs (default eval/results)")
    args = ap.parse_args()
    if args.action == "list":
        for m in sorted(CACHE.glob("*/*/meta.json")):
            meta = json.loads(m.read_text())
            print(f"{m.parent.parent.name:12} {meta['suite']:18} {meta['cases']:4} outputs  engine {meta['engine']}")
    elif args.action == "save":
        save(args.name, args.suite, args.root)
    else:
        compare(args.name, args.suite, args.root)


if __name__ == "__main__":
    main()

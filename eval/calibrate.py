"""How well does eval.score agree with the hand labels, and which weights agree best?

    python -m eval.calibrate          # agreement of the recorded scores and of the current scorer
    python -m eval.calibrate --fit    # also fit the weights to the labels (paste them into PENALTIES)
    python -m eval.calibrate --suite crello-dev   # the labels given on that suite's review page
    python -m eval.calibrate --fit --suite samples,crello-dev,crello-retrieval-review   # several label sets together

Every label in eval/results/labels.json, current or in a case's history, carries
the exact output JSON it was given on, so labels from every engine version are
used, and the scorer can change without them going stale. A case labelled on
two different outputs counts twice. Agreement metrics:
  pairwise  - of all case pairs with different labels, the share the score orders
              the same way (ties count half); 0.5 = chance, 1 = perfect
  tau_b     - Kendall's tau-b between score and label (-1..1)
  auc       - chance a random Good case outscores a random Bad case
Fitting maximises a smooth version of `pairwise` (pairwise logistic loss,
non-negative weights). Leave-one-design-out shows how it does on a design it
was not fitted on, which is the honest number: there are only 45 labels.
"""

import argparse
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

from eval.score import PENALTIES, score

REPO = Path(__file__).resolve().parents[1]
LABELS = REPO / "eval" / "results" / "labels.json"
RANK = {"bad": 0, "okay": 1, "good": 2}


def _source(sample):
    if sample.startswith("crello_"):
        from eval.crello_convert import OUT
        os.environ["CHAMELEON_ASSETS"] = str(OUT / "assets")
        return json.loads((OUT / "designs" / f"{sample[len('crello_'):]}.json").read_text())["source"]
    os.environ["CHAMELEON_ASSETS"] = str(REPO / "brand_kit" / "assets")  # not left pointing at the crello assets
    return json.loads((REPO / "input" / f"chameleon_sample_{sample}.json").read_text())["source"]


def load(suite="samples"):
    labels = LABELS if suite == "samples" else REPO / "eval" / "results" / suite / "labels.json"
    sources = {}
    latest = {}  # (case, output) -> label; the latest verdict on the same output wins
    for case, entry in json.loads(labels.read_text()).items():
        for lab in [*entry.get("history", []), entry]:
            if lab.get("verdict") and lab.get("output"):
                key = (case, json.dumps(lab["output"], sort_keys=True))
                if key not in latest or lab.get("at", "") >= latest[key].get("at", ""):
                    latest[key] = lab
    cases = []
    for (case, _o), lab in sorted(latest.items()):
        sample, out = case.split("__")[0], lab["output"]
        if sample not in sources:
            sources[sample] = _source(sample)
        b = score(sources[sample], out["canvas"], out, contrast=True)["breakdown"]
        cases.append({"case": case, "sample": sample, "label": RANK[lab["verdict"]], "recorded": lab["score"],
                      "engine": lab.get("engine"), "p": np.array([b[k]["penalty"] for k in PENALTIES])})
    return cases


def agreement(scores, labels):
    scores, labels = np.asarray(scores, float), np.asarray(labels)
    right = n = 0.0
    conc = disc = ties_s = ties_l = 0
    for i, j in combinations(range(len(scores)), 2):
        ds, dl = np.sign(scores[i] - scores[j]), np.sign(labels[i] - labels[j])
        if dl:
            n += 1
            right += 1.0 if ds == dl else 0.5 if ds == 0 else 0.0
        if ds == 0 and dl == 0:
            continue
        if ds == 0:
            ties_s += 1
        elif dl == 0:
            ties_l += 1
        elif ds == dl:
            conc += 1
        else:
            disc += 1
    tau = (conc - disc) / np.sqrt((conc + disc + ties_s) * (conc + disc + ties_l))
    good, bad = scores[labels == 2], scores[labels == 0]
    auc = np.mean([(g > b) + 0.5 * (g == b) for g in good for b in bad]) if len(good) and len(bad) else float("nan")
    return {"pairwise": right / n if n else float("nan"), "tau_b": tau, "auc": auc}  # nan: every label the same


def totals(cases, w):
    P = np.stack([c["p"] for c in cases])
    return 100 * (1 - P @ w / w.sum())


MIN_WEIGHT = 0.05  # share of the weight sum: a penalty the labelled outputs barely vary on (the engine
                   # already avoids overlaps) still has to guard the engine's choice between candidates


def fit(cases, l2=1e-3, steps=4000, lr=0.5):
    """Weights minimising pairwise logistic loss over differently-labelled pairs, each at
    least MIN_WEIGHT of the total, scaled to sum to 10."""
    P = np.stack([c["p"] for c in cases])
    y = np.array([c["label"] for c in cases])
    i, j = np.array([(a, b) for a, b in combinations(range(len(cases)), 2) if y[a] != y[b]]).T
    sign = np.sign(y[i] - y[j])
    D = (P[j] - P[i]) * sign[:, None]  # w @ D > 0 when the better-labelled case has the lower weighted penalty
    w = np.ones(P.shape[1])
    for _ in range(steps):
        m = D @ w
        g = -(D * (1 / (1 + np.exp(np.clip(m, -50, 50))))[:, None]).mean(0) + l2 * w
        w = np.maximum(MIN_WEIGHT * w.sum(), w - lr * g)
    return 10 * w / w.sum()


def report(scores, cases, title):
    labels = np.array([c["label"] for c in cases])
    scores = np.asarray(scores)
    a = agreement(scores, labels)
    a["within"] = np.nanmean([agreement(scores[m], labels[m])["pairwise"]
                           for m in (np.array([c["sample"] == s for c in cases]) for s in sorted({c["sample"] for c in cases}))])
    print(f"{title:<46} {a['pairwise']:.3f}     {a['within']:.3f}     {a['tau_b']:+.3f}   {a['auc']:.3f}")
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--suite", default="samples", help="whose labels: samples (eval/results) or e.g. crello-dev")
    args = ap.parse_args()
    cases = [c for s in args.suite.split(",") for c in load(s)]
    print(f"{'':<46} pairwise  pairwise  tau_b    auc")
    print(f"{'':<46} all       within design")
    report([c["recorded"] for c in cases], cases, "old scorer (the scores you labelled against)")
    report(totals(cases, np.array([w for _f, w, _r in PENALTIES.values()])), cases, "this scorer, weights in PENALTIES")
    if not args.fit:
        return
    w = fit(cases)
    report(totals(cases, w), cases, "this scorer, freshly fitted (in-sample)")
    loo = np.zeros(len(cases))
    for s in sorted({c["sample"] for c in cases}):  # fit on two designs, score the third
        idx = [k for k, c in enumerate(cases) if c["sample"] == s]
        loo[idx] = totals([cases[k] for k in idx], fit([c for c in cases if c["sample"] != s]))
    # each held-out design is scored with different weights: only the within-design column is fair here
    report(loo, cases, "freshly fitted, leave-one-design-out")
    print("\nfitted weights:")
    for k, x in zip(PENALTIES, w):
        print(f"  {k:<14} {x:.2f}")


if __name__ == "__main__":
    main()

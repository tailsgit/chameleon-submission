"""Local review page for benchmark results.

    python -m eval.bench --render       # produce eval/results/{cases.csv,renders/}
    python -m eval.review_server        # then open http://127.0.0.1:8765
    python -m eval.review_server --results eval/results/crello-dev --worst 30

Serves eval/review.html, the renders, the per-case results, and saves Good / Okay /
Bad verdicts to eval/results/labels.json (written atomically on every click).
Each verdict stores the exact output JSON it was given on (from
eval/results/outputs/<case>.json), plus its score and engine commit, so
eval.calibrate can use labels from any engine version. Labelling a case whose
output has changed moves the old verdict into that case's `history` instead of
overwriting it. Binds to 127.0.0.1 only. Standard library only.
"""

import argparse
import csv
import json
import os
import re
import tempfile
import threading
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

EVAL = Path(__file__).resolve().parent
RESULTS = EVAL / "results"
RENDERS = RESULTS / "renders"
OUTPUTS = RESULTS / "outputs"
LABELS = RESULTS / "labels.json"
VERDICTS = {"good", "okay", "bad"}
WORST = None  # --worst N: only the N worst cases (failures first, then lowest score)
_lock = threading.Lock()


def load_cases():
    reader = csv.DictReader((RESULTS / "cases.csv").open())
    cols = reader.fieldnames
    penalty_cols = cols[cols.index("total") + 1:cols.index("rules")]  # bench writes the penalties between these
    rows = list(reader)
    cases = []
    for r in rows:
        cases.append({
            "case": r["case"], "sample": r["sample"], "format": r["format"],
            "width": int(r["width"]), "height": int(r["height"]),
            "passed": r["passed"] == "True", "total": float(r["total"] or 0),
            "penalties": {k: float(r[k]) for k in penalty_cols if r.get(k)},
            "violations": [{"rule": v.split(":", 1)[0], "id": v.split(":", 1)[1] if ":" in v else ""}
                           for v in r["rules"].split(";") if v],
            "output_png": f"/renders/{r['case']}.png" if (RENDERS / f"{r['case']}.png").is_file() else None,
            "source_png": f"/renders/source__{r['sample']}.png" if (RENDERS / f"source__{r['sample']}.png").is_file() else None,
            "stretch_png": f"/renders/stretch__{r['case']}.png" if (RENDERS / f"stretch__{r['case']}.png").is_file() else None,
            # a comparison folder can render an earlier engine's output instead of the stretch
            "before_png": f"/renders/before__{r['case']}.png" if (RENDERS / f"before__{r['case']}.png").is_file() else None,
        })
    if WORST:
        cases = sorted(cases, key=lambda c: (c["passed"], c["total"]))[:WORST]
    return cases


def engine_commit():
    m = re.search(r"Engine commit `([^`]+)`", (RESULTS / "summary.md").read_text()) if (RESULTS / "summary.md").is_file() else None
    return m.group(1) if m else "?"


def read_labels():
    try:
        return json.loads(LABELS.read_text())
    except (OSError, ValueError):
        return {}


def current_output(case):
    try:
        return json.loads((OUTPUTS / f"{case}.json").read_text())
    except (OSError, ValueError):
        return None


def public_labels(labels):
    """What the page needs: current verdicts, and whether each was given on the output shown now."""
    out = {}
    for case, e in labels.items():
        if e.get("verdict"):
            out[case] = {k: e.get(k) for k in ("verdict", "score", "engine", "at")}
            out[case]["current"] = e.get("output") is not None and e.get("output") == current_output(case)
    return out


def apply_verdict(labels, case, verdict, output, score, engine):
    """New verdict on `output`; an earlier verdict on a different output goes to history."""
    entry = labels.get(case, {})
    history = entry.get("history", [])
    if entry.get("verdict") and entry.get("output") != output:
        history = history + [{k: v for k, v in entry.items() if k != "history"}]
    if verdict is None:
        labels.pop(case, None)
        if history:
            labels[case] = {"history": history}
        return labels
    labels[case] = {"verdict": verdict, "score": score, "engine": engine, "output": output,
                    "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    if history:
        labels[case]["history"] = history
    return labels


def write_labels(labels):
    fd, tmp = tempfile.mkstemp(dir=RESULTS, prefix=".labels.", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(labels, f, indent=2, sort_keys=True)
    os.replace(tmp, LABELS)  # atomic: a crash mid-write never leaves a half-written file


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in ("/", "/review.html"):
            return self._send(200, (EVAL / "review.html").read_bytes(), "text/html; charset=utf-8")
        if path == "/api/cases":
            return self._json({"commit": engine_commit(), "cases": load_cases()})
        if path == "/api/labels":
            return self._json(public_labels(read_labels()))
        m = re.fullmatch(r"/renders/([A-Za-z0-9_\-]+\.png)", path)  # plain file names only: no traversal
        if m and (RENDERS / m.group(1)).is_file():
            return self._send(200, (RENDERS / m.group(1)).read_bytes(), "image/png")
        self._send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path != "/api/labels":
            return self._send(404, b"not found", "text/plain")
        try:
            body = json.loads(self.rfile.read(min(int(self.headers.get("Content-Length", 0)), 10_000)))
            case, verdict = body["case"], body["verdict"]
        except (ValueError, KeyError, TypeError):
            return self._send(400, b"expected {case, verdict}", "text/plain")
        cases = {c["case"]: c for c in load_cases()}
        if case not in cases or (verdict is not None and verdict not in VERDICTS):
            return self._send(400, b"unknown case or verdict", "text/plain")
        output = current_output(case)
        if output is None:
            return self._send(409, b"no output snapshot for this case; rerun python -m eval.bench --render", "text/plain")
        with _lock:
            labels = apply_verdict(read_labels(), case, verdict, output, cases[case]["total"], engine_commit())
            write_labels(labels)
        self._json(public_labels(labels))

    def _json(self, obj):
        self._send(200, json.dumps(obj).encode(), "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):  # keep the terminal quiet except for saves
        if self.command == "POST":
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--results", type=Path, default=RESULTS,
                    help="a bench results folder, e.g. eval/results/crello-dev (its labels.json lives there too)")
    ap.add_argument("--worst", type=int, help="show only the N worst cases")
    args = ap.parse_args()
    RESULTS = args.results.resolve()
    RENDERS, OUTPUTS, LABELS, WORST = RESULTS / "renders", RESULTS / "outputs", RESULTS / "labels.json", args.worst
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Review page: http://127.0.0.1:{args.port}  (labels -> {LABELS})", flush=True)
    server.serve_forever()

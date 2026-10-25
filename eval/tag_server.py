"""Local server for the diagnosis pages (eval/diagnose_pages.py): serves them and saves their tags.

    python -m eval.tag_server --port 8780       # http://127.0.0.1:8780/failures/  and  /recall/

Serves the static files under eval/results/crello-diagnose/ (pages and renders, built by eval.diagnose_pages).
GET /api/tags/<name> returns, and POST /api/tags/<name> replaces (atomically), eval/results/<name>_tags.json for
name in {failure, recall}. Binds to 127.0.0.1 only. Standard library only.
"""

import argparse
import json
import mimetypes
import os
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ROOT = RESULTS / "crello-diagnose"
TAGS = {n: RESULTS / f"{n}_tags.json" for n in ("failure", "recall", "recall2", "recall3")}  # recall2: the rules branch's re-run
MAX_BODY = 2_000_000
_lock = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        path = self.path.split("?", 1)[0]
        name = path[len("/api/tags/"):] if path.startswith("/api/tags/") else None
        if name is not None:
            if name not in TAGS:
                return self._send(404, b"unknown tag file", "text/plain")
            f = TAGS[name]
            return self._send(200, f.read_bytes() if f.exists() else b"{}", "application/json")
        rel = path.lstrip("/") + ("index.html" if path.endswith("/") else "")
        target = (ROOT / rel).resolve()
        if ROOT.resolve() not in target.parents or not target.is_file():  # no traversal outside the pages folder
            return self._send(404, b"not found", "text/plain")
        self._send(200, target.read_bytes(), mimetypes.guess_type(target.name)[0] or "application/octet-stream")

    def do_POST(self):
        name = self.path[len("/api/tags/"):] if self.path.startswith("/api/tags/") else None
        if name not in TAGS:
            return self._send(404, b"not found", "text/plain")
        n = int(self.headers.get("Content-Length", 0))
        if n > MAX_BODY:
            return self._send(413, b"too large", "text/plain")
        try:
            body = json.loads(self.rfile.read(n))
        except ValueError:
            return self._send(400, b"expected JSON", "text/plain")
        if not isinstance(body, dict):
            return self._send(400, b"expected a JSON object", "text/plain")
        with _lock:
            fd, tmp = tempfile.mkstemp(dir=RESULTS, prefix=f".{name}_tags.", suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump(body, f, indent=1, sort_keys=True)
            os.replace(tmp, TAGS[name])  # atomic: a crash mid-write never leaves a half-written file
        self._send(200, b'{"ok": true}', "application/json")

    def _send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if self.command == "POST":
            super().log_message(fmt, *args)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8780)
    args = ap.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"Diagnosis pages: http://127.0.0.1:{args.port}/failures/  and  http://127.0.0.1:{args.port}/recall/", flush=True)
    server.serve_forever()

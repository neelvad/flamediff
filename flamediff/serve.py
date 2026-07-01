"""flamediff serve: a thin HTTP server that renders the live report and refreshes as checkpoints
arrive -- the live counterpart to ``report --html``. Stdlib ``http.server`` (no dependency): a
background thread polls the run dir via a Watcher and caches the current Report; the page fetches
``/data.json`` on an interval and re-renders in place (no reload, keeps your selection).
"""
from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from flamediff.report import Watcher


def make_server(run_dir: str, *, host: str = "127.0.0.1", port: int = 8000,
                interval: float = 60.0, table: str | None = None,
                min_severity: float = 1.0) -> ThreadingHTTPServer:
    """Build (but don't start) the serving HTTP server; call ``serve_forever()`` on the result."""
    watcher = Watcher(run_dir, table=table, min_severity=min_severity)
    lock = threading.Lock()
    state: dict = {"report": None}
    poll_ms = max(1000, int(interval * 1000))

    def refresh() -> None:
        watcher.poll()
        rep = watcher.current_report()
        with lock:
            state["report"] = rep

    refresh()  # initial synchronous build so the first request has data

    def loop() -> None:
        while True:
            time.sleep(interval)
            try:
                refresh()
            except Exception:  # a transient read error shouldn't kill the watch thread
                pass

    threading.Thread(target=loop, daemon=True).start()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            return

        def do_GET(self):
            with lock:
                rep = state["report"]
            if self.path.startswith("/data.json"):
                body = (rep.to_json() if rep else "{}").encode()
                ctype = "application/json"
            else:
                body = (rep.to_html(live_poll_ms=poll_ms) if rep
                        else "<h1>flamediff: no checkpoints yet…</h1>").encode()
                ctype = "text/html; charset=utf-8"
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((host, port), Handler)

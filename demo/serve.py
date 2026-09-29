"""Serve the demo folder for local viewing.

The demo loads its objects with fetch and starts a Web Worker, which browsers
refuse from a file:// page, so it needs a server. A plain
``python -m http.server`` also works on most systems; this one pins the MIME
types, because some systems map ``.js`` to ``text/plain`` and the page then
comes up blank.

    python demo/serve.py [--port 8000]
"""
from __future__ import annotations

import argparse
import functools
import http.server
from pathlib import Path

TYPES = {
    ".js": "text/javascript", ".json": "application/json",
    ".splat": "application/octet-stream", ".html": "text/html; charset=utf-8",
    ".jpg": "image/jpeg", ".png": "image/png", ".css": "text/css",
}


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **TYPES}

    def log_message(self, fmt, *args):  # quiet: one line per request is noise here
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    handler = functools.partial(Handler, directory=str(root))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as srv:
        print(f"Fast Splat Morphing demo: http://127.0.0.1:{args.port}/")
        try:
            srv.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()

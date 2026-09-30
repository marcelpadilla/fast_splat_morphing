"""Bundle the demo so it runs from a file:// page, with no server.

Reads demo/ and writes the few files OPEN_ME.html needs to double-click and run:

    offline/manifest.js        objects.json, as a global
    offline/data/<f>.splat.js  each .splat, base64, one global per file
    offline/worker-source.js   fsm-worker.js with the paper's four library files
                               spliced in where its importScripts() call was, so
                               the worker can be started from a blob URL

The .splat payloads are the bulk of it and are loaded lazily by offline/boot.js,
so regenerating is only needed when demo/assets or the worker changes (the
worker source is the one piece of demo code copied here, and it is copied, not
rewritten).

    python offline/build_offline.py
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEMO = HERE.parent / "demo"
DATA = HERE / "data"
LIBS = ["trajectories.js", "cloud.js", "pairings.js", "renderer.js"]
IMPORT = "importScripts('trajectories.js', 'cloud.js', 'pairings.js', 'renderer.js');"


def main() -> None:
    DATA.mkdir(exist_ok=True)
    manifest = json.loads((DEMO / "assets" / "objects.json").read_text())

    (HERE / "manifest.js").write_text(
        "window.FSM_MANIFEST = " + json.dumps(manifest, separators=(",", ":")) + ";\n")
    print("manifest.js")

    for o in manifest["objects"]:
        raw = (DEMO / "assets" / o["file"]).read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        (DATA / (o["file"] + ".js")).write_text(
            "window.__FSM_B64 = window.__FSM_B64 || {};\n"
            f'window.__FSM_B64["assets/{o["file"]}"] = "{b64}";\n')
        print(f"data/{o['file']}.js  ({len(raw)} bytes)")

    worker = (DEMO / "fsm-worker.js").read_text()
    if IMPORT not in worker:
        raise SystemExit(f"expected to find the importScripts call in {DEMO / 'fsm-worker.js'}")
    spliced = worker.replace(IMPORT, "\n".join((DEMO / n).read_text() for n in LIBS))
    (HERE / "worker-source.js").write_text(
        "window.__FSM_WORKER_SRC = " + json.dumps(spliced) + ";\n")
    print("worker-source.js")


if __name__ == "__main__":
    main()

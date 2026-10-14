"""TEST-ONLY stand-in for a live 3-D window that CANNOT open (e.g. no display on the machine).

It reports the failure exactly like the real viewer does (viewer_status.json state ERROR with the reason) and
exits non-zero, so the dashboard's error path can be verified.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from src.simulation.live_io import atomic_write_json

STANDIN_FAILING_VIEWER_MODULE = "tests.helpers.standin_viewer_fail"
REASON = "could not open a PyBullet GUI window on this computer (stand-in: no display available)"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--override", default=None)
    ap.add_argument("--run-id", default="standalone")
    ap.add_argument("--parent-pid", type=int, default=None)
    a = ap.parse_args(argv)
    Path(a.runtime, "viewer_status.json")
    atomic_write_json(Path(a.runtime) / "viewer_status.json", {
        "state": "ERROR", "open": False, "pid": os.getpid(), "run_id": a.run_id, "error": REASON,
        "frames": 0, "camera": None, "command_log": [], "updated": time.time()})
    print(f"[standin_viewer_fail] {REASON}", file=sys.stderr, flush=True)
    time.sleep(0.05)
    return 3


if __name__ == "__main__":
    raise SystemExit(main())

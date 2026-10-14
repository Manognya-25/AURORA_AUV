"""TEMPORARY PROBE (not part of the app): does the live 3-D window really render the moving run?

Starts a real dashboard run through SimulationManager, asks for the live PyBullet window, and logs
viewer_status.json plus the true pose in state.json over time. Deleted after the check.

    python tools/probe_live_window.py --seconds 40
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.simulation_manager import SimulationManager  # noqa: E402
from src.simulation.live_io import read_json_retry          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=40.0)
    ap.add_argument("--realtime-factor", type=float, default=1.0)
    ap.add_argument("--navigation", action="store_true")
    ap.add_argument("--show-3d", action="store_true", help="start the run with its OWN PyBullet window (dashboard.headless=false)")
    a = ap.parse_args()
    rt = Path(tempfile.mkdtemp(prefix="probe3d_"))
    mgr = SimulationManager(runtime_dir=rt)
    print("runtime dir:", rt, flush=True)
    cfg = {"realtime_factor": a.realtime_factor, "navigation_enabled": bool(a.navigation)}
    if a.navigation:
        cfg["navigation_goal"] = [30.0, 28.0, -5.0]
    if a.show_3d:
        cfg["show_3d"] = True
    print("configure ->", json.dumps(mgr.configure(cfg))[:400], flush=True)
    print("start ->", json.dumps(mgr.start())[:300], flush=True)
    if not a.show_3d:
        while mgr.status().get("window", {}).get("can_open") is False and mgr.status().get("state") != "RUNNING":
            time.sleep(0.3)
        print("status:", json.dumps({"state": mgr.status().get("state"), "window": mgr.status().get("window")}), flush=True)
        try:
            print("open_window ->", json.dumps(mgr.open_window())[:400], flush=True)
        except Exception as exc:                  # noqa: BLE001 - the probe reports, the app already does
            print("open_window FAILED:", type(exc).__name__, exc, flush=True)
    run_dir = mgr._run_dir                                    # noqa: SLF001 - probe only
    print("run_dir:", run_dir, "viewer.log:", (run_dir / "viewer.log").exists(), flush=True)
    last = None
    t0 = time.time()
    while time.time() - t0 < a.seconds:
        vs = mgr.viewer_status() or {}
        try:
            snap = read_json_retry(run_dir / "state.json") or {}
        except OSError:
            snap = {}
        gt = snap.get("debug_ground_truth") or {}
        pos = gt.get("position")
        run_state = snap.get("run_status")
        frames = vs.get("frames")
        moving = None if (pos is None or last is None) else round(sum((x - y) ** 2 for x, y in zip(pos, last)) ** 0.5, 4)
        print(f"t={time.time() - t0:5.1f} viewer={vs.get('state')} frames={frames} run={run_state} "
              f"sim_time={snap.get('sim_time')} pose={None if pos is None else [round(v, 3) for v in pos]} "
              f"moved={moving} err={vs.get('error')}", flush=True)
        last = pos
        if run_state in ("COMPLETED", "ENDED_BEFORE_GOAL", "STOPPED", "FAILED"):
            break
        time.sleep(2.0)
    log = run_dir / "viewer.log"
    if log.exists():
        print("--- viewer.log ---", flush=True)
        print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]), flush=True)
    mgr.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

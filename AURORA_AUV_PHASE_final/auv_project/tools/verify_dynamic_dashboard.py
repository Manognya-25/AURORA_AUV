"""REAL PyBullet verification of the dynamic dashboard behaviour (no browser, no stand-ins).

Starts a real simulation through the real dashboard backend (dashboard.simulation_process), opens the real live
PyBullet window (src.simulation.live_viewer) and drives the dashboard controls, checking the ACTUAL simulation:

  1 auto-open 3-D view : with dashboard.auto_open_3d, Start opens the 3-D view of THIS run (same simulation pid)
  2 goal consistency   : configured goal == mission.goal == navigator.goal (exact), estimate vs truth distance
  3 live telemetry     : sim time, EKF pose and (if mapping) coverage keep changing between polls
  4 live sensor control: IMU/DVL FAILED -> 0.0 readings, ENABLED -> the real readings come back; sonar scan rays
  5 safety stop        : every sensor FAILED/UNHEALTHY stops the run (robot stopped, state STOPPED)
  6 reset/restart      : reset -> IDLE, a fresh start -> RUNNING again

Everything is measured from the run's own published telemetry (state.json) and the viewer's own status file.
    python tools/verify_dynamic_dashboard.py [--realtime-factor 4] [--no-window]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dashboard.simulation_manager import SimulationManager          # noqa: E402
from src.simulation.live_io import read_json_retry                  # noqa: E402

SENSORS = ("imu", "dvl", "depth", "compass", "sonar")


class Checker:
    def __init__(self):
        self.results: list[tuple[str, bool, str]] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        self.results.append((name, bool(ok), detail))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""), flush=True)
        return bool(ok)

    def summary(self) -> int:
        n_fail = sum(1 for _, ok, _ in self.results if not ok)
        print(f"\n==== {len(self.results) - n_fail}/{len(self.results)} checks passed ====", flush=True)
        return 1 if n_fail else 0


def snap_now(mgr: SimulationManager) -> dict:
    return mgr.state_snapshot() or {}


def wait_for(cond, timeout=90.0, step=0.2):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    return None


def wait_running(mgr: SimulationManager, timeout=120.0) -> bool:
    return wait_for(lambda: (mgr.status()["state"] in ("RUNNING", "STALE") and snap_now(mgr).get("sensors_enabled")),
                    timeout=timeout) is not None


def sensor(mgr: SimulationManager, name: str) -> dict:
    return (snap_now(mgr).get("sensors") or {}).get(name) or {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--realtime-factor", type=float, default=4.0)
    ap.add_argument("--no-window", action="store_true", help="skip the live 3-D window checks")
    a = ap.parse_args()

    runtime = Path(__file__).resolve().parents[1] / "runtime" / "verify_dynamic"
    if runtime.exists():
        import shutil
        shutil.rmtree(runtime, ignore_errors=True)
    ck = Checker()
    mgr = SimulationManager(runtime_dir=runtime)
    goal = [30.0, 30.0, -5.0]
    try:
        print("configure ->", json.dumps(mgr.configure({
            "scenario": "mapping_validation", "seed": 3, "realtime_factor": a.realtime_factor,
            "navigation_enabled": True, "navigation_goal": goal, "auto_open_3d": not a.no_window,
        }))[:300], flush=True)

        started = mgr.start()
        print("start ->", json.dumps({k: started.get(k) for k in ("state", "pid", "run_id")}), flush=True)
        pid = mgr.status()["pid"]
        if not wait_running(mgr):
            ck.check("run reaches RUNNING with telemetry", False, str(mgr.status().get("state")))
            return 1

        # ---- 1. auto-open the 3-D view of THIS run ------------------------------------------------
        if not a.no_window:
            w = wait_for(lambda: mgr.window_state() if mgr.window_state()["open"] else None, timeout=60) or mgr.window_state()
            ck.check("auto-open: a 3-D PyBullet window is open for this run", w.get("open") is True, json.dumps(w))
            ck.check("auto-open: it belongs to THIS run (same run_id, no second simulation)",
                     w.get("run_id") == mgr.run_id and mgr.status()["pid"] == pid,
                     f"window owner={w.get('owner')} run_id={w.get('run_id')} pid={mgr.status()['pid']}")
            vs = mgr.viewer_status() or {}
            f0 = vs.get("frames")
            wait_for(lambda: (mgr.viewer_status() or {}).get("frames", 0) > (f0 or 0), timeout=20)
            f1 = (mgr.viewer_status() or {}).get("frames")
            ck.check("3-D view is really rendering (frame count grows)", bool(f0 is not None and f1 and f1 > f0),
                     f"frames {f0} -> {f1}")

        # ---- 2. goal consistency ------------------------------------------------------------------
        snap = snap_now(mgr)
        mission_goal = (snap.get("mission") or {}).get("goal")
        nav_goal = (snap.get("navigation") or {}).get("goal")
        ck.check("goal: mission.goal == configured goal (exact)",
                 mission_goal == goal, f"{mission_goal} vs {goal}")
        ck.check("goal: navigator.goal == configured goal (exact)", nav_goal == goal, f"{nav_goal}")
        gd_est = (snap.get("navigation") or {}).get("goal_distance_m")
        gd_true = (snap.get("debug_ground_truth") or {}).get("goal_distance_true")
        ck.check("goal distance: estimate and ground-truth evaluation reported separately",
                 gd_est is not None and gd_true is not None, f"estimate={gd_est} truth={gd_true}")

        # ---- 3. live telemetry --------------------------------------------------------------------
        s0 = snap_now(mgr)
        time.sleep(2.0)
        s1 = snap_now(mgr)
        ck.check("live: simulation time advances between polls",
                 (s1.get("sim_time") or 0) > (s0.get("sim_time") or 0),
                 f"{s0.get('sim_time')} -> {s1.get('sim_time')}")
        p0 = ((s0.get("estimate") or {}).get("position"))
        p1 = ((s1.get("estimate") or {}).get("position"))
        ck.check("live: EKF pose changes between polls (graphs have real data)",
                 bool(p0 and p1 and p0 != p1), f"{p0} -> {p1}")
        so0 = (s0.get("sonar") or {})
        so1 = (s1.get("sonar") or {})
        ck.check("live: sonar scans update from the real sensor",
                 so0.get("available") and so1.get("available") and so0.get("timestamp") != so1.get("timestamp"),
                 f"t {so0.get('timestamp')} -> {so1.get('timestamp')}, rays={so1.get('n_rays')}")

        # ---- 4. live sensor control ---------------------------------------------------------------
        healthy_readings = {}
        for name in ("imu", "dvl"):
            wait_for(lambda n=name: sensor(mgr, n).get("value"), timeout=30)
            r = mgr.command({"cmd": "set_sensor_health", "sensor": name, "health": "FAILED"})
            got_zero = wait_for(lambda n=name: (lambda e: e.get("value") if e.get("zero") and e.get("value") else None)(sensor(mgr, n)),
                                timeout=30)
            ck.check(f"live control: {name} FAILED -> output stuck at 0.0",
                     bool(got_zero) and all(v == 0.0 for v in got_zero), f"value={got_zero} ({r.get('detail')})")
            mgr.command({"cmd": "set_sensor_health", "sensor": name, "health": "ENABLED"})
            back = wait_for(lambda n=name: (lambda e: e.get("value") if e.get("value") and any(v != 0.0 for v in e["value"]) else None)(sensor(mgr, n)),
                            timeout=30)
            healthy_readings[name] = back
            ck.check(f"live control: {name} ENABLED -> real readings return", bool(back), f"value={back}")

        # sonar scan: failure removes every valid ray, recovery brings the rays back
        before = snap_now(mgr).get("sonar") or {}
        mgr.command({"cmd": "set_sensor_health", "sensor": "sonar", "health": "FAILED"})
        dead = wait_for(lambda: (lambda s: s if s.get("available") and s.get("valid_rays") == 0 else None)(snap_now(mgr).get("sonar") or {}),
                        timeout=30)
        ck.check("live control: sonar FAILED -> 0 valid rays (no fake ranges)",
                 bool(dead) and dead.get("hit_rays") == 0 and all(r is None for r in (dead.get("range_m") or [])),
                 f"valid_rays={None if not dead else dead.get('valid_rays')}")
        mgr.command({"cmd": "set_sensor_health", "sensor": "sonar", "health": "ENABLED"})
        alive = wait_for(lambda: (lambda s: s if s.get("available") and s.get("valid_rays") > 0 else None)(snap_now(mgr).get("sonar") or {}),
                         timeout=30)
        ck.check("live control: sonar ENABLED -> valid rays return", bool(alive),
                 f"valid_rays={None if not alive else alive.get('valid_rays')} (was {before.get('valid_rays')})")

        # ---- 5. safety stop -----------------------------------------------------------------------
        for name in (n for n in SENSORS if sensor(mgr, n)):
            mgr.command({"cmd": "set_sensor_health", "sensor": name, "health": "FAILED"})
        stopped = wait_for(lambda: (lambda s: s if s["state"] == "STOPPED" and not mgr.alive else None)(mgr.status()), timeout=60)
        snap = snap_now(mgr)
        safety = snap.get("safety") or {}
        ck.check("safety: every sensor failed -> the simulation and robot stop",
                 bool(stopped) and safety.get("stopped") is True and "failed/unhealthy" in (safety.get("reason") or ""),
                 f"state={None if not stopped else stopped['state']} reason={safety.get('reason')}")

        # ---- 6. reset -> IDLE, fresh start -> RUNNING ---------------------------------------------
        mgr.reset()
        ck.check("reset -> IDLE (no run)", mgr.status()["state"] == "IDLE", mgr.status()["state"])
        mgr.configure({"auto_open_3d": False})          # a fresh run without opening another window
        mgr.start()
        ck.check("restart after reset -> RUNNING again with real telemetry", wait_running(mgr),
                 mgr.status()["state"])
        ck.check("restart: the new run is a NEW process", mgr.status()["pid"] != pid)
        return ck.summary()
    finally:
        mgr.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())

"""TEST-ONLY stand-in for the live 3-D window process (src/simulation/live_viewer.py).

It speaks the exact same file IPC -- viewer_status.json (STARTING -> OPEN -> CLOSED/ERROR),
viewer_commands.jsonl (camera commands, acked in viewer_status.json's command_log), viewer_stop.request -- so the
dashboard's open / close / camera paths can be exercised without PyBullet and without a display. It never touches
the run's files; it only reads child.json/status.json to know whether there is still anything to show.

Like the real viewer it KEEPS the window open when the run finishes on its own (visualization.keep_open_after_run,
true by default) and only closes when the run vanished, the user closes it, or the dashboard asks it to.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from src.simulation.live_io import CommandChannel, atomic_write_json, read_json_retry
from src.utils.procutil import pid_alive

STANDIN_VIEWER_MODULE = "tests.helpers.standin_viewer"
STATUS_FILE = "viewer_status.json"
COMMANDS_FILE = "viewer_commands.jsonl"
STOP_FILE = "viewer_stop.request"
TERMINAL_RUN_STATES = ("COMPLETED", "ENDED_BEFORE_GOAL", "FAILED", "STOPPED")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--override", default=None)
    ap.add_argument("--run-id", default="standalone")
    ap.add_argument("--parent-pid", type=int, default=None)
    a = ap.parse_args(argv)
    run_dir = Path(a.runtime)
    frames, log, cmd_id, held_open = 0, [], 0, False

    def write(state: str, error=None) -> None:
        atomic_write_json(run_dir / STATUS_FILE, {
            "state": state, "open": state == "OPEN", "pid": os.getpid(), "run_id": a.run_id, "error": error,
            "frames": frames, "camera": {"mode": "follow", "stand_in": True} if state == "OPEN" else None,
            "command_log": log[-20:], "updated": time.time()})

    def run_gone() -> str:
        """Nothing left to show because the run VANISHED before finishing (no terminal status): the real viewer
        closes for these too, so a crashed run cannot leave an orphan window behind."""
        if a.parent_pid is not None and not pid_alive(a.parent_pid):
            return "dashboard process exited"
        child = read_json_retry(run_dir / "child.json") or {}
        if child.get("pid") is not None and not pid_alive(int(child["pid"])) and not run_ended():
            return f"the simulation process ({child['pid']}) exited"
        return ""

    def run_ended() -> str:
        """The run's own terminal status, else '' -- mirrors live_viewer.run_ended_status, which keeps the window
        open instead of closing it (status.json is the run's final word; child.json keeps its pid)."""
        if a.parent_pid is not None and not pid_alive(a.parent_pid):
            return ""
        st = read_json_retry(run_dir / "status.json") or {}
        status = str(st.get("status", "")).upper()
        return status if status in TERMINAL_RUN_STATES else ""

    write("STARTING")
    try:
        commands = CommandChannel(run_dir / COMMANDS_FILE)
        write("OPEN")
        print("[standin_viewer] window open", file=sys.stderr, flush=True)
        while True:
            stop = run_dir / STOP_FILE
            if stop.exists():
                try:
                    content = stop.read_text(encoding="utf-8").strip()
                except OSError:
                    content = a.run_id
                if content in ("", a.run_id):
                    write("CLOSED")
                    return 0
            for cmd in commands.poll():
                cmd_id = int(cmd.get("id", cmd_id))
                name = str(cmd.get("cmd", ""))
                if name == "stop":
                    log.append({"id": cmd_id, "cmd": name, "ok": True, "detail": "window closing", "sim_time": time.time()})
                    write("CLOSED")
                    return 0
                if name == "set_camera":
                    log.append({"id": cmd_id, "cmd": name, "ok": True,
                                "detail": f"stand-in viewer camera {cmd.get('action')}", "sim_time": time.time()})
                else:
                    log.append({"id": cmd_id, "cmd": name, "ok": False, "detail": f"unknown viewer command {name!r}",
                                "sim_time": time.time()})
                write("OPEN")
            gone = run_gone()
            if gone:
                write("CLOSED", error=f"closed: {gone} (nothing left to show)")
                return 0
            ended = run_ended()
            if ended and not held_open:
                held_open = True
                print(f"[standin_viewer] the run ended ({ended}); keeping the window open on the last frame",
                      file=sys.stderr, flush=True)
            frames += 1
            if frames % 5 == 0:
                write("OPEN")
            time.sleep(0.05)
    except BaseException as exc:                    # noqa: BLE001 - reported like the real viewer does
        write("ERROR", error=f"{type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

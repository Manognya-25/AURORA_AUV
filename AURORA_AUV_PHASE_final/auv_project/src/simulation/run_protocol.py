"""Child-side run protocol shared by dashboard/simulation_process.py and the test stand-in.

Every run owns a private directory (runtime/runs/<run_id>/). Nothing in it can be touched by another
run, so stale files from an earlier run, a crashed run or an orphaned process can never be mistaken for
the current run's state. Files: child.json (pid/run_id), status.json, state.json, maps.json,
commands.jsonl, stop.request, simulation.log.

PyBullet-free on purpose (importable and testable without PyBullet).
"""
from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

from src.simulation.live_io import IpcWriteError, atomic_write_json, read_json_retry
from src.utils.procutil import pid_alive


class ChildRun:
    def __init__(self, runtime, run_id: str, parent_pid: int | None = None, poll_s: float = 0.05):
        self.dir = Path(runtime).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.parent_pid = parent_pid
        self.poll_s = poll_s
        self.stop_event = threading.Event()
        self.stop_reason: str | None = None
        self._done = threading.Event()
        atomic_write_json(self.dir / "child.json", {"pid": os.getpid(), "run_id": run_id, "started": time.time(),
                                                    "parent_pid": parent_pid})

    # ------------------------------------------------------------------ status
    def write_status(self, status: str, result=None, error=None) -> None:
        """status.json is how the dashboard learns the outcome. A failure to write it is reported on
        stderr (never swallowed) but does not mask the simulation's own result."""
        try:
            atomic_write_json(self.dir / "status.json", {"status": status, "result": result, "error": error,
                                                         "run_id": self.run_id, "pid": os.getpid(),
                                                         "time": time.time()})
        except IpcWriteError as exc:
            print(f"[run_protocol] could not write status {status!r}: {exc}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ stop / parent watch
    def _stop_requested(self) -> bool:
        p = self.dir / "stop.request"
        try:
            content = p.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            return False
        except OSError:
            return False                      # transient (Windows) read problem: look again next poll
        return content in ("", self.run_id)   # a request addressed to another run is ignored

    def start_watchers(self) -> threading.Event:
        def watch():
            while not self._done.is_set():
                if self._stop_requested():
                    self.stop_reason = "stop requested"
                    self.stop_event.set()
                    return
                if self.parent_pid is not None and not pid_alive(self.parent_pid):
                    self.stop_reason = "dashboard process exited"
                    self.stop_event.set()
                    return
                time.sleep(self.poll_s)
        threading.Thread(target=watch, name="run-protocol-watch", daemon=True).start()
        return self.stop_event

    def finish(self) -> None:
        self._done.set()


def live_child_pids(runs_root) -> list[tuple[Path, int]]:
    """Directories under runs_root whose child.json names a still-alive process (orphan detection)."""
    out = []
    root = Path(runs_root)
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        try:
            info = read_json_retry(d / "child.json", attempts=3)
        except OSError:
            continue
        if info and pid_alive(info.get("pid")):
            out.append((d, int(info["pid"])))
    return out

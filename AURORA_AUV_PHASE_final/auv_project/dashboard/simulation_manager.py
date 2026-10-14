"""Owns the simulation child process and derives the dashboard state from REAL facts.

State is never stored as a UI flag. It is computed on every request from:
    * the OS process (alive / return code),
    * status.json written by the child,
    * the age of the newest state.json snapshot (telemetry timeout -> STALE).
States: IDLE, STARTING, RUNNING, STALE (process alive but telemetry stopped), STOPPING, STOPPED,
COMPLETED, ENDED_BEFORE_GOAL, FAILED.

Robustness (Windows in particular):
    * Every run gets its OWN directory runtime/runs/<run_id>/. Nothing is shared with or deleted from a
      previous run, so stale files, locked files and orphaned children cannot corrupt a new run.
    * Reads/writes/deletes of runtime files retry on transient sharing errors (WinError 5/32/33); a read
      that stays locked returns the last good snapshot instead of flickering to "no data".
    * The parent never keeps the child's log handle open; start/stop/reset are serialized by a lock.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import yaml

from dashboard.recorder import BASE_COLUMNS, build_run_summary, flatten_snapshot, rows_to_csv
from src.simulation.commands import operator_xyz_to_world as _operator_xyz_to_world
from src.simulation.live_io import atomic_write_json, read_json_retry, rmtree_retry, send_command, unlink_retry
from src.simulation.run_protocol import live_child_pids
from src.utils.procutil import pid_alive

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME = PROJECT_ROOT / "runtime"
CHILD_MODULE = "dashboard.simulation_process"
VIEWER_MODULE = "src.simulation.live_viewer"      # the live 3-D window process (see src/simulation/live_viewer.py)
KEEP_OLD_RUNS = 2
MAX_RECORDED_ROWS = 12000           # bounded telemetry recording for CSV export (no unbounded growth)
MAX_RUN_HISTORY = 50                # finished runs kept in runtime/run_history.json
TERMINAL_STATES = ("COMPLETED", "ENDED_BEFORE_GOAL", "FAILED", "STOPPED")


class ApiError(Exception):
    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class SimulationManager:
    def __init__(self, runtime_dir=None, config_path=None, child_module: str = CHILD_MODULE,
                 telemetry_timeout_s: float | None = None, viewer_module: str = VIEWER_MODULE):
        self.runtime = Path(runtime_dir or DEFAULT_RUNTIME)
        self.runs_root = self.runtime / "runs"
        self.config_path = Path(config_path or PROJECT_ROOT / "configs" / "config.yaml")
        self.child_module = child_module
        self.viewer_module = viewer_module
        self._lock = threading.RLock()
        self._proc: subprocess.Popen | None = None
        # Live 3-D window (PyBullet GUI) for a run that was started headless: a separate viewer process that
        # mirrors the RUN's published telemetry. Opening/closing it never touches the simulation process.
        self._viewer: subprocess.Popen | None = None
        self._viewer_cmd_id = 0
        self._viewer_error: str | None = None
        self._viewer_started_wall: float | None = None
        self._run_id: str | None = None
        self._run_dir: Path | None = None
        self._started_wall: float | None = None
        self._stop_requested = False
        self._stop_forced = False
        self._cmd_id = 0
        self._timeout_override = telemetry_timeout_s
        self._cfg_cache: dict | None = None
        self._good: dict = {}                      # last successfully read payload per file name
        self.warnings: list[str] = []
        self.pending: dict = {}                    # applied on the NEXT start only
        self.active_overrides: dict = {}           # what the current/last run was started with
        self._run_headless = True                  # dashboard.headless the current run was started with
        self._auto_open_3d = False                 # dashboard.auto_open_3d the current run was started with
        # Live telemetry recording (CSV export + run history): one flattened row per published snapshot,
        # bounded so a long run can never grow memory without limit.
        self._rows: deque = deque(maxlen=MAX_RECORDED_ROWS)
        self._cols: list[str] = list(BASE_COLUMNS)
        self._record_key = None
        self._finalized_run_id: str | None = None
        self._history_path = self.runtime / "run_history.json"
        self.runs_root.mkdir(parents=True, exist_ok=True)
        self._stop_orphans()                       # children left over from a previous dashboard process

    # ------------------------------------------------------------------ run directory + files
    @property
    def run_dir(self) -> Path | None:
        return self._run_dir

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def _read(self, name: str):
        d = self._run_dir
        if d is None:
            return None
        try:
            data = read_json_retry(d / name)
        except OSError:
            return self._good.get(name)            # persistently locked: keep showing the last good value
        if data is None:
            self._good.pop(name, None)
            return None
        self._good[name] = data
        if name == "state.json" and isinstance(data, dict):
            self._record_telemetry(data)
        return data

    # ------------------------------------------------------------------ telemetry recording / export
    def _record_telemetry(self, snap: dict) -> None:
        key = (self._run_id, snap.get("step"), snap.get("run_status"))
        with self._lock:
            if key == self._record_key:              # dedupe: the same snapshot may be read by many routes
                return
            self._record_key = key
            row = flatten_snapshot(snap, self._run_id)
            for k in row:
                if k not in self._cols:
                    self._cols.append(k)
            self._rows.append(row)

    def _clear_recording(self) -> None:
        with self._lock:
            self._rows.clear()
            self._cols = list(BASE_COLUMNS)
            self._record_key = None

    def has_telemetry(self) -> bool:
        return bool(self._rows)

    def recorded_row_count(self) -> int:
        return len(self._rows)

    def telemetry_csv(self) -> str:
        with self._lock:
            cols = list(self._cols)
            for r in self._rows:
                for k in r:
                    if k not in cols:
                        cols.append(k)
            return rows_to_csv(list(self._rows), cols)

    # ------------------------------------------------------------------ run history (baseline vs fault runs)
    def _load_history(self) -> list:
        try:
            data = json.loads(self._history_path.read_text("utf-8"))
            return data if isinstance(data, list) else []
        except (OSError, ValueError):
            return []

    def _finalize_run(self, state: str | None = None) -> None:
        """Persist one run's mission summary when the run ends (stop/reset/restart/observed terminal state).
        Called at most once per run_id; the summary is what /api/runs compares (baseline vs fault runs)."""
        with self._lock:
            rid = self._run_id
            if rid is None or rid == self._finalized_run_id:
                return
            snap = self.state_snapshot() or {}
            child = self._read("status.json") or {}
            if not snap and not self._rows:
                return                                # nothing was ever recorded for this run
            state = state or child.get("status") or snap.get("run_status")
            if state == "ERROR":
                state = "FAILED"
            entry = build_run_summary(run_id=rid, snapshot=snap, rows=list(self._rows),
                                      result=child.get("result") or {}, state=state)
            if entry is None:
                return
            self._finalized_run_id = rid
            history = self._load_history()
            if any(h.get("run_id") == rid for h in history):
                return
            history.append(entry)
            atomic_write_json(self._history_path, history[-MAX_RUN_HISTORY:])

    def run_history(self) -> list:
        """Finished runs (persisted) plus the current run as a live entry and any orphaned run directories."""
        with self._lock:
            entries = [dict(e) for e in self._load_history()]
            known = {e.get("run_id") for e in entries}
            if self._run_id and self._run_id not in known and (self._rows or self.state_snapshot()):
                child = self._read("status.json") or {}
                live = build_run_summary(run_id=self._run_id, snapshot=self.state_snapshot() or {},
                                         rows=list(self._rows), result=child.get("result") or {},
                                         state=(child.get("status") or "RUNNING"))
                if live is not None:
                    live["live"] = True
                    entries.append(live)
                    known.add(self._run_id)
            for d in sorted(self.runs_root.iterdir()):       # run dirs a previous dashboard never summarised
                if not d.is_dir() or d.name in known:
                    continue
                try:
                    snap = read_json_retry(d / "state.json")
                except OSError:
                    snap = None
                if not isinstance(snap, dict):
                    continue
                try:
                    child = read_json_retry(d / "status.json") or {}
                except OSError:
                    child = {}
                entry = build_run_summary(run_id=d.name, snapshot=snap, rows=[],
                                          result=child.get("result") or {},
                                          state=child.get("status") or snap.get("run_status"))
                if entry is not None:
                    entries.append(entry)
            entries.sort(key=lambda e: str(e.get("run_id") or ""), reverse=True)
            return entries[:MAX_RUN_HISTORY]

    def _base_cfg(self) -> dict:
        if self._cfg_cache is None:
            with self.config_path.open("r", encoding="utf-8") as f:
                self._cfg_cache = yaml.safe_load(f)
        return self._cfg_cache

    @property
    def telemetry_timeout(self) -> float:
        if self._timeout_override is not None:
            return float(self._timeout_override)
        return float((self._base_cfg().get("dashboard", {}) or {}).get("telemetry_timeout_s", 3.0))

    def _stop_orphans(self, wait_s: float = 3.0) -> None:
        """Ask children of an earlier dashboard process to stop (through THEIR run directory, so the
        request cannot reach the new run). Never kills by PID alone: PIDs get reused."""
        orphans = [(d, pid) for d, pid in live_child_pids(self.runs_root)
                   if self._proc is None or pid != self._proc.pid]
        for d, pid in orphans:
            try:
                (d / "stop.request").write_text("", encoding="utf-8")
            except OSError as exc:
                self.warnings.append(f"could not signal orphan {pid}: {exc}")
        deadline = time.time() + wait_s
        while orphans and time.time() < deadline:
            orphans = [(d, pid) for d, pid in orphans if pid_alive(pid)]
            if orphans:
                time.sleep(0.1)
        for d, pid in orphans:
            self.warnings.append(f"orphaned simulation process {pid} ({d.name}) did not exit; it writes only "
                                 f"to its own run directory and cannot affect this run")

    def _prune_old_runs(self) -> None:
        dirs = sorted(d for d in self.runs_root.iterdir() if d.is_dir())
        alive_dirs = {d for d, _ in live_child_pids(self.runs_root)}
        for d in dirs[:-(KEEP_OLD_RUNS + 1)] if len(dirs) > KEEP_OLD_RUNS + 1 else []:
            if d == self._run_dir or d in alive_dirs:
                continue
            rmtree_retry(d, attempts=5)               # best effort: a locked old run never blocks a new one

    # ------------------------------------------------------------------ state
    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def state_snapshot(self) -> dict | None:
        return self._read("state.json")

    def status(self) -> dict:
        with self._lock:
            now = time.time()
            proc = self._proc
            child = self._read("status.json")
            snap = self._read("state.json")
            age = None if snap is None else max(0.0, now - float(snap.get("wall_time", now)))
            error = (child or {}).get("error")
            if proc is None:
                state = "IDLE"
            elif proc.poll() is None:
                cstat = (child or {}).get("status")
                if self._stop_requested:
                    state = "STOPPING"
                elif (child or {}).get("holding") and cstat in TERMINAL_STATES:
                    # The run finished on its own but its GUI process is still alive, holding the 3-D window it
                    # opened open on the last frame (`visualization.keep_open_after_run`). The run is OVER: report
                    # its real terminal state, not RUNNING. `alive` stays true so the window is still reachable and
                    # the window control (or Stop) can release it.
                    state = cstat
                elif snap is None:
                    state = "STARTING"
                elif age is not None and age > self.telemetry_timeout:
                    state = "STALE"
                else:
                    state = "RUNNING"
            else:
                code = proc.returncode
                cstat = (child or {}).get("status")
                if code == 0 and cstat in ("COMPLETED", "ENDED_BEFORE_GOAL", "STOPPED"):
                    state = cstat
                else:
                    state = "FAILED"
                    error = error or self._log_tail() or f"simulation process exited with code {code}"
            holding = bool((child or {}).get("holding"))
            out = {
                "state": state, "alive": self.alive, "pid": None if proc is None else proc.pid,
                "returncode": None if proc is None else proc.poll(),
                "run_id": self._run_id, "error": error, "telemetry_age_s": age,
                "telemetry_timeout_s": self.telemetry_timeout,
                "uptime_s": None if self._started_wall is None else now - self._started_wall,
                "stop_forced": self._stop_forced, "child_status": (child or {}).get("status"),
                "result": (child or {}).get("result"), "warnings": list(self.warnings),
                "ipc": (snap or {}).get("ipc"),
                "active_overrides": self.active_overrides, "pending_overrides": self.pending,
                "auto_open_3d": self._auto_open_3d, "holding": holding,
            }
            if state in TERMINAL_STATES:              # the run ended: persist its summary for /api/runs
                self._finalize_run(state)
            self._reap_viewer()                       # a window that ended by itself is not reported as open
            out["window"] = self.window_state()
            return out

    def _log_tail(self, n: int = 12) -> str:
        if self._run_dir is None:
            return ""
        try:
            lines = (self._run_dir / "simulation.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> dict:
        with self._lock:
            if self.alive:
                raise ApiError("simulation is already running; stop or reset it first", 409)
            self._finalize_run()                       # summarise the previous run before its directory is replaced
            self._stop_viewer()                        # a live window belongs to the old run's directory
            self._stop_orphans()
            run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]
            run_dir = self.runs_root / run_id
            run_dir.mkdir(parents=True, exist_ok=False)
            self._run_id, self._run_dir, self._good = run_id, run_dir, {}
            self._finalized_run_id = None
            self._clear_recording()                    # CSV export + run history start fresh per run
            self._stop_requested = self._stop_forced = False
            self._cmd_id = 0
            self.active_overrides = json.loads(json.dumps(self.pending))
            self._run_headless = self._headless_from(self.active_overrides)   # GUI run <=> not headless
            self._auto_open_3d = self._auto_open_from(self.active_overrides)  # open the 3-D window for THIS run?
            args = [sys.executable, "-m", self.child_module, "--config", str(self.config_path),
                    "--runtime", str(run_dir), "--run-id", run_id, "--parent-pid", str(os.getpid())]
            if self.active_overrides:
                atomic_write_json(run_dir / "config_override.json", self.active_overrides)
                args += ["--override", str(run_dir / "config_override.json")]
            kwargs = {}
            if os.name == "nt":                     # own process group: a Ctrl-C aimed at the dashboard
                kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP   # does not hit the child
            with open(run_dir / "simulation.log", "wb") as log_fh:
                self._proc = subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL,
                                              stdout=log_fh, stderr=subprocess.STDOUT, **kwargs)
            # the child owns its duplicate of the log handle; the parent keeps none (Windows file locking)
            self._started_wall = time.time()
            self._prune_old_runs()
            auto_open = self._auto_open_3d
        if auto_open:
            # Auto-open the 3-D view for the run that was just started (requirement: a Start reveals the 3-D view
            # of THAT run). Never fails the start: a missing display is reported as a warning.
            try:
                self.open_window()
            except ApiError as exc:
                self.warnings.append(f"auto-open of the 3-D window failed: {exc}")
        return self.status()

    def stop(self, timeout: float = 8.0) -> dict:
        """Ask the child to stop; wait for the OS process to really exit; escalate if it does not."""
        with self._lock:
            if not self.alive:
                raise ApiError("simulation is not running", 409)
            self._stop_requested = True
            (self._run_dir / "stop.request").write_text(self._run_id, encoding="utf-8")
            proc = self._proc
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._stop_forced = True
            proc.terminate()                            # TerminateProcess on Windows
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
        self._stop_viewer()                             # nothing left to show: close the live window too
        return self.status()

    def restart(self) -> dict:
        """Stop the current run (if any) and start a new one with the pending (next-start) settings: how a changed obstacle
        count / scenario / seed / start / goal becomes a NEW real world immediately instead of at the next manual start."""
        with self._lock:
            was_alive = self.alive
        if was_alive:
            self.stop()
        out = self.start()
        out["was_running"] = was_alive
        return out

    def reset(self) -> dict:
        """Stop any run and forget it entirely -> IDLE. Its directory (snapshots, maps, log) is removed
        best-effort; the next start gets a brand-new directory and rebuilds world, vehicle, sensors,
        EKF and map from the configuration."""
        with self._lock:
            was_alive = self.alive
        if was_alive:
            self.stop()
        with self._lock:
            self._stop_viewer()                        # its files live in the directory about to be removed
            self._finalize_run()                       # persist the run summary before its directory is removed
            old = self._run_dir
            self._proc = None
            self._run_id = self._run_dir = None
            self._started_wall = None
            self._stop_requested = self._stop_forced = False
            self._good = {}
            self._finalized_run_id = None
            self._run_headless = True
            self._auto_open_3d = False
            self._clear_recording()
            if old is not None:
                rmtree_retry(old)
        out = self.status()
        out["was_running"] = was_alive
        return out

    # ------------------------------------------------------------------ commands to the running sim
    def command(self, cmd: dict, wait_s: float = 3.0) -> dict:
        """Send a command to the running simulation and return what the simulation REPORTED."""
        with self._lock:
            if not self.alive:
                raise ApiError("simulation is not running; this control acts on a running simulation", 409)
            self._cmd_id += 1
            cid = self._cmd_id
            send_command(self._run_dir / "commands.jsonl", {**cmd, "id": cid})
        deadline = time.time() + wait_s
        while time.time() < deadline:
            snap = self.state_snapshot()
            if snap:
                for entry in snap.get("command_log", []):
                    if entry.get("id") == cid:
                        if not entry.get("ok"):
                            raise ApiError(entry.get("detail", "command rejected"), 422)
                        return {"id": cid, "ok": True, "detail": entry.get("detail"), "sim_time": entry.get("sim_time")}
            if not self.alive:
                raise ApiError("simulation ended before acknowledging the command", 409)
            time.sleep(0.05)
        raise ApiError("simulation did not acknowledge the command in time", 504)

    # ------------------------------------------------------------------ live 3-D window (open/close while running)
    def _headless_from(self, overrides: dict) -> bool:
        """dashboard.headless for a run started with `overrides` (the same config merge the child process does)."""
        from src.simulation.commands import deep_merge
        cfg = deep_merge(self._base_cfg(), overrides)
        return bool((cfg.get("dashboard") or {}).get("headless", True))

    def _auto_open_from(self, overrides: dict) -> bool:
        """dashboard.auto_open_3d for a run started with `overrides`: when true, `start()` opens the live 3-D
        window for the run it just launched (best-effort: a machine without a display gets a warning, not a crash).
        """
        from src.simulation.commands import deep_merge
        cfg = deep_merge(self._base_cfg(), overrides)
        return bool((cfg.get("dashboard") or {}).get("auto_open_3d", False))

    def run_has_its_own_window(self) -> bool:
        """True when the CURRENT run was started with the 3-D PyBullet window itself (dashboard.headless: false).
        Decided from the config the run really uses and cached per run (this is asked on every status poll)."""
        with self._lock:
            return bool(self.alive and not self._run_headless)

    def viewer_status(self) -> dict | None:
        """What the live 3-D window last reported (viewer_status.json in the current run's own directory)."""
        run_dir = self._run_dir
        if run_dir is None:
            return None
        try:
            return read_json_retry(run_dir / "viewer_status.json", attempts=3)
        except OSError:
            return None

    @property
    def viewer_alive(self) -> bool:
        proc = self._viewer
        return proc is not None and proc.poll() is None

    def _viewer_log_tail(self, n: int = 12) -> str:
        if self._run_dir is None:
            return ""
        try:
            lines = (self._run_dir / "viewer.log").read_text(encoding="utf-8", errors="replace").strip().splitlines()
            return "\n".join(lines[-n:])
        except OSError:
            return ""

    def _reap_viewer(self) -> None:
        """Forget a window process that ended on its own (the run finished, or the user closed the window)."""
        with self._lock:
            proc = self._viewer
            if proc is None or proc.poll() is None:
                return
            self._viewer = None
        vs = self.viewer_status() or {}
        if proc.returncode not in (0, None):
            self._viewer_error = self._viewer_error or self._viewer_log_tail() or f"the 3-D window exited with code {proc.returncode}"
        elif vs.get("error"):
            self._viewer_error = vs["error"]

    def window_state(self) -> dict:
        """The real state of the 3-D PyBullet window: nothing, opened by the run itself, or the live viewer."""
        with self._lock:
            proc, run_id = self._viewer, self._run_id
            alive = self.alive
            own = alive and self.run_has_its_own_window()
            holding = alive and bool((self._read("status.json") or {}).get("holding"))
        vs = self.viewer_status() if (proc is not None or (self._run_dir and (self._run_dir / "viewer_status.json").exists())) else None
        viewer_open = bool(proc is not None and proc.poll() is None and (vs or {}).get("state") == "OPEN")
        owner = "run" if own else ("viewer" if viewer_open else None)
        return {"open": bool(own or viewer_open), "owner": owner, "run_id": run_id,
                "viewer_pid": None if proc is None else proc.pid,
                "viewer_alive": bool(proc is not None and proc.poll() is None),
                "state": None if vs is None else vs.get("state"),
                "camera": None if vs is None else vs.get("camera"),
                "error": self._viewer_error or (None if vs is None else vs.get("error")),
                "can_open": bool(alive and not own and not viewer_open),
                # A run that ended while holding its own window open (keep_open_after_run) can be released by
                # closing its window, which is the same as stopping the finished run process.
                "can_close": bool(viewer_open or holding),
                "holding": holding}

    def open_window(self, wait_s: float = 25.0) -> dict:
        """Open a live PyBullet window that SHOWS the running simulation, without stopping or restarting it.

        PyBullet cannot attach a GUI window to a running DIRECT (headless) physics server, so this starts the
        viewer process (src/simulation/live_viewer.py): a real PyBullet window that renders this run's own
        telemetry and world geometry. A run that already opened its own window needs nothing here."""
        with self._lock:
            if not self.alive:
                raise ApiError("start the simulation first: the 3-D window shows a running simulation", 409)
            if self.run_has_its_own_window():
                return self.window_state()                     # this run owns its window already
            vs = self.viewer_status() or {}
            if self.viewer_alive and vs.get("state") == "OPEN":
                return self.window_state()
            run_dir, run_id = self._run_dir, self._run_id
            if run_dir is None:
                raise ApiError("no run directory: start the simulation first", 409)
            spawn = not self.viewer_alive
            if spawn:                                          # a dead handle from an earlier window is discarded
                self._viewer = None
                unlink_retry(run_dir / "viewer_stop.request")   # a close request from an earlier window must not kill this one
                unlink_retry(run_dir / "viewer_status.json")
            self._viewer_error = None
            args = [sys.executable, "-m", self.viewer_module, "--config", str(self.config_path),
                    "--runtime", str(run_dir), "--run-id", str(run_id), "--parent-pid", str(os.getpid())]
            override = run_dir / "config_override.json"
            if override.exists():                              # the SAME world the run was started with
                args += ["--override", str(override)]
            if spawn:                                          # already starting up: never open a second window
                kwargs = {}
                if os.name == "nt":                            # own process group: a Ctrl-C at the dashboard spares it
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                with open(run_dir / "viewer.log", "wb") as log_fh:
                    self._viewer = subprocess.Popen(args, cwd=str(PROJECT_ROOT), stdin=subprocess.DEVNULL,
                                                    stdout=log_fh, stderr=subprocess.STDOUT, **kwargs)
                proc = self._viewer
            self._viewer_started_wall = time.time()
        # Wait OUTSIDE the lock: the dashboard keeps answering polls while the window opens.
        deadline = time.time() + max(wait_s, 1.0)
        while time.time() < deadline:
            vs = self.viewer_status() or {}
            if vs.get("state") == "OPEN":
                return self.window_state()
            if vs.get("state") == "ERROR" or proc.poll() is not None:
                self._viewer_error = vs.get("error") or self._viewer_log_tail() or f"the 3-D window exited with code {proc.returncode}"
                self._kill_viewer()
                raise ApiError(f"the 3-D window could not open: {self._viewer_error}", 502)
            time.sleep(0.1)
        self._viewer_error = f"the 3-D window did not open within {wait_s:.0f} s (see viewer.log in the run directory)"
        self._kill_viewer()
        raise ApiError(self._viewer_error, 504)

    def close_window(self, timeout: float = 8.0) -> dict:
        """Close the live 3-D window. The simulation keeps running untouched. A window the RUN opened itself
        cannot be closed from the dashboard WHILE THE RUN IS GOING (it belongs to the run process and closes when
        the run does); a run that has already FINISHED but is holding that window open on the last frame releases
        it here, which is exactly what Stop does."""
        with self._lock:
            proc, run_dir = self._viewer, self._run_dir
        if (proc is None or proc.poll() is not None) and self.alive \
                and bool((self._read("status.json") or {}).get("holding")):
            self.stop()                                        # the held window belongs to a finished run process
            return self.window_state()
        with self._lock:
            proc, run_dir = self._viewer, self._run_dir
            if proc is None or proc.poll() is not None:
                self._viewer = None
                if self.alive and self.run_has_its_own_window():
                    raise ApiError("this run opened its own PyBullet window at start; the dashboard cannot close it "
                                   "while the run is going (it closes when the run ends). 'off' is saved for the next start.", 409)
                return self.window_state()
            try:
                (run_dir / "viewer_stop.request").write_text(self._run_id or "", encoding="utf-8")
            except OSError as exc:
                raise ApiError(f"could not signal the 3-D window: {exc}", 500)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=3.0)
        with self._lock:
            self._viewer = None
        return self.window_state()

    def _kill_viewer(self, timeout: float = 3.0) -> None:
        with self._lock:
            proc = self._viewer
            self._viewer = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=timeout)

    def _stop_viewer(self) -> None:
        """Ask the live window to close through its OWN file, then wait (never a bare PID kill)."""
        with self._lock:
            proc, run_dir = self._viewer, self._run_dir
            if proc is None:
                return
            if proc.poll() is None and run_dir is not None:
                try:
                    (run_dir / "viewer_stop.request").write_text(self._run_id or "", encoding="utf-8")
                except OSError:
                    pass
        if proc.poll() is None:
            try:
                proc.wait(timeout=6.0)
            except subprocess.TimeoutExpired:
                self._viewer_error = self._viewer_error or "the 3-D window did not close in time; it was terminated"
                self._kill_viewer()
        with self._lock:
            if self._viewer is proc:
                self._viewer = None

    def apply_window_live(self, open_: bool) -> dict:
        """Apply a 3-D window change to a RUNNING simulation immediately. Never raises: the next-start setting is
        already stored, so a display problem is reported next to it instead of failing the save."""
        try:
            if open_:
                self.open_window()
            else:
                self.close_window()
            return self.window_state()
        except ApiError as exc:
            return {**self.window_state(), "error": str(exc)}

    def camera_command(self, action: str, wait_s: float = 3.0) -> dict:
        """Camera controls act on the window that is really open: the live viewer if one is running, otherwise the
        run's own PyBullet window (which answers through the normal command channel)."""
        with self._lock:
            proc, run_dir = self._viewer, self._run_dir
            vs = self.viewer_status() or {}
            live = proc is not None and proc.poll() is None and vs.get("state") == "OPEN"
            cid = None
            if live:
                self._viewer_cmd_id += 1
                cid = self._viewer_cmd_id
                send_command(run_dir / "viewer_commands.jsonl", {"cmd": "set_camera", "action": action, "id": cid})
        if cid is None:
            return self.command({"cmd": "set_camera", "action": action})
        deadline = time.time() + wait_s
        while time.time() < deadline:
            for entry in (self.viewer_status() or {}).get("command_log", []):
                if entry.get("id") == cid:
                    if not entry.get("ok"):
                        raise ApiError(entry.get("detail", "camera command rejected"), 422)
                    return {"id": cid, "ok": True, "detail": entry.get("detail"), "window": "live 3-D window"}
            time.sleep(0.05)
        raise ApiError("the 3-D window did not acknowledge the camera command in time", 504)

    # ------------------------------------------------------------------ pending (next-start) config
    def configure(self, body: dict) -> dict:
        """Validate and store settings that apply on the NEXT start. Real validation: a procedural
        scenario is actually generated (pure NumPy) so impossible configurations are rejected now."""
        if not isinstance(body, dict):
            raise ApiError("body must be a JSON object")
        from src.simulation.world_generator import ALL_SCENARIOS, SCENARIOS, generate_world
        base = self._base_cfg()
        new = json.loads(json.dumps(self.pending))
        env = new.setdefault("environment", {})
        if "scenario" in body:
            if body["scenario"] not in ALL_SCENARIOS:
                raise ApiError(f"unknown scenario {body['scenario']!r}; choose from {list(ALL_SCENARIOS)}")
            env["scenario"] = body["scenario"]
        if "seed" in body:
            try:
                seed = int(body["seed"])
            except (TypeError, ValueError):
                raise ApiError("seed must be an integer")
            if not 0 <= seed < 2 ** 31:
                raise ApiError("seed must be in [0, 2^31)")
            env["seed"] = seed
        if "obstacle_count" in body and body["obstacle_count"] not in (None, ""):
            try:
                n = int(body["obstacle_count"])
            except (TypeError, ValueError):
                raise ApiError("obstacle_count must be an integer")
            if not 0 <= n <= 200:
                raise ApiError("obstacle_count must be in [0, 200]")
            env.setdefault("obstacles", {})["count"] = n
            scenario = env.get("scenario") or (base.get("environment") or {}).get("scenario")
            if scenario not in SCENARIOS:                    # legacy_week4 builds its world from the Week-4 map
                raise ApiError(f"obstacle_count has no effect on scenario {scenario!r}: that world is built from the "
                               f"Week-4 occupancy map. Choose a procedural scenario first ({', '.join(SCENARIOS)}).", 422)
        elif "obstacle_count" in body:
            env.get("obstacles", {}).pop("count", None)
        if "dynamics_model" in body:
            if body["dynamics_model"] not in ("legacy", "6dof"):
                raise ApiError("dynamics_model must be 'legacy' or '6dof'")
            new.setdefault("simulation", {}).setdefault("dynamics", {})["model"] = body["dynamics_model"]
        if "mapping_enabled" in body:
            if not isinstance(body["mapping_enabled"], bool):
                raise ApiError("mapping_enabled must be true or false")
            new.setdefault("mapping", {})["enabled"] = body["mapping_enabled"]
        if "navigation_enabled" in body:
            if not isinstance(body["navigation_enabled"], bool):
                raise ApiError("navigation_enabled must be true or false")
            new.setdefault("navigation", {})["enabled"] = body["navigation_enabled"]
        if "navigation_goal" in body:
            # The goal input is stored as navigation.goal_position (the one key the simulation resolves the mission
            # goal from - see src.simulation.commands.resolve_mission_goal), so it applies with navigation ON and
            # OFF. simulation.goal_position / simulation.waypoints are deliberately NOT touched: the world
            # generator keeps the mission path (start -> waypoints -> goal) clear of obstacles, so writing them
            # here would silently regenerate a different world than the configured mission describes.
            #
            # The z field is a DEPTH below the surface, positive down, exactly as the operator reads it next to the
            # "depth (m)" row of the Estimated navigation panel; it is converted to world z once, here
            # (src.simulation.commands.operator_xyz_to_world). Typing the goal as 18.5 / 13.5 / 7 therefore means
            # 7 m deep, and it is accepted instead of being rejected as "above water".
            g = body["navigation_goal"]
            if g in (None, ""):
                new.setdefault("navigation", {})["goal_position"] = None   # blank = the config goal again
            else:
                try:
                    g = [float(v) for v in g]
                    if len(g) != 3 or not all(v == v and abs(v) < 1e6 for v in g):
                        raise ValueError
                except (TypeError, ValueError):
                    raise ApiError("navigation_goal must be [x, y, depth] numbers")
                new.setdefault("navigation", {})["goal_position"] = _operator_xyz_to_world(g)
        if "start_position" in body:
            s = body["start_position"]
            if s in (None, ""):
                new.setdefault("simulation", {}).pop("start_position", None)   # blank = use the config value
            else:
                try:
                    s = [float(v) for v in s]
                    if len(s) != 3 or not all(v == v and abs(v) < 1e6 for v in s):
                        raise ValueError
                except (TypeError, ValueError):
                    raise ApiError("start_position must be [x, y, depth] numbers")
                new.setdefault("simulation", {})["start_position"] = _operator_xyz_to_world(s)
        if "unknown_policy" in body:
            if body["unknown_policy"] not in ("inherit", "allow", "block"):
                raise ApiError("unknown_policy must be inherit, allow or block")
            new.setdefault("planning", {})["unknown_policy"] = body["unknown_policy"]
        if "show_3d" in body:
            if not isinstance(body["show_3d"], bool):
                raise ApiError("show_3d must be true or false")
            new.setdefault("dashboard", {})["headless"] = not body["show_3d"]       # GUI window <=> not headless
        if "auto_open_3d" in body:
            # Auto-open the 3-D view whenever a simulation is started (the run's own window, or the live viewer
            # for a headless run). Stored for the next start; harmless when no display is available.
            if not isinstance(body["auto_open_3d"], bool):
                raise ApiError("auto_open_3d must be true or false")
            new.setdefault("dashboard", {})["auto_open_3d"] = body["auto_open_3d"]
        if "realtime_factor" in body:
            try:
                rf = float(body["realtime_factor"])
            except (TypeError, ValueError):
                raise ApiError("realtime_factor must be a number")
            if not 0.0 <= rf < 1e4:
                raise ApiError("realtime_factor must be in [0, 10000)")
            new.setdefault("dashboard", {})["realtime_factor"] = rf
        if (new.get("navigation") or {}).get("enabled"):
            from src.planning.navigator import navigation_prerequisites
            from src.simulation.commands import deep_merge as _dm
            why = navigation_prerequisites(_dm(base, new))
            if why:
                raise ApiError(f"navigation cannot be enabled: {why}", 422)
        # validate the resulting environment by generating it
        from src.simulation.commands import deep_merge
        merged_env = deep_merge(base.get("environment") or {}, new.get("environment", {}))   # same merge the child uses
        summary = None
        if merged_env.get("scenario", "legacy_week4") in SCENARIOS:
            try:
                spec = generate_world(merged_env, base["simulation"],
                                      (int(merged_env["grid_size"]["rows"]), int(merged_env["grid_size"]["cols"]),
                                       float(merged_env["resolution"])))
            except (ValueError, KeyError) as exc:
                raise ApiError(f"environment configuration rejected: {exc}", 422)
            summary = spec.summary()
        with self._lock:
            self.pending = new
        out = {"pending_overrides": self.pending, "applies": "on next simulation start",
               "preview_ground_truth_summary": summary, "running": self.alive}
        if "show_3d" in body and self.alive:
            # Changing the 3-D window control while a run is going really opens/closes the window NOW (the run
            # itself is untouched); the next-start value above is stored either way.
            out["window"] = self.apply_window_live(bool(body["show_3d"]))
        return out

    def scene(self) -> dict | None:
        return self._read("scene.json")

    def maps(self) -> dict | None:
        return self._read("maps.json")

    def shutdown(self) -> None:
        self._stop_viewer()
        if self.alive:
            try:
                self.stop(timeout=3.0)
            except ApiError:
                pass

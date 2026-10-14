"""Runtime-state / process-lifecycle robustness (written for the Windows failures; they run on every OS).

Windows facts these tests encode:
  * os.replace() onto a file another process has open -> PermissionError (WinError 5/32/33)
  * os.kill(pid, 0) TERMINATES the process on Windows, so it must never be used as a liveness probe
  * a file cannot be unlinked while any handle to it is open
Where the OS cannot be reproduced here, the exact call site is exercised with injected errors instead.
"""
import ast
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

import src.simulation.live_io as live_io
from dashboard.simulation_manager import ApiError, SimulationManager
from src.simulation.live_io import (IpcWriteError, LivePublisher, atomic_write_json, is_transient_io_error,
                                    read_json_retry, retry_io, unlink_retry)
from src.simulation.run_protocol import ChildRun, live_child_pids
from src.utils import procutil
from src.utils.procutil import pid_alive
from tests.helpers.proc_signals import terminate_pid

ROOT = Path(__file__).resolve().parent.parent
STANDIN = "tests.helpers.standin_sim"


class WinErr(OSError):
    """OSError carrying a Windows error code (what a real Windows failure looks like)."""
    def __init__(self, code):
        super().__init__(f"[WinError {code}]")
        self.winerror = code


def wait_for(cond, timeout=40.0, step=0.05):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting for condition")


# ------------------------------------------------------------------------------ retry primitives
def test_transient_error_classification():
    assert is_transient_io_error(PermissionError(13, "denied"))
    for code in (5, 32, 33):
        assert is_transient_io_error(WinErr(code))
    assert not is_transient_io_error(WinErr(2))
    assert not is_transient_io_error(FileNotFoundError(2, "no"))
    assert not is_transient_io_error(ValueError("x"))


def test_retry_io_retries_transient_errors_then_succeeds_and_backs_off():
    calls, sleeps = [], []

    def flaky():
        calls.append(1)
        if len(calls) < 4:
            raise PermissionError(13, "denied")
        return "ok"
    assert retry_io(flaky, attempts=10, sleep=sleeps.append) == "ok"
    assert len(calls) == 4 and len(sleeps) == 3
    assert sleeps == sorted(sleeps) and sleeps[-1] > sleeps[0]                   # exponential backoff


def test_retry_io_propagates_non_transient_errors_immediately():
    sleeps = []

    def missing():
        raise FileNotFoundError(2, "gone")
    with pytest.raises(FileNotFoundError):
        retry_io(missing, attempts=10, sleep=sleeps.append)
    assert sleeps == []


def test_retry_io_gives_up_after_the_attempt_budget_with_the_last_error():
    n = []

    def locked():
        n.append(1)
        raise WinErr(32)
    with pytest.raises(OSError) as ei:
        retry_io(locked, attempts=5, sleep=lambda s: None)
    assert len(n) == 5 and getattr(ei.value, "winerror", None) == 32


# ------------------------------------------------------------------------------ atomic JSON writes
def flaky_replace(monkeypatch, fail_times, exc_factory=lambda: PermissionError(13, "[WinError 5] Access is denied")):
    real, state = os.replace, {"n": 0}

    def replace(src, dst):
        if state["n"] < fail_times:
            state["n"] += 1
            raise exc_factory()
        return real(src, dst)
    monkeypatch.setattr(live_io.os, "replace", replace)
    monkeypatch.setattr(live_io.time, "sleep", lambda s: None)
    return state


def test_atomic_write_survives_replace_being_denied_like_windows(tmp_path, monkeypatch):
    """THE reported failure: os.replace(tmp, state.json) -> PermissionError [WinError 5]."""
    target = tmp_path / "state.json"
    atomic_write_json(target, {"v": 0})
    state = flaky_replace(monkeypatch, fail_times=7)
    atomic_write_json(target, {"v": 1, "x": [1, 2]})
    assert state["n"] == 7
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": 1, "x": [1, 2]}
    assert not list(tmp_path.glob("*.tmp"))                                       # no temp-file litter


def test_atomic_write_survives_sharing_violation_winerror_32(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    flaky_replace(monkeypatch, 5, exc_factory=lambda: WinErr(32))
    atomic_write_json(target, {"ok": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"ok": True}


def test_atomic_write_failure_is_reported_leaves_old_file_and_cleans_up(tmp_path, monkeypatch):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"v": "old"})
    flaky_replace(monkeypatch, fail_times=10 ** 6)
    with pytest.raises(IpcWriteError):
        atomic_write_json(target, {"v": "new"}, attempts=4)
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": "old"}          # never half-written
    assert not list(tmp_path.glob("*.tmp"))


def test_temp_names_are_unique_per_call(tmp_path):
    names = set()
    real = Path.write_text

    def spy(self, *a, **k):
        names.add(self.name)
        return real(self, *a, **k)
    try:
        Path.write_text = spy
        for i in range(5):
            atomic_write_json(tmp_path / "state.json", {"i": i})
    finally:
        Path.write_text = real
    assert len(names) == 5


def test_non_finite_values_are_written_as_null_never_as_nan(tmp_path):
    atomic_write_json(tmp_path / "a.json", {"x": float("nan"), "y": float("inf"), "z": [float("-inf"), 1.5]})
    text = (tmp_path / "a.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text) == {"x": None, "y": None, "z": [None, 1.5]}


def test_publisher_failures_never_raise_are_counted_and_recover(tmp_path, monkeypatch):
    pub = LivePublisher(tmp_path, state_hz=1000, map_hz=1000)
    flaky_replace(monkeypatch, fail_times=10 ** 6)
    monkeypatch.setattr(live_io, "retry_io", lambda fn, attempts=60, **k: fn())   # fail fast for this test
    assert pub.publish_state({"wall_time": 1.0}) is False
    assert pub.publish_maps({"a": 1}, version=3) is False
    assert pub.failures == 2 and pub.consecutive_failures == 2 and pub.last_error.startswith("IpcWriteError") and "Access is denied" in pub.last_error
    t0 = time.monotonic()
    while time.monotonic() - t0 < 0.01:                                             # (time.sleep is patched out here)
        pass
    assert pub.due_maps(3) is True                                                  # a failed map publish stays due (retried)
    monkeypatch.undo()
    assert pub.publish_state({"wall_time": 2.0}) is True
    first = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert first["ipc"]["consecutive_failures"] == 2 and first["ipc"]["failures"] == 2   # reports what happened up to this write
    assert pub.publish_state({"wall_time": 3.0}) is True
    second = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert second["ipc"]["consecutive_failures"] == 0 and second["ipc"]["published"] == 1  # recovery is visible next


def test_unlink_retry_retries_then_reports_truthfully(tmp_path, monkeypatch):
    f = tmp_path / "x"
    f.write_text("a", encoding="utf-8")
    real, n = Path.unlink, {"n": 0}

    def flaky(self, *a, **k):
        if n["n"] < 3:
            n["n"] += 1
            raise PermissionError(13, "denied")
        return real(self, *a, **k)
    monkeypatch.setattr(Path, "unlink", flaky)
    monkeypatch.setattr(live_io.time, "sleep", lambda s: None)
    assert unlink_retry(f) is True and not f.exists()
    monkeypatch.setattr(Path, "unlink", real)                                         # undo the flaky patch
    assert unlink_retry(tmp_path / "never_existed") is True                          # nothing to delete == success
    g = tmp_path / "y"
    g.write_text("a", encoding="utf-8")
    monkeypatch.setattr(Path, "unlink", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError(13, "locked")))
    assert unlink_retry(g, attempts=3) is False and g.exists()                       # truthful: still there


def test_reader_and_writer_racing_never_see_a_partial_or_invalid_file(tmp_path):
    target = tmp_path / "state.json"
    atomic_write_json(target, {"n": 0, "pad": "x" * 20000})
    stop, errors, seen = threading.Event(), [], []

    def reader():
        last = -1
        while not stop.is_set():
            try:
                d = read_json_retry(target)
            except Exception as exc:                        # noqa: BLE001
                errors.append(repr(exc))
                return
            if d is None:
                errors.append("file vanished")
                return
            if d["n"] < last or len(d["pad"]) != 20000:
                errors.append(f"went backwards/partial: {d['n']} < {last}")
                return
            last = d["n"]
        seen.append(last)

    readers = [threading.Thread(target=reader) for _ in range(4)]
    for t in readers:
        t.start()
    for i in range(1, 400):
        atomic_write_json(target, {"n": i, "pad": "x" * 20000})
    stop.set()
    for t in readers:
        t.join(10)
    assert errors == [] and len(seen) == 4 and max(seen) > 0


# ------------------------------------------------------------------------------ process liveness
def test_pid_alive_reports_real_process_state_and_never_alters_it():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        for _ in range(25):                                   # on Windows os.kill(pid, 0) would have killed it
            assert pid_alive(child.pid)
        assert child.poll() is None                           # still running after 25 probes
    finally:
        child.kill()
        child.wait(10)
    assert not pid_alive(child.pid)
    done = subprocess.Popen([sys.executable, "-c", "pass"])
    done.wait(30)
    assert not pid_alive(done.pid)
    assert pid_alive(os.getpid()) and not pid_alive(None) and not pid_alive(0) and not pid_alive(-4) and not pid_alive("x")


class FakeK32:
    def __init__(self, handle, last_error=0, exit_code=None):
        self.handle, self.err, self.code, self.closed = handle, last_error, exit_code, 0

    def OpenProcess(self, access, inherit, pid):
        self.access = access
        return self.handle

    def get_last_error(self):
        return self.err

    def get_exit_code(self, h):
        return self.code

    def CloseHandle(self, h):
        self.closed += 1


def test_windows_liveness_logic_with_a_fake_kernel32():
    f = FakeK32(handle=7, exit_code=procutil.STILL_ACTIVE)
    assert procutil._pid_alive_windows(123, f) is True and f.closed == 1
    assert f.access == procutil.PROCESS_QUERY_LIMITED_INFORMATION                  # least-privilege query, no terminate right
    f = FakeK32(handle=7, exit_code=0)
    assert procutil._pid_alive_windows(123, f) is False and f.closed == 1          # exited: handle closed, not alive
    f = FakeK32(handle=7, exit_code=None)
    assert procutil._pid_alive_windows(123, f) is False and f.closed == 1
    assert procutil._pid_alive_windows(123, FakeK32(handle=0, last_error=5)) is True       # access denied => exists
    assert procutil._pid_alive_windows(123, FakeK32(handle=0, last_error=87)) is False     # invalid parameter => gone


def test_os_kill_is_only_used_where_it_is_safe():
    """Regression guard for the Windows bug: os.kill(pid, 0) kills the process on Windows."""
    allowed = {"src/utils/procutil.py", "tests/helpers/proc_signals.py"}
    offenders = []
    for base in ("src", "dashboard", "tests"):
        for p in (ROOT / base).rglob("*.py"):
            rel = p.relative_to(ROOT).as_posix()
            if rel in allowed or rel == "tests/test_windows_runtime.py":
                continue
            for node in ast.walk(ast.parse(p.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "kill" \
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "os":
                    offenders.append(f"{rel}:{node.lineno}")
    assert offenders == []


# ------------------------------------------------------------------------------ child protocol
def test_stop_request_for_another_run_is_ignored_and_own_or_empty_is_honoured(tmp_path):
    c = ChildRun(tmp_path, "run-A", poll_s=0.01)
    (tmp_path / "stop.request").write_text("run-OLD", encoding="utf-8")           # stale file from another run
    assert c._stop_requested() is False
    (tmp_path / "stop.request").write_text("run-A", encoding="utf-8")
    assert c._stop_requested() is True
    (tmp_path / "stop.request").write_text("", encoding="utf-8")
    assert c._stop_requested() is True
    (tmp_path / "stop.request").unlink()
    assert c._stop_requested() is False


def test_child_stops_when_its_parent_process_is_gone(tmp_path):
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    c = ChildRun(tmp_path, "run-B", parent_pid=parent.pid, poll_s=0.02)
    ev = c.start_watchers()
    assert not ev.wait(0.3)
    parent.kill()
    parent.wait(10)
    assert ev.wait(5) and c.stop_reason == "dashboard process exited"
    c.finish()


def test_child_json_identifies_the_process(tmp_path):
    ChildRun(tmp_path, "run-C")
    info = json.loads((tmp_path / "child.json").read_text(encoding="utf-8"))
    assert info["pid"] == os.getpid() and info["run_id"] == "run-C"
    assert [pid for _, pid in live_child_pids(tmp_path.parent) if _ == tmp_path] == [os.getpid()]


# ------------------------------------------------------------------------------ manager lifecycle (real processes)
@pytest.fixture()
def mgr(tmp_path):
    m = SimulationManager(runtime_dir=tmp_path / "rt", child_module=STANDIN, telemetry_timeout_s=3.0)
    m.configure({"realtime_factor": 1.0, "scenario": "mapping_validation"})
    yield m
    m.shutdown()


def state_of(m):
    return m.status()["state"]


def test_returned_pid_stays_alive_while_state_is_polled_hard(mgr):
    """The reported failure: the PID returned by start must remain alive while the dashboard reads the
    runtime files continuously (on Windows those reads raced the child's os.replace and killed it)."""
    st = mgr.start()
    pid = st["pid"]
    assert pid and pid_alive(pid)
    stop, problems = threading.Event(), []

    def hammer():
        while not stop.is_set():
            try:
                mgr.status(); mgr.state_snapshot(); mgr.maps()
            except Exception as exc:                          # noqa: BLE001
                problems.append(repr(exc))
                return
    threads = [threading.Thread(target=hammer) for _ in range(6)]
    for t in threads:
        t.start()
    try:
        wait_for(lambda: state_of(mgr) == "RUNNING")
        t0 = time.time()
        while time.time() - t0 < 5.0:
            assert pid_alive(pid) and mgr.alive, mgr.status()
            assert state_of(mgr) in ("RUNNING", "STARTING")
            time.sleep(0.05)
    finally:
        stop.set()
        for t in threads:
            t.join(10)
    st = mgr.status()
    assert problems == [] and st["state"] == "RUNNING" and st["pid"] == pid and pid_alive(pid)
    assert st["ipc"]["consecutive_failures"] == 0
    assert st["error"] is None


def test_concurrent_start_requests_start_exactly_one_process(mgr):
    results, barrier = [], threading.Barrier(8)

    def go():
        barrier.wait()
        try:
            results.append(("ok", mgr.start()["pid"]))
        except ApiError as e:
            results.append(("err", e.status))
    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    ok = [r for r in results if r[0] == "ok"]
    assert len(ok) == 1 and [r for r in results if r[0] == "err"] == [("err", 409)] * 7
    wait_for(lambda: state_of(mgr) == "RUNNING")
    assert len(live_child_pids(mgr.runs_root)) == 1                  # one run directory, one live process
    assert len(list(mgr.runs_root.iterdir())) == 1


def test_repeated_start_stop_reset_cycles_are_reliable(mgr):
    pids, dirs = [], []
    for i in range(3):
        st = mgr.start()
        pids.append(st["pid"]); dirs.append(mgr.run_dir)
        wait_for(lambda: state_of(mgr) == "RUNNING")
        assert pid_alive(st["pid"])
        if i % 2 == 0:
            out = mgr.stop()
            assert out["state"] == "STOPPED" and out["returncode"] == 0 and not pid_alive(st["pid"])
            assert mgr.reset()["state"] == "IDLE"
        else:
            out = mgr.reset()                                 # reset straight from RUNNING
            assert out["state"] == "IDLE" and out["was_running"] is True and not pid_alive(st["pid"])
        assert mgr.run_dir is None and not dirs[-1].exists()
    assert len(set(pids)) == 3 and len(set(dirs)) == 3
    assert live_child_pids(mgr.runs_root) == []


def test_reset_immediately_after_start_does_not_hang_or_leave_a_process(mgr):
    st = mgr.start()
    t0 = time.time()
    out = mgr.reset()
    assert time.time() - t0 < 20 and out["state"] == "IDLE"
    wait_for(lambda: not pid_alive(st["pid"]), 10)
    assert live_child_pids(mgr.runs_root) == []


def test_each_run_gets_a_private_directory_and_stale_files_cannot_leak_in(mgr):
    old = mgr.runs_root / "20000101-000000-stale1"
    old.mkdir(parents=True)
    atomic_write_json(old / "state.json", {"wall_time": time.time(), "sim_time": 9999.0, "mapping": {"update_count": 9999}})
    atomic_write_json(old / "status.json", {"status": "RUNNING"})
    (old / "stop.request").write_text("", encoding="utf-8")
    st = mgr.start()
    assert mgr.run_dir.parent == mgr.runs_root and mgr.run_dir != old
    wait_for(lambda: state_of(mgr) == "RUNNING")                  # not stopped by the stale stop.request
    snap = mgr.state_snapshot()
    assert snap["sim_time"] < 60 and snap["mapping"]["update_count"] < 9999   # not the stale snapshot
    assert pid_alive(st["pid"])


@pytest.mark.skipif(os.name == "nt", reason="uses /proc/self/fd (POSIX); the Windows hazard it models is closed by design")
def test_manager_keeps_no_handle_open_on_the_child_log(mgr):
    mgr.start()
    wait_for(lambda: state_of(mgr) == "RUNNING")
    log = str((mgr.run_dir / "simulation.log").resolve())
    fds = [os.path.realpath(f"/proc/self/fd/{fd}") for fd in os.listdir("/proc/self/fd")]
    assert log not in fds        # on Windows an open handle makes the log (and its directory) undeletable
    assert mgr.reset()["state"] == "IDLE" and not Path(log).exists()


def test_orphaned_child_from_an_earlier_dashboard_is_stopped_on_manager_start(tmp_path):
    root = tmp_path / "rt"
    run_dir = root / "runs" / "20000101-000000-orph01"
    cmd = [sys.executable, "-m", STANDIN, "--config", str(ROOT / "configs" / "config.yaml"), "--runtime", str(run_dir),
           "--run-id", "orph01"]
    orphan = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: (run_dir / "state.json").exists(), 60)
        assert pid_alive(orphan.pid)
        m = SimulationManager(runtime_dir=root, child_module=STANDIN)       # a "restarted dashboard"
        wait_for(lambda: orphan.poll() is not None, 10)
        assert not pid_alive(orphan.pid) and m.warnings == []
        assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "STOPPED"
    finally:
        if orphan.poll() is None:
            orphan.kill()
        orphan.wait(10)


def test_child_exits_on_its_own_when_the_dashboard_process_dies(tmp_path):
    parent = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    run_dir = tmp_path / "rt" / "runs" / "r1"
    cmd = [sys.executable, "-m", STANDIN, "--config", str(ROOT / "configs" / "config.yaml"), "--runtime", str(run_dir),
           "--run-id", "r1", "--parent-pid", str(parent.pid)]
    child = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(lambda: (run_dir / "state.json").exists(), 60)
        parent.kill()
        parent.wait(10)
        assert child.wait(30) == 0
        assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "STOPPED"
    finally:
        for pr in (parent, child):
            if pr.poll() is None:
                pr.kill()
                pr.wait(10)


def test_start_works_again_after_the_child_was_killed_from_outside(mgr):
    st = mgr.start()
    wait_for(lambda: state_of(mgr) == "RUNNING")
    terminate_pid(st["pid"])
    wait_for(lambda: state_of(mgr) == "FAILED", 15)
    assert not mgr.status()["alive"]
    st2 = mgr.start()
    assert st2["pid"] != st["pid"]
    wait_for(lambda: state_of(mgr) == "RUNNING")
    assert pid_alive(st2["pid"])


def test_rmtree_retry_survives_transient_denials_and_reports_truthfully(tmp_path, monkeypatch):
    d = tmp_path / "run"
    (d / "sub").mkdir(parents=True)
    (d / "sub" / "state.json").write_text("{}", encoding="utf-8")
    real, n = live_io.shutil.rmtree, {"n": 0}

    def flaky(path, *a, **k):
        if n["n"] < 4:
            n["n"] += 1
            raise PermissionError(13, "[WinError 32] being used by another process")
        return real(path, *a, **k)
    monkeypatch.setattr(live_io.shutil, "rmtree", flaky)
    monkeypatch.setattr(live_io.time, "sleep", lambda s: None)
    assert live_io.rmtree_retry(d) is True and not d.exists() and n["n"] == 4
    assert live_io.rmtree_retry(tmp_path / "never_existed") is True
    e = tmp_path / "locked"
    e.mkdir()
    monkeypatch.setattr(live_io.shutil, "rmtree", lambda *a, **k: (_ for _ in ()).throw(PermissionError(13, "locked")))
    assert live_io.rmtree_retry(e, attempts=3) is False and e.exists()                # truthful when it truly cannot delete


# ------------------------------------------------------------------ path lock (the Windows rename-vs-open-reader fix)
class FakeMutexK32:
    """In-memory stand-in for the mutex half of kernel32 (one mutex per name, plain threading.Lock)."""
    locks: dict = {}

    def CreateMutexW(self, attrs, owned, name):
        self.locks.setdefault(name, threading.Lock())
        return name

    def WaitForSingleObject(self, handle, ms):
        return 0 if self.locks[handle].acquire(timeout=ms / 1000.0) else live_io.WAIT_TIMEOUT

    def ReleaseMutex(self, handle):
        self.locks[handle].release()
        return 1


def _with_fake_lock(monkeypatch):
    FakeMutexK32.locks = {}
    monkeypatch.setattr(live_io, "_lock_enabled", lambda: True)
    monkeypatch.setattr(live_io, "_mutexes", {})
    monkeypatch.setattr(live_io, "_mutex_factory", lambda name: live_io._WinMutex(name, FakeMutexK32()))


def test_replace_and_reads_are_mutually_exclusive_under_the_path_lock(tmp_path, monkeypatch):
    """With the lock on, no reader ever has the file open while os.replace runs (the Windows WinError 5 cause)."""
    _with_fake_lock(monkeypatch)
    target = tmp_path / "state.json"
    atomic_write_json(target, {"n": 0})
    open_readers, overlaps = [0], []
    real_replace, real_read = live_io.os.replace, live_io.read_bytes_shared

    def guarded_replace(a, b):
        if open_readers[0]:
            overlaps.append(open_readers[0])
        return real_replace(a, b)

    def guarded_read(path):
        open_readers[0] += 1
        try:
            time.sleep(0.0005)
            return real_read(path)
        finally:
            open_readers[0] -= 1

    monkeypatch.setattr(live_io.os, "replace", guarded_replace)
    monkeypatch.setattr(live_io, "read_bytes_shared", guarded_read)
    stop, errors = threading.Event(), []

    def reader():
        while not stop.is_set():
            try:
                read_json_retry(target)
            except Exception as exc:                    # noqa: BLE001
                errors.append(repr(exc)); return

    ts = [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in ts]
    for i in range(1, 120):
        atomic_write_json(target, {"n": i})
    stop.set(); [t.join(10) for t in ts]
    assert errors == [] and overlaps == []
    assert json.loads(target.read_text(encoding="utf-8")) == {"n": 119}


def test_lock_timeout_is_a_transient_error_then_a_truthful_ipc_write_error(tmp_path, monkeypatch):
    _with_fake_lock(monkeypatch)
    monkeypatch.setattr(live_io, "LOCK_WAIT_S", 0.01)
    monkeypatch.setattr(live_io.time, "sleep", lambda s: None)
    target = tmp_path / "state.json"
    atomic_write_json(target, {"v": "old"})
    held = live_io._mutex_for(target)
    held.acquire(1.0)                                   # somebody keeps the lock
    try:
        with pytest.raises(IpcWriteError):
            atomic_write_json(target, {"v": "new"}, attempts=3)
    finally:
        held.release()
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": "old"}        # never half-written
    assert not list(tmp_path.glob("*.tmp"))                                        # temp file cleaned up
    atomic_write_json(target, {"v": "new"})                                         # and it recovers
    assert json.loads(target.read_text(encoding="utf-8")) == {"v": "new"}


def test_abandoned_mutex_is_acquired_not_treated_as_an_error():
    class K(FakeMutexK32):
        def WaitForSingleObject(self, handle, ms):
            return live_io.WAIT_ABANDONED
    m = live_io._WinMutex("Local\\x", K())
    m.acquire(0.1)                                      # must not raise
    with pytest.raises(OSError):
        class Bad(FakeMutexK32):
            def WaitForSingleObject(self, handle, ms):
                return 0xFFFFFFFF
        live_io._WinMutex("Local\\y", Bad()).acquire(0.1)


READER_PROC = r"""
import sys, time
sys.path.insert(0, sys.argv[1])
from pathlib import Path
from src.simulation.live_io import read_json_retry
target, seconds = Path(sys.argv[2]), float(sys.argv[3])
end, last, n = time.time() + seconds, -1, 0
while time.time() < end:
    d = read_json_retry(target)
    if d is None or len(d["pad"]) != 20000 or d["n"] < last:
        print("BAD", d and d["n"], last); sys.exit(1)
    last, n = d["n"], n + 1
print("OK", n, last)
"""


def test_separate_reader_processes_and_a_writer_never_conflict(tmp_path):
    """The real production topology: Flask (other process) polls state.json while the simulation writes it."""
    target = tmp_path / "state.json"
    atomic_write_json(target, {"n": 0, "pad": "x" * 20000})
    root = str(Path(__file__).resolve().parents[1])
    procs = [subprocess.Popen([sys.executable, "-c", READER_PROC, root, str(target), "4"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for _ in range(3)]
    try:
        end = time.time() + 3.5
        i = 0
        while time.time() < end:
            i += 1
            atomic_write_json(target, {"n": i, "pad": "x" * 20000})          # raises IpcWriteError on any failure
    finally:
        outs = [p.communicate(timeout=60) for p in procs]
    for p, (out, err) in zip(procs, outs):
        assert p.returncode == 0 and out.startswith("OK"), (out, err)
    assert i > 20

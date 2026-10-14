"""File-based IPC between the simulation child process and the dashboard (Flask) process.

  child  -> parent : state.json (throttled snapshot), maps.json (throttled map payloads), status.json
  parent -> child  : commands.jsonl (append-only, each line one JSON command with an integer id),
                     stop.request (empty file => stop)
Snapshots are written to a unique temp file and os.replace()d, so a reader never sees a partial file.
WINDOWS: os.replace() onto a file that another process currently has open fails with PermissionError
(WinError 5/32/33) because Python opens files without FILE_SHARE_DELETE, and the dashboard polls these
files constantly. Every replace/unlink/read here therefore goes through retry_io() (short exponential
backoff). A publish that still fails is counted and reported, never allowed to kill the simulation. No
PyBullet / numpy-object leakage: everything passes through jsonsafe() and is written with
allow_nan=False, so NaN/Infinity can never reach a browser.
"""
from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import shutil
import threading
import time
from pathlib import Path

from src.utils.jsonsafe import jsonsafe


# Windows sharing/locking errors that are transient: ACCESS_DENIED, SHARING_VIOLATION, LOCK_VIOLATION.
TRANSIENT_WINERRORS = (5, 32, 33)
_tmp_counter = itertools.count()


class IpcWriteError(OSError):
    """A runtime file could not be written/replaced even after all retries."""


def is_transient_io_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionError):
        return True
    return isinstance(exc, OSError) and getattr(exc, "winerror", None) in TRANSIENT_WINERRORS


def retry_io(fn, attempts: int = 60, first_delay: float = 0.004, max_delay: float = 0.1, sleep=time.sleep):
    """Run fn(); on a transient sharing/permission error back off and retry (about 5 s in total with the
    defaults). Non-transient errors (e.g. FileNotFoundError) propagate immediately and unchanged."""
    delay, last = first_delay, None
    for _ in range(max(1, attempts)):
        try:
            return fn()
        except OSError as exc:
            if not is_transient_io_error(exc):
                raise
            last = exc
            sleep(delay)
            delay = min(delay * 1.6, max_delay)
    raise last


# ------------------------------------------------------------------------------------------ path lock
# WHY A LOCK. On Windows a rename-over (os.replace / MoveFileEx) of a file that ANY other handle has open fails
# with WinError 5 "Access is denied" -- even when that handle was opened with FILE_SHARE_DELETE, because the
# destination goes delete-pending and cannot be replaced until the handle closes. A reader that polls in a loop
# (four threads in the test, the Flask dashboard in production) keeps the file open nearly continuously, so no
# amount of retrying or share-flag tuning reliably lets the writer in.
# The fix is to make "a handle is open on the file" and "the file is being replaced" mutually exclusive:
# readers and the writer take a per-path, CROSS-PROCESS Windows named mutex. A reader holds it only while it
# opens, reads fully into memory and closes the file (a few hundred microseconds); the writer holds it only for
# the os.replace. Windows grants a released mutex to the first waiter (FIFO), so neither side can starve.
# POSIX needs no lock (rename-over is atomic there and open readers keep their old inode).
# A wait that times out raises PermissionError, which retry_io treats as transient, and after the retry budget
# atomic_write_json raises IpcWriteError: nothing is ever swallowed.
WAIT_OBJECT_0, WAIT_ABANDONED, WAIT_TIMEOUT = 0x0, 0x80, 0x102
LOCK_WAIT_S = 1.0


class _WinMutex:
    """A named Win32 mutex (kernel32 injectable so the logic is unit-testable on any OS)."""

    def __init__(self, name: str, k32=None):
        self.name = name
        self.k32 = k32 if k32 is not None else self._real_kernel32()
        self.handle = self.k32.CreateMutexW(None, False, name)
        if not self.handle:
            raise OSError(f"CreateMutexW({name!r}) failed")

    @staticmethod
    def _real_kernel32():
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        k32.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
        k32.CreateMutexW.restype = wintypes.HANDLE
        k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        k32.WaitForSingleObject.restype = wintypes.DWORD
        k32.ReleaseMutex.argtypes = [wintypes.HANDLE]
        k32.ReleaseMutex.restype = wintypes.BOOL
        return k32

    def acquire(self, timeout_s: float) -> None:
        rc = self.k32.WaitForSingleObject(self.handle, int(timeout_s * 1000))
        if rc in (WAIT_OBJECT_0, WAIT_ABANDONED):      # ABANDONED: a holder died; we now own it and the data is intact
            return
        if rc == WAIT_TIMEOUT:
            raise PermissionError(13, f"timed out after {timeout_s:.1f}s waiting for the IPC lock {self.name}")
        raise OSError(f"WaitForSingleObject({self.name}) failed with {rc:#x}")

    def release(self) -> None:
        self.k32.ReleaseMutex(self.handle)


_mutexes: dict = {}
_mutexes_guard = threading.Lock()
_mutex_factory = _WinMutex            # tests substitute a fake; None-able via _lock_enabled()


def _lock_enabled() -> bool:
    return os.name == "nt"


def _mutex_for(path) -> _WinMutex:
    key = os.path.normcase(os.path.abspath(str(path)))
    with _mutexes_guard:
        m = _mutexes.get(key)
        if m is None:
            m = _mutexes[key] = _mutex_factory("Local\\aurora_ipc_" + hashlib.sha1(key.encode("utf-8")).hexdigest())
        return m


@contextlib.contextmanager
def path_lock(path, timeout_s: float = LOCK_WAIT_S):
    """Mutual exclusion between readers and the replacing writer of `path`, across threads AND processes."""
    if not _lock_enabled():
        yield
        return
    m = _mutex_for(path)
    m.acquire(timeout_s)
    try:
        yield
    finally:
        m.release()


def _open_shared_windows(path):
    """Open `path` for reading on Windows WITH FILE_SHARE_DELETE (plus READ/WRITE sharing).

    Python's open() does not pass FILE_SHARE_DELETE, so while any reader holds the file open, the writer's
    os.replace() (MoveFileEx) onto it fails with WinError 5/32. Four threads polling in a loop keep the file
    open nearly continuously, so retrying alone can starve the writer. A reader that shares DELETE lets the
    rename-over succeed immediately; the reader keeps seeing the complete OLD content through its handle."""
    import ctypes
    import msvcrt
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
                                wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
    k32.CreateFileW.restype = wintypes.HANDLE
    GENERIC_READ, SHARE_ALL, OPEN_EXISTING, NORMAL = 0x80000000, 0x1 | 0x2 | 0x4, 3, 0x80
    handle = k32.CreateFileW(str(path), GENERIC_READ, SHARE_ALL, None, OPEN_EXISTING, NORMAL, None)
    if handle in (None, wintypes.HANDLE(-1).value, 0xFFFFFFFF, -1):
        err = ctypes.get_last_error()
        if err in (2, 3):                                   # ERROR_FILE_NOT_FOUND / PATH_NOT_FOUND
            raise FileNotFoundError(2, "file not found", str(path))
        raise ctypes.WinError(err)                          # OSError with .winerror => retry_io classifies it
    fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    return os.fdopen(fd, "rb")


def read_bytes_shared(path) -> bytes:
    """Read a whole file without blocking the writer's atomic replace (see _open_shared_windows)."""
    if os.name == "nt":
        with _open_shared_windows(path) as f:
            return f.read()
    with open(path, "rb") as f:
        return f.read()


def atomic_write_json(path: Path, payload: dict, attempts: int = 60) -> None:
    """Write JSON atomically: unique temp file in the same directory, then retried os.replace()."""
    path = Path(path)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{next(_tmp_counter)}.tmp")
    text = json.dumps(jsonsafe(payload), allow_nan=False, separators=(",", ":"))
    try:
        retry_io(lambda: tmp.write_text(text, encoding="utf-8"), attempts)
        def _replace():
            with path_lock(path):                      # no reader can hold the destination open while we rename over it
                os.replace(tmp, path)
        retry_io(_replace, attempts)
    except OSError as exc:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise IpcWriteError(f"could not write {path.name}: {exc}") from exc


def read_json_retry(path: Path, attempts: int = 8):
    """Read a JSON file written by atomic_write_json. Returns None if the file does not exist.
    Transient sharing errors are retried; a persistent one raises OSError (the caller decides)."""
    path = Path(path)

    def _read():
        with path_lock(path):                          # open + read fully + close while no replace can start
            data = read_bytes_shared(path)
        return json.loads(data.decode("utf-8"))

    try:
        return retry_io(_read, attempts, first_delay=0.003, max_delay=0.03)
    except FileNotFoundError:
        return None


def unlink_retry(path: Path, attempts: int = 25) -> bool:
    """Best-effort delete with retries. Returns True if the file is gone (or never existed)."""
    path = Path(path)
    try:
        retry_io(lambda: path.unlink(), attempts, first_delay=0.004, max_delay=0.05)
        return True
    except FileNotFoundError:
        return True
    except OSError:
        return False


def rmtree_retry(path: Path, attempts: int = 25) -> bool:
    """Remove a directory tree, retrying transient Windows sharing errors (antivirus / indexer holding a freshly
    written file). Returns True iff the directory is gone; never raises."""
    path = Path(path)

    def _rm():
        if path.exists():
            shutil.rmtree(path)
    try:
        retry_io(_rm, attempts, first_delay=0.01, max_delay=0.15)
    except OSError:
        try:
            shutil.rmtree(path, ignore_errors=True)       # last resort: remove whatever can still be removed
        except OSError:
            pass
    return not path.exists()


class LivePublisher:
    """Wall-clock throttled snapshot writer (so a fast headless run does not write 100 files/s).

    A failed publish never raises into the simulation loop: it is counted and the last error is kept so
    the snapshot can report it (see diagnostics()) and the dashboard shows STALE if it persists."""

    def __init__(self, runtime_dir, state_hz: float = 10.0, map_hz: float = 2.0):
        self.dir = Path(runtime_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.state_period = 1.0 / max(state_hz, 1e-6)
        self.map_period = 1.0 / max(map_hz, 1e-6)
        self._t_state = self._t_map = -1e18
        self._map_version = None
        self.published = 0
        self.failures = 0
        self.consecutive_failures = 0
        self.last_error = None

    def due_state(self) -> bool:
        return time.monotonic() - self._t_state >= self.state_period

    def due_maps(self, version) -> bool:
        return version != self._map_version and time.monotonic() - self._t_map >= self.map_period

    def _publish(self, name: str, payload: dict) -> bool:
        try:
            atomic_write_json(self.dir / name, payload)
        except OSError as exc:
            self.failures += 1
            self.consecutive_failures += 1
            self.last_error = f"{type(exc).__name__}: {exc}"
            return False
        self.published += 1
        self.consecutive_failures = 0
        return True

    def publish_state(self, snapshot: dict) -> bool:
        self._t_state = time.monotonic()
        snapshot["ipc"] = self.diagnostics()
        return self._publish("state.json", snapshot)

    def publish_scene(self, scene: dict) -> bool:
        """Static per run (world geometry + conditions); written once, read by /api/scene."""
        return self._publish("scene.json", scene)

    def publish_maps(self, maps: dict, version) -> bool:
        self._t_map = time.monotonic()
        ok = self._publish("maps.json", maps)
        if ok:
            self._map_version = version          # on failure the same version stays due and is retried
        return ok

    def diagnostics(self) -> dict:
        return {"published": self.published, "failures": self.failures,
                "consecutive_failures": self.consecutive_failures, "last_error": self.last_error}


class CommandChannel:
    """Child-side reader of commands.jsonl (tracks its byte offset; malformed lines are reported)."""

    def __init__(self, path):
        self.path = Path(path)
        self._pos = 0
        self.rejected: list = []

    def poll(self) -> list:
        def _read():
            return read_bytes_shared(self.path)[self._pos:]
        try:
            data = retry_io(_read, 10, first_delay=0.003, max_delay=0.03)
        except FileNotFoundError:
            return []
        except OSError:
            return []                       # still locked (Windows): try again on the next poll
        if not data.endswith(b"\n") and b"\n" in data:      # leave a half-written trailing line for next poll
            data = data[: data.rindex(b"\n") + 1]
        elif b"\n" not in data:
            return []
        self._pos += len(data)
        out = []
        for line in data.decode("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            try:
                cmd = json.loads(line)
                if not isinstance(cmd, dict) or "cmd" not in cmd:
                    raise ValueError("command must be an object with a 'cmd' field")
                out.append(cmd)
            except (json.JSONDecodeError, ValueError) as exc:
                self.rejected.append(str(exc))
        return out


def send_command(path, cmd: dict) -> None:
    """Parent-side: append one command line (single write => effectively atomic for small lines)."""
    line = (json.dumps(jsonsafe(cmd), allow_nan=False) + "\n").encode("utf-8")

    def _append():
        with open(path, "ab") as f:
            f.write(line)
    retry_io(_append)

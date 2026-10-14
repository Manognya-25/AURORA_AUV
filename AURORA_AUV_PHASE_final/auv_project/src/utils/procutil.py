"""Cross-platform, NON-DESTRUCTIVE process helpers.

Why this exists: on Windows ``os.kill(pid, 0)`` does not probe a process, it calls TerminateProcess and
kills it (Python maps every signal except CTRL_C_EVENT / CTRL_BREAK_EVENT to TerminateProcess). Liveness
checks therefore must never use ``os.kill`` on Windows. Everything here only *queries* the process.
"""
from __future__ import annotations

import os

STILL_ACTIVE = 259                       # GetExitCodeProcess: the process has not exited
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
ERROR_ACCESS_DENIED = 5


def _pid_alive_windows(pid: int, k32) -> bool:
    """Pure logic over a kernel32-like object (so it is unit-testable on any OS).

    OpenProcess failing with ACCESS_DENIED means the process exists but belongs to someone else (alive);
    any other failure (typically ERROR_INVALID_PARAMETER = 87) means there is no such process.
    """
    handle = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not handle:
        return k32.get_last_error() == ERROR_ACCESS_DENIED
    try:
        code = k32.get_exit_code(handle)
        return code is not None and code == STILL_ACTIVE
    finally:
        k32.CloseHandle(handle)


class _Kernel32:
    """Thin ctypes adapter exposing exactly what _pid_alive_windows needs."""

    def __init__(self):
        import ctypes
        from ctypes import wintypes
        self._ct, self._wt = ctypes, wintypes
        self._k = ctypes.WinDLL("kernel32", use_last_error=True)
        self._k.OpenProcess.restype = wintypes.HANDLE
        self._k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self._k.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        self._k.GetExitCodeProcess.restype = wintypes.BOOL
        self._k.CloseHandle.argtypes = [wintypes.HANDLE]

    def OpenProcess(self, access, inherit, pid):
        return self._k.OpenProcess(access, inherit, pid)

    def get_last_error(self):
        return self._ct.get_last_error()

    def get_exit_code(self, handle):
        code = self._wt.DWORD()
        if not self._k.GetExitCodeProcess(handle, self._ct.byref(code)):
            return None
        return int(code.value)

    def CloseHandle(self, handle):
        self._k.CloseHandle(handle)


def _pid_alive_posix(pid: int) -> bool:
    try:
        os.kill(pid, 0)                    # signal 0 = existence probe on POSIX (never delivers a signal)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True                        # exists, owned by another user
    try:                                   # an exited-but-unreaped child (zombie) is not alive
        with open(f"/proc/{pid}/stat", encoding="utf-8", errors="replace") as f:
            return f.read().rsplit(")", 1)[-1].split()[0] != "Z"
    except OSError:
        return True


def pid_alive(pid) -> bool:
    """True iff a process with this PID exists and has not exited. Never signals or alters the process."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    if pid <= 0:
        return False
    if os.name == "nt":
        return _pid_alive_windows(pid, _Kernel32())
    return _pid_alive_posix(pid)

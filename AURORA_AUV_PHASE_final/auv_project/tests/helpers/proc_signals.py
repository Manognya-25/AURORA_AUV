"""TEST-ONLY cross-platform process control (the product code never needs these).

POSIX:    SIGKILL / SIGSTOP / SIGCONT.
Windows:  os.kill(pid, SIGTERM) is TerminateProcess; suspend/resume use ntdll NtSuspendProcess /
          NtResumeProcess (PROCESS_SUSPEND_RESUME access). signal.SIGKILL/SIGSTOP/SIGCONT do not exist there.
NOTE: never call os.kill(pid, 0) on Windows to "check" a process -- it terminates it. Use
src.utils.procutil.pid_alive.
"""
import os
import signal


def terminate_pid(pid: int) -> None:
    if os.name == "nt":
        os.kill(pid, signal.SIGTERM)          # TerminateProcess(handle, SIGTERM)
    else:
        os.kill(pid, signal.SIGKILL)


def _nt_call(pid: int, fn_name: str) -> None:
    import ctypes
    from ctypes import wintypes
    PROCESS_SUSPEND_RESUME = 0x0800
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    nt = ctypes.WinDLL("ntdll")
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    fn = getattr(nt, fn_name)
    fn.argtypes = [wintypes.HANDLE]
    fn.restype = ctypes.c_long
    handle = k32.OpenProcess(PROCESS_SUSPEND_RESUME, False, pid)
    if not handle:
        raise OSError(ctypes.get_last_error(), f"OpenProcess({pid}) failed for {fn_name}")
    try:
        status = fn(handle)
        if status != 0:
            raise OSError(status, f"{fn_name}({pid}) returned NTSTATUS {status:#x}")
    finally:
        k32.CloseHandle(handle)


def suspend_pid(pid: int) -> None:
    if os.name == "nt":
        _nt_call(pid, "NtSuspendProcess")
    else:
        os.kill(pid, signal.SIGSTOP)


def resume_pid(pid: int) -> None:
    if os.name == "nt":
        _nt_call(pid, "NtResumeProcess")
    else:
        os.kill(pid, signal.SIGCONT)

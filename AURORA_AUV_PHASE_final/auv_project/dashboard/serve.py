"""Server entry point for the dashboard: exactly one backend per port, on purpose.

The dashboard is one process that owns the simulation child, the telemetry snapshot and the run-history file.
On Windows a second process CAN bind an address that is already in use (unlike Linux), so starting
`python dashboard.py` while another dashboard runs does not fail: it prints "Running on http://127.0.0.1:5000"
and then requests are answered by whichever process bound first. That silently produces the worst kind of
failure — a page served by an OLD process (with its OLD template already read into memory) while the static
files on disk are NEW, so the browser gets a document whose chart canvases do not match the JS that draws them
and the console dies with a JavaScript error. `main()` refuses that situation instead of hiding it.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 5000


def port_in_use(host: str, port: int, timeout: float = 0.5) -> bool:
    """True when something is already accepting TCP connections on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def resolve(host: str | None, port: int | None) -> tuple[str, int]:
    """Command line wins over the environment, which wins over the defaults.

    `port=0` is honoured as-is (the OS picks a free port), so tests and tools can ask for an ephemeral one.
    """
    host = host or os.environ.get("DASHBOARD_HOST") or DEFAULT_HOST
    if port is None:
        port = int(os.environ.get("DASHBOARD_PORT") or DEFAULT_PORT)
    return host, port


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AURORA AUV dashboard (simulation + live telemetry).")
    p.add_argument("--host", default=None, help=f"bind address (default: {DEFAULT_HOST}, or $DASHBOARD_HOST)")
    p.add_argument("--port", type=int, default=None, help=f"TCP port (default: {DEFAULT_PORT}, or $DASHBOARD_PORT)")
    return p.parse_args(argv)


def already_running_message(host: str, port: int) -> str:
    return (
        f"a dashboard is already serving on http://{host}:{port}/ — refusing to start a second one.\n"
        "Two dashboards on one port do not fail on Windows: the newer one also binds the address and then\n"
        "the browser may be answered by the older process, which still has the page and the Python code it\n"
        "loaded at start-up. That shows up as a console that renders but never updates its charts.\n"
        "Stop the running instance, or start this one on another port:\n"
        f"  Windows:    netstat -ano | findstr :{port}      then  taskkill /PID <pid> /F\n"
        f"  Linux/mac:  lsof -i :{port}                     then  kill <pid>\n"
        f"  or:         python dashboard.py --port {port + 1}\n"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    host, port = resolve(args.host, args.port)
    if port and port_in_use(host, port):
        sys.stderr.write(already_running_message(host, port))
        return 2
    from dashboard.app import app                     # imported here so the check above stays instant

    where = f"http://{host}:{port}/" if port else "an OS-chosen port"
    print(f"AURORA AUV console on {where}   (Ctrl+C to stop)", flush=True)
    app.run(host=host, port=port, debug=False, threaded=True)
    return 0

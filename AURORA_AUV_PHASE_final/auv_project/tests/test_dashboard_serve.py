"""The server entry point: one backend per port.

The dashboard owns the simulation child and the run-history file, so two servers on one port is not a supported
setup: on Windows the second one binds anyway and then requests are answered by whichever process bound first,
which is how a console ends up served by an old process while the files on disk are new. These tests pin the guard
that refuses that, and the port resolution it uses.
"""
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from dashboard.serve import DEFAULT_HOST, DEFAULT_PORT, already_running_message, main, parse_args, port_in_use, resolve

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture()
def listening_port():
    """A real listening socket on a port the OS picked, closed at the end of the test."""
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind((DEFAULT_HOST, 0))
    srv.listen(1)
    yield srv.getsockname()[1]
    srv.close()


def test_a_free_port_is_free_and_a_listening_one_is_not(listening_port):
    assert port_in_use(DEFAULT_HOST, listening_port) is True
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as free:
        free.bind((DEFAULT_HOST, 0))
        spare = free.getsockname()[1]
    assert port_in_use(DEFAULT_HOST, spare) is False


def test_host_and_port_come_from_the_arguments_then_the_environment(monkeypatch):
    assert resolve(None, None) == (DEFAULT_HOST, DEFAULT_PORT)
    monkeypatch.setenv("DASHBOARD_PORT", "5111")
    monkeypatch.setenv("DASHBOARD_HOST", "0.0.0.0")
    assert resolve(None, None) == ("0.0.0.0", 5111)
    assert resolve("127.0.0.1", 5222) == ("127.0.0.1", 5222)          # the command line wins over the environment
    assert resolve(None, 0) == ("0.0.0.0", 0)                          # 0 = let the OS choose, honoured as asked
    assert parse_args([]).port is None and parse_args(["--port", "5333"]).port == 5333


def test_starting_a_second_server_on_a_taken_port_is_refused(listening_port, capsys):
    """main() returns 2 and explains itself instead of starting a server that would shadow the running one."""
    assert main(["--port", str(listening_port)]) == 2
    err = capsys.readouterr().err
    assert err == already_running_message(DEFAULT_HOST, listening_port)
    assert f"already serving on http://{DEFAULT_HOST}:{listening_port}/" in err
    assert "taskkill /PID" in err and f"lsof -i :{listening_port}" in err
    assert f"--port {listening_port + 1}" in err                       # the way out is offered, not just the refusal


def test_the_real_command_refuses_a_taken_port_and_serves_nothing_on_it(listening_port):
    """`python dashboard.py` — the command people actually type — exits 2 without touching the running server."""
    proc = subprocess.run([sys.executable, "dashboard.py", "--port", str(listening_port)],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 2, proc.stderr
    assert "refusing to start a second one" in proc.stderr
    assert "AURORA AUV console on" not in proc.stdout                 # it never reached the point of serving

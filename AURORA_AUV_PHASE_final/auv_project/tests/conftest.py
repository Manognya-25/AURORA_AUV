"""pytest configuration: make PyBullet execution (vs. skipping) impossible to miss.

* Every run ends with an "AURORA PyBullet execution report": how many PyBullet-dependent tests PASSED and how
  many were SKIPPED because PyBullet is missing.
* AURORA_REQUIRE_PYBULLET=1  ->  a missing PyBullet aborts the run instead of silently skipping those tests.
  Windows (cmd):  set AURORA_REQUIRE_PYBULLET=1 && python -m pytest -v
  PowerShell:     $env:AURORA_REQUIRE_PYBULLET=1; python -m pytest -v
Hooks are defensive: a reporting problem can never fail or hide a test result.
"""
import os

import pytest

PYBULLET_MODULES = ("test_pybullet_world", "test_simulation", "test_dynamics_pybullet", "test_sonar_pybullet",
                    "test_estimation_integration", "test_dashboard_api", "test_dashboard_ui", "test_windows_runtime",
                    "test_navigation_pybullet", "test_dashboard_navigation", "test_live_window")


def _pybullet_status():
    try:
        import pybullet  # noqa: F401
        return True, ""
    except Exception as exc:                              # noqa: BLE001 - any import failure means "not usable"
        return False, f"{type(exc).__name__}: {exc}"


def pytest_configure(config):
    if os.environ.get("AURORA_REQUIRE_PYBULLET") == "1":
        ok, why = _pybullet_status()
        if not ok:
            raise pytest.UsageError(f"AURORA_REQUIRE_PYBULLET=1 but PyBullet cannot be imported ({why}). "
                                    f"Install it (pip install pybullet) or unset the variable.")


def _skip_reason(report) -> str:
    lr = getattr(report, "longrepr", None)
    if isinstance(lr, tuple) and len(lr) == 3:             # (file, line, "Skipped: reason")
        return str(lr[2])
    return str(lr)


def _is_missing_pybullet_skip(report) -> bool:
    """A skip that is really 'PyBullet could not be imported' (importorskip). NOT merely a skip whose reason or
    file name mentions PyBullet: e.g. the 'PyBullet is installed, so the missing-PyBullet failure path cannot
    be produced' skipif is the OPPOSITE situation and is a correct skip."""
    r = _skip_reason(report).lower()
    return "could not import 'pybullet'" in r or "pybullet is not installed" in r


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    try:
        tr = terminalreporter
        ok, why = _pybullet_status()
        passed = [r for r in tr.stats.get("passed", []) if any(m in r.nodeid for m in PYBULLET_MODULES)]
        failed = [r for r in tr.stats.get("failed", []) if any(m in r.nodeid for m in PYBULLET_MODULES)]
        all_skipped = tr.stats.get("skipped", [])
        missing = [r for r in all_skipped if _is_missing_pybullet_skip(r)]
        tr.write_sep("=", "AURORA PyBullet execution report")
        tr.write_line(f"PyBullet importable in this interpreter: {ok}" + ("" if ok else f"  ({why})"))
        tr.write_line(f"PyBullet-dependent tests: {len(passed)} passed, {len(failed)} failed, "
                      f"{len(missing)} skipped because PyBullet is missing")
        for r in all_skipped:
            kind = "PYBULLET MISSING" if _is_missing_pybullet_skip(r) else "legitimate skip"
            tr.write_line(f"  skipped [{kind}]: {r.nodeid} -- {_skip_reason(r)}")
        if missing:
            tr.write_line("WARNING: PyBullet tests were SKIPPED, not run. The real-geometry / real-sonar / "
                          "real-mission checks are UNVERIFIED in this run.")
        elif ok and passed:
            tr.write_line("PyBullet tests actually executed (no test was skipped for lack of PyBullet).")
    except Exception as exc:                              # noqa: BLE001
        print(f"[conftest] could not build the PyBullet report: {exc}")

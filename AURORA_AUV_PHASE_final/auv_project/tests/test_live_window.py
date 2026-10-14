"""Live 3-D window: the dashboard opens and closes the PyBullet window of a RUNNING simulation.

The whole point of these tests is that the ROBOT KEEPS RUNNING: every one of them asserts that the simulation
process, its state and its telemetry are untouched while the window opens and closes around it. The window
process is the stand-in (tests/helpers/standin_viewer.py) so no display is needed; the real one is
src/simulation/live_viewer.py, which speaks exactly the same file IPC.
"""
import json
import os
import subprocess
import sys
import time

import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from tests.helpers.standin_viewer import STANDIN_VIEWER_MODULE
from tests.helpers.standin_viewer_fail import STANDIN_FAILING_VIEWER_MODULE
from tests.test_dashboard_api import CHILD, Client, wait_for


def make(tmp_path, viewer_module=STANDIN_VIEWER_MODULE, child=CHILD):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=child, viewer_module=viewer_module,
                            telemetry_timeout_s=3.0)
    cl = Client(create_app(mgr).test_client())
    cl.post("/api/environment/configure", {"realtime_factor": 4.0, "scenario": "mapping_validation"})
    return cl, mgr


@pytest.fixture()
def env(tmp_path):
    cl, mgr = make(tmp_path)
    yield cl, mgr
    mgr.shutdown()


def running(cl):
    """A live snapshot: the simulation is really publishing telemetry with sensors on."""
    def f():
        code, t = cl.get("/api/telemetry")
        return t if code == 200 and t["state"] == "RUNNING" and t["snapshot"].get("sensors_enabled") else None
    return wait_for(f, timeout=90)


def test_the_window_opens_and_closes_around_a_run_that_keeps_running(env):
    cl, mgr = env
    assert cl.post("/api/visual/window", {"open": True})[0] == 409            # nothing is running yet: nothing to show
    assert cl.post("/api/simulation/start")[0] == 202
    running(cl)
    pid = mgr.status()["pid"]

    code, out = cl.post("/api/visual/window", {"open": True})
    assert code == 200 and out["ok"] is True
    assert out["open"] is True and out["owner"] == "viewer" and out["can_close"] is True
    assert out["viewer_pid"] and mgr.viewer_alive                            # a real viewer process is running
    assert mgr.status()["pid"] == pid and mgr.status()["state"] == "RUNNING"  # the robot was NOT restarted

    t1 = float((cl.get("/api/telemetry")[1]["snapshot"] or {})["sim_time"])
    wait_for(lambda: cl.get("/api/telemetry")[1]["snapshot"]["sim_time"] > t1, timeout=30)   # still advancing
    assert cl.get("/api/visual/window")[1]["open"] is True
    assert cl.get("/api/status")[1]["window"]["open"] is True

    code, out = cl.post("/api/visual/window", {"open": False})
    assert code == 200 and out["open"] is False and out["owner"] is None and out["can_close"] is False
    assert not mgr.viewer_alive
    st = mgr.status()
    assert st["pid"] == pid and st["state"] == "RUNNING"                     # and it is STILL the same run
    assert st["telemetry_age_s"] is not None and st["telemetry_age_s"] < 3.0


def test_the_window_endpoint_validates_its_input_and_is_harmless_when_idle(env):
    cl, _ = env
    code, out = cl.post("/api/visual/window", {})
    assert code == 400 and "open" in out["error"]
    code, out = cl.post("/api/visual/window", {"open": 1})
    assert code == 400 and "true or false" in out["error"]
    code, out = cl.post("/api/visual/window", {"open": False})                # nothing open, nothing running
    assert code == 200 and out["open"] is False
    assert cl.get("/api/visual/window")[1]["open"] is False


def test_the_camera_buttons_act_on_the_live_window_not_on_the_simulation(env):
    cl, mgr = env
    cl.post("/api/simulation/start")
    running(cl)
    cl.post("/api/visual/window", {"open": True})
    for action in ("top", "follow", "orbit_left", "zoom_in", "reset"):
        code, out = cl.post(f"/api/camera/{action}")
        assert code == 200 and out["ok"] and action in out["detail"], (action, out)
        assert out["window"] == "live 3-D window"                             # it really went to the window
    cmds = [e.get("cmd") for e in (mgr.state_snapshot() or {}).get("command_log", [])]
    assert "set_camera" not in cmds                                          # nothing was sent to the robot


def test_the_3d_window_switch_acts_on_a_running_simulation_immediately(env):
    cl, mgr = env
    cl.post("/api/simulation/start")
    running(cl)
    pid = mgr.status()["pid"]

    code, out = cl.post("/api/environment/configure", {"show_3d": True})
    assert code == 200 and out["running"] is True
    assert out["window"]["open"] is True and out["window"]["owner"] == "viewer"
    assert mgr.pending["dashboard"]["headless"] is False                      # and it is saved for the next start too
    assert mgr.status()["pid"] == pid and mgr.status()["state"] == "RUNNING"  # the run was not restarted

    code, out = cl.post("/api/environment/configure", {"show_3d": False})
    assert code == 200 and out["window"]["open"] is False
    assert mgr.pending["dashboard"]["headless"] is True
    assert mgr.status()["pid"] == pid and mgr.status()["state"] == "RUNNING"


def test_a_run_that_opened_its_own_window_reports_it_and_cannot_be_closed_from_here(tmp_path):
    """A run started with the 3-D window on owns its PyBullet window; the dashboard cannot close that one without
    stopping the run, and says so instead of pretending. (The stand-in vehicle is used so no window really opens.)"""
    cl, mgr = make(tmp_path, child="tests.helpers.standin_sim")
    try:
        cl.post("/api/environment/configure", {"show_3d": True})              # next start: the run opens the window
        cl.post("/api/simulation/start")
        wait_for(lambda: mgr.status()["state"] == "RUNNING", timeout=60)
        assert mgr.run_has_its_own_window() is True
        code, out = cl.post("/api/visual/window", {"open": True})
        assert code == 200 and out["open"] is True and out["owner"] == "run" and out["can_close"] is False
        assert out["viewer_pid"] is None                                     # no viewer process was started
        code, out = cl.post("/api/visual/window", {"open": False})
        assert code == 409 and "own PyBullet window" in out["error"]
        assert mgr.status()["state"] == "RUNNING"                            # the run kept going, as promised
    finally:
        mgr.shutdown()


def test_a_window_that_cannot_open_is_reported_with_the_real_reason(tmp_path):
    cl, mgr = make(tmp_path, viewer_module=STANDIN_FAILING_VIEWER_MODULE)
    try:
        cl.post("/api/simulation/start")
        running(cl)
        pid = mgr.status()["pid"]
        code, out = cl.post("/api/visual/window", {"open": True})
        assert code == 502 and "could not open" in out["error"] and "no display" in out["error"]
        assert mgr.status()["pid"] == pid and mgr.status()["state"] == "RUNNING"
        w = cl.get("/api/status")[1]["window"]
        assert w["open"] is False and w["error"]                             # the failure is reported, not hidden
    finally:
        mgr.shutdown()


def test_the_window_closes_when_the_operator_stops_the_run(env):
    cl, mgr = env
    cl.post("/api/simulation/start")
    running(cl)
    cl.post("/api/visual/window", {"open": True})
    assert mgr.viewer_alive
    code, out = cl.post("/api/simulation/stop")
    assert code == 200 and out["state"] == "STOPPED"
    wait_for(lambda: not mgr.viewer_alive, timeout=20)
    assert mgr.status()["window"]["open"] is False


def test_the_live_viewer_keeps_the_window_open_only_when_the_run_ends_by_itself(tmp_path):
    """`run_ended_status` is the real viewer's decision (the stand-in mirrors it): it separates 'the mission
    finished on its own' -- which keeps the window on the last frame -- from 'the run / dashboard vanished',
    which still closes it so a dead simulation cannot leave an orphan window behind."""
    from src.simulation.live_viewer import run_ended_status
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "child.json").write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")   # pid is kept here
    (run_dir / "status.json").write_text(json.dumps({"status": "RUNNING"}), encoding="utf-8")
    assert run_ended_status(run_dir, os.getpid()) is None                 # still running: nothing to decide
    assert run_ended_status(run_dir) is None                              # and no run file at all -> None
    for status in ("COMPLETED", "STOPPED", "ENDED_BEFORE_GOAL", "FAILED"):
        (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")
        assert run_ended_status(run_dir, os.getpid()) == status           # the mission ended -> keep the window
    (run_dir / "status.json").write_text(json.dumps({"status": "ERROR"}), encoding="utf-8")
    assert run_ended_status(run_dir, os.getpid()) is None                 # not a run end: it still closes
    (run_dir / "status.json").write_text(json.dumps({"status": "COMPLETED"}), encoding="utf-8")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    assert run_ended_status(run_dir, gone.pid) is None                    # dashboard gone -> still close it


def test_the_window_stays_open_after_the_robot_reaches_the_goal(tmp_path):
    """`visualization.keep_open_after_run` (default true): a run that finishes on its own does NOT close the 3-D
    window -- it stays on the last frame, and only the explicit close ends it. The run's own telemetry is untouched."""
    cl, mgr = make(tmp_path)
    try:
        assert cl.post("/api/environment/configure", {"scenario": "open_terrain", "seed": 3,
                                                       "start_position": [5.0, 5.0, -5.0],
                                                       "navigation_goal": [7.5, 5.0, -5.0]})[0] == 200
        assert cl.post("/api/simulation/start")[0] == 202
        running(cl)
        assert cl.post("/api/visual/window", {"open": True})[0] == 200
        assert mgr.viewer_alive
        wait_for(lambda: mgr.status()["state"] == "COMPLETED", timeout=240)      # the robot reaches the goal
        assert mgr.viewer_alive and mgr.window_state()["open"] is True           # the window is STILL there
        assert cl.post("/api/visual/window", {"open": False})[0] == 200          # ... and closes when asked to
        wait_for(lambda: not mgr.viewer_alive, timeout=20)
        assert mgr.window_state()["open"] is False
    finally:
        mgr.shutdown()


def test_a_finished_run_keeps_its_own_window_open_until_it_is_released(tmp_path):
    """`visualization.keep_open_after_run` for the window the RUN ITSELF opened: when the mission ends the run
    process stays alive holding that window open on the last frame. The dashboard must report the run's REAL
    terminal state (not RUNNING) with a fresh snapshot (not STALE), and the window control must release it."""
    cl, mgr = make(tmp_path)
    try:
        assert cl.post("/api/environment/configure", {"scenario": "open_terrain", "seed": 3,
                                                       "show_3d": True,                                # the run's own window
                                                       "navigation_goal": [7.5, 5.0, -5.0],
                                                       "realtime_factor": 1.0})[0] == 200
        assert cl.post("/api/simulation/start")[0] == 202
        running(cl)
        assert mgr.run_has_its_own_window() is True
        assert mgr.window_state()["owner"] == "run" and mgr.window_state()["open"] is True
        # The run finished on its own and is holding its window on the last frame (what run_simulation writes).
        (mgr.run_dir / "status.json").write_text(json.dumps(
            {"status": "COMPLETED", "result": {"goal_reached": True}, "error": None, "holding": True,
             "pid": mgr.status()["pid"], "time": time.time()}), encoding="utf-8")
        st = mgr.status()
        assert st["state"] == "COMPLETED" and st["alive"] is True and st["holding"] is True   # not RUNNING
        assert st["telemetry_age_s"] is not None and st["telemetry_age_s"] < mgr.telemetry_timeout   # not STALE
        assert st["window"]["open"] is True and st["window"]["can_close"] is True
        # Release it through the SAME control the dashboard uses: the finished run's own window closes.
        code, out = cl.post("/api/visual/window", {"open": False})
        assert code == 200 and out["open"] is False
        wait_for(lambda: not mgr.alive, timeout=20)
        assert mgr.window_state()["open"] is False
    finally:
        mgr.shutdown()


def test_the_page_has_the_window_control_wired_to_the_live_endpoint(env):
    cl, _ = env
    html = cl.c.get("/").get_data(as_text=True)
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    assert 'id="selShow3d"' in html and 'id="windowNote"' in html
    assert "/api/visual/window" in js and 'selShow3d").onchange' in js
    assert 'windowNote' in js and "syncWindowControl" in js


def test_dashboard_auto_open_3d_opens_the_view_of_the_run_that_was_just_started(tmp_path):
    """dashboard.auto_open_3d: starting a run also reveals the 3-D view of THAT run -- the live viewer for a
    headless run, no second simulation and no second environment. The run keeps its own process and telemetry."""
    cl, mgr = make(tmp_path)                      # stand-in viewer, so no display is needed
    try:
        assert cl.post("/api/environment/configure", {"auto_open_3d": True})[0] == 200
        assert mgr.pending["dashboard"]["auto_open_3d"] is True
        code, out = cl.post("/api/simulation/start")
        assert code == 202, out
        pid = mgr.status()["pid"]
        # start() itself opened the live window for this run (owner "viewer"), before any manual open call
        wait_for(lambda: mgr.window_state()["open"] is True, timeout=40)
        w = mgr.window_state()
        assert w["owner"] == "viewer" and w["run_id"] == mgr.run_id, w
        assert mgr.viewer_alive
        # ... and it is the SAME run: the simulation process was neither restarted nor duplicated
        assert mgr.status()["pid"] == pid and mgr.status()["state"] in ("STARTING", "RUNNING")
        assert cl.get("/api/status")[1]["auto_open_3d"] is True
        # restart keeps the dashboard and the visualization synchronized: the window follows the NEW run
        code, out = cl.post("/api/simulation/restart")
        assert code == 202 and out["was_running"] is True
        wait_for(lambda: mgr.window_state()["open"] is True, timeout=40)
        assert mgr.window_state()["run_id"] == mgr.run_id
        # with it off, the next start opens nothing (and the setting is really stored)
        assert cl.post("/api/environment/configure", {"auto_open_3d": False})[0] == 200
        assert mgr.pending["dashboard"]["auto_open_3d"] is False
        assert cl.post("/api/simulation/stop")[0] == 200
        code, out = cl.post("/api/simulation/start")
        assert code == 202
        wait_for(lambda: mgr.status()["state"] in ("RUNNING", "STARTING"), timeout=40)
        assert mgr.window_state()["open"] is False and not mgr.viewer_alive
    finally:
        mgr.shutdown()


def test_auto_open_3d_is_reported_and_validated_by_the_configure_endpoint(tmp_path):
    cl, mgr = make(tmp_path)
    try:
        assert cl.post("/api/environment/configure", {"auto_open_3d": "yes"})[0] == 400
        assert cl.get("/api/status")[1]["auto_open_3d"] is False        # default: no auto-open
    finally:
        mgr.shutdown()


def test_the_page_auto_opens_the_3d_view_on_start(env):
    cl, _ = env
    html = cl.c.get("/").get_data(as_text=True)
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    assert 'id="chkAuto3d"' in html                              # the operator control exists
    assert "checked" in html.split('id="chkAuto3d"')[1][:40]     # and is on by default
    assert "autoOpen3dForRun" in js and "chkAuto3d" in js       # the Start handler calls it
    assert "auto_open_3d" in js                                  # and the stored next-start setting is applied

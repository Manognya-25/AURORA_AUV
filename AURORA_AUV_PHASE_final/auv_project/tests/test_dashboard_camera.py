"""Dashboard camera controls: Follow / Top / Reset / Orbit / Zoom -> API -> command channel -> the live CameraRig.

The rig is exercised against the recording fake pybullet of tests/test_gui_camera.py (no PyBullet needed); the dashboard
route is exercised against the real child process. With no 3-D window the control reports that, honestly, instead of
pretending to act."""
import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from src.simulation.commands import apply_command
from tests.test_dashboard_api import CHILD, Client, wait_for
from tests.test_gui_camera import World
from src.simulation.gui_camera import CameraRig

ACTIONS = ("follow", "top", "reset", "orbit_left", "orbit_right", "zoom_in", "zoom_out")


def test_set_camera_command_drives_the_real_camera_rig():
    w = World()
    rig = CameraRig(0, {"poll_every_n_updates": 1}, w, auv_body_id=w.auv_id)
    state = {"camera_rig": rig}
    ok, detail = apply_command({"cmd": "set_camera", "action": "top"}, None, None, state)
    assert ok and rig.mode == "TOP" or rig.mode.lower() == "top"
    yaw0, scale0 = rig.user["yaw"], rig.user["scale"]
    assert apply_command({"cmd": "set_camera", "action": "orbit_right"}, None, None, state)[0] and rig.user["yaw"] > yaw0
    assert apply_command({"cmd": "set_camera", "action": "zoom_in"}, None, None, state)[0] and rig.user["scale"] < scale0
    assert apply_command({"cmd": "set_camera", "action": "reset"}, None, None, state)[0]
    assert rig.user == {"yaw": 0.0, "pitch": 0.0, "scale": 1.0}
    assert apply_command({"cmd": "set_camera", "action": "follow"}, None, None, state)[0]
    ok, why = apply_command({"cmd": "set_camera", "action": "fly"}, None, None, state)
    assert not ok and "unknown camera action" in why


def test_set_camera_without_a_3d_window_is_rejected_honestly():
    ok, why = apply_command({"cmd": "set_camera", "action": "top"}, None, None, {"camera_rig": None})
    assert not ok and "3-D window" in why


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    yield Client(create_app(mgr).test_client()), mgr
    mgr.shutdown()


def test_every_camera_button_exists_and_is_wired_to_its_api_route(env):
    cl, _ = env
    rules = {r.rule for r in cl.c.application.url_map.iter_rules()}
    assert "/api/camera/<action>" in rules
    html = cl.c.get("/").get_data(as_text=True)
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    for a in ACTIONS:
        assert f'data-cam="{a}"' in html, a
    assert "/api/camera/" in js and "dataset.cam" in js


def test_camera_route_rejects_unknown_actions_and_reports_no_window_on_a_headless_run(env):
    cl, _ = env
    assert cl.post("/api/camera/fly")[0] == 404
    assert cl.post("/api/simulation/start")[0] == 202
    wait_for(lambda: cl.get("/api/telemetry")[0] == 200 or None)
    code, out = cl.post("/api/camera/top")
    assert code == 422 and "3-D window" in out["error"]

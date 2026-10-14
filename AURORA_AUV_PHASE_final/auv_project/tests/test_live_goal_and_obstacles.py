"""User-controlled goal and obstacle count.

* in-window goal keys (arrows / G) -> the same nav_set_goal command the dashboard sends (fake PyBullet; no window needed);
* dashboard "Apply & restart": a changed obstacle count becomes a NEW real world immediately (real child process);
* the demo launcher starts the real navigator with a user goal."""
import numpy as np
import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from run_visual_demo import build_override
from src.simulation.gui_goal import GoalKeyEditor
from tests.test_dashboard_api import CHILD, Client, wait_for, with_telemetry


class KeyPB:
    B3G_LEFT_ARROW, B3G_RIGHT_ARROW, B3G_UP_ARROW, B3G_DOWN_ARROW = 1, 2, 3, 4
    B3G_PAGE_UP, B3G_PAGE_DOWN, B3G_RETURN, B3G_SHIFT, KEY_WAS_TRIGGERED, GEOM_SPHERE = 5, 6, 7, 8, 2, 2

    def __init__(self):
        self.events, self.bodies, self.texts, self._n = {}, {}, {}, 0

    def getKeyboardEvents(self, physicsClientId=0): return dict(self.events)
    def createVisualShape(self, *a, **k): return 1

    def createMultiBody(self, **k):
        self._n += 1; self.bodies[self._n] = list(k["basePosition"]); return self._n

    def resetBasePositionAndOrientation(self, b, pos, orn, physicsClientId=0): self.bodies[b] = list(pos)

    def addUserDebugText(self, text, pos, **k):
        self._n += 1; self.texts[self._n] = text; return self._n

    def removeUserDebugItem(self, i, physicsClientId=0): self.texts.pop(i, None)
    def removeBody(self, b, physicsClientId=0): self.bodies.pop(b, None)


def press(pb, ed, step, key, current):
    pb.events = {key: 2}
    return ed.poll(step, current, lambda g: sent.append(g) or (True, "ok"))


sent: list = []


def test_arrow_keys_move_a_pending_goal_and_g_applies_it_through_the_navigator_command():
    sent.clear()
    pb = KeyPB(); ed = GoalKeyEditor(0, pb, poll_every_n_steps=1)
    cur = [35.0, 28.0, -5.0]
    press(pb, ed, 0, pb.B3G_RIGHT_ARROW, cur); press(pb, ed, 1, pb.B3G_RIGHT_ARROW, cur); press(pb, ed, 2, pb.B3G_UP_ARROW, cur)
    press(pb, ed, 3, pb.B3G_PAGE_DOWN, cur)
    assert np.allclose(ed.pending, [37.0, 29.0, -6.0]) and len(pb.bodies) == 1 and "NEW GOAL" in list(pb.texts.values())[0]
    assert sent == []                                              # nothing changes until it is applied
    msg = press(pb, ed, 4, ord("g"), cur)
    assert sent == [[37.0, 29.0, -6.0]] and "goal set" in msg and ed.pending is None and not pb.bodies   # preview removed


def test_h_cancels_and_a_rejected_goal_is_reported_not_silently_applied():
    pb = KeyPB(); ed = GoalKeyEditor(0, pb, poll_every_n_steps=1)
    cur = [35.0, 28.0, -5.0]
    press(pb, ed, 0, pb.B3G_LEFT_ARROW, cur)
    assert "cancelled" in press(pb, ed, 1, ord("h"), cur) and ed.pending is None
    press(pb, ed, 2, pb.B3G_LEFT_ARROW, cur)
    pb.events = {pb.B3G_RETURN: 2}
    msg = ed.poll(3, cur, lambda g: (False, "goal cell is occupied"))
    assert "rejected" in msg and "occupied" in msg and ed.pending is not None     # stays pending so the user can adjust it


def test_demo_starts_the_real_navigator_towards_the_users_goal():
    ov = build_override("mapping_validation", 3, "default", {}, None, 20, navigate=True, goal=[40.0, 30.0, -6.0])
    assert ov["navigation"]["enabled"] and ov["navigation"]["goal_position"] == [40.0, 30.0, -6.0]
    assert ov["navigation"]["linger_after_terminal_s"] >= 600                       # stays alive for further goal changes
    assert ov["environment"]["obstacles"] == {"count": 20}
    assert "navigation" not in build_override("mapping_validation", 3, "default", {}, None)       # old behaviour is unchanged


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    yield Client(create_app(mgr).test_client()), mgr
    mgr.shutdown()


def test_apply_and_restart_turns_a_new_obstacle_count_into_a_new_real_world_immediately(env):
    cl, _ = env
    cl.post("/api/environment/configure", {"scenario": "dense_obstacles", "seed": 11, "obstacle_count": 10})
    assert cl.post("/api/simulation/start")[0] == 202
    assert with_telemetry(cl, 1)["snapshot"]["environment"]["obstacle_count"] == 10
    for n in (30, 53):
        cl.post("/api/environment/configure", {"obstacle_count": n})
        code, out = cl.post("/api/simulation/restart")                     # what the "Apply & restart now" button calls
        assert code == 202 and out["was_running"] is True, out
        en = with_telemetry(cl, 1)["snapshot"]["environment"]
        assert en["obstacle_count"] == n and en["obstacles_requested"] == n


def test_the_goal_input_is_the_mission_goal_in_both_control_modes(env):
    """The dashboard's goal input is the MISSION goal, and the run resolves it in both control modes: with
    navigation ON the planner flies to it, and with navigation OFF the Phase-4 waypoint mission ends at it. The
    stored key is navigation.goal_position only - simulation.goal_position / simulation.waypoints are deliberately
    left alone, because the world generator derives the obstacle-free mission corridor from them. Blank restores
    the config goal."""
    import yaml
    from src.simulation.commands import deep_merge, resolve_mission_goal
    cl, _ = env
    code, out = cl.post("/api/environment/configure", {"navigation_goal": [30.0, 12.0, -4.0]})
    assert code == 200, out
    p = out["pending_overrides"]
    assert p["navigation"]["goal_position"] == [30.0, 12.0, -4.0]
    assert "goal_position" not in p.get("simulation", {})                # the configured world stays untouched
    assert "waypoints" not in p.get("simulation", {})
    base = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    assert resolve_mission_goal(deep_merge(base, p)) == (30.0, 12.0, -4.0)     # both modes resolve the typed goal
    assert base["simulation"]["goal_position"] == [20, 20, -5]                 # the config itself is not modified

    code, out = cl.post("/api/environment/configure", {"navigation_goal": None})
    assert code == 200
    p = out["pending_overrides"]
    assert p.get("navigation", {}).get("goal_position") is None           # blank = the config goal again
    assert resolve_mission_goal(deep_merge(base, p)) == (20.0, 20.0, -5.0)


def test_restart_works_from_idle_too_and_the_page_wires_the_button(env):
    cl, _ = env
    assert cl.post("/api/simulation/restart")[1]["was_running"] is False
    html = cl.c.get("/").get_data(as_text=True)
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    assert 'id="btnApplyRestart"' in html and "/api/simulation/restart" in js

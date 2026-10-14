"""Phase 5 dashboard: planner / navigation APIs and controls against a REAL child process.

Child = tests/helpers/standin_sim.py without PyBullet (real sensors / EKF / mapper / NAVIGATOR; stand-in rigid body) or the
real dashboard.simulation_process with PyBullet. Every test follows UI -> API -> backend -> real state -> telemetry:
a control is only counted as working if the telemetry that comes BACK reflects it."""
import json
import re

import numpy as np
import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from tests.test_dashboard_api import CHILD, Client, strict_loads, wait_for

GOAL = [45.0, 40.0, -5.0]            # far enough that the mission is still running while the controls are exercised


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    cl = Client(create_app(mgr).test_client())
    cl.post("/api/environment/configure", {"realtime_factor": 1.5, "scenario": "mapping_validation",
                                           "navigation_enabled": True, "navigation_goal": GOAL})
    yield cl, mgr
    mgr.shutdown()


def start_and_wait(cl, state=None, min_updates=1):
    code, body = cl.post("/api/simulation/start")
    assert code == 202, body

    def ready():
        code, t = cl.get("/api/telemetry")
        if code != 200 or t["state"] != "RUNNING":
            return None
        n = t["snapshot"].get("navigation") or {}
        if n.get("enabled") and n.get("state") in (state or ("FOLLOW_PATH", "AVOID_OBSTACLE", "REPLAN")) \
                and t["snapshot"]["mapping"].get("update_count", 0) >= min_updates:
            return t["snapshot"]
    return wait_for(ready, timeout=60)


def nav(cl):
    code, body = cl.get("/api/navigation")
    assert code == 200, body
    return body["data"], body["path"]


def wait_nav(cl, cond, timeout=40):
    def f():
        d, p = nav(cl)
        return (d, p) if cond(d, p) else None
    return wait_for(f, timeout=timeout)


# ---------------------------------------------------------------- wiring
def test_phase5_routes_exist(env):
    cl, _ = env
    rules = {r.rule for r in cl.c.application.url_map.iter_rules()}
    for need in ("/api/navigation", "/api/planner", "/api/path", "/api/navigation/goal", "/api/navigation/start",
                 "/api/navigation/cancel", "/api/navigation/replan", "/api/navigation/unknown_policy"):
        assert need in rules, need


def test_html_and_js_wire_every_navigation_control_to_a_real_api_call(env):
    cl, _ = env
    html = cl.c.get("/").get_data(as_text=True)
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    for cid in ("navState", "plannerTable", "followTable", "avoidTable", "replanTable", "navHistory", "btnGoal", "btnNavStart",
                "btnNavReplan", "btnNavCancel", "selPolicyLive", "goalX", "goalY", "goalZ", "selNavStart", "inGoalX", "chkPath"):
        assert f'id="{cid}"' in html, cid
    for url in ("/api/navigation/goal", "/api/navigation/start", "/api/navigation/replan", "/api/navigation/cancel",
                "/api/navigation/unknown_policy"):
        assert url in js, url
    for bid in ("btnGoal", "btnNavStart", "btnNavReplan", "btnNavCancel"):
        assert f'$("{bid}").onclick' in js
    assert "snap.navigation_path" in js and "drawNavOverlay" in js and "navigation_enabled" in js
    assert "Math.random" not in js
    for field in ("planning_time", "last_planning_time_ms", "replanning", "avoidance", "goal_distance_m", "current_waypoint"):
        assert field in js, field                                           # every requested figure is rendered from the payload


# ---------------------------------------------------------------- configure validation
@pytest.mark.parametrize("body,fragment", [
    ({"navigation_enabled": "yes"}, "true or false"),
    ({"navigation_goal": [1, 2]}, "[x, y, depth]"),
    ({"navigation_goal": ["a", 2, -5]}, "[x, y, depth]"),
    ({"unknown_policy": "maybe"}, "inherit, allow or block"),
])
def test_configure_rejects_invalid_navigation_settings(env, body, fragment):
    cl, _ = env
    code, out = cl.post("/api/environment/configure", body)
    assert code == 400 and fragment in out["error"]


def test_configure_stores_navigation_settings_for_the_next_start(env):
    cl, _ = env
    code, out = cl.post("/api/environment/configure", {"unknown_policy": "allow", "navigation_goal": [30, 12, -4]})
    assert code == 200
    code, e = cl.get("/api/environment")
    p = e["pending_overrides"]
    assert p["navigation"]["enabled"] is True and p["navigation"]["goal_position"] == [30.0, 12.0, -4.0]
    assert p["planning"]["unknown_policy"] == "allow"


@pytest.mark.parametrize("typed,stored", [
    ([18.5, 13.5, 7], [18.5, 13.5, -7.0]),        # a DEPTH typed as it reads in the panel (7 m deep)
    ([18.5, 13.5, -7], [18.5, 13.5, -7.0]),       # the same point written as an explicit world z
    ([18.5, 13.5, 7.5], [18.5, 13.5, -7.5]),
])
def test_a_goal_depth_typed_positive_is_accepted_and_stored_as_the_same_point(env, typed, stored):
    """The z field is a DEPTH in metres below the surface (positive down), the unit the dashboard reports as
    \"depth (m)\". Typing a positive depth is accepted - it used to be rejected as \"above water\" - and it stores
    exactly the same world point as the older negative form, so nothing that worked before changes meaning."""
    cl, _ = env
    assert cl.post("/api/environment/configure", {"navigation_goal": typed})[0] == 200
    _, e = cl.get("/api/environment")
    assert e["pending_overrides"]["navigation"]["goal_position"] == stored


def test_a_start_depth_typed_positive_is_accepted_and_stored_as_the_same_point(env):
    cl, _ = env
    assert cl.post("/api/environment/configure", {"start_position": [-1, 1, 1.5]})[0] == 200
    _, e = cl.get("/api/environment")
    assert e["pending_overrides"]["simulation"]["start_position"] == [-1.0, 1.0, -1.5]


# ---------------------------------------------------------------- disabled navigation
def test_navigation_disabled_is_reported_honestly_and_commands_are_rejected(env):
    cl, _ = env
    cl.post("/api/environment/configure", {"navigation_enabled": False})
    code, _ = cl.post("/api/simulation/start")
    assert code == 202
    wait_for(lambda: cl.get("/api/telemetry")[0] == 200)
    code, body = cl.get("/api/navigation")
    assert code == 200 and body["data"]["enabled"] is False
    code, out = cl.post("/api/navigation/goal", {"goal": [20, 20, -5]})
    assert code == 422 and "not enabled" in out["error"]
    assert cl.post("/api/navigation/replan")[0] == 422 and cl.post("/api/navigation/start")[0] == 422


# ---------------------------------------------------------------- live planner telemetry
def test_running_navigation_publishes_real_planner_output(env):
    cl, _ = env
    snap = start_and_wait(cl)
    d, p = nav(cl)
    assert d["enabled"] and d["state"] in ("FOLLOW_PATH", "AVOID_OBSTACLE", "REPLAN") and d["pose_source"] == "EKF estimate"
    pl = d["planner"]
    assert pl["status"] == "OK" and pl["plans"] >= 1 and pl["plan_successes"] >= 1 and pl["collision_free"] is True
    assert pl["path_length_m"] > 40.0 and pl["last_planning_time_ms"] > 0 and pl["expansions"] > 0
    assert d["goal"] == GOAL and d["goal_distance_m"] > 0
    g = np.asarray(p["global"], float)
    assert g.shape[1] == 2 and len(g) >= 2
    assert g[-1] == pytest.approx(GOAL[:2], abs=1e-6)                       # the real planner path ends at the real goal
    ext = snap["mapping"]["map_size"], snap["mapping"]["map_resolution"]
    assert (g > -5.0).all() and (g < 55.0).all()                           # inside the mapped world
    assert p["waypoint"] is not None and p["following"] in ("global", "detour")
    cl_json = json.dumps(d)                                                  # strict JSON already enforced by the client
    assert "NaN" not in cl_json
    code, planner = cl.get("/api/planner")
    assert code == 200 and planner["data"]["planner"]["status"] == "OK" and planner["data"]["goal"] == GOAL
    code, path = cl.get("/api/path")
    assert code == 200 and path["data"]["global"] == p["global"] and path["enabled"] is True


def test_telemetry_is_live_goal_distance_shrinks_and_progress_advances(env):
    cl, _ = env
    start_and_wait(cl)
    d0, _ = nav(cl)
    d1, _ = wait_nav(cl, lambda d, p: d["state_since"] >= 0 and d["goal_distance_m"] < d0["goal_distance_m"] - 1.5, timeout=40)
    assert d1["estimate_position"] != d0["estimate_position"]
    assert d1["following"]["progress_m"] >= 0.0 and d1["control"]["forward"] != 0.0 or d1["control"]["lateral"] != 0.0


def test_the_navigation_section_never_contains_ground_truth(env):
    cl, _ = env
    start_and_wait(cl)
    d0, _ = nav(cl)
    wait_nav(cl, lambda d, p: d["goal_distance_m"] < d0["goal_distance_m"] - 2.0)       # moved: estimate != initial truth exactly
    code, t = cl.get("/api/telemetry")
    snap = t["snapshot"]
    n, path = snap["navigation"], snap["navigation_path"]
    est = snap["estimate"]["position"]
    assert n["pose_source"] == "EKF estimate"
    assert n["estimate_position"] == pytest.approx(est, abs=1e-6)            # the pose the navigator used IS the EKF estimate
    truth = snap["debug_ground_truth"]["position"]                           # labelled debug section
    floats = []

    def walk(o):
        if isinstance(o, float):
            floats.append(o)
        elif isinstance(o, dict):
            [walk(v) for v in o.values()]
        elif isinstance(o, list):
            [walk(v) for v in o]
    walk(n)
    walk(path)
    for c in truth:
        assert not any(abs(f - c) < 1e-9 for f in floats), f"truth coordinate {c} appears in the operational navigation section"
    assert "debug_ground_truth" not in json.dumps(n) and "truth" not in json.dumps(path).lower()


# ---------------------------------------------------------------- controls
def test_set_goal_control_changes_the_goal_and_replans_to_it(env):
    cl, _ = env
    start_and_wait(cl)
    d0, p0 = nav(cl)
    new = [20.0, 30.0, -5.0]
    code, out = cl.post("/api/navigation/goal", {"goal": new})
    assert code == 200 and out["ok"] and "goal set" in out["detail"]
    d1, p1 = wait_nav(cl, lambda d, p: d["goal"] == new and d["state"] in ("FOLLOW_PATH", "AVOID_OBSTACLE")
                      and d["replanning"]["reasons"].get("goal_changed") == 1)
    assert d1["replanning"]["goal_changes"] == 1 and d1["replanning"]["count"] >= 1
    assert p1["global"][-1] == pytest.approx(new[:2], abs=1e-6) and p1["global"] != p0["global"]
    assert p1["version"] > p0["version"]
    code, t = cl.get("/api/telemetry")
    assert any(c["cmd"] == "nav_set_goal" and c["ok"] for c in t["snapshot"]["command_log"])        # acknowledged by the sim


def test_set_goal_takes_a_positive_depth_and_steers_to_the_same_point_as_its_negative_form(env):
    """The live Set-goal z is a depth too: 4 means 4 m below the surface, i.e. world z -4 - the same point the
    older "-4" form selects. A positive depth used to be rejected as "underwater"."""
    cl, _ = env
    start_and_wait(cl)
    d0, _ = nav(cl)
    code, out = cl.post("/api/navigation/goal", {"goal": [30.0, 12.0, 4.0]})
    assert code == 200 and out["ok"], out
    d1, _ = wait_nav(cl, lambda d, p: d["goal"] == [30.0, 12.0, -4.0])
    assert d1["goal"][2] == -4.0 and d1["goal"] != d0["goal"]
    assert d1["replanning"]["goal_changes"] == 1


def test_set_goal_accepts_x_y_and_defaults_depth_to_the_current_goal(env):
    cl, _ = env
    start_and_wait(cl)
    code, out = cl.post("/api/navigation/goal", {"x": 25.0, "y": 33.0})
    assert code == 200
    d, _ = wait_nav(cl, lambda d, p: d["goal"] == [25.0, 33.0, GOAL[2]])
    assert d["goal"][2] == GOAL[2]


@pytest.mark.parametrize("body,code_want,fragment", [
    ({"goal": [500.0, 5.0, -5.0]}, 422, "outside the map"),
    ({"goal": [30.0, 5.0]}, 400, "[x, y, depth]"),
    ({"goal": "north"}, 400, "[x, y, depth]"),
    ({"goal": ["a", "b", "c"]}, 400, "numbers"),
    ({}, 400, "goal"),
])
def test_invalid_goals_are_rejected_with_the_real_reason_and_do_not_change_the_goal(env, body, code_want, fragment):
    cl, _ = env
    start_and_wait(cl)
    d0, _ = nav(cl)
    code, out = cl.post("/api/navigation/goal", body)
    assert code == code_want and fragment in out["error"]
    d1, _ = nav(cl)
    assert d1["goal"] == d0["goal"] and d1["replanning"]["goal_changes"] == 0


def test_replan_control_triggers_a_real_replan_and_reports_the_reason(env):
    cl, _ = env
    start_and_wait(cl)
    d0, p0 = nav(cl)
    code, out = cl.post("/api/navigation/replan")
    assert code == 200 and "replan requested" in out["detail"]
    d1, p1 = wait_nav(cl, lambda d, p: d["replanning"]["reasons"].get("operator_request") == 1
                      and d["state"] in ("FOLLOW_PATH", "AVOID_OBSTACLE"))
    assert d1["replanning"]["count"] == d0["replanning"]["count"] + 1 and d1["planner"]["plans"] > d0["planner"]["plans"]


def test_unknown_policy_control_reaches_the_planner(env):
    cl, _ = env
    start_and_wait(cl)
    assert cl.post("/api/navigation/unknown_policy", {"policy": "banana"})[0] == 400
    code, out = cl.post("/api/navigation/unknown_policy", {"policy": "block"})
    assert code == 200
    d, _ = wait_nav(cl, lambda d, p: d["planner"]["unknown_policy"] == "block")
    assert d["replanning"]["reasons"].get("unknown_policy_changed") == 1


def test_cancel_and_start_controls_drive_the_state_machine(env):
    cl, _ = env
    start_and_wait(cl)
    code, out = cl.post("/api/navigation/cancel")
    assert code == 200
    d, p = wait_nav(cl, lambda d, p: d["state"] == "IDLE")
    assert p["global"] is None and d["history"][-1]["reason"] == "cancelled_by_operator"
    assert cl.post("/api/navigation/cancel")[0] == 422                       # already idle: the backend says so
    code, out = cl.post("/api/navigation/start")
    assert code == 200
    d, p = wait_nav(cl, lambda d, p: d["state"] in ("FOLLOW_PATH", "PLAN") and p["global"])
    assert [h["to"] for h in d["history"]][-2:] == ["PLAN", "FOLLOW_PATH"] or d["state"] == "PLAN"


def test_navigation_commands_need_a_running_simulation(env):
    cl, _ = env
    for url in ("/api/navigation/start", "/api/navigation/cancel", "/api/navigation/replan"):
        code, out = cl.post(url)
        assert code == 409 and "not running" in out["error"]
    assert cl.get("/api/navigation")[0] == 404                               # no telemetry yet


def test_reset_clears_navigation_telemetry(env):
    cl, _ = env
    start_and_wait(cl)
    assert cl.post("/api/simulation/reset")[0] == 200
    wait_for(lambda: cl.get("/api/status")[1]["state"] == "IDLE")
    assert cl.get("/api/navigation")[0] == 404

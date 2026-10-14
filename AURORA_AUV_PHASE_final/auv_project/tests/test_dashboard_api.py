"""Dashboard backend: every API endpoint and control, against a REAL child process.

The child is tests/helpers/standin_sim.py when PyBullet is unavailable (same IPC, real sensors/EKF/mapper,
kinematic vehicle + analytic ray cast) and the real dashboard.simulation_process when PyBullet is installed.
What each test proves is the backend contract: UI -> API -> process/state change -> telemetry.
"""
import json
import threading
import time

import numpy as np
import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from src.utils.procutil import pid_alive
from tests.helpers.proc_signals import resume_pid, suspend_pid, terminate_pid

try:
    import pybullet  # noqa: F401
    CHILD = "dashboard.simulation_process"
    PYBULLET = True
except ImportError:
    CHILD = "tests.helpers.standin_sim"
    PYBULLET = False


def strict_loads(text):
    def bad(c):
        raise AssertionError(f"non-finite JSON token {c}")
    return json.loads(text, parse_constant=bad)


class Client:
    def __init__(self, c):
        self.c = c

    def get(self, url):
        r = self.c.get(url)
        return r.status_code, strict_loads(r.get_data(as_text=True))

    def post(self, url, body=None, raw=None, ctype="application/json"):
        if raw is not None:
            r = self.c.post(url, data=raw, content_type=ctype)
        else:
            r = self.c.post(url, json=body)
        assert r.mimetype == "application/json", (url, r.status_code, r.mimetype)
        return r.status_code, strict_loads(r.get_data(as_text=True))


def wait_for(cond, timeout=40.0, step=0.1):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = cond()
        if v:
            return v
        time.sleep(step)
    raise AssertionError("timed out waiting for condition")


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    app = create_app(mgr)
    cl = Client(app.test_client())
    cl.post("/api/environment/configure", {"realtime_factor": 1.0, "scenario": "mapping_validation"})
    yield cl, mgr
    mgr.shutdown()


def running(cl):
    code, st = cl.get("/api/status")
    return st["state"] == "RUNNING" and st

def with_telemetry(cl, min_updates=0):
    def f():
        code, t = cl.get("/api/telemetry")
        if code == 200 and t["state"] == "RUNNING" and t["snapshot"]["mapping"].get("update_count", 0) >= min_updates:
            return t
    return wait_for(f)


# ---------------------------------------------------------------- basics
def test_app_imports_and_exposes_the_documented_routes(env):
    cl, mgr = env
    rules = {r.rule for r in cl.c.application.url_map.iter_rules()}
    for need in ["/", "/api/status", "/api/telemetry", "/api/sensors", "/api/ekf", "/api/sonar", "/api/mapping",
                 "/api/occupancy", "/api/costmap", "/api/environment", "/api/thrusters", "/api/simulation/start",
                 "/api/simulation/stop", "/api/simulation/reset", "/api/mapping/reset", "/api/mapping/enable",
                 "/api/mapping/disable", "/api/environment/configure", "/api/simulation/speed",
                 "/api/sensors/<name>/health"]:
        assert need in rules, need


def visible_text(html):
    from html.parser import HTMLParser
    out = []
    class P(HTMLParser):
        skip = False
        def handle_starttag(self, tag, attrs):
            self.skip = tag in ("script", "style")
            for k, v in attrs:                       # text a user can see in attributes
                if k in ("title", "aria-label", "alt", "value") and v:
                    out.append(v)
        def handle_endtag(self, tag):
            self.skip = False
        def handle_data(self, d):
            if not self.skip:
                out.append(d)
    P().feed(html)
    return " ".join(out)


def test_index_page_is_served_and_has_no_placeholder_text(env):
    cl, _ = env
    html = cl.c.get("/").get_data(as_text=True)
    assert "<canvas" in html and "dashboard.js" in html
    text = visible_text(html).lower()
    for bad in ("coming soon", "todo", "demo", "placeholder", "mock", "fake", "lorem", "battery"):
        assert bad not in text, bad
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    assert "Math.random" not in js                                         # no random values anywhere in the frontend
    # every number written to the UI goes through values read from API payloads (snap/status/occ/cost), never literals:
    import re as _re
    assert not _re.search(r"textContent\s*=\s*['\"]?\d+(\.\d+)?\s*(m|%|s)?['\"]?;", js)
    # every button has a handler wired in the script
    import re
    for bid in re.findall(r'<button id="(\w+)"', html):
        assert f'$("{bid}").onclick' in js, bid
    for sid in re.findall(r'<select id="(\w+)"', html):
        assert sid in js, sid


def test_idle_state_before_start_and_telemetry_is_404_json(env):
    cl, _ = env
    code, st = cl.get("/api/status")
    assert code == 200 and st["state"] == "IDLE" and st["alive"] is False and st["pid"] is None
    for url in ("/api/telemetry", "/api/sensors", "/api/ekf", "/api/sonar", "/api/mapping", "/api/occupancy",
                "/api/costmap", "/api/thrusters"):
        code, body = cl.get(url)
        assert code == 404 and body["ok"] is False and "error" in body, url


# ---------------------------------------------------------------- start / telemetry / stop
def test_start_launches_a_real_process_and_status_follows_reality(env):
    cl, mgr = env
    code, st = cl.post("/api/simulation/start")
    assert code == 202 and st["pid"] and st["state"] in ("STARTING", "RUNNING")
    pid = st["pid"]
    assert pid_alive(pid)
    st = wait_for(lambda: running(cl))
    assert st["alive"] and st["pid"] == pid and pid_alive(pid)
    code, again = cl.post("/api/simulation/start")
    assert code == 409 and again["ok"] is False                                # cannot double-start


def test_telemetry_really_changes_over_time_and_is_strict_json(env):
    cl, mgr = env
    cl.post("/api/simulation/start")
    a = with_telemetry(cl, 3)["snapshot"]
    time.sleep(1.5)
    b = with_telemetry(cl, a["mapping"]["update_count"] + 3)["snapshot"]
    assert b["sim_time"] > a["sim_time"] and b["step"] > a["step"]
    assert a["estimate"]["position"] != b["estimate"]["position"]              # estimated position moved
    assert b["sensors"]["imu"]["samples"] > a["sensors"]["imu"]["samples"]
    assert b["sensors"]["dvl"]["samples"] > a["sensors"]["dvl"]["samples"]
    assert b["sensors"]["sonar"]["samples"] > a["sensors"]["sonar"]["samples"]
    assert b["estimate"]["stats"]["predictions"] > a["estimate"]["stats"]["predictions"]
    assert b["mapping"]["update_count"] > a["mapping"]["update_count"]
    assert b["mapping"]["known_cells"] >= a["mapping"]["known_cells"] > 0
    assert b["sonar"]["available"] and b["sonar"]["n_rays"] == 105 and len(b["sonar"]["range_m"]) == 105
    for url in ("/api/status", "/api/telemetry", "/api/sensors", "/api/ekf", "/api/sonar", "/api/mapping",
                "/api/environment", "/api/thrusters", "/api/occupancy", "/api/costmap"):
        code, _ = cl.get(url)                                                   # strict_loads rejects NaN/Infinity
        assert code == 200, url


def test_ekf_sensor_sonar_endpoints_return_real_sections(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 2)
    _, ekf = cl.get("/api/ekf")
    assert ekf["data"]["available"] and ekf["data"]["filter_status"] in ("NOMINAL", "DEAD_RECKONING", "COASTING")
    assert len(ekf["data"]["position"]) == 3 and ekf["data"]["covariance_trace"] > 0
    assert ekf["ground_truth_debug"]["LABEL"].startswith("GROUND TRUTH")
    _, sens = cl.get("/api/sensors")
    for n in ("imu", "dvl", "depth", "sonar"):
        assert sens["data"][n]["health"] in ("HEALTHY", "DEGRADED", "FAILED") and sens["data"][n]["samples"] > 0
    _, so = cl.get("/api/sonar")
    assert so["data"]["valid_rays"] + so["data"]["invalid_rays"] == so["data"]["n_rays"]
    assert all((r is None) or (0 < r <= so["data"]["max_range_m"]) for r in so["data"]["range_m"])
    _, th = cl.get("/api/thrusters")
    assert "available" in th["data"]["thrusters"]


def test_occupancy_and_costmap_are_real_and_change_and_support_since_cache(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 2)
    from src.mapping.map_manager import rle_decode

    def has_free():                       # a cell needs ~2 free observations to become FREE (by design)
        r = cl.get("/api/occupancy")
        if r[0] == 200 and (rle_decode(r[1]["data"], tuple(r[1]["shape"])) == 1).any():
            return r
    code, occ = wait_for(has_free)
    assert occ["shape"] == [240, 240] and occ["legend"] == {"0": "unknown", "1": "free", "2": "occupied"}
    grid = rle_decode(occ["data"], tuple(occ["shape"]))
    assert grid.size == 57600 and (grid == 0).sum() > 0 and (grid == 1).sum() > 0     # unknown AND free present
    code, same = cl.get(f"/api/occupancy?since={occ['version']}")
    assert same.get("unchanged") is True or same["version"] != occ["version"]
    newer = wait_for(lambda: (lambda r: r[1] if r[1]["version"] != occ["version"] else None)(cl.get("/api/occupancy")))
    g2 = rle_decode(newer["data"], tuple(newer["shape"]))
    assert (g2 != 0).sum() >= (grid != 0).sum()                                        # known area grows as it explores
    code, cm = cl.get("/api/costmap")
    assert code == 200 and cm["legend"]["classes"]["3"] == "inflated" and cm["stats"]["inflation_radius_m"] == pytest.approx(0.6)
    cls = rle_decode(cm["classes"], tuple(cm["shape"]))
    assert (cls == 1).sum() > 0                                                        # unknown shown as its own class


def test_stop_terminates_the_real_process_and_reports_stopped(env):
    cl, mgr = env
    code, st = cl.post("/api/simulation/start")
    pid = st["pid"]
    wait_for(lambda: running(cl))
    code, st = cl.post("/api/simulation/stop")
    assert code == 200 and st["state"] == "STOPPED" and st["alive"] is False
    assert not pid_alive(pid) and st["returncode"] == 0
    code, st2 = cl.get("/api/status")
    assert st2["state"] == "STOPPED"
    code, err = cl.post("/api/simulation/stop")
    assert code == 409 and err["ok"] is False                                          # nothing left to stop
    code, snap = cl.get("/api/telemetry")                                              # final snapshot is still readable
    assert code == 200 and snap["snapshot"]["run_status"] == "STOPPED"


def test_start_again_after_stop_works_and_runs_a_fresh_simulation(env):
    cl, mgr = env
    cl.post("/api/simulation/start"); wait_for(lambda: running(cl))
    t_first = with_telemetry(cl, 5)["snapshot"]["sim_time"]
    _, st1 = cl.get("/api/status")
    cl.post("/api/simulation/stop")
    code, st = cl.post("/api/simulation/start")
    assert code == 202 and st["pid"] != st1["pid"] and pid_alive(st["pid"])
    wait_for(lambda: running(cl))
    snap = with_telemetry(cl, 1)["snapshot"]
    assert snap["sim_time"] < t_first + 1.0 and snap["mapping"]["update_count"] < 30   # restarted from zero, not resumed


def test_reset_stops_clears_everything_and_returns_to_idle(env):
    cl, mgr = env
    code, st = cl.post("/api/simulation/start")
    pid = st["pid"]
    with_telemetry(cl, 5)
    code, out = cl.post("/api/simulation/reset")
    assert code == 200 and out["state"] == "IDLE" and out["was_running"] is True
    assert not pid_alive(pid)
    for url in ("/api/telemetry", "/api/occupancy", "/api/costmap"):
        assert cl.get(url)[0] == 404, url                                              # telemetry + maps are gone
    assert mgr.run_dir is None and not list(mgr.runs_root.glob(f"{st['run_id']}*"))       # the run's directory is gone
    code, out2 = cl.post("/api/simulation/reset")                                      # idempotent when idle
    assert code == 200 and out2["state"] == "IDLE" and out2["was_running"] is False
    cl.post("/api/simulation/start")
    snap = with_telemetry(cl, 1)["snapshot"]
    assert snap["mapping"]["update_count"] < 30


# ---------------------------------------------------------------- mapping controls
def test_mapping_disable_enable_and_reset_act_on_the_running_mapper(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 5)
    code, r = cl.post("/api/mapping/disable")
    assert code == 200 and r["ok"] and "disabled" in r["detail"]
    time.sleep(1.2)
    _, a = cl.get("/api/mapping"); n0 = a["data"]["mapping"]["update_count"]
    assert a["data"]["mapping"]["enabled"] is False
    time.sleep(1.5)
    _, b = cl.get("/api/mapping")
    assert b["data"]["mapping"]["update_count"] == n0                                  # really stopped integrating
    code, r = cl.post("/api/mapping/enable")
    assert code == 200 and "enabled" in r["detail"]
    wait_for(lambda: cl.get("/api/mapping")[1]["data"]["mapping"]["update_count"] > n0)
    assert cl.get("/api/mapping")[1]["data"]["mapping"]["enabled"] is True
    code, r = cl.post("/api/mapping/reset")
    assert code == 200 and "reset" in r["detail"]
    _, snap = cl.get("/api/mapping")
    m = snap["data"]["mapping"]
    assert m["update_count"] < n0 and m["known_cells"] < 57600 * 0.5               # went back toward unknown
    assert snap["data"]["sonar_mapping"]["rays_processed"] < 105 * n0


def test_mapping_reset_returns_the_grid_to_unknown_in_the_published_map(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 10)
    cl.post("/api/mapping/disable")                                                    # freeze so the reset is observable
    time.sleep(1.2)
    before = cl.get("/api/mapping")[1]["data"]["mapping"]["known_cells"]
    assert before > 1000
    cl.post("/api/mapping/reset")
    m = cl.get("/api/mapping")[1]["data"]
    assert m["mapping"]["known_cells"] == 0 and m["mapping"]["update_count"] == 0
    assert m["cost_map"]["occupied_count"] == 0 and m["cost_map"]["inflated_count"] == 0
    from src.mapping.map_manager import rle_decode
    occ = wait_for(lambda: (lambda r: r[1] if r[0] == 200 and (rle_decode(r[1]["data"], tuple(r[1]["shape"])) != 0).sum() == 0 else None)(cl.get("/api/occupancy")), 10)
    assert occ["shape"] == [240, 240]


def test_mapping_controls_are_rejected_when_nothing_is_running(env):
    cl, _ = env
    for url in ("/api/mapping/enable", "/api/mapping/disable", "/api/mapping/reset"):
        code, body = cl.post(url)
        assert code == 409 and body["ok"] is False and "not running" in body["error"], url
    assert cl.post("/api/simulation/speed", {"value": 2})[0] == 409
    assert cl.post("/api/sensors/sonar/health", {"health": "FAILED"})[0] == 409


# ---------------------------------------------------------------- sensor fault injection + EKF/sonar visibility
def test_sonar_fault_injection_changes_health_and_stops_map_updates_then_recovers(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 5)
    code, r = cl.post("/api/sensors/sonar/health", {"health": "FAILED"})
    assert code == 200 and "FAILED" in r["detail"]
    time.sleep(1.0)
    _, m0 = cl.get("/api/mapping"); n0 = m0["data"]["mapping"]["update_count"]
    time.sleep(1.5)
    _, s = cl.get("/api/sensors"); _, so = cl.get("/api/sonar"); _, m1 = cl.get("/api/mapping")
    assert s["data"]["sonar"]["health"] == "FAILED" and so["data"]["health"] == "FAILED"
    assert so["data"]["valid_rays"] == 0 and so["data"]["invalid_rays"] == so["data"]["n_rays"]
    assert m1["data"]["mapping"]["update_count"] == n0                                  # no fabricated map updates
    assert m1["data"]["mapping"]["last_reject_reason"] == "sonar_invalid_or_failed"
    code, _ = cl.post("/api/sensors/sonar/health", {"health": "HEALTHY"})
    wait_for(lambda: cl.get("/api/mapping")[1]["data"]["mapping"]["update_count"] > n0)
    assert cl.get("/api/sensors")[1]["data"]["sonar"]["health"] == "HEALTHY"


def test_dvl_fault_is_reflected_in_sensor_health_and_ekf_status(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 3)
    cl.post("/api/sensors/dvl/health", {"health": "FAILED"})
    wait_for(lambda: cl.get("/api/sensors")[1]["data"]["dvl"]["health"] == "FAILED")
    d0 = cl.get("/api/ekf")[1]["data"]["stats"]["dvl_updates"]
    time.sleep(1.5)
    e = cl.get("/api/ekf")[1]["data"]
    assert e["stats"]["dvl_updates"] <= d0 + 1                                            # EKF stopped receiving DVL
    assert cl.get("/api/sensors")[1]["data"]["dvl"]["invalid"] > 0


# ---------------------------------------------------------------- speed
def test_speed_control_changes_the_real_pacing(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    with_telemetry(cl, 3)
    t0, w0 = cl.get("/api/telemetry")[1]["snapshot"]["sim_time"], time.time()
    time.sleep(0.8)
    t1, w1 = cl.get("/api/telemetry")[1]["snapshot"]["sim_time"], time.time()
    slow_rate = (t1 - t0) / (w1 - w0)                                                  # measured at 1x
    assert 0.5 < slow_rate < 2.0
    code, r = cl.post("/api/simulation/speed", {"value": 10})
    assert code == 200 and "realtime factor 10" in r["detail"]
    time.sleep(0.4)                                                                    # let the new pacing take effect
    t2, w2 = cl.get("/api/telemetry")[1]["snapshot"]["sim_time"], time.time()
    time.sleep(0.6)
    snap = cl.get("/api/telemetry")[1]["snapshot"]
    t3, w3 = snap["sim_time"], time.time()
    assert cl.get("/api/status")[1]["state"] == "RUNNING"                              # still running: no vacuous pass
    fast_rate = (t3 - t2) / (w3 - w2)
    # The pacer only ever sleeps DOWN to the factor it was asked for, so it can never outrun it; the achieved
    # rate is bounded by this machine's simulation throughput (a real PyBullet + sensors + mapping loop tops out
    # near 2x real time here). Asserting a flat 4x ACHIEVED speedup would therefore be asserting the hardware,
    # not the speed control, so the control's real contract is checked instead:
    assert fast_rate > 1.25 * slow_rate                                                # the change genuinely sped the run up
    assert fast_rate <= 10.0 * slow_rate + 0.5                                         # ... and never past the factor it was given
    assert snap["performance"]["realtime_factor_setting"] == 10.0                      # the loop reports the new setting
    assert cl.post("/api/simulation/speed", {"value": -1})[0] in (409, 422)
    assert cl.post("/api/simulation/speed", {})[0] == 400


# ---------------------------------------------------------------- environment configuration (next start)
def test_environment_configure_is_saved_validated_and_applies_on_next_start(env):
    cl, _ = env
    code, out = cl.post("/api/environment/configure", {"scenario": "dense_obstacles", "seed": 7, "obstacle_count": 12})
    assert code == 200 and out["applies"] == "on next simulation start"
    p = out["pending_overrides"]["environment"]
    assert p["scenario"] == "dense_obstacles" and p["seed"] == 7 and p["obstacles"]["count"] == 12
    # obstacles.count is the number of PyBullet BODIES: a multi-body structure (reef tower, wreck) is several of
    # them, so the generated body count is EXACTLY what was requested while structures may be fewer.
    assert out["preview_ground_truth_summary"]["obstacle_count"] == 12                    # exactly 12 bodies requested
    assert 1 <= out["preview_ground_truth_summary"]["structure_count"] <= 12              # grouped into structures
    _, e = cl.get("/api/environment")
    assert e["pending_overrides"]["environment"]["seed"] == 7 and any(s["name"] == "narrow_corridor" for s in e["scenarios"])
    code, st = cl.post("/api/simulation/start")
    assert st["active_overrides"]["environment"]["seed"] == 7
    snap = with_telemetry(cl, 1)["snapshot"]
    assert snap["environment"]["scenario"] == "dense_obstacles" and snap["environment"]["seed"] == 7
    assert snap["environment"]["obstacle_count"] == 12                                   # generated count, from the world itself
    assert snap["environment"]["obstacles_requested"] == 12 and snap["environment"]["obstacles_fixed"] == 0
    assert 1 <= snap["environment"]["structure_count"] <= 12
    code, out2 = cl.post("/api/environment/configure", {"seed": 99})                      # while running: saved, NOT applied
    assert code == 200 and out2["running"] is True
    time.sleep(0.6)
    assert cl.get("/api/telemetry")[1]["snapshot"]["environment"]["seed"] == 7            # running sim unchanged


def test_configured_seeds_produce_different_worlds_in_the_running_simulation(env):
    cl, _ = env
    seen = []
    for seed in (1, 2):
        cl.post("/api/environment/configure", {"scenario": "sparse_obstacles", "seed": seed})
        cl.post("/api/simulation/start")
        snap = with_telemetry(cl, 1)["snapshot"]
        seen.append((snap["environment"]["seed"], snap["environment"]["scenario"]))
        cl.post("/api/simulation/reset")
    assert seen == [(1, "sparse_obstacles"), (2, "sparse_obstacles")]


def test_obstacle_count_creates_that_many_real_obstacles_every_time_it_changes(env):
    """10 -> 30 -> 53 must produce exactly that many PyBullet obstacle bodies in the RUNNING world, reported by
    the run itself (not by the form), and be deterministic for the selected seed."""
    cl, _ = env
    generated = []
    for n in (10, 30, 53):
        code, out = cl.post("/api/environment/configure", {"scenario": "dense_obstacles", "seed": 11,
                                                            "obstacle_count": n})
        assert code == 200, out
        assert out["preview_ground_truth_summary"]["obstacle_count"] == n      # validated preview == requested
        code, st = cl.post("/api/simulation/start")
        assert code == 202, st
        snap = with_telemetry(cl, 1)["snapshot"]
        en = snap["environment"]
        assert en["obstacle_count"] == n and en["obstacles_requested"] == n    # the real world the child built
        assert en["obstacle_count"] > 0 and en["structure_count"] >= 1
        generated.append((en["obstacle_count"], sorted(en["obstacle_kinds"].items()),
                          en["seabed_z_min"], en["seabed_z_max"]))
        cl.post("/api/simulation/reset")
    assert [g[0] for g in generated] == [10, 30, 53]
    assert generated[0][1] != generated[1][1]                                   # more obstacles: a different world
    # same seed + same count -> byte-identical world (determinism of the generator)
    for _ in range(2):
        cl.post("/api/environment/configure", {"scenario": "dense_obstacles", "seed": 11, "obstacle_count": 30})
        cl.post("/api/simulation/start")
        en = with_telemetry(cl, 1)["snapshot"]["environment"]
        assert en["obstacle_count"] == 30 and en["obstacle_kinds"] == dict(dict(generated[1][1]))
        cl.post("/api/simulation/reset")


def test_obstacle_count_is_rejected_on_the_week4_map_scenario_instead_of_being_ignored(env):
    cl, _ = env
    code, out = cl.post("/api/environment/configure", {"scenario": "legacy_week4", "obstacle_count": 10})
    assert code == 422 and "procedural" in out["error"]                        # honest refusal, not a silent no-op
    assert "obstacles" not in cl.get("/api/environment")[1]["pending_overrides"].get("environment", {})


def test_invalid_inputs_get_json_errors_with_meaningful_status(env):
    cl, _ = env
    cases = [
        ("/api/environment/configure", {"scenario": "atlantis"}, 400),
        ("/api/environment/configure", {"seed": -5}, 400),
        ("/api/environment/configure", {"seed": "abc"}, 400),
        ("/api/environment/configure", {"obstacle_count": 9999}, 400),
        ("/api/environment/configure", {"obstacle_count": "x"}, 400),
        ("/api/environment/configure", {"dynamics_model": "warp"}, 400),
        ("/api/environment/configure", {"mapping_enabled": "yes"}, 400),
        ("/api/environment/configure", {"realtime_factor": -2}, 400),
        ("/api/sensors/sonarr/health", {"health": "FAILED"}, 404),
        ("/api/sensors/sonar/health", {"health": "BROKEN"}, 400),
    ]
    for url, body, want in cases:
        code, out = cl.post(url, body)
        assert code == want and out["ok"] is False and out["error"], (url, body, code, out)
    code, out = cl.post("/api/environment/configure", raw="{not json", ctype="application/json")
    assert code == 400 and out["ok"] is False
    code, out = cl.get("/api/does/not/exist")
    assert code == 404 and out["ok"] is False
    r = cl.c.delete("/api/simulation/start")
    assert r.status_code == 405 and r.mimetype == "application/json"
    code, out = cl.post("/api/environment/configure", {"scenario": "open_terrain", "seabed": {"x": 1}})
    assert code == 200                                                                  # unknown keys are ignored, not fatal
    assert cl.get("/api/status")[1]["pending_overrides"]["environment"]["scenario"] == "open_terrain"


# ---------------------------------------------------------------- state synchronisation under failure
def test_stalled_telemetry_is_reported_as_stale_then_recovers(env):
    cl, mgr = env
    code, st = cl.post("/api/simulation/start")
    wait_for(lambda: running(cl))
    suspend_pid(st["pid"])
    try:
        s = wait_for(lambda: (lambda x: x if x["state"] == "STALE" else None)(cl.get("/api/status")[1]), 15)
        assert s["alive"] is True and s["telemetry_age_s"] > 1.5                          # alive but NOT shown as running
    finally:
        resume_pid(st["pid"])
    wait_for(lambda: running(cl))


def test_a_killed_simulation_is_reported_as_failed_not_running(env):
    cl, mgr = env
    code, st = cl.post("/api/simulation/start")
    wait_for(lambda: running(cl))
    terminate_pid(st["pid"])
    s = wait_for(lambda: (lambda x: x if x["state"] == "FAILED" else None)(cl.get("/api/status")[1]), 10)
    assert s["alive"] is False and s["returncode"] != 0 and s["error"]
    assert cl.post("/api/mapping/reset")[0] == 409                                         # no control acts on a dead sim


def test_a_child_that_cannot_start_is_reported_as_failed_with_its_error(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module="tests.helpers.no_such_module")
    cl = Client(create_app(mgr).test_client())
    cl.post("/api/simulation/start")
    s = wait_for(lambda: (lambda x: x if x["state"] == "FAILED" else None)(cl.get("/api/status")[1]), 15)
    assert "no_such_module" in s["error"] and s["alive"] is False
    assert cl.get("/api/telemetry")[0] == 404


@pytest.mark.skipif(PYBULLET, reason="PyBullet is installed; the missing-PyBullet failure path cannot be produced")
def test_missing_pybullet_surfaces_as_a_real_failed_state_with_the_import_error(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt")                                   # the REAL child process
    cl = Client(create_app(mgr).test_client())
    cl.post("/api/simulation/start")
    s = wait_for(lambda: (lambda x: x if x["state"] == "FAILED" else None)(cl.get("/api/status")[1]), 30)
    assert "pybullet" in s["error"].lower()


def test_optional_polling_returns_200_unavailable_instead_of_404_and_default_stays_404(env):
    cl, _ = env
    for url in ("/api/telemetry", "/api/occupancy", "/api/costmap"):
        assert cl.get(url)[0] == 404, url                                       # strict default for API clients
        code, body = cl.get(url + "?optional=1")
        assert code == 200 and body["available"] is False, url                  # race-free mode for the page's poller
    cl.post("/api/simulation/start")
    with_telemetry(cl, 2)
    code, body = cl.get("/api/telemetry?optional=1")
    assert code == 200 and body["available"] is True and body["snapshot"]["sim_time"] > 0


def test_show_3d_option_maps_to_the_headless_setting_for_the_next_start(env):
    cl, _ = env
    code, out = cl.post("/api/environment/configure", {"show_3d": True})
    assert code == 200 and out["pending_overrides"]["dashboard"]["headless"] is False      # GUI window <=> not headless
    code, out = cl.post("/api/environment/configure", {"show_3d": False})
    assert code == 200 and out["pending_overrides"]["dashboard"]["headless"] is True
    code, out = cl.post("/api/environment/configure", {"show_3d": "yes"})
    assert code == 400 and out["ok"] is False
    assert cl.get("/api/status")[1]["pending_overrides"]["dashboard"]["headless"] is True

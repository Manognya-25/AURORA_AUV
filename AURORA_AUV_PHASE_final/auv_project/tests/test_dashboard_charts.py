"""Live charts: all 20 exist, each is registered exactly once, is fed from a real snapshot field, and
navigation events are drawn as markers."""
import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from tests.test_dashboard_api import CHILD, Client, wait_for

CHARTS = ("err", "cov", "goal", "speed", "depth", "yaw", "clr", "rpl",           # original 8
          "roll", "pitch", "estX", "estY", "estZ",                                     # attitude + estimated position
          "vel", "imuAcc", "angVel", "health", "thrust",                               # components, sensors, thrusters
          "adapt", "adaptTrack")                                                       # the driving controller's gains
MULTI = ("vel", "imuAcc", "angVel", "health", "thrust", "adapt", "adaptTrack")


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    cl = Client(create_app(mgr).test_client())
    cl.post("/api/environment/configure", {"realtime_factor": 1.5, "scenario": "mapping_validation",
                                           "navigation_enabled": True, "navigation_goal": [45.0, 40.0, -5.0]})
    yield cl, mgr
    mgr.shutdown()


def test_every_chart_is_in_the_page_and_registered_with_a_live_series():
    html = open("dashboard/templates/index.html", encoding="utf-8").read()
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    charts = open("dashboard/static/js/charts.js", encoding="utf-8").read()
    for c in CHARTS:
        for suffix in ("Chart", "Now", "Stats"):
            assert f'id="{c}{suffix}"' in html, (c, suffix)
        assert f'canvasId: "{c}Chart"' in js, c
    assert "getEvents" in charts and "getEvents" in js            # navigation event markers
    assert "Math.random" not in js and "Math.random" not in charts


def test_exactly_20_charts_are_registered_once_with_unique_ids():
    html = open("dashboard/templates/index.html", encoding="utf-8").read()
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    assert len(CHARTS) == 20 and len(set(CHARTS)) == 20
    assert js.count("canvasId:") == 20, "every chart must be registered exactly once"
    for c in CHARTS:
        assert js.count(f'canvasId: "{c}Chart"') == 1, c
        assert html.count(f'id="{c}Chart"') == 1, c
        assert html.count(f'id="{c}Now"') == 1 and html.count(f'id="{c}Stats"') == 1, c
    # component charts are multi-series; the rest are single-series
    import re
    for c in MULTI:
        assert re.search(rf'canvasId: "{c}Chart".{{0,500}}?series', js, re.S), c
    for c in CHARTS:
        if c not in MULTI:
            assert re.search(rf'canvasId: "{c}Chart".{{0,300}}?getSeries', js, re.S), c


def test_the_registered_canvases_and_the_page_agree_in_both_directions():
    """A canvas the JS registers but the page does not carry (or the other way round) is not cosmetic: every chart is
    built at boot, so a page whose DOM does not match dashboard.js used to abort the whole boot sequence — no charts
    AND no polling, i.e. a console that renders once and never updates. The two files must agree exactly, which is
    what this pins: an added chart that is not in the template, or a removed canvas still registered, fails here
    instead of only in a browser."""
    import re
    html = open("dashboard/templates/index.html", encoding="utf-8").read()
    js = open("dashboard/static/js/dashboard.js", encoding="utf-8").read()
    registered = set(re.findall(r'canvasId: "([^"]+)"', js))
    on_page = {c for c in re.findall(r'<canvas id="([^"]+)"', html) if c.endswith("Chart")}
    assert registered == on_page, (f"registered in the JS only: {sorted(registered - on_page)}; "
                                   f"on the page only: {sorted(on_page - registered)}")


def test_the_chart_series_come_from_the_live_snapshot_fields(env):
    cl, _ = env
    assert cl.post("/api/simulation/start")[0] == 202

    def ready():
        code, t = cl.get("/api/telemetry")
        if code != 200 or t["state"] != "RUNNING":
            return None
        s = t["snapshot"]
        return s if (s.get("navigation") or {}).get("enabled") and (s.get("estimate") or {}).get("available") else None
    snap = wait_for(ready, timeout=60)
    est, nav = snap["estimate"], snap["navigation"]
    assert isinstance(est["speed"], float) and isinstance(est["depth"], float) and len(est["euler_rad"]) == 3
    assert nav["goal_distance_m"] is not None and "count" in nav["replanning"] and "count" in nav["avoidance"]
    assert "min_valid_range_m" in snap["sonar"]
    assert "estimate_error_norm" in snap["debug_ground_truth"]       # the only chart that uses truth, and it is labelled
    # --- fields feeding the ten new charts (all must be present in a real snapshot) ---
    assert len(est["position"]) == 3 and len(est["velocity_world"]) == 3 and len(est["angular_velocity_body"]) == 3
    assert all(isinstance(v, (int, float)) for v in est["euler_rad"])       # roll + pitch + yaw in one array
    sens = snap["sensors"]
    for name in ("imu", "dvl", "depth", "sonar", "compass"):             # sensor-health chart + source flags
        if name in sens:
            assert "health" in sens[name] and "forced" in sens[name]
            # the CURRENT sample (None while the sensor produces none) that the dashboard's reading column and the
            # sensor-fed charts follow, plus how old it is, so no stale number is ever displayed for a dead sensor
            assert "value" in sens[name] and "value_time" in sens[name] and "value_age_s" in sens[name], name
    imu = sens.get("imu") or {}
    assert "last_value" in imu and "last_value_time" in imu                # raw IMU acceleration series
    if imu.get("last_value") is not None:
        assert len(imu["last_value"]) == 6                                  # [accel xyz, gyro xyz]
        assert all(isinstance(v, float) for v in imu["last_value"])         # finite: invalid samples are never stored
    if imu.get("value") is not None:                                        # healthy IMU: a real, fresh reading
        assert len(imu["value"]) == 6 and all(isinstance(v, float) for v in imu["value"])
        assert imu["value_age_s"] is not None and imu["value_age_s"] <= 1.0
    assert "available" in snap["thrusters"]                                # thruster chart (or its honest empty state)
    if snap["thrusters"]["available"]:
        assert len(snap["thrusters"]["force_n"]) == snap["thrusters"]["count"]

"""Stage 3: CSV telemetry export, run history (baseline vs fault runs) and mission statistics.

Everything is produced by the REAL simulation process: the backend records each snapshot it reads, the run
summary comes from those rows plus the run's own result, and the endpoints only report what was measured.
"""
import csv
import io
import time

import numpy as np
import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from tests.test_dashboard_api import CHILD, Client, wait_for


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    cl = Client(create_app(mgr).test_client())
    cl.post("/api/environment/configure", {"realtime_factor": 4.0, "scenario": "mapping_validation"})
    yield cl, mgr
    mgr.shutdown()


def wait_running(cl, min_updates=2):
    def f():
        code, t = cl.get("/api/telemetry")
        if code != 200 or t["state"] != "RUNNING":
            return None
        s = t["snapshot"]
        return s if s["mapping"].get("update_count", 0) >= min_updates else None
    return wait_for(f, timeout=60)


def poll_recording(cl, seconds=1.2):
    """Read telemetry a few times so the backend recorder sees several snapshots (it records on read)."""
    end = time.time() + seconds
    while time.time() < end:
        cl.get("/api/telemetry?optional=1")
        time.sleep(0.15)


# ---------------------------------------------------------------- adaptive P controller (live)
def test_telemetry_reports_the_adaptive_controller_gains_live(env):
    """The dashboard's adaptive-gain panel is fed by this block: the effective proportional gain the controller
    actually used, its two multipliers, and the errors that produced them - all real numbers from the running
    controller, not recomputed in the browser."""
    cl, _ = env
    cl.post("/api/simulation/start")
    wait_running(cl, 2)

    def gains():
        code, t = cl.get("/api/telemetry")
        if code != 200 or t["state"] != "RUNNING":
            return None
        ctl = t["snapshot"].get("controller") or {}
        return ctl.get("adaptive") or None

    ad = wait_for(gains, timeout=60)
    assert ad["adaptive"] is True
    for k in ("kp_position_nominal", "kp_position_effective", "kp_position_multiplier", "schedule_multiplier",
              "self_tuning_gain", "kp_yaw_nominal", "kp_yaw_effective", "kp_yaw_multiplier",
              "position_error_m", "heading_error_rad"):
        assert isinstance(ad[k], float) and np.isfinite(ad[k]), (k, ad.get(k))
    # the controller's TRACKED gain history travels with the same block (the dashboard's Gain tracking panel and
    # its chart read it, and the CSV records it): counters, the mode it is in and why, and the tracked peak / mean.
    tr = ad["tracking"]
    for k in ("steps", "episodes", "reseeds"):
        assert isinstance(tr[k], int) and tr[k] >= 0, (k, tr.get(k))
    for k in ("elapsed_s", "ramp_s", "relax_s", "hold_s", "boosted_s", "at_cap_s", "peak_active",
              "peak_self_tuning", "mean_active"):
        assert isinstance(tr[k], float) and np.isfinite(tr[k]), (k, tr.get(k))
    assert tr["action"] in ("ramping", "relaxing", "holding", "off")
    assert tr["ramp_s"] + tr["relax_s"] + tr["hold_s"] == pytest.approx(tr["elapsed_s"], rel=1e-9)
    assert 0.0 <= tr["at_cap_s"] <= tr["boosted_s"] <= tr["elapsed_s"]
    assert 1.0 <= tr["mean_active"] <= tr["peak_active"] <= 3.0
    assert tr["peak_self_tuning"] <= tr["peak_active"]
    assert tr["closing_speed_mps"] is None or np.isfinite(tr["closing_speed_mps"])
    # the reported numbers are internally consistent with the controller's own definition
    assert ad["kp_position_effective"] == pytest.approx(ad["kp_position_nominal"] * ad["kp_position_multiplier"], rel=1e-9)
    assert ad["kp_position_multiplier"] == pytest.approx(max(ad["schedule_multiplier"], ad["self_tuning_gain"]), rel=1e-9)
    assert 1.0 <= ad["kp_position_multiplier"] <= 3.0                      # bounded by adaptive.gain_boost
    # ... and they are LIVE: the same block reports a different error as the vehicle moves
    e0 = ad["position_error_m"]

    def moved():
        a = gains()
        return a if a and abs(a["position_error_m"] - e0) > 1e-6 else None

    assert wait_for(moved, timeout=40) is not None


def test_telemetry_reports_the_path_follower_gains_under_autonomous_navigation(env):
    """Autonomous navigation has no adaptive P controller: the PATH FOLLOWER drives, and it publishes its own gains
    and live speed schedule in `controller.follower`, so the dashboard's gain panel and charts have real data in the
    mode the operator actually runs (they used to be empty there)."""
    cl, _ = env
    code, _ = cl.post("/api/environment/configure", {"scenario": "open_terrain", "seed": 3, "realtime_factor": 4.0,
                                                     "navigation_enabled": True, "navigation_goal": [30.0, 28.0, 5.0]})
    assert code == 200
    cl.post("/api/simulation/start")
    wait_running(cl, 2)

    def gains():
        code, t = cl.get("/api/telemetry")
        if code != 200 or t["state"] != "RUNNING":
            return None
        fol = (t["snapshot"].get("controller") or {}).get("follower")
        return fol if fol and fol.get("tracking", {}).get("steps", 0) > 0 else None

    fol = wait_for(gains, timeout=90)
    assert fol["available"] is True and fol["source"] == "path follower (Phase 5)"
    assert fol["mode"] in ("follow", "hold", "station_keep")
    for k in ("kv_n_per_mps", "kp_yaw_nm_per_rad", "kd_yaw_nm_per_rad_s", "kp_depth_mps_per_m",
              "ki_depth_n_per_m_s", "station_kp_per_s", "cruise_speed_mps", "speed_factor"):
        assert isinstance(fol[k], float) and np.isfinite(fol[k]) and fol[k] > 0.0, (k, fol.get(k))
    # the live schedule is bounded by cruise speed, and the commanded speed matches the factor reported
    assert 0.0 <= fol["speed_factor"] <= 1.0
    assert fol["desired_speed_mps"] == pytest.approx(fol["speed_factor"] * fol["cruise_speed_mps"], rel=1e-6, abs=1e-9)
    for k in ("goal_factor", "heading_factor", "sonar_factor"):
        assert fol[k] is None or (0.0 <= fol[k] <= 1.0), (k, fol[k])
    tr = fol["tracking"]
    assert tr["steps"] > 0 and tr["elapsed_s"] > 0.0
    assert tr["follow_s"] + tr["hold_s"] + tr["station_s"] == pytest.approx(tr["elapsed_s"], rel=1e-9)
    assert tr["full_speed_s"] + tr["slowed_s"] + tr["braked_s"] == pytest.approx(tr["elapsed_s"], rel=1e-9)
    assert 0.0 <= tr["speed_factor_mean"] <= 1.0 and 0.0 <= tr["speed_factor_min"] <= 1.0
    assert isinstance(tr["slows"], int) and tr["slows"] >= 0 and isinstance(tr["brakes"], int)


def test_csv_export_records_the_path_follower_gains_and_its_tracked_schedule(env):
    """The follower's gains are recorded as data too: a navigation run's exported CSV carries the loop gains, the
    live speed schedule (with its real factors) and the tracked schedule history."""
    cl, _ = env
    assert cl.post("/api/environment/configure", {"scenario": "open_terrain", "seed": 3, "realtime_factor": 4.0,
                                                  "navigation_enabled": True, "navigation_goal": [30.0, 28.0, 5.0]})[0] == 200
    cl.post("/api/simulation/start")
    wait_running(cl, 2)
    poll_recording(cl, 1.5)
    r = cl.c.get("/api/telemetry/export")
    assert r.status_code == 200 and r.mimetype == "text/csv"
    rows = list(csv.reader(io.StringIO(r.get_data(as_text=True))))
    header, data = rows[0], rows[1:]
    for col in ("follow_available", "follow_source", "follow_mode", "follow_frame", "follow_kv_n_per_mps",
                "follow_kp_yaw_nm_per_rad", "follow_kd_yaw_nm_per_rad_s", "follow_kp_depth_mps_per_m",
                "follow_ki_depth_n_per_m_s", "follow_station_kp_per_s", "follow_cruise_speed_mps",
                "follow_speed_factor", "follow_goal_factor", "follow_heading_factor", "follow_sonar_factor",
                "follow_desired_speed_mps", "follow_lookahead_m", "follow_cross_track_m", "follow_depth_error_m",
                "follow_depth_integral_n", "follow_track_steps", "follow_track_elapsed_s", "follow_track_follow_s",
                "follow_track_full_speed_s", "follow_track_slowed_s", "follow_track_braked_s",
                "follow_track_speed_factor_mean", "follow_track_slows", "follow_track_brakes"):
        assert col in header, col
    idx = {c: i for i, c in enumerate(header)}
    kv = [row[idx["follow_kv_n_per_mps"]] for row in data if row[idx["follow_kv_n_per_mps"]].strip()]
    assert kv and all(20.0 <= float(v) <= 25.0 for v in kv), kv[:3]     # the real configured velocity gain
    factors = [float(row[idx["follow_speed_factor"]]) for row in data if row[idx["follow_speed_factor"]].strip()]
    assert factors and all(0.0 <= v <= 1.0 for v in factors)
    steps = [int(row[idx["follow_track_steps"]]) for row in data if row[idx["follow_track_steps"]].strip()]
    assert steps and steps == sorted(steps) and steps[-1] > 0
    # the follower's rows really belong to a navigation run
    assert all(row[idx["follow_mode"]] in ("", "follow", "hold", "station_keep") for row in data)


# ---------------------------------------------------------------- CSV export
def test_csv_export_is_rejected_before_anything_is_recorded(env):
    cl, _ = env
    code, body = cl.get("/api/telemetry/export")
    assert code == 404 and body["ok"] is False and "start a simulation" in body["error"]


def test_csv_export_returns_recorded_real_telemetry(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    wait_running(cl, 3)
    poll_recording(cl, 1.5)
    r = cl.c.get("/api/telemetry/export")
    assert r.status_code == 200 and r.mimetype == "text/csv"
    assert "attachment" in r.headers.get("Content-Disposition", "")
    text = r.get_data(as_text=True)
    rows = list(csv.reader(io.StringIO(text)))
    header, data = rows[0], rows[1:]
    for col in ("sim_time", "run_status", "est_x", "est_y", "est_z", "est_roll_deg", "est_pitch_deg",
                "est_yaw_deg", "est_vx", "est_vy", "est_vz", "est_omega_x", "imu_health", "imu_forced",
                "imu_accel_x", "compass_health", "map_coverage_percent"):
        assert col in header, col
    assert len(data) >= 2
    idx = {c: i for i, c in enumerate(header)}
    times = [float(row[idx["sim_time"]]) for row in data]
    assert times == sorted(times) and times[-1] > times[0]              # monotonic simulation time
    for row in data:                                                    # strict numeric hygiene
        for cell in row:
            assert cell.strip().lower() not in ("nan", "inf", "-inf", "infinity")
    for row in data:                                                    # estimate columns come from real EKF samples
        assert row[idx["run_status"]] in ("RUNNING", "STOPPED", "COMPLETED", "ENDED_BEFORE_GOAL", "FAILED")
        if row[idx["est_x"]]:
            assert abs(float(row[idx["est_x"]])) < 1e6
    # export reflects NEW recordings (the recorder keeps working while the run continues)
    n0 = len(data)
    poll_recording(cl, 1.0)
    r2 = cl.c.get("/api/telemetry/export")
    assert r2.status_code == 200
    assert len(list(csv.reader(io.StringIO(r2.get_data(as_text=True))))) - 1 >= n0


def test_csv_export_records_the_adaptive_controller_gains(env):
    """The adaptive gains are recorded as data, not only drawn: the exported CSV carries the effective proportional
    gain and both multipliers for every row, so the adaptation can be analysed off-line."""
    cl, _ = env
    cl.post("/api/simulation/start")
    wait_running(cl, 3)
    poll_recording(cl, 1.5)
    r = cl.c.get("/api/telemetry/export")
    assert r.status_code == 200 and r.mimetype == "text/csv"
    rows = list(csv.reader(io.StringIO(r.get_data(as_text=True))))
    header, data = rows[0], rows[1:]
    for col in ("ctrl_type", "ctrl_adaptive", "ctrl_kp_position_nominal", "ctrl_kp_position_effective",
                "ctrl_kp_position_multiplier", "ctrl_gain_schedule", "ctrl_self_tuning_gain",
                "ctrl_kp_yaw_nominal", "ctrl_kp_yaw_effective", "ctrl_kp_yaw_multiplier",
                "ctrl_position_error_m", "ctrl_heading_error_rad", "ctrl_closing_speed_mps",
                "ctrl_adapt_action", "ctrl_adapt_reason", "ctrl_track_steps", "ctrl_track_elapsed_s",
                "ctrl_track_ramp_s", "ctrl_track_relax_s", "ctrl_track_hold_s", "ctrl_track_boosted_s",
                "ctrl_track_at_cap_s", "ctrl_track_peak_active", "ctrl_track_peak_self_tuning",
                "ctrl_track_mean_active", "ctrl_track_episodes", "ctrl_track_reseeds"):
        assert col in header, col
    idx = {c: i for i, c in enumerate(header)}
    eff = [row[idx["ctrl_kp_position_effective"]] for row in data]
    mult = [row[idx["ctrl_kp_position_multiplier"]] for row in data]
    assert any(v.strip() for v in eff), "the adaptive gains were never recorded"
    nums = [float(v) for v in eff if v.strip()]
    assert nums and all(np.isfinite(n) for n in nums)
    assert min(nums) >= 1.0                                     # effective = nominal * multiplier >= nominal
    assert all(1.0 <= float(v) <= 3.0 for v in mult if v.strip())   # bounded by adaptive.gain_boost
    # The TRACKED gain history is recorded too, and it behaves like a tracked quantity: the step count and the
    # simulated adaptation time only grow within a run, the action is one of the controller's own modes, and the
    # tracked peak / mean stay inside the configured gain band.
    steps = [int(v) for v in (row[idx["ctrl_track_steps"]] for row in data) if v.strip()]
    assert steps and steps == sorted(steps) and steps[-1] > 0, "the tracked gain history was never recorded"
    elapsed = [float(v) for v in (row[idx["ctrl_track_elapsed_s"]] for row in data) if v.strip()]
    assert elapsed == sorted(elapsed) and elapsed[-1] > 0.0
    assert {row[idx["ctrl_adapt_action"]] for row in data} - {""} <= {"ramping", "relaxing", "holding", "off"}
    assert all(row[idx["ctrl_adapt_reason"]].strip().lower() not in ("nan", "none") for row in data)
    for col in ("ctrl_track_peak_active", "ctrl_track_peak_self_tuning", "ctrl_track_mean_active"):
        vals = [float(row[idx[col]]) for row in data if row[idx[col]].strip()]
        assert vals and all(1.0 <= v <= 3.0 for v in vals), (col, vals)
    peak = [float(row[idx["ctrl_track_peak_active"]]) for row in data if row[idx["ctrl_track_peak_active"]].strip()]
    assert peak == sorted(peak)                                 # the tracked peak never goes down


# ---------------------------------------------------------------- run history / comparison + mission stats
def test_run_history_lists_runs_with_mission_statistics_and_faults_for_comparison(env):
    cl, _ = env
    code, body = cl.get("/api/runs")
    assert code == 200 and body["runs"] == [] and body["recorded_rows"] == 0

    # baseline run: no faults
    cl.post("/api/simulation/start")
    wait_running(cl, 2)
    poll_recording(cl, 1.0)
    cl.post("/api/simulation/stop")
    runs = wait_for(lambda: cl.get("/api/runs")[1]["runs"], timeout=20)
    base = [r for r in runs if not r["faults"]]
    assert base, runs
    b = base[0]
    assert b["state"] == "STOPPED" and b["scenario"] == "mapping_validation"
    assert b["duration_sim_s"] is not None and b["duration_sim_s"] > 0
    assert b["distance_estimated_m"] is not None and b["distance_estimated_m"] > 0
    assert b["recorded_rows"] >= 2 and b["live"] is False
    assert b["run_id"] and b["navigation_enabled"] is False

    # fault run: inject a DVL failure, record it, stop
    cl.post("/api/simulation/start")
    wait_running(cl, 2)
    cl.post("/api/sensors/dvl/health", {"health": "FAILED"})
    wait_for(lambda: cl.get("/api/sensors")[1]["data"]["dvl"]["health"] == "FAILED", timeout=20)
    poll_recording(cl, 1.5)                          # capture rows where the health is FAILED + forced
    cl.post("/api/sensors/dvl/health", {"health": "HEALTHY"})
    poll_recording(cl, 1.0)
    cl.post("/api/simulation/stop")
    runs = wait_for(lambda: cl.get("/api/runs")[1]["runs"], timeout=20)
    assert len([r for r in runs if not r["live"]]) >= 2
    faulty = [r for r in runs if r["faults"]]
    assert faulty, [r["run_id"] for r in runs]
    fl = faulty[0]["faults"][0]
    assert fl["sensor"] == "dvl" and fl["health"] == "FAILED" and fl["source"] == "forced"
    assert fl["sim_time"] is not None
    # both sides of the comparison exist: a baseline AND a fault run
    assert any(not r["faults"] for r in runs)


def test_run_summary_reports_terminal_state_and_result_metrics_when_the_run_finishes(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    wait_running(cl, 2)
    poll_recording(cl, 0.8)
    cl.post("/api/simulation/stop")
    runs = wait_for(lambda: cl.get("/api/runs")[1]["runs"], timeout=20)
    assert runs and runs[0]["state"] == "STOPPED"
    assert runs[0]["recorded_rows"] >= 1
    # the current run id is reported so the page can highlight it
    code, body = cl.get("/api/runs")
    assert code == 200 and "current_run_id" in body and "recorded_rows" in body

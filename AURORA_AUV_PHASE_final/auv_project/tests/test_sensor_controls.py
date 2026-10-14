"""Stage 1: sensor health controls against the REAL running simulation.

Covers: compass parity (the dashboard offers a compass selector, so the API must accept it), forced vs automatic
health reporting, real consequences of DEGRADED/FAILED in the sensor -> EKF chain, recovery after a fault is
cleared, the RESET alias, and safe handling of invalid measurements (strict JSON: no NaN token ever reaches
the dashboard).
"""
import math
import time

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


def running_snapshot(cl, min_updates=2):
    def f():
        code, t = cl.get("/api/telemetry")
        if code != 200 or t["state"] != "RUNNING":
            return None
        s = t["snapshot"]
        return s if s["mapping"].get("update_count", 0) >= min_updates and s.get("sensors_enabled") else None
    return wait_for(f, timeout=60)


def sensor_entry(cl, name):
    code, body = cl.get("/api/sensors")
    assert code == 200, body
    return body["data"].get(name)


def wait_sensor(cl, name, **want):
    def f():
        s = sensor_entry(cl, name)
        return s if s and all(s.get(k) == v for k, v in want.items()) else None
    return wait_for(f, timeout=20)


def ekf_stats(cl):
    code, body = cl.get("/api/ekf")
    assert code == 200
    return body["data"]["stats"]


# ---------------------------------------------------------------- compass parity + recovery
def test_compass_health_control_is_accepted_and_a_cleared_fault_recovers(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    snap = running_snapshot(cl)
    assert "compass" in snap["sensors"], "compass is enabled in configs/config.yaml, so telemetry must list it"
    wait_sensor(cl, "compass", health="HEALTHY")
    assert sensor_entry(cl, "compass")["forced"] is None            # healthy + NOT operator-forced

    code, r = cl.post("/api/sensors/compass/health", {"health": "FAILED"})
    assert code == 200 and r["ok"] and "FAILED" in r["detail"]      # the mismatch fix: compass IS accepted
    wait_sensor(cl, "compass", health="FAILED", forced="FAILED")
    u0 = ekf_stats(cl)["compass_updates"]                           # failed -> EKF receives no heading updates
    time.sleep(1.2)
    assert ekf_stats(cl)["compass_updates"] <= u0 + 1

    code, r = cl.post("/api/sensors/compass/health", {"health": "HEALTHY"})
    assert code == 200 and "cleared" in r["detail"]
    wait_sensor(cl, "compass", health="HEALTHY")
    assert sensor_entry(cl, "compass")["forced"] is None            # forced flag cleared with the fault
    u1 = ekf_stats(cl)["compass_updates"]                           # recovery: heading aiding resumes
    wait_for(lambda: ekf_stats(cl)["compass_updates"] > u1, timeout=20)


def test_reset_alias_clears_an_injected_fault_and_validation_is_unchanged(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    code, _ = cl.post("/api/sensors/imu/health", {"health": "FAILED"})
    assert code == 200
    wait_sensor(cl, "imu", health="FAILED", forced="FAILED")
    code, r = cl.post("/api/sensors/imu/health", {"health": "RESET"})    # explicit "reset to healthy" option
    assert code == 200 and r["ok"]
    wait_sensor(cl, "imu", health="HEALTHY")
    assert sensor_entry(cl, "imu")["forced"] is None
    assert cl.post("/api/sensors/magnetometer/health", {"health": "FAILED"})[0] == 404
    assert cl.post("/api/sensors/compass/health", {"health": "BROKEN"})[0] == 400


def test_sensor_health_endpoint_refuses_to_act_when_nothing_is_running(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    cl = Client(create_app(mgr).test_client())
    try:
        for name in ("imu", "dvl", "depth", "sonar", "compass"):        # every configured sensor incl. compass
            code, body = cl.post(f"/api/sensors/{name}/health", {"health": "FAILED"})
            assert code == 409 and body["ok"] is False, name
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------- DEGRADED / FAILED consequences + recovery
def test_degraded_still_aids_the_ekf_failed_stops_it_and_clearing_recovers(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)

    # DEGRADED: forced flag set, measurements keep flowing (inflated noise in the sensor model), EKF keeps aiding
    code, r = cl.post("/api/sensors/dvl/health", {"health": "DEGRADED"})
    assert code == 200
    wait_sensor(cl, "dvl", health="DEGRADED", forced="DEGRADED")
    d0 = ekf_stats(cl)["dvl_updates"]
    time.sleep(1.0)
    d1 = ekf_stats(cl)["dvl_updates"]
    assert d1 > d0, "DEGRADED must degrade quality, not silence the sensor"
    v0 = sensor_entry(cl, "dvl")["valid"]

    # FAILED: measurements become invalid, the EKF stops receiving DVL, dropouts are counted
    code, r = cl.post("/api/sensors/dvl/health", {"health": "FAILED"})
    assert code == 200 and "FAILED" in r["detail"]
    wait_sensor(cl, "dvl", health="FAILED", forced="FAILED")
    f0 = ekf_stats(cl)["dvl_updates"]
    i0 = sensor_entry(cl, "dvl")["invalid"]
    time.sleep(1.2)
    assert ekf_stats(cl)["dvl_updates"] <= f0 + 1
    assert sensor_entry(cl, "dvl")["invalid"] > i0

    # clearing the fault: forced flag drops, health returns to HEALTHY, valid samples and EKF aiding resume
    code, r = cl.post("/api/sensors/dvl/health", {"health": "HEALTHY"})
    assert code == 200 and "cleared" in r["detail"]
    wait_sensor(cl, "dvl", health="HEALTHY")
    assert sensor_entry(cl, "dvl")["forced"] is None
    wait_for(lambda: sensor_entry(cl, "dvl")["valid"] > v0, timeout=20)
    u0 = ekf_stats(cl)["dvl_updates"]
    wait_for(lambda: ekf_stats(cl)["dvl_updates"] > u0, timeout=20)


def test_failed_imu_is_coasted_safely_and_no_invalid_value_reaches_the_dashboard(env):
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    wait_sensor(cl, "imu", health="HEALTHY")
    assert sensor_entry(cl, "imu")["last_value_time"] is not None     # a valid IMU reading was recorded

    cl.post("/api/sensors/imu/health", {"health": "FAILED"})
    wait_sensor(cl, "imu", health="FAILED", forced="FAILED")
    time.sleep(1.0)
    est = cl.get("/api/ekf")[1]["data"]
    assert est["filter_status"] == "COASTING" or est["stats"]["imu_missing"] > 0     # safe fallback, never a bad update
    assert est["available"] is True                                              # the estimate survives the outage
    entry = sensor_entry(cl, "imu")
    assert entry["last_value"] is None or all(isinstance(v, float) for v in entry["last_value"])
    # strict_loads in the client already rejects NaN/Infinity tokens in every payload above

    t0 = entry["last_value_time"]
    cl.post("/api/sensors/imu/health", {"health": "HEALTHY"})
    wait_sensor(cl, "imu", health="HEALTHY")
    wait_for(lambda: (sensor_entry(cl, "imu") or {}).get("last_value_time") not in (None, t0), timeout=20)


# ---------------------------------------------------------------- live readings published per sensor
def test_every_sensor_publishes_its_live_reading_and_a_failed_one_publishes_zeros(env):
    """The dashboard's reading column is fed by the sensor's CURRENT sample: it changes while the sensor is healthy;
    while FAILED the sensor still samples but its output is stuck at 0.0, so the published reading is exactly zero
    and stays zero (valid=False, so the EKF ignores it); clearing the fault brings the real readings back.
    The 105-ray sonar is summarised by the sonar section instead of a raw value array."""
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)

    def value_of(name):
        return (sensor_entry(cl, name) or {}).get("value")

    for name in ("imu", "dvl", "depth", "compass"):
        first = wait_for(lambda: value_of(name) or None, timeout=20)
        assert all(isinstance(v, float) and math.isfinite(v) for v in first), (name, first)
        wait_for(lambda: value_of(name) != first, timeout=20)        # dynamic: the reading really keeps changing
        e = sensor_entry(cl, name)
        assert e["value_age_s"] is not None and e["value_age_s"] <= 1.0, (name, e["value_age_s"])
        assert e["value_time"] is not None and e["last_value_time"] is not None, name
        assert e["zero"] is False and e["stopped"] is False

    for name in ("imu", "dvl", "depth", "compass"):
        assert cl.post(f"/api/sensors/{name}/health", {"health": "FAILED"})[0] == 200
        wait_sensor(cl, name, health="FAILED", forced="FAILED", zeroed=True)
        e = wait_for(lambda: (lambda x: x if x and x.get("zero") and x.get("value") else None)(sensor_entry(cl, name)), timeout=20)
        assert e["value"] and all(v == 0.0 for v in e["value"]), (name, e["value"])   # 0.0 on every channel
        assert e["invalid"] > 0 and e["stopped"] is False                            # a real (invalid) sample
        t0 = e["value_time"]
        time.sleep(0.6)
        e2 = sensor_entry(cl, name)
        assert e2["value"] and all(v == 0.0 for v in e2["value"]), name              # ... and it stays 0.0
        assert e2["value_time"] > t0, (name, "a failed sensor still outputs a sample every cycle")
        cl.post(f"/api/sensors/{name}/health", {"health": "HEALTHY"})
        wait_sensor(cl, name, health="HEALTHY")
        wait_for(lambda: value_of(name) and any(v != 0.0 for v in value_of(name)), timeout=20)

    # sonar: the summary in the sonar section must carry nothing while the scan is failed
    assert cl.post("/api/sensors/sonar/health", {"health": "FAILED"})[0] == 200
    wait_sensor(cl, "sonar", health="FAILED", forced="FAILED")
    so = wait_for(lambda: (lambda s: s if s and not s["scan_valid"] and not s["valid_rays"] else None)(
        cl.get("/api/sonar")[1]["data"]), timeout=20)
    assert so["min_valid_range_m"] is None and so["max_valid_range_m"] is None
    assert so["hit_rays"] == 0 and so["valid_rays"] == 0                    # 0 hits: a 0.0 m range would be a collision
    assert all(r is None for r in so["range_m"]) and so["invalid_rays"] == so["n_rays"]
    assert cl.post("/api/sensors/sonar/health", {"health": "HEALTHY"})[0] == 200
    cl.get("/api/sonar")                                                      # poll once more before waiting
    wait_for(lambda: cl.get("/api/sonar")[1]["data"]["valid_rays"] > 0, timeout=20)


# ---------------------------------------------------------------- discarded / link cut / every sensor stopped
def test_discarded_and_link_cut_stop_the_values_and_all_discarded_delivers_nothing(env):
    """DISCARDED and LINK CUT produce no sample at all: the sensor's reading stops changing (same value, same
    timestamp, no new samples) while the run keeps going, and clearing the state resumes sampling. With EVERY
    configured sensor discarded / link cut the sensor system delivers nothing, which the snapshot reports as
    sensors_all_stopped (that is what makes the dashboard show no sensor values)."""
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    wait_sensor(cl, "depth", health="HEALTHY")
    v0 = wait_for(lambda: (sensor_entry(cl, "depth") or {}).get("value") or None, timeout=20)

    for state, label in (("DISCARDED", "discarded"), ("LINK_CUT", "link cut")):
        assert cl.post("/api/sensors/depth/health", {"health": state})[0] == 200
        e = wait_sensor(cl, "depth", forced=state, stopped=True)
        assert e["health"] == state
        s0, t0, v0 = e["samples"], e["last_time"], e["value"]
        time.sleep(1.2)
        e2 = sensor_entry(cl, "depth")
        assert e2["samples"] == s0, (label, "no new sample may be taken while discarded / link cut")
        assert e2["value"] == v0 and e2["value_time"] == e["value_time"], (label, "the reading stays frozen")
        assert e2["last_time"] == t0, (label, "nothing arrives from the sensor")
        cl.post("/api/sensors/depth/health", {"health": "HEALTHY"})
        e3 = wait_sensor(cl, "depth", health="HEALTHY", forced=None)
        wait_for(lambda: (sensor_entry(cl, "depth") or {}).get("value") != v0, timeout=20)   # sampling resumed
        assert e3["stopped"] is False and e3["health"] == "HEALTHY"

    # every configured sensor discarded: the snapshot says so, and no measurement reaches the EKF any more
    names = [n for n in ("imu", "dvl", "depth", "compass", "sonar") if sensor_entry(cl, n)]
    for n in names:
        assert cl.post(f"/api/sensors/{n}/health", {"health": "DISCARDED"})[0] == 200
    def all_stopped():
        code, body = cl.get("/api/telemetry")
        s = body["snapshot"] if code == 200 else None
        return s if s and s.get("sensors_all_stopped") else None
    snap = wait_for(all_stopped, timeout=25)
    assert set(snap["sensors_stopped"]) == set(names)
    assert all(snap["sensors"][n]["stopped"] for n in names)
    pred0 = snap["estimate"]["stats"]["predictions"] if snap["estimate"].get("available") else 0
    time.sleep(1.0)
    snap2 = wait_for(all_stopped, timeout=25)
    if snap2["estimate"].get("available"):
        assert snap2["estimate"]["stats"]["predictions"] <= pred0 + 2, "no sensor: the EKF gets nothing to predict from"
    for n in names:                                   # clearing them all brings the readings back
        cl.post(f"/api/sensors/{n}/health", {"health": "HEALTHY"})
    wait_for(lambda: (lambda t: t if t and not t["snapshot"].get("sensors_all_stopped") else None)(
        {"snapshot": cl.get("/api/telemetry?optional=1")[1].get("snapshot") or {}}), timeout=25)


def test_degraded_keeps_publishing_values_but_with_the_degraded_noise_scale(env):
    """DEGRADED must NOT silence a sensor: readings keep arriving and keep changing (the model inflates its noise),
    which is the difference the dashboard shows between 'degraded' (numbers, worse) and 'failed' (no numbers)."""
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    assert cl.post("/api/sensors/depth/health", {"health": "DEGRADED"})[0] == 200
    wait_sensor(cl, "depth", health="DEGRADED", forced="DEGRADED")
    v0 = wait_for(lambda: (sensor_entry(cl, "depth") or {}).get("value"), timeout=20)
    assert v0 is not None and math.isfinite(v0[0])
    wait_for(lambda: (sensor_entry(cl, "depth") or {}).get("value") != v0, timeout=20)
    e = sensor_entry(cl, "depth")
    assert e["valid"] > 0 and e["value_age_s"] is not None and e["value_age_s"] <= 1.0


def test_suite_skips_discarded_and_link_cut_sensors_and_failed_outputs_zeros():
    """Model level (no simulation process): the suite samples every sensor normally, a FAILED one keeps producing
    samples that are stuck at zero, and a DISCARDED / LINK_CUT one produces NO sample at all - so nothing reaches
    the EKF, the mapper or the telemetry and the last reading can never change."""
    import numpy as np
    from src.sensors.health import SensorHealth
    from src.sensors.models import TruthSnapshot
    from src.sensors.suite import SensorSuite
    cfg = {"simulation": {"time_step": 0.01, "gravity": -9.8, "seabed_z": -15.0},
           "sensors": {"imu": {"enabled": True}, "depth": {"enabled": True}}}
    suite = SensorSuite(cfg)
    truth = TruthSnapshot(0.0, [0, 0, -5], (0, 0, 0, 1), [0, 0, 0], [0, 0, 0])

    assert set(suite.sensors) == {"imu", "depth", "dvl"}      # dvl is enabled by default when the config omits it
    out = suite.sample(truth)
    assert set(out) == set(suite.sensors) and out["imu"].valid

    suite.sensors["imu"].inject_fault("failed")
    truth.time += 0.11                                              # slow sensors (dvl/depth) run at 10 Hz
    out = suite.sample(truth)
    assert "imu" in out and not out["imu"].valid and out["imu"].health == SensorHealth.FAILED
    assert out["imu"].value.shape == (6,) and np.all(out["imu"].value == 0.0)      # 0.0 on every channel
    assert out["depth"].valid

    for state, health in (("discarded", SensorHealth.DISCARDED), ("link_cut", SensorHealth.LINK_CUT)):
        suite.sensors["imu"].inject_fault(state)
        truth.time += 0.11
        out = suite.sample(truth)
        assert "imu" not in out, (state, "a discarded / link-cut sensor must not produce a sample")
        assert "depth" in out
        assert suite.sensors["imu"].health_monitor.health == health
        assert suite.sensors["imu"].health_monitor.stopped is True

    suite.sensors["imu"].inject_fault(None)
    truth.time += 0.11
    out = suite.sample(truth)
    assert "imu" in out and out["imu"].valid and out["imu"].health == SensorHealth.HEALTHY


def test_compass_model_degraded_and_failed_have_real_consequences_and_clear():
    """Model-level (no simulation process): DEGRADED inflates noise, FAILED outputs a stuck-at-zero sample, clearing recovers."""
    import numpy as np
    from src.sensors.compass import Compass, CompassConfig
    from src.sensors.health import SensorHealth
    from src.sensors.models import TruthSnapshot
    truth = TruthSnapshot(0.0, [0, 0, -5], (0, 0, 0, 1), [0, 0, 0], [0, 0, 0])
    c = Compass(CompassConfig(noise_std=0.01, bias=0.0, dropout_probability=0.0, seed=7))
    healthy = [c.measure(truth) for _ in range(50)]
    assert all(m.valid and m.health == SensorHealth.HEALTHY for m in healthy)
    c.inject_fault("degraded")
    m = c.measure(truth)
    assert m.valid and m.health == SensorHealth.DEGRADED and m.covariance[0, 0] > 0.01 ** 2 * 20   # x25 noise
    c.inject_fault("failed")
    m = c.measure(truth)
    assert not m.valid and m.health == SensorHealth.FAILED and m.value[0] == 0.0      # stuck at zero, not truth
    c.inject_fault(None)                                             # clear the injected fault
    m = c.measure(truth)
    assert m.valid and m.health == SensorHealth.HEALTHY and np.isfinite(m.value[0])


# ---------------------------------------------------------------- operator-facing state names
# (ENABLED / DISABLED / FAILED / UNHEALTHY) applied live to the RUNNING sensor
def test_operator_state_names_enabled_disabled_unhealthy_failed_are_all_real(env):
    """The four operator-facing names act on the RUNNING sensor at once and really change its output:
    DISABLED -> no sample at all (frozen reading), ENABLED -> normal output again, UNHEALTHY -> degraded
    (samples keep coming, still valid), FAILED -> a real sample stuck at 0.0 (rejected by the EKF)."""
    cl, _ = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    wait_sensor(cl, "dvl", health="HEALTHY")
    wait_for(lambda: (sensor_entry(cl, "dvl") or {}).get("value") or None, timeout=20)
    v0 = sensor_entry(cl, "dvl")["valid"]

    # DISABLED (== DISCARDED): no output at all, so the readings stop changing
    code, r = cl.post("/api/sensors/dvl/health", {"health": "DISABLED"})
    assert code == 200 and r["ok"], r
    e = wait_sensor(cl, "dvl", forced="DISCARDED", stopped=True)
    assert e["health"] == "DISCARDED"
    s0 = e["samples"]
    time.sleep(0.8)
    assert sensor_entry(cl, "dvl")["samples"] == s0, "a DISABLED sensor must produce no sample"

    # ENABLED (== HEALTHY, clears the fault): sampling and the real readings resume
    code, r = cl.post("/api/sensors/dvl/health", {"health": "ENABLED"})
    assert code == 200 and r["ok"], r
    wait_sensor(cl, "dvl", health="HEALTHY", forced=None)
    wait_for(lambda: sensor_entry(cl, "dvl")["samples"] > s0, timeout=20)
    wait_for(lambda: sensor_entry(cl, "dvl")["valid"] > v0, timeout=20)

    # UNHEALTHY (== DEGRADED): noisier but still valid and still arriving (the EKF keeps using it)
    code, r = cl.post("/api/sensors/dvl/health", {"health": "UNHEALTHY"})
    assert code == 200 and r["ok"], r
    e = wait_sensor(cl, "dvl", health="DEGRADED", forced="DEGRADED")
    assert e["stopped"] is False and e["zeroed"] is False
    u0 = e["valid"]
    wait_for(lambda: sensor_entry(cl, "dvl")["valid"] > u0, timeout=20)

    # FAILED: a real sample every cycle whose output is stuck at 0.0 (the EKF rejects it). Wait for the tracker's
    # own `zero` flag (the newest sample really is a stuck-at-zero output), not just the forced-state label.
    assert cl.post("/api/sensors/dvl/health", {"health": "FAILED"})[0] == 200
    wait_sensor(cl, "dvl", health="FAILED", forced="FAILED")
    e = wait_for(lambda: (lambda x: x if x and x.get("zero") and x.get("value") else None)(sensor_entry(cl, "dvl")), timeout=20)
    assert all(v == 0.0 for v in e["value"]), e["value"]

    # ENABLED clears it again -> real readings come back (recovery)
    assert cl.post("/api/sensors/dvl/health", {"health": "ENABLED"})[0] == 200
    wait_sensor(cl, "dvl", health="HEALTHY", forced=None)
    wait_for(lambda: sensor_entry(cl, "dvl")["value"] and any(v != 0.0 for v in sensor_entry(cl, "dvl")["value"]), timeout=20)


def test_invalid_state_name_is_rejected_and_aliases_stay_canonical(tmp_path):
    """The API still rejects an unknown state; the four operator names resolve to the canonical states so the rest
    of the pipeline (EKF / mapping / telemetry) never has to know the aliases exist."""
    from src.sensors.health import canonical_state
    assert canonical_state("ENABLED") == "HEALTHY" and canonical_state("enabled") == "HEALTHY"
    assert canonical_state("DISABLED") == "DISCARDED" and canonical_state("UNHEALTHY") == "DEGRADED"
    assert canonical_state("FAILED") == "FAILED" and canonical_state("RESET") == "HEALTHY"
    assert canonical_state("BROKEN") == "BROKEN"
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    cl = Client(create_app(mgr).test_client())
    try:
        cl.post("/api/simulation/start")
        running_snapshot(cl)
        assert cl.post("/api/sensors/imu/health", {"health": "BROKEN"})[0] == 400
    finally:
        mgr.shutdown()


# ---------------------------------------------------------------- safety stop: every sensor failed/unhealthy
def _configured_sensor_names(cl):
    return [n for n in ("imu", "dvl", "depth", "compass", "sonar") if sensor_entry(cl, n)]


def _run_stopped(mgr, timeout=45):
    return wait_for(lambda: (lambda s: s if s["state"] in ("STOPPED", "COMPLETED", "ENDED_BEFORE_GOAL", "FAILED")
                            and not mgr.alive else None)(mgr.status()), timeout=timeout)


def test_every_sensor_failed_stops_the_simulation_and_the_robot(env):
    """With EVERY configured sensor FAILED the vehicle has no usable navigation input, so the run stops itself:
    the robot is commanded to stop, the process exits with STOPPED, and the final telemetry says why."""
    cl, mgr = env
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    names = _configured_sensor_names(cl)
    assert names, "the run must have configured sensors"
    for n in names:
        code, r = cl.post(f"/api/sensors/{n}/health", {"health": "FAILED"})
        assert code == 200 and r["ok"], (n, r)
    st = _run_stopped(mgr)
    assert st["state"] == "STOPPED", st
    snap = mgr.state_snapshot() or {}
    safety = snap.get("safety") or {}
    assert safety.get("stopped") is True and "failed/unhealthy" in (safety.get("reason") or ""), safety
    assert any(e.get("cmd") == "safety_stop" for e in snap.get("command_log", [])), snap.get("command_log")
    # not a crash: a fresh start rebuilds the world/sensors and runs again
    code, out = cl.post("/api/simulation/start")
    assert code == 202, out
    running_snapshot(cl)


def test_every_sensor_unhealthy_also_stops_the_run_but_disabled_does_not(env):
    """UNHEALTHY (degraded) on every sensor is also a stop; DISABLED (no samples: the EKF simply coasts) is not."""
    cl, mgr = env
    # DISABLED everywhere must NOT stop the run (the existing contract: no sample arrives, the EKF coasts)
    cl.post("/api/simulation/start")
    running_snapshot(cl)
    names = _configured_sensor_names(cl)
    for n in names:
        assert cl.post(f"/api/sensors/{n}/health", {"health": "DISABLED"})[0] == 200
    time.sleep(2.0)
    assert mgr.alive and mgr.status()["state"] in ("RUNNING", "STALE"), mgr.status()
    for n in names:                                   # clear them again
        assert cl.post(f"/api/sensors/{n}/health", {"health": "ENABLED"})[0] == 200
    wait_for(lambda: not (mgr.state_snapshot() or {}).get("sensors_all_stopped"), timeout=25)

    # UNHEALTHY everywhere stops the run
    for n in names:
        assert cl.post(f"/api/sensors/{n}/health", {"health": "UNHEALTHY"})[0] == 200
    st = _run_stopped(mgr)
    assert st["state"] == "STOPPED", st
    snap = mgr.state_snapshot() or {}
    assert (snap.get("safety") or {}).get("stopped") is True
    assert "DEGRADED" in ((snap.get("safety") or {}).get("reason") or "")

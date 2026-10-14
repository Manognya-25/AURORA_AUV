"""Assemble the dashboard's live snapshot from REAL simulation components (no numbers invented here).

Sections and their sources
  estimate      EKF EstimatedState (+ EKF stats / last-update times)               OPERATIONAL
  sensors       the Measurement objects the sensor suite produced (health, validity, counts)
  sonar         the latest SonarMeasurement (per-ray status / range; invalid rays are null)
  mapping, cost_map, sonar_mapping   MappingSystem.telemetry()
  thrusters, dynamics, control       AUV thruster array / allocator / controller command
  environment   Environment.describe()  (generation parameters - ground-truth world description)
  navigation    AutonomousNavigator.telemetry() + path payload (Phase 5)                 OPERATIONAL (EKF/map/sonar only)
  debug_ground_truth  PyBullet truth + estimation error                              EVALUATION ONLY
  performance   wall-clock timings measured by the run loop
This module imports neither PyBullet nor the simulation package.
"""
from __future__ import annotations

import time

import numpy as np

from src.sensors.health import STOPPED_STATES, SensorHealth


class SensorStatsTracker:
    """Counts what each sensor actually delivered (per measurement object seen).

    Value fields, deliberately different - they are what the dashboard displays for each operator state:
      value / value_time            what the sensor output in its MOST RECENT sample. None when that sample carried
                                    no output at all (dropout / out of range / blanked). A FAILED sensor really
                                    outputs 0.0 on every sample (stuck at zero), so `value` is 0.0 with `zero`=True
                                    and the dashboard shows 0.00 that never changes. A DISCARDED / LINK_CUT sensor
                                    produces no sample at all, so `value` keeps the last one it did output: the
                                    readings stop changing (the forced state explains why).
      last_value / last_value_time  the most recent VALID reading. Never overwritten by an invalid sample, so a
                                    history chart can keep the real samples it already has and simply stop plotting
                                    while the sensor is failed (no fabricated or carried-over numbers).
      zero                          True while the newest sample is a stuck-at-zero output (injected FAILED).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.stats: dict = {}

    def observe(self, measurements: dict) -> None:
        for name, m in measurements.items():
            s = self.stats.setdefault(name, {"samples": 0, "valid": 0, "invalid": 0, "dropouts": 0,
                                             "consecutive_invalid": 0, "last_time": None, "last_valid_time": None,
                                             "health": "HEALTHY", "last_valid": None, "zero": False,
                                             "last_value": None, "last_value_time": None,
                                             "value": None, "value_time": None})
            s["samples"] += 1
            s["last_time"] = float(m.timestamp)
            s["health"] = SensorHealth(m.health).name
            s["last_valid"] = bool(m.valid)
            arr = _small_finite_vector(m.value)
            if m.valid:
                s["valid"] += 1
                s["consecutive_invalid"] = 0
                s["last_valid_time"] = float(m.timestamp)
                # Keep the last VALID raw reading (small vectors only; the 105-ray sonar stays out of the
                # snapshot). Never overwritten by an invalid sample, so the dashboard can chart real sensor
                # output and simply stop plotting while a sensor is failed (no fabricated values).
                if arr is not None:
                    s["last_value"] = [float(v) for v in arr]
                    s["last_value_time"] = float(m.timestamp)
                    s["value"] = s["last_value"]                  # the CURRENT output of the sensor
                    s["value_time"] = s["last_value_time"]
                s["zero"] = False
            else:
                s["invalid"] += 1
                s["dropouts"] += 1
                s["consecutive_invalid"] += 1
                if arr is not None and not np.any(arr):
                    # Stuck-at-zero output (injected FAILED): this 0.0 IS the sensor's real sample, so it is
                    # published as the current reading - it stays 0.0 and never changes. valid is False, so the
                    # EKF still rejects it (a stuck sensor is a fault, not a measurement of zero motion).
                    s["value"] = [0.0] * arr.size
                    s["value_time"] = float(m.timestamp)
                    s["zero"] = True
                else:
                    s["value"] = None                                 # this sample carries no output (NaN by model)
                    s["value_time"] = None
                    s["zero"] = False

    def snapshot(self, configured: list, now: float) -> dict:
        out = {}
        for name in configured:
            s = self.stats.get(name)
            if s is None:
                out[name] = {"configured": True, "samples": 0, "valid": 0, "invalid": 0, "dropouts": 0,
                             "health": None, "last_valid": None, "age_s": None, "zero": False,
                             "last_value": None, "last_value_time": None,
                             "value": None, "value_time": None, "value_age_s": None}
            else:
                out[name] = {"configured": True, **s,
                             "age_s": None if s["last_time"] is None else now - s["last_time"],
                             # age of the CURRENT reading: the dashboard marks the reading frozen once this grows
                             "value_age_s": None if s["value_time"] is None else now - s["value_time"]}
        return out


def _ekf_status(ekf, est, now: float, imu_gap: bool) -> dict:
    """Filter status derived from what the EKF actually did (not a label made up for the UI)."""
    ages = {}
    for k, t in ekf.last_update_time.items():
        ages[k] = None if t is None else float(now - t)
    stale = [k for k, a in ages.items() if a is None or a > 2.0]
    if imu_gap:
        status = "COASTING"
    elif stale:
        status = "DEAD_RECKONING"
    else:
        status = "NOMINAL"
    return {"filter_status": status, "last_aiding_age_s": ages, "stats": dict(ekf.stats)}


def _small_finite_vector(value) -> np.ndarray | None:
    """A sensor's raw output as a small finite vector (None for the 105-ray sonar or non-numeric values)."""
    try:
        arr = np.asarray(value, float).ravel()
    except (TypeError, ValueError):
        return None
    return arr if 0 < arr.size <= 6 and np.all(np.isfinite(arr)) else None


def sonar_section(meas, health_stats: dict | None) -> dict:
    if meas is None:
        return {"available": False}
    status = meas.ray_status()
    rng = np.asarray(meas.value, float)
    ranges = [float(r) if st == 1 and np.isfinite(r) else None for r, st in zip(rng, status)]
    hits = rng[status == 1]
    return {
        "available": True, "timestamp": float(meas.timestamp), "scan_valid": bool(meas.valid),
        "health": SensorHealth(meas.health).name,
        "n_rays": int(len(status)), "valid_rays": int(np.sum(status >= 0)), "invalid_rays": int(np.sum(status < 0)),
        "hit_rays": int(np.sum(status == 1)), "no_return_rays": int(np.sum(status == 0)),
        "min_valid_range_m": None if len(hits) == 0 else float(hits.min()),
        "max_valid_range_m": None if len(hits) == 0 else float(hits.max()),
        "max_range_m": float(meas.max_range), "min_range_m": float(meas.min_range),
        "azimuth_deg": np.rad2deg(meas.azimuths).round(3).tolist(),
        "elevation_deg": np.rad2deg(meas.elevations).round(3).tolist(),
        "status": status.tolist(),                 # 1 hit, 0 valid no-return, -1 invalid (dropout/blanked/failed)
        "range_m": ranges,                         # hit range, null otherwise
        "mount_position_m": None if meas.mount_position is None else np.asarray(meas.mount_position).tolist(),
    }


def thruster_section(auv, control: dict | None) -> tuple[dict, dict]:
    ctl = {"available": control is not None, **(control or {})}
    if auv is None:
        return ({"available": False, "reason": "no vehicle dynamics in this run (stand-in test vehicle)"},
                {"mode": "unavailable", "control": ctl})
    if getattr(auv, "thrusters", None) is None:
        return ({"available": False, "reason": "dynamics.model is 'legacy': no thruster layer (world-frame force model)"},
                {"mode": "legacy", "control": ctl})
    th, alloc = auv.thrusters, auv.last_allocation
    n = th.n
    section = {
        "available": True, "count": n,
        "command": np.asarray(th.last_command).tolist(),          # normalised, post-allocation
        "force_n": np.asarray(th.force).tolist(),                 # actual thrust after lag/noise/faults
        "max_thrust_n": np.asarray(th.max_thrust).tolist(),
        "fault": [str(getattr(f, "value", f)) for f in th.fault],
        "health": np.asarray(th.health()).tolist(),
        "allocator_available": np.asarray(auv.thruster_available).tolist(),
        "saturated": None if alloc is None else bool(alloc.saturated),
        "allocation_residual_norm": None if alloc is None else float(np.linalg.norm(alloc.residual)),
        "wrench_requested": None if alloc is None else np.asarray(alloc.achieved + alloc.residual).tolist(),
        "wrench_achieved": None if alloc is None else np.asarray(alloc.achieved).tolist(),
    }
    return section, {"mode": "6dof", "control": ctl}


def build_snapshot(*, step: int, sim_time: float, dt: float, status: str, dynamics_mode: str, estimation,
                   estimate, measurements: dict, sensor_tracker: SensorStatsTracker, mapping,
                   env_desc: dict, truth_position, truth_euler, auv, control: dict | None,
                   controller_info: dict, perf: dict, last_command_id: int, mapping_unavailable_reason: str | None,
                   trajectory_tail: list | None = None, navigation: dict | None = None,
                   navigation_path: dict | None = None) -> dict:
    now = float(sim_time)
    snap: dict = {
        "schema": 1, "wall_time": time.time(), "step": int(step), "sim_time": now, "time_step": float(dt),
        "run_status": status, "dynamics_mode": dynamics_mode, "last_command_id": int(last_command_id),
        "environment": env_desc, "controller": controller_info, "performance": perf,
    }
    cfg_sensors = list(estimation.suite.sensors.keys()) if estimation is not None else []
    snap["sensors"] = sensor_tracker.snapshot(cfg_sensors, now) if estimation is not None else {}
    if estimation is not None:
        # Report whether a state was MANUALLY FORCED by the operator or detected automatically from consecutive
        # invalid samples (HealthMonitor.forced is the single source of truth for that), and say what the forced
        # state really does to the output: `stopped` = no sample at all (DISCARDED / LINK_CUT), `zeroed` = a
        # stuck-at-zero sample (FAILED). The dashboard needs both to display the operator's fault correctly.
        stopped_names = []
        for name, entry in snap["sensors"].items():
            sensor_obj = estimation.suite.sensors.get(name)
            forced = getattr(getattr(sensor_obj, "health_monitor", None), "forced", None)
            mode = None if forced is None else SensorHealth(forced).name
            if mode is not None:
                # A DISCARDED / LINK_CUT sensor produces no sample, so the tracker has no fresh measurement to
                # read the state from: the operator-forced state IS the sensor's state right now (and the reason
                # its reading is frozen). Reporting the stale measured health here would contradict the panel.
                entry["health"] = mode
            entry["forced"] = mode
            entry["stopped"] = bool(mode in STOPPED_STATES)
            entry["zeroed"] = bool(mode == "FAILED")
            if entry["stopped"]:
                stopped_names.append(name)
        # Every configured sensor discarded / link cut: the sensor system delivers nothing at all, so the
        # dashboard shows no sensor values anywhere (and the EKF has no measurement to aid it).
        snap["sensors_stopped"] = stopped_names
        snap["sensors_stopped_count"] = len(stopped_names)
        snap["sensors_all_stopped"] = bool(snap["sensors"]) and len(stopped_names) == len(snap["sensors"])
    else:
        snap["sensors_stopped"] = []
        snap["sensors_stopped_count"] = 0
        snap["sensors_all_stopped"] = False
    snap["sensors_enabled"] = estimation is not None
    snap["sonar"] = sonar_section(estimation.latest.get("sonar") if estimation is not None else None, None)

    if estimate is not None:
        e = estimate.euler
        st = estimation.ekf
        imu_gap = estimation.last_measurements.get("imu") is None or not estimation.last_measurements["imu"].valid
        snap["estimate"] = {
            "available": True, "time": float(estimate.time),
            "position": estimate.position.tolist(), "depth": float(estimate.depth),
            "euler_rad": [float(e[0]), float(e[1]), float(e[2])],
            "velocity_world": estimate.velocity.tolist(), "velocity_body": estimate.velocity_body.tolist(),
            "speed": float(np.linalg.norm(estimate.velocity)),
            "angular_velocity_body": estimate.angular_velocity.tolist(),
            "position_std": estimate.position_std.tolist(), "velocity_std": estimate.velocity_std.tolist(),
            "attitude_std_rad": estimate.attitude_std.tolist(),
            "accel_bias": estimate.accel_bias.tolist(), "gyro_bias": estimate.gyro_bias.tolist(),
            "covariance_trace": float(np.trace(estimate.covariance)),
            **_ekf_status(st, estimate, float(estimate.time), imu_gap),
        }
    else:
        snap["estimate"] = {"available": False, "reason": "sensors/EKF disabled" if estimation is None else "no estimate yet"}

    if mapping is not None:
        snap.update(mapping.telemetry())
        snap["mapping_available"] = True
    else:
        snap["mapping_available"] = False
        snap["mapping"] = {"enabled": False, "unavailable_reason": mapping_unavailable_reason}

    snap["thrusters"], dyn = thruster_section(auv, control)
    snap["dynamics"] = dyn
    if truth_position is not None:
        tp = np.asarray(truth_position, float)
        gt = {"LABEL": "GROUND TRUTH - DEBUG/EVALUATION ONLY", "position": tp.tolist(),
              "euler_rad": None if truth_euler is None else [float(v) for v in truth_euler]}
        if estimate is not None:
            err = estimate.position - tp
            gt["estimate_error"] = err.tolist()
            gt["estimate_error_norm"] = float(np.linalg.norm(err))
        snap["debug_ground_truth"] = gt
    if trajectory_tail is not None:
        snap["trajectory_estimated"] = trajectory_tail
    # Phase 5: navigation section (operational: computed from the EKF estimate, the cost map and the sonar only)
    snap["navigation"] = navigation if navigation is not None else {"enabled": False}
    if navigation_path is not None:
        snap["navigation_path"] = navigation_path
    return snap

"""Flatten live telemetry snapshots into CSV rows, and summarise finished runs for the run history.

Everything here is derived from REAL snapshots produced by the simulation (src/simulation/telemetry_snapshot.py);
nothing is synthesised. Values that are unavailable in a run (no EKF, legacy dynamics without thrusters, stand-in
vehicle without collisions) are left empty in the CSV and None in the summary - never a fabricated zero.
"""
from __future__ import annotations

import csv
import io
from datetime import datetime, timezone

from src.sensors.health import STOPPED_STATES

HEALTH_SENSORS = ("imu", "dvl", "depth", "sonar", "compass")
# Every state that counts as a fault episode in the run history: degraded, failed (stuck-at-zero output) and the
# two states that stop the output entirely - discarded and link cut.
FAULT_STATES = ("DEGRADED", "FAILED") + tuple(STOPPED_STATES)

# Preferred column order; dynamic extras (e.g. thruster_force_N) are appended as they are first seen.
BASE_COLUMNS = [
    "run_id", "step", "sim_time", "wall_time", "run_status", "dynamics_mode",
    "ekf_available", "ekf_filter_status",
    "est_x", "est_y", "est_z", "est_depth",
    "est_roll_deg", "est_pitch_deg", "est_yaw_deg",
    "est_vx", "est_vy", "est_vz", "est_speed",
    "est_omega_x", "est_omega_y", "est_omega_z",
    "est_pos_std_x", "est_pos_std_y", "est_pos_std_z", "ekf_covariance_trace",
    "truth_x", "truth_y", "truth_z", "est_error_norm_m",
    "mapping_enabled", "map_updates", "map_coverage_percent",
    "nav_enabled", "nav_state", "nav_goal_distance_m", "nav_replans", "nav_avoidance",
    # Adaptive P controller (legacy Phase-4 mission): the EFFECTIVE proportional gains of the last command, so the
    # adaptation is recorded as data and not only drawn. Empty when the adaptive controller was not driving.
    "ctrl_type", "ctrl_adaptive", "ctrl_kp_position_nominal", "ctrl_kp_position_effective",
    "ctrl_kp_position_multiplier", "ctrl_gain_schedule", "ctrl_self_tuning_gain",
    "ctrl_kp_yaw_nominal", "ctrl_kp_yaw_effective", "ctrl_kp_yaw_multiplier",
    "ctrl_position_error_m", "ctrl_heading_error_rad", "ctrl_closing_speed_mps",
    # ... and the TRACKED gain history the controller keeps for itself: what the gain loop is doing (and why), how
    # much simulated time it has spent in each mode, and the tracked peak / mean multiplier.
    "ctrl_adapt_action", "ctrl_adapt_reason", "ctrl_track_steps", "ctrl_track_elapsed_s",
    "ctrl_track_ramp_s", "ctrl_track_relax_s", "ctrl_track_hold_s", "ctrl_track_boosted_s",
    "ctrl_track_at_cap_s", "ctrl_track_peak_active", "ctrl_track_peak_self_tuning", "ctrl_track_mean_active",
    "ctrl_track_episodes", "ctrl_track_reseeds",
    # Phase-5 PATH FOLLOWER gains (autonomous navigation: no adaptive P controller drives there). The loop gains
    # in the units they are applied in, the live speed schedule that shapes the command, and its tracked history.
    "follow_available", "follow_source", "follow_mode", "follow_frame",
    "follow_kv_n_per_mps", "follow_kp_yaw_nm_per_rad", "follow_kd_yaw_nm_per_rad_s",
    "follow_kp_depth_mps_per_m", "follow_ki_depth_n_per_m_s", "follow_station_kp_per_s",
    "follow_cruise_speed_mps", "follow_speed_factor", "follow_goal_factor", "follow_heading_factor",
    "follow_sonar_factor", "follow_desired_speed_mps", "follow_lookahead_m", "follow_cross_track_m",
    "follow_heading_error_rad", "follow_depth_error_m", "follow_depth_integral_n",
    "follow_track_steps", "follow_track_elapsed_s", "follow_track_follow_s", "follow_track_hold_s",
    "follow_track_station_s", "follow_track_full_speed_s", "follow_track_slowed_s", "follow_track_braked_s",
    "follow_track_speed_factor_now", "follow_track_speed_factor_mean", "follow_track_speed_factor_min",
    "follow_track_heading_factor_min", "follow_track_sonar_factor_min", "follow_track_slows", "follow_track_brakes",
    "sonar_health", "sonar_forced", "sonar_min_range_m", "sonar_valid_rays",
    "imu_health", "imu_forced", "imu_accel_x", "imu_accel_y", "imu_accel_z",
    "dvl_health", "dvl_forced", "dvl_velocity_x", "dvl_velocity_y", "dvl_velocity_z",
    "depth_health", "depth_forced", "depth_measurement_m",
    "compass_health", "compass_forced", "compass_heading_rad",
    "thruster_count", "collisions",
]


def _get(d, *keys):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def _deg(rad):
    return None if rad is None else rad * 57.29577951308232


def flatten_snapshot(snap: dict, run_id: str | None) -> dict:
    """One CSV row per published snapshot. None -> empty cell (missing data is never turned into 0)."""
    if not isinstance(snap, dict):
        return {}
    row: dict = {
        "run_id": run_id,
        "step": snap.get("step"),
        "sim_time": snap.get("sim_time"),
        "wall_time": snap.get("wall_time"),
        "run_status": snap.get("run_status"),
        "dynamics_mode": snap.get("dynamics_mode"),
    }
    est = _get(snap, "estimate") or {}
    if est.get("available"):
        pos, eul, vw, w = est.get("position"), est.get("euler_rad"), est.get("velocity_world"), est.get("angular_velocity_body")
        row["ekf_available"] = True
        row["ekf_filter_status"] = est.get("filter_status")
        if isinstance(pos, list) and len(pos) == 3:
            row["est_x"], row["est_y"], row["est_z"] = pos
        row["est_depth"] = est.get("depth")
        if isinstance(eul, list) and len(eul) == 3:
            row["est_roll_deg"], row["est_pitch_deg"], row["est_yaw_deg"] = (_deg(eul[0]), _deg(eul[1]), _deg(eul[2]))
        if isinstance(vw, list) and len(vw) == 3:
            row["est_vx"], row["est_vy"], row["est_vz"] = vw
        row["est_speed"] = est.get("speed")
        if isinstance(w, list) and len(w) == 3:
            row["est_omega_x"], row["est_omega_y"], row["est_omega_z"] = w
        std = est.get("position_std")
        if isinstance(std, list) and len(std) == 3:
            row["est_pos_std_x"], row["est_pos_std_y"], row["est_pos_std_z"] = std
        row["ekf_covariance_trace"] = est.get("covariance_trace")
    else:
        row["ekf_available"] = False

    gt = _get(snap, "debug_ground_truth") or {}
    tp, terr = gt.get("position"), gt.get("estimate_error")
    if isinstance(tp, list) and len(tp) == 3:
        row["truth_x"], row["truth_y"], row["truth_z"] = tp
    row["est_error_norm_m"] = gt.get("estimate_error_norm")

    mapping = _get(snap, "mapping") or {}
    row["mapping_enabled"] = mapping.get("enabled")
    row["map_updates"] = mapping.get("update_count")
    row["map_coverage_percent"] = mapping.get("coverage_percent")

    nav = _get(snap, "navigation") or {}
    row["nav_enabled"] = nav.get("enabled")
    if nav.get("enabled"):
        row["nav_state"] = nav.get("state")
        row["nav_goal_distance_m"] = nav.get("goal_distance_m")
        row["nav_replans"] = _get(nav, "replanning", "count")
        row["nav_avoidance"] = _get(nav, "avoidance", "count")

    # Adaptive P controller: the effective gains the controller actually used for the last command. Written from
    # the real `controller.adaptive` block, only when the run really reported it (a navigation run has none).
    ctl = _get(snap, "controller") or {}
    row["ctrl_type"] = ctl.get("type")
    adapt = ctl.get("adaptive")
    if isinstance(adapt, dict):
        row["ctrl_adaptive"] = adapt.get("adaptive")
        row["ctrl_kp_position_nominal"] = adapt.get("kp_position_nominal")
        row["ctrl_kp_position_effective"] = adapt.get("kp_position_effective")
        row["ctrl_kp_position_multiplier"] = adapt.get("kp_position_multiplier")
        row["ctrl_gain_schedule"] = adapt.get("schedule_multiplier")
        row["ctrl_self_tuning_gain"] = adapt.get("self_tuning_gain")
        row["ctrl_kp_yaw_nominal"] = adapt.get("kp_yaw_nominal")
        row["ctrl_kp_yaw_effective"] = adapt.get("kp_yaw_effective")
        row["ctrl_kp_yaw_multiplier"] = adapt.get("kp_yaw_multiplier")
        row["ctrl_position_error_m"] = adapt.get("position_error_m")
        row["ctrl_heading_error_rad"] = adapt.get("heading_error_rad")
        # The tracked gain history, flattened as it was published (empty when the run reported no tracking).
        track = adapt.get("tracking")
        if isinstance(track, dict):
            row["ctrl_adapt_action"] = track.get("action")
            row["ctrl_adapt_reason"] = track.get("reason")
            row["ctrl_closing_speed_mps"] = track.get("closing_speed_mps")
            row["ctrl_track_steps"] = track.get("steps")
            row["ctrl_track_elapsed_s"] = track.get("elapsed_s")
            row["ctrl_track_ramp_s"] = track.get("ramp_s")
            row["ctrl_track_relax_s"] = track.get("relax_s")
            row["ctrl_track_hold_s"] = track.get("hold_s")
            row["ctrl_track_boosted_s"] = track.get("boosted_s")
            row["ctrl_track_at_cap_s"] = track.get("at_cap_s")
            row["ctrl_track_peak_active"] = track.get("peak_active")
            row["ctrl_track_peak_self_tuning"] = track.get("peak_self_tuning")
            row["ctrl_track_mean_active"] = track.get("mean_active")
            row["ctrl_track_episodes"] = track.get("episodes")
            row["ctrl_track_reseeds"] = track.get("reseeds")

    # Path follower (Phase 5 autonomous navigation): the gains that were actually driving, in their own units.
    fol = ctl.get("follower")
    if isinstance(fol, dict) and fol.get("available"):
        row["follow_available"] = True
        row["follow_source"] = fol.get("source")
        row["follow_mode"] = fol.get("mode")
        row["follow_frame"] = fol.get("command_frame")
        for key in ("kv_n_per_mps", "kp_yaw_nm_per_rad", "kd_yaw_nm_per_rad_s", "kp_depth_mps_per_m",
                    "ki_depth_n_per_m_s", "station_kp_per_s", "cruise_speed_mps", "speed_factor", "goal_factor",
                    "heading_factor", "sonar_factor", "desired_speed_mps", "lookahead_m", "cross_track_m",
                    "heading_error_rad", "depth_error_m", "depth_integral_n"):
            row[f"follow_{key}"] = fol.get(key)
        ftrack = fol.get("tracking")
        if isinstance(ftrack, dict):
            for key in ("steps", "elapsed_s", "follow_s", "hold_s", "station_s", "full_speed_s", "slowed_s",
                        "braked_s", "speed_factor_now", "speed_factor_mean", "speed_factor_min",
                        "heading_factor_min", "sonar_factor_min", "slows", "brakes"):
                row[f"follow_track_{key}"] = ftrack.get(key)

    sonar = _get(snap, "sonar") or {}
    if sonar.get("available"):
        row["sonar_min_range_m"] = sonar.get("min_valid_range_m")
        row["sonar_valid_rays"] = sonar.get("hit_rays")

    sensors = _get(snap, "sensors") or {}
    for name in HEALTH_SENSORS:
        s = sensors.get(name)
        if not isinstance(s, dict):
            continue
        row[f"{name}_health"] = s.get("health")
        row[f"{name}_forced"] = s.get("forced")
        # The sensor's CURRENT sample (None when that sample was invalid, e.g. while the sensor is failed), so a
        # recorded row never carries a value the sensor did not output at that moment.
        val = s.get("value")
        if name == "imu" and isinstance(val, list) and len(val) >= 3:
            row["imu_accel_x"], row["imu_accel_y"], row["imu_accel_z"] = val[0], val[1], val[2]
        elif name == "dvl" and isinstance(val, list) and len(val) >= 3:
            row["dvl_velocity_x"], row["dvl_velocity_y"], row["dvl_velocity_z"] = val[0], val[1], val[2]
        elif name == "depth" and isinstance(val, list) and len(val) >= 1:
            row["depth_measurement_m"] = val[0]
        elif name == "compass" and isinstance(val, list) and len(val) >= 1:
            row["compass_heading_rad"] = val[0]

    thr = _get(snap, "thrusters") or {}
    if thr.get("available") and isinstance(thr.get("force_n"), list):
        forces = thr["force_n"]
        row["thruster_count"] = len(forces)
        for i, fv in enumerate(forces):
            row[f"thruster_force_{i}"] = fv

    collisions = _get(snap, "controller", "collisions")
    if collisions is None:
        collisions = gt.get("collisions")
    row["collisions"] = collisions
    return row


def _cell(v):
    if v is None:
        return ""
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, float):
        return "" if v != v else round(v, 6)          # NaN/inf -> empty (never a non-finite CSV token)
    return v


def rows_to_csv(rows: list, columns: list) -> str:
    buf = io.StringIO(newline="")
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(columns)
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in columns])
    return buf.getvalue()


def _episode_faults(rows: list) -> list:
    """Detect sensor-health transitions in the recorded rows: every entry in DEGRADED/FAILED starts an episode.
    `source` is 'forced' when the snapshot's forced flag matches the new state, otherwise 'automatic'."""
    faults = []
    prev: dict = {}
    for r in rows:
        for name in HEALTH_SENSORS:
            h = r.get(f"{name}_health")
            if not h:
                continue
            old = prev.get(name)
            if old is not None and old != h and h in FAULT_STATES:
                faults.append({"sensor": name, "health": h, "previous": old,
                               "sim_time": r.get("sim_time"),
                               "source": "forced" if r.get(f"{name}_forced") == h else "automatic"})
            prev[name] = h
    return faults


def build_run_summary(*, run_id: str, snapshot: dict, rows: list, result: dict, state: str | None) -> dict | None:
    """Mission statistics for one run, from its recorded rows + final snapshot + the run's own result
    (src/simulation/metrics.py) when the child reported one. Truth-derived figures are labelled evaluation."""
    snapshot = snapshot or {}
    result = result or {}
    metrics = result.get("metrics") or {}
    env = _get(snapshot, "environment") or {}
    last = rows[-1] if rows else {}

    duration = metrics.get("simulation_duration_seconds_simulated")
    if duration is None:
        duration = last.get("sim_time") or snapshot.get("sim_time")

    dist_est = None
    pts = [r for r in rows if r.get("est_x") is not None and r.get("est_y") is not None and r.get("est_z") is not None]
    for a, b in zip(pts, pts[1:]):
        dx, dy, dz = b["est_x"] - a["est_x"], b["est_y"] - a["est_y"], b["est_z"] - a["est_z"]
        dist_est = (dist_est or 0.0) + (dx * dx + dy * dy + dz * dz) ** 0.5

    collisions = metrics.get("collision_count")
    if collisions is None:
        collisions = last.get("collisions")

    goal_reached = result.get("goal_reached")
    if goal_reached is None:
        goal_reached = (_get(snapshot, "controller") or {}).get("goal_reached")
    if goal_reached is None and _get(snapshot, "navigation", "enabled"):
        goal_reached = _get(snapshot, "navigation", "state") == "GOAL_REACHED"

    nav = _get(snapshot, "navigation") or {}
    return {
        "run_id": run_id,
        "state": state or snapshot.get("run_status") or result.get("final_status"),
        "ended": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "scenario": env.get("scenario"),
        "seed": env.get("seed"),
        "dynamics_mode": snapshot.get("dynamics_mode"),
        "navigation_enabled": bool(nav.get("enabled")),
        "duration_sim_s": duration,
        "distance_estimated_m": None if dist_est is None else round(dist_est, 3),
        "distance_truth_m": metrics.get("trajectory_length_m"),
        "goal_reached": goal_reached,
        "final_goal_distance_m": nav.get("goal_distance_m"),
        "collisions": collisions,
        "coverage_percent": last.get("map_coverage_percent"),
        "replans": last.get("nav_replans") or (nav.get("replanning") or {}).get("count"),
        "avoidance": last.get("nav_avoidance") or (nav.get("avoidance") or {}).get("count"),
        "recorded_rows": len(rows),
        "faults": _episode_faults(rows),
        "live": False,
    }

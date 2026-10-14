"""Pybullet-free helpers shared by the simulation loop and its tests: config merge + dashboard commands.

Every command acts on a REAL component (the mapper, a sensor's fault injector, the pacing setting) and
returns (ok, detail) describing what actually happened; the dashboard shows that, not an assumption."""
from __future__ import annotations

import copy

from src.sensors.health import FORCED_STATES, canonical_state


def deep_merge(base: dict, over: dict | None) -> dict:
    """Return base updated recursively with over (used for dashboard-supplied config overrides)."""
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out



def _apply_nav_command(name: str, cmd: dict, mapping, navigator, state: dict) -> tuple[bool, str]:
    """Phase 5 navigation commands. They act on the REAL navigator; goals are validated against the live map."""
    if navigator is None:
        return False, "autonomous navigation is not enabled in this run (navigation.enabled: false)"
    now = float(state.get("sim_time", 0.0))
    if name == "nav_set_goal":
        goal = cmd.get("goal")
        if goal is None and all(k in cmd for k in ("x", "y")):
            goal = [cmd["x"], cmd["y"], cmd.get("z", float(navigator.goal[2]) if navigator.goal is not None else -5.0)]
        ok, why = navigator.check_goal(goal, mapping)
        if not ok:
            return False, why
        return navigator.set_goal(goal, now)
    if name == "nav_start":
        return navigator.start(now)
    if name == "nav_cancel":
        return navigator.cancel(now)
    if name == "nav_replan":
        return navigator.request_replan(now, "operator_request")
    if name == "nav_set_unknown_policy":
        return navigator.set_unknown_policy(str(cmd.get("policy", "")), now)
    return False, f"unknown command {name!r}"


NAV_COMMANDS = ("nav_set_goal", "nav_start", "nav_cancel", "nav_replan", "nav_set_unknown_policy")


def operator_z_to_world(depth_or_z: float) -> float:
    """The z an operator TYPES is a DEPTH in metres below the surface (positive down) -- the same unit the
    dashboard reports as "depth (m)" -- while the simulation works in world z, which is up. A depth of 7 m is
    therefore world z = -7 m.

    The magnitude is what is read, so an input that is already a world z (the older "z is depth, negative" form,
    e.g. -7) selects exactly the same point: -7 -> world z -7 (depth 7 m). That keeps every existing typed value
    meaning what it meant before while letting 7 work as well, and it makes an above-water goal unrepresentable
    instead of a hard error the operator has to decode.
    """
    return -abs(float(depth_or_z))


def operator_xyz_to_world(xyz) -> list:
    """An operator-typed [x, y, z] with z read as a depth (see operator_z_to_world). x and y are world coordinates."""
    x, y, z = (float(v) for v in xyz)
    return [x, y, operator_z_to_world(z)]


def resolve_mission_goal(cfg: dict) -> tuple:
    """The mission goal in BOTH control modes: `navigation.goal_position` when set (that is what the dashboard's
    goal input stores), otherwise `simulation.goal_position` - the same resolution the Phase-5 planner uses.

    The simulation loop uses this instead of reading `simulation.goal_position` directly, so a run with navigation
    DISABLED also ends at the goal the operator typed instead of silently flying to the config goal. Neither
    `simulation.goal_position` nor `simulation.waypoints` is rewritten by the dashboard, because the world
    generator derives the obstacle-free mission corridor from them.
    """
    nav_goal = (cfg.get("navigation") or {}).get("goal_position")
    raw = nav_goal if isinstance(nav_goal, (list, tuple)) and len(nav_goal) == 3 else cfg["simulation"]["goal_position"]
    return tuple(float(v) for v in raw)


def apply_command(cmd: dict, mapping, estimation, state: dict, navigator=None) -> tuple[bool, str]:
    """Execute one dashboard command against the REAL running components. Returns (ok, detail)."""
    name = cmd.get("cmd")
    if name == "set_camera":
        rig = state.get("camera_rig")                       # the live 3-D camera rig (None when no PyBullet window is open)
        if rig is None:
            return False, "no 3-D window in this run (start with '3D window: on'); the camera controls act on the PyBullet view"
        return rig.do(str(cmd.get("action", "")))
    if name in NAV_COMMANDS:
        return _apply_nav_command(name, cmd, mapping, navigator, state)
    if name == "reset_mapping":
        if mapping is None:
            return False, "mapping unavailable (sensors/sonar disabled)"
        mapping.reset()
        return True, "occupancy grid, cost map and mapping counters reset to UNKNOWN/zero"
    if name == "set_mapping_enabled":
        if mapping is None:
            return False, "mapping unavailable (sensors/sonar disabled)"
        mapping.set_enabled(bool(cmd.get("enabled")))
        return True, f"mapping {'enabled' if mapping.enabled else 'disabled'}"
    if name == "set_sensor_health":
        # Live state change on the RUNNING sensor (never a stop/restart): the sensor model itself changes behaviour,
        # and the detail says what really happens so the dashboard can show the consequence, not a label.
        sensor = str(cmd.get("sensor"))
        health = canonical_state(cmd.get("health", ""))
        if estimation is None:
            return False, "sensors are disabled in this run (sensors.enabled: false)"
        if sensor not in estimation.suite.sensors:
            avail = ", ".join(estimation.suite.sensors) or "none"
            return False, f"sensor {sensor!r} not available in this run (configured sensors: {avail})"
        if health not in FORCED_STATES:
            return False, f"invalid state {health!r}; choose one of {', '.join(FORCED_STATES)}"
        estimation.suite.sensors[sensor].inject_fault(None if health == "HEALTHY" else health)
        if health == "HEALTHY":
            return True, f"{sensor}: injected fault cleared; automatic health monitoring stays active"
        if health == "FAILED":
            return True, f"{sensor} FAILED: its output is now stuck at 0.0 on every sample (invalid to the EKF)"
        if health == "DISCARDED":
            return True, f"{sensor} DISCARDED: it produces no output, so its readings stop changing"
        if health == "LINK_CUT":
            return True, f"{sensor} LINK CUT: no data reaches the computer, so its readings stop changing"
        return True, f"{sensor} fault injection set to {health}"
    if name == "set_realtime_factor":
        try:
            v = float(cmd.get("value"))
        except (TypeError, ValueError):
            return False, "value must be a number"
        if not (v >= 0.0 and v < 1e4):
            return False, "value must be in [0, 10000) (0 = as fast as possible)"
        state["realtime_factor"] = v
        return True, f"realtime factor {v}"
    return False, f"unknown command {name!r}"


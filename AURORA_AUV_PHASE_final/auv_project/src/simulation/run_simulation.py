"""
run_simulation.py
-------------------
Week 5 entry point: builds the PyBullet world from Week 4's occupancy map,
spawns the AUV, and drives it through a manually-defined waypoint sequence
(START -> waypoint(s) -> GOAL) using the basic proportional controller.
No path planning (A*) happens here -- that is Week 6, which will read
navigation_cost_map.npy (already loaded onto Environment.cost_map here,
but unused for control until then).

    1. Load configuration + connect Week 4 mapping outputs
    2. Connect to PyBullet (GUI or DIRECT, per config.simulation.mode/gui)
    3. Build environment (seabed + obstacles from occupancy_grid_inflated)
    4. Load the AUV at the configured start position
    5. Step the simulation, driving toward each waypoint in sequence
    6. Read proximity sensors + detect collisions each step; log everything
    7. Save the trajectory log/metrics/figure
    8. Print a summary report

Run with:
    python -m src.simulation.run_simulation
"""

from __future__ import annotations

import copy
import math
import os
import time as _time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import pybullet as p
import pybullet_data
import yaml

from src.simulation.adaptive_controller import AdaptivePController
from src.simulation.auv import AUV
from src.simulation.controller import ControllerParams, WaypointController
from src.simulation.environment import Environment
from src.simulation.commands import apply_command as _apply_command, deep_merge as _deep_merge, resolve_mission_goal
from src.simulation.estimation import EstimationPipeline
from src.simulation.metrics import compute_metrics, save_metrics
from src.planning.navigator import NavState, TERMINAL_STATES, AutonomousNavigator, authoritative_goal, navigation_prerequisites
from src.simulation.sensors import read_proximity
from src.simulation.telemetry_snapshot import SensorStatsTracker, build_snapshot
from src.utils.visualization import plot_auv_trajectory

CONFIG_PATH = "configs/config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and return the project YAML config as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_mode(sim_cfg: dict, mode_override: str | None) -> str:
    """
    "mode" (GUI/DIRECT string) and "gui" (bool) are both accepted (STEP 13
    asks for `simulation.gui: true/false`); mode_override (used by tests)
    takes precedence over both, then explicit "mode", then "gui".
    """
    if mode_override is not None:
        return mode_override.upper()
    if "mode" in sim_cfg:
        return str(sim_cfg["mode"]).upper()
    return "GUI" if sim_cfg.get("gui", False) else "DIRECT"



def _build_terrain_risk(cfg: dict, mapping_cfg):
    """Optional a-priori terrain-risk layer from the EXISTING bathymetry + Random Forest pipeline.
    Returns an array on the map grid, or None when mapping.terrain_risk.enabled is false."""
    tr = (cfg.get("mapping", {}) or {}).get("terrain_risk", {}) or {}
    if not tr.get("enabled", False):
        return None
    from src.mapping.coordinate_utils import GridGeometry
    from src.mapping.terrain_prior import classes_from_depth, resample_to_geometry, risk_from_classes
    processed = cfg["output"]["processed_dir"]
    depth = np.load(f"{processed}/processed_bathymetry.npy")
    classes = classes_from_depth(depth, float(cfg["environment"]["resolution"]),
                                 cfg["terrain"]["model_path"], int(cfg["terrain"].get("roughness_window", 3)))
    risk = risk_from_classes(classes, {int(k): float(v) for k, v in tr.get("class_risk", {}).items()})
    geom = GridGeometry.from_extent(mapping_cfg.resolution_m, mapping_cfg.width_m, mapping_cfg.height_m,
                                    mapping_cfg.origin_x_m, mapping_cfg.origin_y_m)
    return resample_to_geometry(risk, (0.0, 0.0), float(cfg["environment"]["resolution"]), geom)



def _controller_info(navigator, controller, position, target, waypoint_index, waypoints, collisions) -> dict:
    """Snapshot 'controller' section. Navigation mode reports ONLY operational (estimate-based) quantities here; the
    truth-based distance and the physics collision count move to the labelled debug_ground_truth section."""
    if navigator is not None:
        info = {"type": "Phase 5 autonomous navigation (EKF estimate only)", "state": navigator.state.value,
                "target": [float(v) for v in target], "waypoint_index": int(waypoint_index)}
        # Autonomous navigation has no adaptive P controller: the PATH FOLLOWER drives the vehicle, so it reports
        # its own gains and live speed schedule here. Without this the dashboard's gain panel and charts would have
        # nothing to show whenever navigation is on.
        report = getattr(navigator.follower, "gain_report", None)
        if callable(report):
            block = report()
            if block:
                info["follower"] = block
        return info
    info = {"type": "P waypoint controller on TRUE state (legacy Phase 4 mode; navigation disabled)", "target": list(target), "waypoint_index": waypoint_index,
            "total_waypoints": len(waypoints), "distance_to_target_true": float(controller.distance_to(position, target)),
            "collisions": collisions}
    adaptive = getattr(controller, "last_info", None)         # AdaptivePController reports its effective gains
    if adaptive:
        info["adaptive"] = dict(adaptive)
    return info


def gui_window_options(vis_cfg: dict | None) -> str:
    """PyBullet GUI start-up options: a larger window gives the side panel room for the full button labels (rendering only)."""
    w = (vis_cfg or {}).get("window") or {}
    return f"--width={int(w.get('width', 1600))} --height={int(w.get('height', 900))}"


def hold_window_open(p, client, publisher, snapshot, stop_event, live_dir, final_status, result,
                     period: float = 1.0 / 30.0, publish_period: float = 0.25) -> None:
    """Keep the run's OWN PyBullet window on the last frame after the mission has ended.

    `visualization.keep_open_after_run` (default true): the window the run itself opened is NOT closed
    automatically when the robot reaches the goal (or the run stops / fails). The run is already finished --
    nothing is simulated any more -- so the window is only kept alive and responsive, and the FINAL telemetry
    snapshot is republished (with a fresh wall_time) so the dashboard keeps reading a live snapshot instead of
    reporting the finished run as STALE.

    The window is released as soon as the operator closes it, or a stop is requested (the dashboard's Stop, or a
    new run about to start); the caller then disconnects and the window goes away. Nothing about the finished run
    changes: its telemetry, status and result were already published, and no physics is stepped for the vehicle.
    """
    print("\n[run_simulation] mission ended: keeping the 3-D window on the last frame "
          "(visualization.keep_open_after_run). Close it, or press Stop, to release it.", flush=True)
    if live_dir:
        # Report the run's terminal status NOW, with its result, while the window is still held: the dashboard
        # must read the finished run (COMPLETED / STOPPED / ...), not a running one, for as long as it is held.
        try:
            from src.simulation.live_io import atomic_write_json
            atomic_write_json(Path(live_dir) / "status.json",
                              {"status": final_status, "result": result, "error": None, "holding": True,
                               "pid": os.getpid(), "time": _time.time()})
        except Exception as exc:                                  # noqa: BLE001 - reporting only
            print(f"[run_simulation] could not write the holding status: {exc}", flush=True)
    last_publish = 0.0
    while True:
        if not p.isConnected(physicsClientId=client):
            print("[run_simulation] 3-D window closed by the operator.", flush=True)
            return
        if stop_event is not None and stop_event.is_set():
            print("[run_simulation] stop requested while the window was held: releasing it.", flush=True)
            return
        try:
            p.stepSimulation(physicsClientId=client)             # only keeps the GUI responsive
        except Exception as exc:                                 # noqa: BLE001 - the window is gone
            print(f"[run_simulation] the 3-D window is no longer usable: {exc}", flush=True)
            return
        now = _time.monotonic()
        if snapshot is not None and publisher is not None and now - last_publish >= publish_period:
            last_publish = now
            try:
                snapshot["wall_time"] = _time.time()             # the dashboard must not call the held run STALE
                publisher.publish_state(snapshot)
            except Exception as exc:                             # noqa: BLE001 - telemetry only
                print(f"[run_simulation] could not republish the final snapshot: {exc}", flush=True)
        _time.sleep(period)


def _setup_gui_visuals(start_position, waypoints, goal_position, environment, client, vis_cfg=None):
    """Set a useful camera and add visual-only mission markers for GUI demos.

    These objects have no collision shapes, so they do not change Week 5
    physics or collision results. They only make the PyBullet demonstration
    easy to see: seabed/obstacles remain physical, while start/waypoint/goal
    markers and labels are visual aids.
    """
    # Initial camera. With camera_follow on (default) the first frame is already the third-person follow view
    # behind the start position, looking toward the first waypoint (src/simulation/gui_camera.py takes over from
    # there). With camera_follow off the legacy whole-mission overview is used and the camera is left to the user.
    from src.simulation import gui_camera                              # noqa: PLC0415
    vis_cfg = vis_cfg or {}
    # Underwater atmosphere (background colour, one light + shadows, the water surface at z = 0). Rendering only:
    # no collision shapes are created, so sensors / physics / mapping / planning are untouched.
    from src.simulation.underwater_look import apply as _underwater_look      # noqa: PLC0415
    _underwater = _underwater_look(client, vis_cfg, world_bounds=getattr(environment, "bounds", None))
    xs = [float(start_position[0]), float(goal_position[0])] + [float(w[0]) for w in waypoints]
    ys = [float(start_position[1]), float(goal_position[1])] + [float(w[1]) for w in waypoints]
    target = [sum(xs) / len(xs), sum(ys) / len(ys), -5.0]
    if vis_cfg.get("camera_follow", True):
        first = waypoints[0] if len(waypoints) else goal_position
        p.resetDebugVisualizerCamera(physicsClientId=client,
                                     **gui_camera.initial_follow_camera(start_position, first, vis_cfg.get("camera")))
    else:
        p.resetDebugVisualizerCamera(physicsClientId=client, **gui_camera.overview_free_camera(start_position, waypoints, goal_position))

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=client)
    p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=client)
    for _flag in ("COV_ENABLE_RGB_BUFFER_PREVIEW", "COV_ENABLE_DEPTH_BUFFER_PREVIEW"):     # PyBullet's default preview panels: clutter
        if hasattr(p, _flag):
            p.configureDebugVisualizer(getattr(p, _flag), 0, physicsClientId=client)
    if _underwater.get("applied"):
        print(f"\nUnderwater look: background {_underwater.get('background')}, shadows "
              f"{'on' if _underwater.get('shadows') else 'off'}, "
              f"water surface body {_underwater.get('surface_body')} (visual only, no collision shape)")

    # Visual-only marker helper: no collision geometry is created.
    def marker(position, radius, rgba):
        visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=rgba,
            physicsClientId=client,
        )
        return p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=visual,
            basePosition=position,
            physicsClientId=client,
        )

    marker(start_position, 0.45, [0.1, 0.9, 0.2, 1.0])
    for i, wp in enumerate(waypoints[:-1], start=1):
        marker(wp, 0.30, [1.0, 0.75, 0.1, 1.0])
        p.addUserDebugText(
            f"WP {i}", [wp[0], wp[1], wp[2] + 0.8],
            textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=0,
            physicsClientId=client,
        )
    # the GOAL marker + label are owned by GoalMarker (gui_visuals.py) so they follow the authoritative goal

    p.addUserDebugText(
        "START", [start_position[0], start_position[1], start_position[2] + 0.9],
        textColorRGB=[0.1, 1, 0.2], textSize=1.3, lifeTime=0,
        physicsClientId=client,
    )

    return target

def run(
    config_path: str = CONFIG_PATH,
    mode_override: str | None = None,
    telemetry_callback=None,
    stop_event=None,
    cfg_override: dict | None = None,
    live_dir: str | None = None,
    command_path: str | None = None,
) -> dict:
    """
    Execute the Week 5 waypoint-following demonstration.

    telemetry_callback: optional callable receiving each live simulation record.
    stop_event: optional threading.Event; when set, the simulation exits cleanly.
    cfg_override: optional nested dict merged over the loaded config (dashboard 'apply on next start').
    live_dir / command_path: Phase 4 dashboard IPC (throttled state/map snapshots out, commands in).

    Input:
        config_path: path to configs/config.yaml.
        mode_override: optionally force "GUI" or "DIRECT" (tests force
                       "DIRECT" so headless/CI runs never open a window).
    Processing:
        See module docstring for the 8-step pipeline.
    Output:
        dict: {"goal_reached", "waypoints_reached", "total_waypoints",
        "collisions", "metrics", "log_path", "trajectory_path",
        "metrics_path", "figure_path"}.
    """
    print("=" * 40)
    print("WEEK 5 - AUV SIMULATION")
    print("=" * 40)

    cfg = _deep_merge(load_config(config_path), cfg_override)
    sim_cfg = cfg["simulation"]
    mode = _resolve_mode(sim_cfg, mode_override)
    max_steps = int(sim_cfg.get("max_steps", 20000))
    # Phase 5: autonomous navigation is opt-in (navigation.enabled). When on, the control loop is driven by the
    # navigator (EKF estimate + online map + sonar only); the Phase-4 true-state waypoint controller is not used.
    nav_cfg_enabled = bool((cfg.get("navigation") or {}).get("enabled", False))
    if nav_cfg_enabled:
        why_not = navigation_prerequisites(cfg)
        if why_not:
            raise ValueError(f"navigation.enabled is true but it cannot run: {why_not}")

    print("\nLoading Week 4 environment...")
    connection = p.GUI if mode == "GUI" else p.DIRECT
    client = p.connect(connection, options=gui_window_options(cfg.get("visualization"))) if mode == "GUI" else p.connect(connection)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, float(sim_cfg["gravity"]))
    p.setTimeStep(float(sim_cfg["time_step"]))

    environment = Environment(cfg, physics_client=client)
    print(f"\nGrid:\n  {environment.grid_info.rows} x {environment.grid_info.cols} cells")
    print(f"\nResolution:\n  {environment.grid_info.resolution} m/cell")
    n_obstacle_cells = int(np.sum(environment.occupancy_grid == environment.grid_info.obstacle_value))
    print(f"\nLoading occupancy map...\n  {n_obstacle_cells} obstacle cells "
          f"-> {len(environment.obstacle_ids)} merged obstacle bodies")

    print("\nCreating PyBullet environment...\n  seabed + obstacles created.")

    print("\nLoading AUV...")
    start_position = tuple(float(v) for v in sim_cfg["start_position"])
    # Mission goal in BOTH modes: navigation.goal_position when set (the dashboard's goal input lives there), else
    # simulation.goal_position. Without this a run with navigation disabled flew to the config goal, ignoring the
    # goal the operator had typed.
    goal_position = resolve_mission_goal(cfg)
    if (cfg.get("navigation") or {}).get("goal_position") is not None:
        # The operator set the goal (dashboard / 3-D window): fly straight to it. Keeping the config's hand-picked
        # route in front of it would send the AUV to the old waypoints first, which is not the mission the operator
        # asked for and does not match the goal the dashboard shows.
        waypoints = [goal_position]
    else:
        raw_waypoints = sim_cfg.get("waypoints") or [list(goal_position)]
        waypoints = [tuple(float(v) for v in wp) for wp in raw_waypoints]
        if waypoints[-1] != goal_position:
            waypoints.append(goal_position)  # goal is always the final waypoint
    if nav_cfg_enabled:
        waypoints = [goal_position]       # the planner generates the waypoints; the hand-picked list is not used

    dynamics_mode_name = str((sim_cfg.get("dynamics", {}) or {}).get("model", "legacy")).lower()
    auv = AUV(sim_cfg["urdf_path"], start_position, sim_cfg, physics_client=client)
    if mode == "GUI":
        _setup_gui_visuals(start_position, waypoints, goal_position, environment, client, cfg.get("visualization"))
        print("\nPyBullet GUI ready: AUV + seabed + obstacles + mission markers visible.")
    print(f"\nStart:\n  {start_position}")
    print(f"\nWaypoints:\n  {' -> '.join(str(w) for w in waypoints)}")
    print(f"\nGoal:\n  {goal_position}")

    controller_params = ControllerParams.from_config(sim_cfg["controller"])
    # Week 9: with adaptation enabled the legacy waypoint mission is driven by the ADAPTIVE P controller (same
    # interface and the same mid-mission P law, but the proportional gain is scheduled near the target and
    # self-tuned while stalled). Disabled -> the plain WaypointController, bit-for-bit as before.
    controller = AdaptivePController(controller_params) if controller_params.adaptive.enabled \
        else WaypointController(controller_params)
    # Final approach to the GOAL (legacy waypoint mission): the last waypoint IS the goal, and the mission is only
    # complete once the vehicle is within goal_tolerance of it, so a finished run really ends on the goal x/y/z the
    # operator configured. Bounded by goal_settle_timeout_s so a vehicle that cannot creep closer still completes.
    goal_tolerance_m = controller_params.goal_tolerance
    goal_settle_timeout_s = controller_params.goal_settle_timeout_s
    # Station-keeping trim for that final approach: the legacy buoyancy model is deliberately slightly POSITIVE
    # (see tests/test_dynamics.py), so holding a depth needs a small permanent vertical command. A feed-forward
    # term cancels exactly that known residual (it is not an integral term); the P term keeps the rest of the
    # mission proportional-only, as documented.
    _pp = auv.physics_params
    _net_up_n = (_pp.water_density * abs(_pp.gravity) * _pp.volume * _pp.buoyancy_ratio
                 - _pp.mass * abs(_pp.gravity))
    vertical_trim = float(np.clip(-_net_up_n / controller_params.max_thrust, -1.0, 1.0))
    sensor_range = float(sim_cfg["sensors"]["max_range"])

    # Phase 3: simulated sensors + EKF. With navigation disabled (legacy Phase 4 waypoint demo) the P controller below
    # still uses the TRUE state and the estimate is only logged. With navigation enabled the EKF estimate is the ONLY
    # state the navigator / path follower / controller see.
    estimation = None
    if (cfg.get("sensors", {}) or {}).get("enabled", False):
        estimation = EstimationPipeline(cfg, auv.body_id, start_position, physics_client=client)
        print("\nSensors + EKF: enabled ("
              + ("navigation drives the vehicle from the EKF estimate only)" if nav_cfg_enabled
                 else "estimate logged only; legacy waypoint controller uses true state)"))

    # Phase 4: online mapping. The mapper is handed ONLY the sonar Measurement and the EKF estimate
    # (see src/mapping/map_manager.py); it never sees PyBullet, truth, or the environment description.
    mapping = None
    mapping_unavailable_reason = None
    if estimation is None:
        mapping_unavailable_reason = "sensors disabled (sensors.enabled: false)"
    elif estimation.ekf is None:
        mapping_unavailable_reason = "EKF disabled (ekf.enabled: false): mapping needs an estimated pose"
    elif "sonar" not in estimation.suite.sensors:
        mapping_unavailable_reason = "sonar disabled (sensors.sonar.enabled: false)"
    else:
        from src.mapping.map_manager import MappingSystem
        from src.mapping.map_types import MapConfig
        mapping = MappingSystem(cfg, terrain_risk=_build_terrain_risk(cfg, MapConfig.from_config(cfg)))
        print(f"\nMapping: {'enabled' if mapping.enabled else 'disabled'} "
              f"({mapping.grid.geom.rows}x{mapping.grid.geom.cols} cells @ {mapping.grid.geom.resolution} m, "
              f"fed by sonar + EKF estimate only)")
    print(f"Environment scenario: {environment.scenario} (seed {environment.seed})")

    # Phase 5: the navigator. Its ONLY inputs are (sim time, EKF estimate, mapping, latest sonar measurement).
    navigator = None
    if nav_cfg_enabled:
        frame = "body" if dynamics_mode_name == "6dof" else "world"
        navigator = AutonomousNavigator(cfg, command_frame=frame)
        navigator.set_goal(goal_position, 0.0, start=bool((cfg.get("navigation") or {}).get("auto_start", True)))
        print(f"\nAutonomous navigation: ENABLED (EKF-only, {frame}-frame commands). Goal {list(goal_position)}")

    dash_cfg = cfg.get("dashboard", {}) or {}
    publisher = commands = None
    if live_dir is not None:
        from src.simulation.live_io import CommandChannel, LivePublisher
        publisher = LivePublisher(live_dir, float(dash_cfg.get("state_publish_hz", 10.0)),
                                  float(dash_cfg.get("map_publish_hz", 2.0)))
    if command_path is not None:
        commands = CommandChannel(command_path)
    # real-time pacing applies only to dashboard-launched runs (live_dir set); standalone runs stay full speed
    vis_cfg = cfg.get("visualization", {}) or {}
    # Real-time pacing: GUI windows run at visualization.realtime_factor (1.0 = watchable) so the motion can be
    # seen; headless dashboard runs use dashboard.realtime_factor; standalone headless runs are unpaced.
    if mode == "GUI":
        rf0 = float(vis_cfg.get("realtime_factor", 1.0))
    elif live_dir is not None:
        rf0 = float(dash_cfg.get("realtime_factor", 0.0))
    else:
        rf0 = 0.0
    run_state = {"realtime_factor": rf0}
    gui_visuals = None
    goal_editor = None
    if mode == "GUI":
        from src.simulation.gui_visuals import GuiVisuals
        gui_visuals = GuiVisuals(client, vis_cfg, environment, auv_body_id=auv.body_id, goal=goal_position)
        if navigator is not None:                  # change the goal from the 3-D window (arrows + G); same command path as the dashboard
            from src.simulation.gui_goal import GoalKeyEditor
            goal_editor = GoalKeyEditor(client)
    run_state["camera_rig"] = getattr(gui_visuals, "camera", None) if gui_visuals is not None else None
    env_desc_cached = environment.describe()      # static for the whole run
    sensor_tracker = SensorStatsTracker()
    command_log = deque(maxlen=20)
    last_command_id = 0
    est_traj, truth_traj = deque(maxlen=800), deque(maxlen=800)
    step_ms_ema = None
    dynamics_mode = str((sim_cfg.get("dynamics", {}) or {}).get("model", "legacy"))
    terminal_since = None
    linger_s = float((cfg.get("navigation") or {}).get("linger_after_terminal_s", 2.0))

    print("\nRunning simulation...")
    records = []
    collisions = 0
    waypoint_index = 0
    dt = float(sim_cfg["time_step"])
    wall_start = _time.time()
    # Pacing anchor: wall time and step index at which the CURRENT realtime factor took effect. Re-anchored
    # whenever the factor changes, so a speed change applies immediately (an anchor fixed at run start would
    # make a 1x -> 10x switch burst unpaced and a 10x -> 1x switch sleep for the whole accumulated difference).
    pace_wall0, pace_step0, pace_rf = _time.perf_counter(), 0, run_state["realtime_factor"]
    stopped_by_request = False
    safety_stop = None           # reason the run stopped for safety (all sensors failed/unhealthy), else None
    goal_settle_from = None      # sim time at which the final approach to the goal started (None: not there yet)

    if publisher is not None:
        _scene = environment.scene_payload()
        if _scene is not None:
            _cur = ((cfg.get("simulation") or {}).get("current") or {}).get("base") or [0.0, 0.0, 0.0]
            _scene["conditions"] = {                     # only quantities this simulation really has (no invented wave height)
                "current_mps": round(float(np.linalg.norm(_cur)), 3), "current_vector": [float(v) for v in _cur],
                "sonar_max_range_m": float((cfg.get("sensors", {}).get("sonar") or {}).get("max_range_m", 0.0) or 0.0),
                "seabed_type": _scene["seabed_type"], "wave_height_m": None,
                "seabed_z_range": [float(np.min(_scene["heights"])), float(np.max(_scene["heights"]))]}
            publisher.publish_scene(_scene)

    for step in range(max_steps):
        if mode == "GUI" and not p.isConnected(physicsClientId=client):
            print("\nPyBullet window closed: stopping the simulation.")
            stopped_by_request = True
            break
        if stop_event is not None and stop_event.is_set():
            print("\nSimulation stop requested.")
            stopped_by_request = True
            break
        if estimation is not None and estimation.suite.all_unusable():
            # SAFETY: every configured sensor is FAILED or UNHEALTHY, so the vehicle has no usable navigation
            # input at all. Checked at the TOP of the step (the previous step already published and acknowledged
            # the operator's fault commands), then stop the robot and end the run instead of flying on a
            # dead-reckoned estimate.
            safety_stop = ("all configured sensors failed/unhealthy "
                           + "(" + ", ".join(f"{n}:{estimation.suite.sensors[n].health_monitor.health.name}"
                                             for n in estimation.suite.sensors) + ")")
            print(f"\nSAFETY STOP at t={step * dt:.2f} s: {safety_stop}")
            auv.stop()
            stopped_by_request = True
            command_log.append({"id": last_command_id, "cmd": "safety_stop", "ok": True, "detail": safety_stop,
                                "sim_time": step * dt})
            break
        target = waypoints[min(waypoint_index, len(waypoints) - 1)]   # nav mode: waypoint_index is the path vertex, not this list

        state = auv.get_state()
        position = state.position
        yaw = state.orientation_euler[2]

        # Sensors sample the TRUE state (that is what a sensor is); the EKF turns the measurements into the ESTIMATE.
        measurements, estimate = ({}, None)
        loop_t0 = _time.perf_counter()
        run_state["sim_time"] = step * dt
        if commands is not None:
            for cmd in commands.poll():
                ok, detail = _apply_command(cmd, mapping, estimation, run_state, navigator)
                last_command_id = int(cmd.get("id", last_command_id))
                command_log.append({"id": last_command_id, "cmd": cmd.get("cmd"), "ok": ok, "detail": detail,
                                    "sim_time": step * dt})
        if goal_editor is not None and navigator is not None:
            msg = goal_editor.poll(step, navigator.goal, lambda g: _apply_command(
                {"cmd": "nav_set_goal", "goal": g}, mapping, estimation, run_state, navigator))
            if msg:
                command_log.append({"id": last_command_id, "cmd": "window_goal_keys", "ok": "rejected" not in msg,
                                    "detail": msg, "sim_time": step * dt})
        if navigator is not None and navigator.goal is not None:
            goal_position = authoritative_goal(navigator, goal_position)       # dashboard goal changes
            waypoints[-1] = goal_position
        if estimation is not None:
            measurements, estimate = estimation.update(step * dt)
            sensor_tracker.observe(measurements)
            if mapping is not None:
                # ONLY the sonar scan + the EKF estimate enter the mapper (ground truth does not).
                mapping.process(step * dt, measurements.get("sonar"), estimate)
            if gui_visuals is not None:
                # 3-D overlays: real sonar ray endpoints, EKF estimate marker, mapped obstacles, trail, camera.
                gui_visuals.update(step, estimation.truth, measurements.get("sonar"), estimate, mapping, step * dt)

        if navigator is not None:
            # EKF estimate + mapped cost map + latest sonar scan: NOTHING from the true state `position` / `yaw`.
            nav_out = navigator.update(step * dt, estimate, mapping, estimation.latest.get("sonar"))
            control = nav_out.control
            wp = navigator.follower.current_waypoint()
            if wp is not None:
                target = tuple(float(v) for v in wp)
            else:
                target = goal_position
            waypoint_index = navigator.follower.waypoint_index if navigator.follower.path is not None else 0
            if gui_visuals is not None:
                gui_visuals.update_path(navigator.path_payload(), float(goal_position[2]))
                gui_visuals.update_goal(goal_position)
        elif goal_settle_from is not None:
            # Terminal approach: damped P + buoyancy trim against the GOAL, so the vehicle settles on the goal
            # values instead of hunting around it (the mid-mission control law above stays purely proportional).
            control = controller.compute_control(position, yaw, goal_position,
                                                current_velocity=state.linear_velocity, vertical_trim=vertical_trim,
                                                dt=dt)
        else:
            control = controller.compute_control(position, yaw, target, dt=dt)
        if navigator is None and dynamics_mode_name == "6dof":
            # The P controller computes WORLD-frame commands (forward = world +X, see controller.py); the 6-DOF
            # model expects a BODY-frame wrench, so rotate the horizontal part by the vehicle's heading. Without
            # this the thrust points the wrong way as soon as the AUV turns away from +X (docs/audit.md: the
            # world-frame commands are correct for the legacy force model only).
            c, s = math.cos(yaw), math.sin(yaw)
            control = {**control, "forward": control["forward"] * c + control["lateral"] * s,
                       "lateral": -control["forward"] * s + control["lateral"] * c}
        auv.apply_control(**control, dt=dt)
        auv.step_underwater_physics()

        p.stepSimulation(physicsClientId=client)

        collided = environment.check_collision(auv.body_id)
        if collided:
            collisions += 1
        if not environment.in_bounds(position):
            auv.stop()

        proximity = read_proximity(auv.body_id, sensor_range, physics_client=client)

        record = {
            "time": step * dt,
            "x": position[0], "y": position[1], "z": position[2],
            "roll": state.orientation_euler[0], "pitch": state.orientation_euler[1],
            "yaw": state.orientation_euler[2],
            "vx": state.linear_velocity[0], "vy": state.linear_velocity[1], "vz": state.linear_velocity[2],
            "energy": state.energy_consumed,
            "target_x": target[0], "target_y": target[1], "target_z": target[2],
            "waypoint_index": waypoint_index,
            "distance_to_target": controller.distance_to(position, target),
            "collision": collided,
            "sensor_front": proximity["front"],
            "sensor_left": proximity["left"],
            "sensor_right": proximity["right"],
        }
        if estimation is not None:
            record.update(estimation.log_fields(estimate, measurements))
        if navigator is not None:
            record["nav_state"] = navigator.state.value
            record["nav_goal_distance_est"] = navigator._goal_distance     # from the ESTIMATE (operational)
            record["nav_replans"] = navigator.counters["replans"]
            record["nav_avoidance_count"] = navigator.counters["avoidance_count"]
        records.append(record)
        if telemetry_callback is not None:
            try:
                telemetry_callback(record)
            except Exception as exc:
                print(f"Telemetry callback warning: {exc}")

        if step % 10 == 0:
            if estimate is not None:
                est_traj.append([float(estimate.position[0]), float(estimate.position[1])])
            truth_traj.append([float(position[0]), float(position[1])])
        step_ms = (_time.perf_counter() - loop_t0) * 1e3
        step_ms_ema = step_ms if step_ms_ema is None else 0.98 * step_ms_ema + 0.02 * step_ms
        if publisher is not None and (publisher.due_state() or step == 0):
            wall = max(_time.time() - wall_start, 1e-9)
            snap = build_snapshot(
                step=step, sim_time=step * dt, dt=dt, status="RUNNING", dynamics_mode=dynamics_mode,
                estimation=estimation, estimate=estimate, measurements=measurements,
                sensor_tracker=sensor_tracker, mapping=mapping, env_desc=env_desc_cached,
                truth_position=position, truth_euler=state.orientation_euler, auv=auv, control=control,
                controller_info=_controller_info(navigator, controller, position, target, waypoint_index, waypoints,
                                                 collisions),
                perf={"sim_seconds_per_wall_second": (step * dt) / wall,
                      "mean_step_ms": step_ms_ema, "realtime_factor_setting": run_state["realtime_factor"],
                      "mapping_last_ms": None if mapping is None else mapping.timing["last_map_ms"],
                      "mapping_mean_ms": None if mapping is None else mapping.timing["mean_map_ms"],
                      "costmap_mean_ms": None if mapping is None else mapping.timing["mean_cost_ms"]},
                last_command_id=last_command_id, mapping_unavailable_reason=mapping_unavailable_reason,
                trajectory_tail=list(est_traj),
                navigation=None if navigator is None else navigator.telemetry(),
                navigation_path=None if navigator is None else navigator.path_payload())
            snap["command_log"] = list(command_log)
            snap["mission"] = {"start": list(start_position), "goal": list(goal_position),
                               "waypoints": [list(w) for w in waypoints]}
            snap["safety"] = {"stopped": safety_stop is not None, "reason": safety_stop}
            if "debug_ground_truth" in snap:
                snap["debug_ground_truth"]["trajectory"] = list(truth_traj)
                # Truth-derived figures live ONLY in the labelled debug/evaluation section. goal_distance_true is the
                # ground-truth AUV-to-goal distance, kept SEPARATE from the operational estimate-based distance
                # (navigation.goal_distance_m) so the operator can compare the two.
                snap["debug_ground_truth"]["goal_distance_true"] = float(np.linalg.norm(
                    np.asarray(position, float) - np.asarray(goal_position, float)))
                if navigator is not None:
                    snap["debug_ground_truth"]["collisions"] = int(collisions)
            publisher.publish_state(snap)
        if publisher is not None and mapping is not None and publisher.due_maps(mapping.version):
            publisher.publish_maps({"occupancy": mapping.occupancy_payload(), "costmap": mapping.costmap_payload()},
                                   mapping.version)
        rf_now = run_state["realtime_factor"]
        if rf_now != pace_rf:                      # speed changed while running: restart the pacing schedule here
            pace_wall0, pace_step0, pace_rf = _time.perf_counter(), step + 1, rf_now
        if rf_now > 0.0:
            ahead = (step + 1 - pace_step0) * dt / rf_now - (_time.perf_counter() - pace_wall0)
            if ahead > 0.0005:
                _time.sleep(ahead)
            elif ahead < -0.25:                    # compute-bound / stalled: do not accumulate an unbounded debt
                pace_wall0, pace_step0 = _time.perf_counter(), step + 1

        if navigator is not None:
            # stop after the navigator reached a terminal state (GOAL_REACHED / FAILED), decided from the estimate
            if navigator.state in TERMINAL_STATES:
                terminal_since = step * dt if terminal_since is None else terminal_since
                if step * dt - terminal_since >= linger_s:
                    break
            else:
                terminal_since = None
        elif controller.reached_waypoint(position, target):
            if waypoint_index < len(waypoints) - 1:
                waypoint_index += 1
            else:
                # At the final waypoint (= the goal): keep driving toward it until it is really reached. Inside the
                # 0.5 m waypoint radius the P controller still closes the remaining error, so the run ends on the
                # configured goal values instead of stopping anywhere in that radius.
                if goal_settle_from is None:
                    goal_settle_from = step * dt
                if controller.distance_to(position, goal_position) <= goal_tolerance_m:
                    break
                if step * dt - goal_settle_from >= goal_settle_timeout_s:
                    break  # cannot get closer; complete here (still reported as goal reached by the 0.5 m criterion)

    simulation_wall_time = _time.time() - wall_start
    goal_reached_truth = bool(controller.reached_waypoint(auv.get_position(), goal_position))   # EVALUATION only
    if navigator is not None:
        reached_goal = navigator.state == NavState.GOAL_REACHED                # operational: decided from the estimate
        total_wp = navigator.global_path.n if navigator.global_path is not None else 1
        waypoints_reached = total_wp if reached_goal else min(waypoint_index, max(total_wp - 1, 0))
        waypoints_total = total_wp
    else:
        reached_goal = goal_reached_truth
        waypoints_reached = waypoint_index + (1 if reached_goal else 0)
        waypoints_total = len(waypoints)

    if safety_stop is not None:
        print(f"\nSAFETY STOP: {safety_stop}")
    print(f"\nWaypoint reached:\n  goal reached = {'YES' if reached_goal else 'NO'} "
          f"({waypoints_reached}/{waypoints_total} waypoints reached)")
    if navigator is not None:
        print(f"Navigation: final state {navigator.state.value}, replans {navigator.counters['replans']}, "
              f"avoidance maneuvers {navigator.counters['avoidance_count']}, "
              f"goal reached by TRUE position (evaluation): {'YES' if goal_reached_truth else 'NO'}")
    print(f"\nSimulation time:\n  {simulation_wall_time:.2f} s (wall clock), "
          f"{len(records) * dt:.2f} s (simulated)")
    print(f"\nCollisions:\n  {collisions}")

    # Phase 4: final live snapshot (the dashboard reads the true end state, not the last throttled one)
    final_status = "STOPPED" if stopped_by_request else ("COMPLETED" if reached_goal else "ENDED_BEFORE_GOAL")
    if publisher is not None and records:
        last_step = len(records) - 1
        snap = build_snapshot(
            step=last_step, sim_time=last_step * dt, dt=dt, status=final_status, dynamics_mode=dynamics_mode,
            estimation=estimation, estimate=estimate, measurements=measurements, sensor_tracker=sensor_tracker,
            mapping=mapping, env_desc=env_desc_cached, truth_position=auv.get_position(),
            truth_euler=state.orientation_euler, auv=auv, control=None,
            # Same helper as every periodic snapshot, so the legacy controller's adaptive gains are reported on the
            # FINAL frame too (a hand-built dict here used to drop them).
            controller_info=({**_controller_info(None, controller, auv.get_position(), goal_position, waypoint_index,
                                                 waypoints, collisions), "goal_reached": bool(reached_goal)}
                             if navigator is None else
                             _controller_info(navigator, controller, auv.get_position(), goal_position, waypoint_index,
                                              waypoints, collisions)),
            perf={"mean_step_ms": step_ms_ema,
                  "mapping_mean_ms": None if mapping is None else mapping.timing["mean_map_ms"],
                  "costmap_mean_ms": None if mapping is None else mapping.timing["mean_cost_ms"]},
            last_command_id=last_command_id, mapping_unavailable_reason=mapping_unavailable_reason,
            trajectory_tail=list(est_traj),
            navigation=None if navigator is None else navigator.telemetry(),
            navigation_path=None if navigator is None else navigator.path_payload())
        snap["command_log"] = list(command_log)
        snap["mission"] = {"start": list(start_position), "goal": list(goal_position),
                           "waypoints": [list(w) for w in waypoints]}
        snap["safety"] = {"stopped": safety_stop is not None, "reason": safety_stop}
        if "debug_ground_truth" in snap:
            snap["debug_ground_truth"]["trajectory"] = list(truth_traj)
            snap["debug_ground_truth"]["goal_distance_true"] = float(np.linalg.norm(
                np.asarray(auv.get_position(), float) - np.asarray(goal_position, float)))
            if navigator is not None:
                snap["debug_ground_truth"]["collisions"] = int(collisions)
        publisher.publish_state(snap)
        if mapping is not None:
            publisher.publish_maps({"occupancy": mapping.occupancy_payload(), "costmap": mapping.costmap_payload()},
                                   mapping.version)

    mapping_artifacts = None
    if mapping is not None and mapping.stats["map_updates"] > 0:
        maps_dir = "results/maps"
        os.makedirs(maps_dir, exist_ok=True)
        np.savez_compressed(os.path.join(maps_dir, "online_map_latest.npz"), **mapping.export_arrays())
        from src.mapping.map_plots import save_map_figures
        mapping_artifacts = save_map_figures(mapping, cfg["output"]["figures_dir"],
                                             est_xy=np.asarray(est_traj, float) if est_traj else None,
                                             truth_xy=np.asarray(truth_traj, float) if truth_traj else None)
        mapping_artifacts["arrays"] = os.path.join(maps_dir, "online_map_latest.npz")

    print("\nSaving trajectory...")
    logs_dir = "results/logs"
    figures_dir = cfg["output"]["figures_dir"]
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    df = pd.DataFrame.from_records(records)
    log_path = os.path.join(logs_dir, "week5_simulation_log.csv")
    df.to_csv(log_path, index=False)

    trajectory_path = os.path.join(logs_dir, "week5_trajectory.npy")
    np.save(trajectory_path, df[["x", "y", "z"]].to_numpy())

    metrics = compute_metrics(
        records, start_position, goal_position, reached_goal,
        waypoints_reached, waypoints_total, collisions, simulation_wall_time, dt,
    )
    metrics_path = os.path.join(logs_dir, "week5_simulation_metrics.json")
    save_metrics(metrics, metrics_path)

    figure_path = os.path.join(figures_dir, "auv_trajectory.png")
    plot_auv_trajectory(
        df, figure_path,
        start=start_position, goal=goal_position,
        occupancy_grid=environment.occupancy_grid,
        resolution=environment.grid_info.resolution,
        obstacle_value=environment.grid_info.obstacle_value,
    )

    print(f"  log:        {log_path}")
    print(f"  trajectory: {trajectory_path}")
    print(f"  metrics:    {metrics_path}")
    print(f"  figure:     {figure_path}")

    final_truth_position = tuple(auv.get_position())

    estimation_summary = None
    if estimation is not None and "est_pos_err" in df.columns:
        estimation_summary = {
            "rms_position_error_m": float(np.sqrt(np.mean(df["est_pos_err"] ** 2))),
            "final_position_error_m": float(df["est_pos_err"].iloc[-1]),
            "ekf_stats": dict(estimation.ekf.stats),
        }
        print(f"\nEKF (logged only): RMS position error {estimation_summary['rms_position_error_m']:.3f} m, "
              f"final {estimation_summary['final_position_error_m']:.3f} m")

    print("\nSimulation complete.")

    result = {
        "goal_reached": reached_goal,
        "waypoints_reached": waypoints_reached,
        "total_waypoints": waypoints_total,
        "collisions": collisions,
        "metrics": metrics,
        "log_path": log_path,
        "trajectory_path": trajectory_path,
        "metrics_path": metrics_path,
        "figure_path": figure_path,
        "estimation": estimation_summary,
        "final_status": final_status,
        "mapping": None if mapping is None else mapping.telemetry(),
        "mapping_artifacts": mapping_artifacts,
        "environment": environment.describe(),
        "navigation": None if navigator is None else {
            **navigator.telemetry(), "goal_reached_true_position_evaluation": goal_reached_truth,
            "state_history": list(navigator.history), "final_true_goal_distance_m": float(np.linalg.norm(
                np.asarray(final_truth_position, float) - np.asarray(goal_position, float)))},
        "goal_reached_true_position": goal_reached_truth,
        "safety_stop": safety_stop,
    }

    # Keep the run's OWN 3-D window on the last frame when the mission ends on its own
    # (`visualization.keep_open_after_run`, default true). The window is released on a close / Stop, then the
    # client is disconnected. Only a GUI run that published a snapshot has a window to keep.
    keep_open = bool((cfg.get("visualization") or {}).get("keep_open_after_run", True))
    if mode == "GUI" and keep_open and publisher is not None and records:
        hold_window_open(p, client, publisher, snap, stop_event, live_dir, final_status, result)
    if p.isConnected(physicsClientId=client):
        p.disconnect(physicsClientId=client)
    return result


if __name__ == "__main__":
    run()

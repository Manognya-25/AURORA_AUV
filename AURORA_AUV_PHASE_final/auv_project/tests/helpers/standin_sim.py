"""TEST-ONLY stand-in simulation process (used when PyBullet is not installed).

It speaks the exact same IPC as dashboard/simulation_process.py + run_simulation.run (status.json,
state.json, maps.json, commands.jsonl, stop.request) and runs the REAL product components:
SensorSuite (IMU/DVL/depth), the Sonar model (analytic ray cast instead of PyBullet), the 15-state EKF,
the MappingSystem, build_snapshot, LivePublisher, CommandChannel and apply_command.
What is NOT real here: the vehicle is a kinematic point following the waypoints (no hydrodynamics, no
thrusters) and the world is the analytic twin of the generated WorldSpec. Results from this process
must never be reported as PyBullet results.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from src.mapping.map_manager import MappingSystem
from src.sensors.models import TruthSnapshot
from src.sensors.sonar import SonarConfig
from src.simulation.commands import apply_command, deep_merge
from src.simulation.estimation import EstimationPipeline
from src.simulation.live_io import CommandChannel, LivePublisher
from src.simulation.run_protocol import ChildRun
from src.simulation.telemetry_snapshot import SensorStatsTracker, build_snapshot
from src.simulation.world_generator import generate_world
from src.utils.transforms import euler_to_quat
from tests.helpers.analytic_sonar import AnalyticSonar
from tests.helpers.analytic_world import world_from_spec
from tests.helpers.closed_loop import NumpyVehicle


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--runtime", required=True)
    ap.add_argument("--override", default=None)
    ap.add_argument("--run-id", default="standalone")
    ap.add_argument("--parent-pid", type=int, default=None)
    a = ap.parse_args(argv)
    child = ChildRun(a.runtime, a.run_id, a.parent_pid)
    rt = child.dir
    status = child.write_status

    status("STARTING")
    try:
        cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
        if a.override:
            cfg = deep_merge(cfg, json.loads(Path(a.override).read_text(encoding="utf-8")))
        sim = cfg["simulation"]
        env = cfg["environment"]
        scenario = env.get("scenario", "legacy_week4")
        if scenario == "legacy_week4":
            scenario = "mapping_validation"      # stand-in has no Week-4 world; use the validation scenario
        env = {**env, "scenario": scenario}
        spec = generate_world(env, sim, (int(env["grid_size"]["rows"]), int(env["grid_size"]["cols"]),
                                         float(env["resolution"])))
        world = world_from_spec(spec)
        start = np.array(sim["start_position"], float)
        wps = [np.array(w, float) for w in sim["waypoints"]]
        dt = float(sim["time_step"])
        est = EstimationPipeline(cfg, 0, start)
        est.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
        mapping = MappingSystem(cfg)
        navigator = veh = None
        if (cfg.get("navigation") or {}).get("enabled", False):       # Phase 5: REAL navigator, stand-in vehicle (NumpyVehicle)
            from src.planning.navigator import AutonomousNavigator
            navigator = AutonomousNavigator(cfg, command_frame="world")
            navigator.set_goal(navigator.nav.resolve_goal(cfg), 0.0, start=bool(cfg["navigation"].get("auto_start", True)))
            veh = NumpyVehicle(cfg, start)
            wps = [np.array(navigator.nav.resolve_goal(cfg), float)]
        dash = cfg.get("dashboard", {}) or {}
        pub = LivePublisher(rt, float(dash.get("state_publish_hz", 10.0)), float(dash.get("map_publish_hz", 2.0)))
        pub.publish_scene({**spec.scene_payload(), "conditions": {"current_mps": 0.0, "sonar_max_range_m": 30.0,
                           "seabed_type": spec.seabed_type, "wave_height_m": None, "seabed_z_range": [float(spec.heights.min()), float(spec.heights.max())]}})
        cmds = CommandChannel(rt / "commands.jsonl")
        tracker, log, est_traj, truth_traj = SensorStatsTracker(), deque(maxlen=20), deque(maxlen=800), deque(maxlen=800)
        run_state = {"realtime_factor": float(dash.get("realtime_factor", 0.0))}
        stop = child.start_watchers()
        sm = spec.summary()           # same source the real Environment.describe() uses
        env_desc = {"label": "GROUND TRUTH ENVIRONMENT (debug)", "scenario": sm["scenario"], "seed": sm["seed"],
                    "terrain_type": sm["seabed_type"], "obstacle_count": sm["obstacle_count"],
                    "structure_count": sm["structure_count"], "seabed_z_min": sm["seabed_z_min"],
                    "seabed_z_max": sm["seabed_z_max"], "obstacle_kinds": sm["obstacle_kinds"],
                    "obstacles_requested": sm["obstacles_requested"], "obstacles_fixed": sm.get("obstacles_fixed", 0),
                    "obstacles_placed": sm.get("obstacles_placed"), "procedural": True, "stand_in": True}

        pos, vel, wi, speed, last_id = start.copy(), np.zeros(3), 0, 1.0, 0
        yaw = 0.0; yaw_rate = 0.0; max_rate = 0.5      # spawn at identity attitude, like the real sim
        status("RUNNING")
        t_wall0, step, final = time.time(), 0, "COMPLETED"
        terminal_since = None
        while True:
            if stop.is_set():
                final = "STOPPED"
                break
            for c in cmds.poll():
                run_state["sim_time"] = step * dt
                ok, detail = apply_command(c, mapping, est, run_state, navigator)
                last_id = int(c.get("id", last_id))
                log.append({"id": last_id, "cmd": c.get("cmd"), "ok": ok, "detail": detail, "sim_time": step * dt})
            if navigator is None:
                tgt = wps[wi]
                d = tgt - pos
                dist = float(np.linalg.norm(d))
                if dist < 0.5:
                    if wi == len(wps) - 1:
                        break
                    wi += 1
                    continue
                # kinematic unicycle: constant surge speed, rate-limited heading change (TEST-ONLY truth)
                want = float(np.arctan2(d[1], d[0]))
                err = (want - yaw + np.pi) % (2 * np.pi) - np.pi
                yaw_rate = float(np.clip(err / dt, -max_rate, max_rate))
                yaw += yaw_rate * dt
                vel = speed * np.array([np.cos(yaw), np.sin(yaw), 0.0])
                pos = pos + vel * dt
                q = euler_to_quat(0.0, 0.0, yaw)
                truth = TruthSnapshot(time=step * dt, position=pos.copy(), quat=q, velocity_world=vel.copy(),
                                      angular_velocity_world=np.array([0.0, 0.0, yaw_rate]))
            else:
                truth = veh.truth(step * dt)
                pos, yaw = veh.p.copy(), veh.yaw
            meas, estimate = est.update_from_truth(truth)
            tracker.observe(meas)
            mapping.process(step * dt, meas.get("sonar"), estimate)
            if navigator is not None:
                run_state["sim_time"] = step * dt
                out = navigator.update(step * dt, estimate, mapping, est.latest.get("sonar"))
                veh.step(out.control, dt)
                if navigator.state.value in ("GOAL_REACHED", "FAILED"):
                    terminal_since = step * dt if terminal_since is None else terminal_since
                    if step * dt - terminal_since >= float((cfg.get("navigation") or {}).get("linger_after_terminal_s", 2.0)):
                        final = "COMPLETED" if navigator.state.value == "GOAL_REACHED" else "ENDED_BEFORE_GOAL"
                        break
                else:
                    terminal_since = None
            if step % 10 == 0:
                est_traj.append([float(estimate.position[0]), float(estimate.position[1])])
                truth_traj.append([float(pos[0]), float(pos[1])])
            if pub.due_state() or step == 0:
                snap = build_snapshot(
                    step=step, sim_time=step * dt, dt=dt, status="RUNNING", dynamics_mode="stand-in (kinematic)",
                    estimation=est, estimate=estimate, measurements=meas, sensor_tracker=tracker, mapping=mapping,
                    env_desc=env_desc, truth_position=pos, truth_euler=(0.0, 0.0, yaw), auv=None, control=None,
                    controller_info={"type": "TEST STAND-IN (kinematic)"},
                    perf={"sim_seconds_per_wall_second": (step * dt) / max(time.time() - t_wall0, 1e-9),
                          "realtime_factor_setting": run_state["realtime_factor"]}, last_command_id=last_id,
                    mapping_unavailable_reason=None, trajectory_tail=list(est_traj),
                    navigation=None if navigator is None else navigator.telemetry(),
                    navigation_path=None if navigator is None else navigator.path_payload())
                snap["command_log"] = list(log)
                snap["mission"] = {"start": start.tolist(), "goal": wps[-1].tolist(), "waypoints": [w.tolist() for w in wps]}
                snap["debug_ground_truth"]["trajectory"] = list(truth_traj)
                pub.publish_state(snap)
            if pub.due_maps(mapping.version):
                pub.publish_maps({"occupancy": mapping.occupancy_payload(), "costmap": mapping.costmap_payload()}, mapping.version)
            step += 1
            rf = run_state["realtime_factor"]
            if rf > 0:
                ahead = step * dt / rf - (time.time() - t_wall0)
                if ahead > 0.002:
                    time.sleep(ahead)
        # final snapshot
        snap = build_snapshot(step=step, sim_time=step * dt, dt=dt, status=final, dynamics_mode="stand-in (kinematic)",
                              estimation=est, estimate=est.ekf.state, measurements={}, sensor_tracker=tracker,
                              mapping=mapping, env_desc=env_desc, truth_position=pos, truth_euler=None, auv=None,
                              control=None, controller_info={"type": "TEST STAND-IN"}, perf={}, last_command_id=last_id,
                              mapping_unavailable_reason=None, trajectory_tail=list(est_traj),
                              navigation=None if navigator is None else navigator.telemetry(),
                              navigation_path=None if navigator is None else navigator.path_payload())
        snap["command_log"] = list(log)
        pub.publish_state(snap)
        pub.publish_maps({"occupancy": mapping.occupancy_payload(), "costmap": mapping.costmap_payload()}, mapping.version)
        status(final, result={"final_status": final, "mapping": mapping.telemetry()})
        return 0
    except BaseException as exc:   # noqa: BLE001
        status("ERROR", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

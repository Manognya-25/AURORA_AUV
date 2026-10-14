"""TEST-ONLY closed-loop harness that runs WITHOUT PyBullet.

Real product components:  SensorSuite (IMU / DVL / depth), analytic-ray-cast Sonar (the product Sonar model with the
PyBullet ray cast replaced), 15-state EKF, MappingSystem (occupancy + cost map), AutonomousNavigator.
Stand-in (NOT PyBullet): the vehicle is a rigid body integrated here with the SAME equations as the legacy
dynamics (world-frame force from the normalised command, linear drag, neutral buoyancy, yaw torque with the
URDF's Izz and angular drag) -- `NumpyVehicle`. Collisions are detected against the analytic footprint of the
generated world (EVALUATION only; the navigator never sees it).
Results from this module must never be reported as PyBullet results.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import yaml

from src.mapping.map_manager import MappingSystem
from src.planning.navigator import AutonomousNavigator, NavState
from src.sensors.models import TruthSnapshot
from src.sensors.sonar import SonarConfig
from src.simulation.commands import deep_merge
from src.simulation.estimation import EstimationPipeline
from src.simulation.world_generator import generate_world
from src.utils.transforms import euler_to_quat
from tests.helpers.analytic_sonar import AnalyticSonar
from tests.helpers.analytic_world import world_from_spec


def load_cfg(override: dict | None = None) -> dict:
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return deep_merge(cfg, override)


class NumpyVehicle:
    """Single rigid body, legacy force model (see simulation/physics.py + auv.py)."""

    def __init__(self, cfg: dict, position, yaw: float = 0.0):
        sim = cfg["simulation"]
        self.m = float(sim["auv"]["mass"])
        self.c_lin = float(sim["physics"]["linear_drag"])
        self.c_ang = float(sim["physics"]["angular_drag"])
        self.Izz = 4.0625                                    # assets/auv/auv.urdf
        self.max_thrust = float(sim["controller"]["max_thrust"])
        self.max_torque = float(sim["controller"]["max_yaw_torque"])
        self.p = np.array(position, float)
        self.v = np.zeros(3)
        self.yaw, self.r = float(yaw), 0.0

    def step(self, cmd: dict, dt: float) -> None:
        f = np.clip([cmd["forward"], cmd["lateral"], cmd["vertical"]], -1, 1) * self.max_thrust
        tq = float(np.clip(cmd["yaw"], -1, 1)) * self.max_torque
        a = (f - self.c_lin * self.v) / self.m                # buoyancy == weight (net zero), legacy world-frame force
        self.v = self.v + a * dt
        self.p = self.p + self.v * dt
        self.r += (tq - self.c_ang * self.r) / self.Izz * dt
        self.yaw += self.r * dt

    def truth(self, t: float) -> TruthSnapshot:
        return TruthSnapshot(time=t, position=self.p.copy(), quat=euler_to_quat(0.0, 0.0, self.yaw),
                             velocity_world=self.v.copy(), angular_velocity_world=np.array([0.0, 0.0, self.r]))


@dataclass
class LoopResult:
    final_state: str
    sim_time: float
    steps: int
    truth_final: np.ndarray
    est_final: np.ndarray
    goal: np.ndarray
    truth_path: np.ndarray
    est_path: np.ndarray
    history: list
    telemetry: dict
    collisions: int
    min_clearance_m: float
    states_visited: list
    navigator: object = None
    mapping: object = None
    nav_calls: int = 0
    truth_goal_distance: float = 0.0
    planned_paths: list = field(default_factory=list)


def circle_clearance(spec, p_xy) -> float:
    """EVALUATION ONLY: distance from p to the nearest generated obstacle footprint (negative = inside)."""
    best = np.inf
    for o in spec.obstacles:
        pts = np.asarray(o.footprint, float)
        d = float(np.min(np.linalg.norm(pts - np.asarray(p_xy)[None, :], axis=1))) - o.footprint_radius
        best = min(best, d)
    return best


def run_closed_loop(override: dict | None = None, goal=None, max_time: float = 140.0,
                    goal_updates: list | None = None, pre_step=None, cfg: dict | None = None, estimate_hook=None) -> LoopResult:
    """Run the full sense -> EKF -> map -> plan -> follow loop. goal_updates: [(sim_time, [x,y,z]), ...] operator goal changes.
    estimate_hook: optional f(EstimatedState) -> EstimatedState applied to the estimate before it reaches the mapper and
    the navigator (used to prove that decisions follow the ESTIMATE, not the truth)."""
    # The navigation-LOGIC stand-in missions were validated with gyro-only yaw (no heading reference), which makes the map
    # and the pose drift enough to force replans/avoidance. Keep that harder case by default; tests of the compass pass
    # {"sensors": {"compass": {"enabled": True}}} (see tests/test_compass.py::test_closed_loop_mission_with_compass_*).
    base = {"navigation": {"enabled": True}, "sensors": {"compass": {"enabled": False}}}
    cfg = cfg or load_cfg(deep_merge(base, override))
    sim, env = cfg["simulation"], cfg["environment"]
    spec = generate_world(env, sim, (int(env["grid_size"]["rows"]), int(env["grid_size"]["cols"]), float(env["resolution"])))
    world = world_from_spec(spec)
    extra = []
    start = np.array(sim["start_position"], float)
    dt = float(sim["time_step"])
    veh = NumpyVehicle(cfg, start)
    est = EstimationPipeline(cfg, 0, start)
    est.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
    mapping = MappingSystem(cfg)
    nav = AutonomousNavigator(cfg, command_frame="world")
    goal = np.array(goal if goal is not None else nav.nav.resolve_goal(cfg), float)
    nav.set_goal(goal, 0.0)
    updates = sorted(goal_updates or [], key=lambda u: u[0])
    tp, ep, states, collisions, minclr, calls = [], [], [], 0, np.inf, 0
    planned = []
    t, step = 0.0, 0
    nsteps = int(max_time / dt)
    for step in range(nsteps):
        t = step * dt
        while updates and updates[0][0] <= t:
            nav.set_goal(updates.pop(0)[1], t)
        if pre_step is not None:
            pre_step(step, t, nav, mapping, est, veh)
        meas, estimate = est.update_from_truth(veh.truth(t))
        if estimate_hook is not None:             # test-only: corrupt the estimate the navigator and mapper see
            estimate = estimate_hook(estimate)
        mapping.process(t, meas.get("sonar"), estimate)
        out = nav.update(t, estimate, mapping, est.latest.get("sonar"))
        calls += 1
        veh.step(out.control, dt)
        if step % 20 == 0:
            tp.append(veh.p.copy()); ep.append(np.asarray(estimate.position).copy())
            c = circle_clearance(spec, veh.p[:2])
            minclr = min(minclr, c)
        if nav.global_path is not None and (not planned or planned[-1][0] != nav.global_path.version):
            planned.append((nav.global_path.version, nav.global_path.xy.copy()))
        if nav.state in (NavState.GOAL_REACHED, NavState.FAILED):
            break
    # evaluation-only collision proxy: inside any obstacle footprint (vehicle radius 0.25 m)
    for p_ in tp:
        if circle_clearance(spec, p_[:2]) < 0.25:
            collisions += 1
    states = [h['to'] for h in nav.history]
    return LoopResult(nav.state.value, t, step + 1, veh.p.copy(), np.asarray(nav._est_pos if nav._est_pos is not None else est.ekf.state.position).copy(), goal,
                      np.array(tp), np.array(ep), list(nav.history), nav.telemetry(), collisions, float(minclr), states,
                      nav, mapping, calls, float(np.linalg.norm(veh.p - goal)), planned)

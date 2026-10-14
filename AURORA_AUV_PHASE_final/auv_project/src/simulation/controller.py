"""
controller.py
--------------
A simple proportional (P) waypoint controller: given the AUV's current
position/yaw and a target waypoint, computes normalized thruster commands
that drive it toward the target.

This module keeps the plain P law (`WaypointController`) and the parameters
it shares with the Week-9 ADAPTIVE controller
(`src/simulation/adaptive_controller.AdaptivePController`), which is a
subclass that leaves the mid-mission command UNCHANGED far from the target
and only re-shapes the PROPORTIONAL GAIN near the set-point / while stalled.
It also does not do any path planning; it drives toward whatever single
target position it's given, one waypoint at a time (Week 6's A* planner
will supply a sequence of these).

Pipeline implemented here:

    current position/yaw, target position
                       |
                       v
      compute_control()   -> forward/lateral/vertical/yaw commands in [-1, 1]
                       |
                       v
      reached_waypoint()  -> bool, True once distance < waypoint_tolerance
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields

import numpy as np


@dataclass
class AdaptiveParams:
    """Adaptive-P knobs (config: simulation.controller.adaptive.*).

    The proportional gain is NOT constant. Two bounded, deterministic mechanisms act on it:

    1. GAIN SCHEDULE (feed-forward, no state): far from the target the nominal gain is used, so the mid-mission
       command is the plain P law; inside `near_m` the gain rises to `kp * gain_boost`. That extra authority is
       what closes the last centimetres against the (deliberately positive) legacy buoyancy and linear drag,
       where a nominal P term alone leaves the vehicle short of the set-point.
    2. SELF-TUNING (state, needs `dt`): while the vehicle is STALLED (closing speed below `stall_progress_mps`)
       the gain ramps up at `adapt_rate_per_s`; while it is TRACKING WELL (closing speed at/above
       `healthy_progress_mps`, or already inside `settle_deadband_m`) the gain relaxes back at
       `relax_rate_per_s`. Both directions are clamped to [1, `gain_boost`], so the controller can never wind
       the gain up without bound and the gains are reproducible step-for-step.

    The yaw loop is gain-scheduled the other way round: full gain when nearly aligned (`yaw_near_rad`) so the
    heading settles, floored at `yaw_gain_min` for large heading errors so the vehicle does not spin while the
    horizontal error is still being reduced.
    """
    enabled: bool = True
    near_m: float = 1.0                      # at/below this horizontal error the schedule is at its maximum
    far_m: float = 6.0                       # at/above this error the nominal gain is used
    gain_boost: float = 3.0                  # max multiplier on kp_position (schedule AND self-tuning share it)
    adapt_rate_per_s: float = 0.6            # gain units/s added while stalled
    relax_rate_per_s: float = 1.5            # gain units/s removed while tracking well
    stall_progress_mps: float = 0.02         # closing speed below this counts as stalled
    healthy_progress_mps: float = 0.15       # closing speed at/above this counts as tracking well
    settle_deadband_m: float = 0.05          # inside this error the gain relaxes (nothing left to push)
    target_change_jump_m: float = 5.0        # a jump larger than this is a NEW target: re-seed, do not adapt
    yaw_near_rad: float = 0.25               # at/below this heading error the yaw gain is nominal
    yaw_far_rad: float = 1.2                 # at/above this heading error the yaw gain is floored
    yaw_gain_min: float = 0.5                # floor multiplier for large heading errors

    def __post_init__(self):
        if not self.near_m > 0.0:
            raise ValueError("adaptive.near_m must be > 0")
        if not self.far_m > self.near_m:
            raise ValueError("adaptive.far_m must be > adaptive.near_m")
        if self.gain_boost < 1.0:
            raise ValueError("adaptive.gain_boost must be >= 1")
        if self.adapt_rate_per_s < 0.0 or self.relax_rate_per_s < 0.0:
            raise ValueError("adaptive rates must be >= 0")
        if not 0.0 <= self.stall_progress_mps < self.healthy_progress_mps:
            raise ValueError("need 0 <= adaptive.stall_progress_mps < adaptive.healthy_progress_mps")
        if self.settle_deadband_m < 0.0:
            raise ValueError("adaptive.settle_deadband_m must be >= 0")
        if not 0.0 < self.yaw_near_rad < self.yaw_far_rad:
            raise ValueError("need 0 < adaptive.yaw_near_rad < adaptive.yaw_far_rad")
        if not 0.0 < self.yaw_gain_min <= 1.0:
            raise ValueError("adaptive.yaw_gain_min must be in (0, 1]")

    @classmethod
    def from_config(cls, section: dict | None) -> "AdaptiveParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (section or {}).items() if k in known})


@dataclass
class ControllerParams:
    """Proportional-controller gains and limits (config: simulation.controller.*).

    `kd_velocity`, `goal_tolerance` and `goal_settle_timeout_s` are used ONLY for the terminal approach to the
    goal (the last waypoint): the mid-mission control law stays purely proportional, and the extra velocity
    damping + buoyancy trim let the vehicle settle ON the configured goal instead of hunting around it.

    `adaptive` selects the Week-9 AdaptivePController instead of the plain WaypointController; with
    `adaptive.enabled: false` the two behave identically.
    """
    kp_position: float
    kp_yaw: float
    max_thrust: float
    max_yaw_torque: float
    waypoint_tolerance: float
    kd_velocity: float = 7.0                 # N per m/s: velocity feedback in the goal approach only
    goal_tolerance: float = 0.1              # m: the mission ends once the goal is really reached
    goal_settle_timeout_s: float = 40.0      # simulated seconds allowed for that final approach
    adaptive: AdaptiveParams = field(default_factory=AdaptiveParams)

    @classmethod
    def from_config(cls, controller_cfg: dict) -> "ControllerParams":
        return cls(
            kp_position=float(controller_cfg["kp_position"]),
            kp_yaw=float(controller_cfg["kp_yaw"]),
            max_thrust=float(controller_cfg["max_thrust"]),
            max_yaw_torque=float(controller_cfg["max_yaw_torque"]),
            waypoint_tolerance=float(controller_cfg["waypoint_tolerance"]),
            kd_velocity=float(controller_cfg.get("kd_velocity", 7.0)),
            goal_tolerance=float(controller_cfg.get("goal_tolerance", 0.1)),
            goal_settle_timeout_s=float(controller_cfg.get("goal_settle_timeout_s", 40.0)),
            adaptive=AdaptiveParams.from_config(controller_cfg.get("adaptive")),
        )


class WaypointController:
    """Proportional controller driving the AUV toward one waypoint at a time."""

    def __init__(self, params: ControllerParams):
        self.params = params

    def compute_control(self, current_position: tuple, current_yaw: float, target_position: tuple,
                        current_velocity: tuple | None = None, vertical_trim: float = 0.0) -> dict:
        """
        Input:
            current_position: (x, y, z) world position of the AUV.
            current_yaw: current heading about world Z, in radians.
            target_position: (x, y, z) waypoint to move toward.
            current_velocity: optional (vx, vy, vz) world velocity. When given, a velocity feedback term
                (-kd_velocity * v) is added: the TERMINAL approach to the goal uses it so the vehicle settles on
                the goal instead of oscillating around it. Mid-mission calls leave it None (pure P).
            vertical_trim: optional normalised vertical command added to the P term, used to cancel a known
                residual buoyancy while station-keeping at the goal (feed-forward, not an integral term).
        Processing:
            1. error = target - current (STEP 19). Proportional control
               control = kp_position * error, converted to a normalized
               [-1, 1] command by dividing by max_thrust and clamping (the
               same [-1, 1] convention AUV.apply_control expects).
            2. desired_yaw = atan2(dy, dx) (STEP 20); yaw_error is
               normalized into [-pi, pi] via atan2(sin, cos) rather than
               plain subtraction, so it always takes the shorter turn.
               yaw command = clip(kp_yaw * yaw_error / max_yaw_torque, -1, 1).
            3. Simplification (documented, not hidden): forward/lateral
               commands use the world-frame X/Y error directly rather than
               rotating it into the AUV's body frame using current_yaw.
               This keeps the controller a pure P-controller as specified
               and is sufficient for the basic waypoint-following
               demonstration -- a heading-aware body-frame controller is a
               natural Week 9 improvement (see docs/simulation.md).
        Output:
            dict with keys "forward", "lateral", "vertical", "yaw", each a
            float clamped to [-1, 1].
        """
        current = np.array(current_position, dtype=float)
        target = np.array(target_position, dtype=float)
        error = target - current

        position_control = self.params.kp_position * error
        if current_velocity is not None:
            position_control = position_control - self.params.kd_velocity * np.asarray(current_velocity, float)
        forward = float(np.clip(position_control[0] / self.params.max_thrust, -1.0, 1.0))
        lateral = float(np.clip(position_control[1] / self.params.max_thrust, -1.0, 1.0))
        vertical = float(np.clip(position_control[2] / self.params.max_thrust + vertical_trim, -1.0, 1.0))

        dx, dy = error[0], error[1]
        desired_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(
            math.sin(desired_yaw - current_yaw), math.cos(desired_yaw - current_yaw)
        )
        yaw = float(np.clip(self.params.kp_yaw * yaw_error / self.params.max_yaw_torque, -1.0, 1.0))

        return {"forward": forward, "lateral": lateral, "vertical": vertical, "yaw": yaw}

    def reached_waypoint(self, current_position: tuple, target_position: tuple) -> bool:
        """
        Input: current_position, target_position: (x, y, z) tuples.
        Processing: Euclidean distance < config.simulation.controller.waypoint_tolerance (STEP 21).
        Output: bool.
        """
        return self.distance_to(current_position, target_position) < self.params.waypoint_tolerance

    def distance_to(self, current_position: tuple, target_position: tuple) -> float:
        """Euclidean distance between current and target positions, in metres."""
        current = np.array(current_position, dtype=float)
        target = np.array(target_position, dtype=float)
        return float(np.linalg.norm(target - current))

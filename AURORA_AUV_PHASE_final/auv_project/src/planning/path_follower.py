"""Path follower: lookahead-carrot + velocity controller driven by the EKF ESTIMATE only.

    estimate (position, yaw, velocity, yaw-rate)  +  PathData
        -> progress along the path (monotone arc length s, projection of the ESTIMATED xy)
        -> carrot point at s + lookahead
        -> desired world velocity  v_d = v_cmd * unit(carrot - p_est)           (horizontal)
                                   vz_d = kp_depth * (z_path - z_est)           (vertical, rate-limited)
        -> force   F = kv * (v_d - v_est) + c_drag * v_d      (+ integral depth hold)
        -> normalised command  F / max_thrust  in [-1, 1]
    Heading (yaw) is a PD loop on the estimated yaw / gyro yaw rate toward the carrot direction, so the forward
    sonar looks where the vehicle is going; speed is reduced while the heading error is large.

FRAME HANDLING (documented, not hidden): the legacy dynamics apply `forward`/`lateral` as WORLD-frame X/Y
forces; the 6-DOF model treats them as BODY-frame force requests. `command_frame` selects which; for "body"
the world force is rotated by the ESTIMATED yaw. The existing thruster / dynamics code is untouched; this class
only produces the same normalised {forward, lateral, vertical, yaw} dictionary the Phase-2 controller produces.
Nothing here imports PyBullet or any truth source.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.planning.config import FollowerConfig
from src.planning.path_data import PathData
from src.utils.transforms import wrap_angle


@dataclass
class FollowCommand:
    forward: float = 0.0
    lateral: float = 0.0
    vertical: float = 0.0
    yaw: float = 0.0
    info: dict = field(default_factory=dict)

    def as_control(self) -> dict:
        return {"forward": self.forward, "lateral": self.lateral, "vertical": self.vertical, "yaw": self.yaw}


class PathFollower:
    def __init__(self, cfg: FollowerConfig | None = None):
        self.cfg = cfg or FollowerConfig()
        self.path: PathData | None = None
        self.s = 0.0                       # progress (arc length of the projection of the ESTIMATED position)
        self.cross_track = 0.0
        self._i_depth = 0.0
        self._last_t: float | None = None
        self._heading_ref: float | None = None
        self._last_gains: dict = {}        # the gains of the last command (see gain_report())
        self._track = self._new_tracking()
        self._was_slowing = self._was_braking = False

    # ------------------------------------------------------------------------------ path handling
    def set_path(self, path: PathData, start_s: float = 0.0) -> None:
        self.path = path
        self.s = float(np.clip(start_s, 0.0, path.length))
        self.cross_track = 0.0
        self._heading_ref = None

    def clear(self) -> None:
        self.path = None
        self.s = 0.0
        self._i_depth = 0.0
        self._last_t = None

    @property
    def remaining_m(self) -> float:
        return 0.0 if self.path is None else max(0.0, self.path.length - self.s)

    @property
    def waypoint_index(self) -> int:
        return 0 if self.path is None else self.path.next_vertex_index(self.s)

    def current_waypoint(self):
        if self.path is None:
            return None
        return self.path.point(self.path.cum[self.waypoint_index])

    # ------------------------------------------------------------------------------ gain reporting
    @staticmethod
    def _new_tracking() -> dict:
        """A fresh record of what the follower's gain schedule has done so far (see gain_report())."""
        return {"steps": 0, "elapsed_s": 0.0, "mode": "idle",
                "follow_s": 0.0, "hold_s": 0.0, "station_s": 0.0,
                "full_speed_s": 0.0, "slowed_s": 0.0, "braked_s": 0.0,
                "speed_factor_now": 0.0, "speed_factor_mean": 0.0, "speed_factor_min": 1.0,
                "heading_factor_min": 1.0, "sonar_factor_min": 1.0, "slows": 0, "brakes": 0}

    def reset_tracking(self) -> None:
        """Forget the tracked schedule history: called when a NEW navigation mission begins (not on a replan, which
        only swaps the path - `clear()` deliberately leaves the tracking alone)."""
        self._track = self._new_tracking()
        self._last_gains = {}
        self._was_slowing = self._was_braking = False

    @property
    def tracking(self) -> dict:
        """The tracked schedule history so far (a copy, so a caller cannot mutate the follower's own record)."""
        return dict(self._track)

    def gain_report(self) -> dict:
        """The gains and the live schedule the follower is ACTUALLY applying (the telemetry `controller.follower` block).

        Two different things are reported, in the units they are really used in:
          * the LOOP GAINS (kv, kp_yaw/kd_yaw, kp_depth/ki_depth, station_kp) - the numbers that turn an error into a
            force, exactly as configured; the follower does not reshape them;
          * the SPEED SCHEDULE that shapes the command dynamically: the commanded speed as a fraction of cruise
            (`speed_factor`) and the three real factors that produced it - distance to the goal (`goal_factor`),
            heading error (`heading_factor`) and the sonar/avoidance scale (`sonar_factor`). Those are read from the
            command that was just produced, so they move as the vehicle does; the min-speed floor can make
            `speed_factor` slightly larger than their product, which is reported rather than hidden.
        Plus the live error/state terms (desired speed, lookahead, cross-track, heading/depth errors, depth integral)
        and the tracked schedule history (`tracking`). Nothing here is a nominal value copied from a config file.
        """
        rep = dict(self._last_gains)
        rep["tracking"] = dict(self._track)
        return rep

    def _track_step(self, dt: float, speed_factor: float, mode: str, info: dict) -> None:
        """Fold one command into the tracked schedule history (cumulative, bounded, deterministic)."""
        t = self._track
        f = float(min(1.0, max(0.0, speed_factor)))
        slowing, braking = f < 1.0 - 1e-9, f <= 1e-9
        if dt > 0.0:
            t["steps"] += 1
            t["elapsed_s"] += float(dt)
            if mode == "hold":
                t["hold_s"] += float(dt)
            elif mode == "station_keep":
                t["station_s"] += float(dt)
            else:
                t["follow_s"] += float(dt)
            if braking:
                t["braked_s"] += float(dt)
            elif slowing:
                t["slowed_s"] += float(dt)
            else:
                t["full_speed_s"] += float(dt)
        t["mode"] = mode
        t["speed_factor_now"] = f
        t["speed_factor_mean"] = ((t["speed_factor_mean"] * (t["steps"] - 1) + f) / t["steps"]) if t["steps"] else f
        t["speed_factor_min"] = min(t["speed_factor_min"], f)
        for key, name in (("heading_factor", "heading_factor_min"), ("sonar_factor", "sonar_factor_min")):
            v = info.get(key)
            if isinstance(v, (int, float)):
                t[name] = min(t[name], float(v))
        # "slows"/"brakes" are EPISODES (a rising edge), so one long slow section is counted once, like a ramp episode
        if slowing and not self._was_slowing:
            t["slows"] += 1
        if braking and not self._was_braking:
            t["brakes"] += 1
        self._was_slowing, self._was_braking = slowing, braking

    # ------------------------------------------------------------------------------ control
    def hold(self, estimate, target_z: float | None = None) -> FollowCommand:
        """Zero-velocity set-point (brake + depth hold) used outside path following."""
        return self._command(estimate, np.zeros(2), target_z if target_z is not None else float(estimate.position[2]),
                             None, {"mode": "hold", "desired_speed": 0.0})

    def station_keep(self, estimate, goal) -> FollowCommand:
        """GOAL_REACHED: hold the goal position (EKF estimate only): gentle P on the horizontal estimate error, depth hold at
        the goal depth, rotation damped. Inside the deadband no horizontal force is applied beyond braking."""
        c = self.cfg
        g = np.asarray(goal, float)
        e = g[:2] - np.asarray(estimate.position, float)[:2]
        d = float(np.linalg.norm(e))
        v_d = np.zeros(2)
        if d > c.station_deadband_m:
            v_d = c.station_kp * e
            n = float(np.linalg.norm(v_d))
            if n > c.station_max_speed_mps:
                v_d *= c.station_max_speed_mps / n
        return self._command(estimate, v_d, float(g[2]), None,
                             {"mode": "station_keep", "desired_speed": float(np.linalg.norm(v_d)), "goal_error_m": d})

    def compute(self, estimate, speed_scale: float = 1.0) -> FollowCommand:
        if self.path is None or self.path.n == 0:
            return self.hold(estimate)
        c, path = self.cfg, self.path
        p = np.asarray(estimate.position, float)
        v = np.asarray(estimate.velocity, float)
        speed = float(np.linalg.norm(v[:2]))
        # progress: closest point at-or-ahead of the current progress (never backwards), searched over a bounded window
        s_new, xt, _ = path.project(p[:2], self.s, self.s + max(8.0, 4.0 * c.lookahead_min_m))
        self.s = max(self.s, s_new)
        self.cross_track = xt
        L = c.lookahead_min_m + c.lookahead_gain_s * speed
        s_look = min(self.s + L, path.length)
        carrot = path.point(s_look)
        to_c = carrot[:2] - p[:2]
        dist_c = float(np.linalg.norm(to_c))
        dist_end = float(np.linalg.norm(path.xy[-1] - p[:2]))
        remaining = max(path.length - self.s, dist_end)

        # desired speed: cruise, smooth stop at the end of the path, slow for large heading error, sonar scale
        goal_factor = min(1.0, max(c.goal_slowdown_gain * remaining, 0.0) / c.cruise_speed_mps)
        v_cmd = min(c.cruise_speed_mps, max(c.goal_slowdown_gain * remaining, 0.0))
        if remaining > 1.0:
            v_cmd = max(v_cmd, c.min_speed_mps)
        yaw = float(estimate.euler[2])
        if dist_c > 1e-6:
            desired_heading = math.atan2(to_c[1], to_c[0])
            self._heading_ref = desired_heading
        elif self._heading_ref is not None:
            desired_heading = self._heading_ref
        else:
            desired_heading = yaw
        yaw_err = wrap_angle(desired_heading - yaw)
        a = abs(yaw_err)
        heading_factor = 1.0
        if a > c.heading_slow_start_rad:
            f = 1.0 - (a - c.heading_slow_start_rad) / max(c.heading_slow_end_rad - c.heading_slow_start_rad, 1e-6)
            heading_factor = float(np.clip(f, c.heading_min_speed_frac, 1.0))
            v_cmd *= heading_factor
        sonar_factor = float(np.clip(speed_scale, 0.0, 1.0))
        v_cmd *= sonar_factor
        v_d = np.zeros(2) if dist_c < 1e-6 else v_cmd * to_c / dist_c
        return self._command(estimate, v_d, float(carrot[2]), yaw_err,
                             {"mode": "follow", "desired_speed": v_cmd, "lookahead_m": L, "carrot": carrot.tolist(),
                              "heading_error_rad": yaw_err, "cross_track_m": xt, "speed_scale": sonar_factor,
                              "goal_factor": goal_factor, "heading_factor": heading_factor,
                              "sonar_factor": sonar_factor})

    # ------------------------------------------------------------------------------ internals
    def _command(self, estimate, v_d_xy: np.ndarray, z_target: float, yaw_err, info: dict) -> FollowCommand:
        c = self.cfg
        p = np.asarray(estimate.position, float)
        v = np.asarray(estimate.velocity, float)
        t = float(estimate.time)
        dt = 0.0 if self._last_t is None else float(np.clip(t - self._last_t, 0.0, 0.1))
        self._last_t = t

        vz_d = float(np.clip(c.kp_depth * (z_target - p[2]), -c.max_vertical_speed_mps, c.max_vertical_speed_mps))
        F_xy = c.kv * (v_d_xy - v[:2]) + c.drag_feedforward * v_d_xy
        ez = z_target - p[2]
        self._i_depth = float(np.clip(self._i_depth + c.ki_depth * ez * dt, -c.depth_i_limit_n, c.depth_i_limit_n))
        F_z = c.kv * (vz_d - v[2]) + c.drag_feedforward * vz_d + self._i_depth

        yaw_rate = float(estimate.angular_velocity[2])
        torque = 0.0 if yaw_err is None else c.kp_yaw * yaw_err - c.kd_yaw * yaw_rate
        if yaw_err is None:
            torque = -c.kd_yaw * yaw_rate                      # hold: damp rotation, do not chase a heading

        if c.command_frame == "body":
            cy, sy = math.cos(float(estimate.euler[2])), math.sin(float(estimate.euler[2]))
            Fx = cy * F_xy[0] + sy * F_xy[1]                   # R_z(yaw)^T applied to the world force
            Fy = -sy * F_xy[0] + cy * F_xy[1]
        else:
            Fx, Fy = float(F_xy[0]), float(F_xy[1])

        n = c.max_thrust_n
        mode = str(info.get("mode") or "follow")
        desired_speed = float(np.linalg.norm(v_d_xy))
        speed_factor = float(min(1.0, desired_speed / c.cruise_speed_mps))
        cmd = FollowCommand(
            forward=float(np.clip(Fx / n, -1.0, 1.0)), lateral=float(np.clip(Fy / n, -1.0, 1.0)),
            vertical=float(np.clip(F_z / n, -1.0, 1.0)), yaw=float(np.clip(torque / c.max_yaw_torque_nm, -1.0, 1.0)),
            info={**info, "depth_error_m": ez, "frame": c.command_frame,
                  "desired_velocity": [float(v_d_xy[0]), float(v_d_xy[1]), vz_d]})
        self._last_gains = {
            "available": True, "source": "path follower (Phase 5)", "mode": mode, "command_frame": c.command_frame,
            # the loop gains, in the units they are applied in
            "kv_n_per_mps": float(c.kv), "kp_yaw_nm_per_rad": float(c.kp_yaw), "kd_yaw_nm_per_rad_s": float(c.kd_yaw),
            "kp_depth_mps_per_m": float(c.kp_depth), "ki_depth_n_per_m_s": float(c.ki_depth),
            "station_kp_per_s": float(c.station_kp), "cruise_speed_mps": float(c.cruise_speed_mps),
            # the live speed schedule (dynamic): the commanded speed as a fraction of cruise and its real factors
            "speed_factor": speed_factor, "goal_factor": info.get("goal_factor"),
            "heading_factor": info.get("heading_factor"), "sonar_factor": info.get("sonar_factor"),
            "desired_speed_mps": desired_speed, "lookahead_m": info.get("lookahead_m"),
            "cross_track_m": info.get("cross_track_m"), "heading_error_rad": info.get("heading_error_rad"),
            "depth_error_m": ez, "depth_integral_n": float(self._i_depth),
            "command": cmd.as_control(),
        }
        self._track_step(dt, speed_factor, mode, info)
        return cmd

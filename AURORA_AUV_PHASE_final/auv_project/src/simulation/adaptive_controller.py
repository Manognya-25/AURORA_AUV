"""
adaptive_controller.py
----------------------
Week 9: an ADAPTIVE proportional waypoint controller.

It is a drop-in replacement for `WaypointController` (same constructor, same
`compute_control` / `reached_waypoint` / `distance_to` interface, same
world-frame, normalised [-1, 1] output), so nothing about the vehicle, the
dynamics or the command convention changes -- only HOW the proportional gain
is chosen. The error -> command pipeline is identical:

    error = target - current
    command = kp_effective * error / max_thrust      (clamped to [-1, 1])

What makes it adaptive (see `AdaptiveParams` in controller.py for the knobs
and the full rationale):

  1. GAIN SCHEDULE.  The gain is a function of the position error: nominal
     far from the target, rising smoothly to `kp * gain_boost` inside
     `near_m`. The mid-mission command is therefore the plain P law; only
     the final approach -- where a nominal P term is too weak to overcome
     the deliberately positive legacy buoyancy and linear drag -- gets more
     authority, so the vehicle actually settles ON the target instead of
     stopping short of it.
  2. SELF-TUNING.  When `dt` is supplied the controller watches its own
     closing speed. While the vehicle is stalled it ramps the gain up; while
     it is tracking well (or sitting inside the settle deadband) it relaxes
     the gain back down. The gain is clamped to [1, gain_boost] in both
     directions, so the adaptation is bounded and reproducible: the same
     inputs produce the same gain sequence every run.

The yaw loop is scheduled the other way round: full gain when nearly aligned
(so the heading settles) and a floor for large heading errors (so the vehicle
does not spin while the horizontal error is still being reduced).

Every command also TRACKS the gain: which way the self-tuning pushed and why,
the measured closing speed, the simulated time spent ramping / relaxing /
boosted / at the gain cap, the tracked peak and mean gain, and the number of
adaptation episodes and re-seeds (`last_info["tracking"]`, shown live in the
dashboard and recorded in the exported CSV).

Nothing here imports PyBullet: this module is pure numpy + the P controller.
"""

from __future__ import annotations

import numpy as np

from src.simulation.controller import AdaptiveParams, ControllerParams, WaypointController


def gain_schedule(error: float, near_m: float, far_m: float, boost: float) -> float:
    """Proportional-gain multiplier from the error magnitude.

    1.0 at/above `far_m`, rising LINEARLY to `boost` at/below `near_m`. Callers pass the shared `near_m`/`far_m`
    for both the position and the depth loop, so a single schedule governs every proportional error in metres.
    """
    e = abs(float(error))
    if e <= near_m:
        return float(boost)
    if e >= far_m:
        return 1.0
    return 1.0 + (float(boost) - 1.0) * (far_m - e) / (far_m - near_m)


def yaw_gain_multiplier(error: float, near_rad: float, far_rad: float, floor: float) -> float:
    """Heading-gain multiplier: 1.0 at/below `near_rad`, falling LINEARLY to `floor` at/above `far_rad`."""
    a = abs(float(error))
    if a <= near_rad:
        return 1.0
    if a >= far_rad:
        return float(floor)
    return 1.0 - (1.0 - float(floor)) * (a - near_rad) / (far_rad - near_rad)


class AdaptivePController(WaypointController):
    """Proportional waypoint controller whose proportional gain is scheduled on the error and self-tuned on stall.

    Besides the command itself, every call publishes what the gain loop DID (`last_info`), including a cumulative
    gain-tracking record (`last_info["tracking"]`): which way the self-tuning pushed and why, how fast the error was
    actually closing, how much simulated time was spent ramping / relaxing / boosted / at the gain cap, the tracked
    peak and mean gain, and how many adaptation episodes / re-seeds have happened. Those are counted here, step by
    step, from the same numbers that moved the gain -- they are never inferred later from the command.
    """

    # A tracked step only counts as boosted / at the cap above this tolerance, so float noise on the clamped cap can
    # never inflate the tracked time.
    CAP_EPS = 1e-9

    def __init__(self, params: ControllerParams):
        super().__init__(params)
        self._gain = 1.0                     # self-tuning multiplier, clamped to [1, gain_boost]
        self._prev_error: float | None = None      # previous horizontal error magnitude (for the closing speed)
        self._time = 0.0                     # controller-local simulated time of the adaptation (sum of the dts seen)
        self._gain_sum = 0.0                 # running sum of the active multiplier (for the tracked mean)
        self._track = self._new_tracking()
        self.last_info: dict = {}            # telemetry of the last command (effective gains, schedule, tracking)

    # ------------------------------------------------------------------------------ adaptation state
    def reset_adaptation(self) -> None:
        """Forget the self-tuning history (new mission / new target): the gain returns to its nominal value and the
        tracked gain history is cleared, so a new mission is never reported with the previous one's numbers."""
        self._gain = 1.0
        self._prev_error = None
        self._time = 0.0
        self._gain_sum = 0.0
        self._track = self._new_tracking()

    @staticmethod
    def _new_tracking() -> dict:
        """A fresh gain-tracking record (see the class docstring; published as `last_info["tracking"]`)."""
        return {
            "steps": 0, "elapsed_s": 0.0,
            "action": "holding", "reason": "no adaptation step yet", "closing_speed_mps": None,
            "ramp_s": 0.0, "relax_s": 0.0, "hold_s": 0.0, "boosted_s": 0.0, "at_cap_s": 0.0,
            "peak_active": 1.0, "peak_self_tuning": 1.0, "peak_time_s": None, "mean_active": 1.0,
            "episodes": 0, "reseeds": 0,
        }

    @property
    def adaptive_gain(self) -> float:
        return float(self._gain)

    @property
    def tracking(self) -> dict:
        """The tracked gain history so far (a copy, so a caller cannot mutate the controller's own record)."""
        return dict(self._track)

    def _adapt(self, error: float, dt: float | None, ap: AdaptiveParams) -> tuple[float, str, str, float | None]:
        """Update the self-tuning gain from the observed closing speed.

        Returns (gain, action, reason, closing_speed_mps). `action` is what the self-tuning did on THIS step --
        "ramping" (pushing the gain up), "relaxing" (letting it back down) or "holding" (it did not move) -- taken
        from the very branch that moved the gain, so the reported direction can never disagree with the gain, and
        `reason` says which measured quantity chose that branch. `closing_speed_mps` is the rate at which the
        horizontal error shrank over this step (None when there was nothing to measure).
        """
        e = float(error)
        if dt is None or dt <= 0.0:
            self._prev_error = e
            return self._gain, "holding", "no adaptation step (no dt)", None
        if self._prev_error is None:
            self._prev_error = e
            return self._gain, "holding", "first sample (no previous error to compare)", None
        if abs(e - self._prev_error) > ap.target_change_jump_m:
            # the target changed under us: the old error is meaningless, so re-seed instead of adapting on a phantom jump
            self._prev_error = e
            self._track["reseeds"] += 1
            return self._gain, "holding", "new target - adaptation re-seeded", None
        closing = (self._prev_error - e) / dt          # > 0 while the error shrinks (moving toward the target)
        self._prev_error = e
        if e <= ap.settle_deadband_m:
            self._gain = max(1.0, self._gain - ap.relax_rate_per_s * dt)
            action, reason = "relaxing", f"inside the settle deadband ({ap.settle_deadband_m:g} m)"
        elif closing >= ap.healthy_progress_mps:
            self._gain = max(1.0, self._gain - ap.relax_rate_per_s * dt)
            action, reason = "relaxing", f"tracking well (closing {closing:.3f} m/s)"
        elif closing <= ap.stall_progress_mps:
            self._gain = min(ap.gain_boost, self._gain + ap.adapt_rate_per_s * dt)
            action, reason = "ramping", f"stalled (closing {closing:.3f} m/s)"
        else:
            action, reason = "holding", f"making progress (closing {closing:.3f} m/s)"
        return self._gain, action, reason, float(closing)

    def _track_step(self, dt: float, active: float, tuned: float, action: str, reason: str,
                    closing: float | None, ap: AdaptiveParams) -> None:
        """Fold one adaptation step into the tracked gain history: cumulative, bounded and reproducible."""
        t = self._track
        prev_action = t["action"]
        self._time += float(dt)
        t["steps"] += 1
        t["elapsed_s"] += float(dt)
        t["action"], t["reason"] = action, reason
        t["closing_speed_mps"] = None if closing is None else float(closing)
        if action == "ramping":
            t["ramp_s"] += float(dt)
            if prev_action != "ramping":
                t["episodes"] += 1                 # a NEW episode: the self-tuning started pushing the gain up
        elif action == "relaxing":
            t["relax_s"] += float(dt)
        else:
            t["hold_s"] += float(dt)
        if active > 1.0 + self.CAP_EPS:
            t["boosted_s"] += float(dt)            # simulated seconds driven with a gain above nominal
        if tuned >= ap.gain_boost - self.CAP_EPS:
            t["at_cap_s"] += float(dt)             # ... and how much of that sat at the clamped maximum
        if active > t["peak_active"]:
            t["peak_active"] = float(active)
            t["peak_time_s"] = float(self._time)
        if tuned > t["peak_self_tuning"]:
            t["peak_self_tuning"] = float(tuned)
        self._gain_sum += float(active)
        t["mean_active"] = self._gain_sum / t["steps"]

    # ------------------------------------------------------------------------------ control
    def compute_control(self, current_position: tuple, current_yaw: float, target_position: tuple,
                        current_velocity: tuple | None = None, vertical_trim: float = 0.0,
                        dt: float | None = None) -> dict:
        """Same contract as `WaypointController.compute_control`, plus the adaptive gain.

        `dt` (simulated seconds since the previous call) enables the self-tuning term; left None the controller
        still applies the gain SCHEDULE but does not adapt (useful for one-shot / stateless callers).
        """
        p = self.params
        ap = p.adaptive
        current = np.array(current_position, dtype=float)
        target = np.array(target_position, dtype=float)
        error = target - current
        e_horiz = float(np.linalg.norm(error[:2]))

        if ap.enabled:
            sched = gain_schedule(e_horiz, ap.near_m, ap.far_m, ap.gain_boost)
            tuned, action, reason, closing = self._adapt(e_horiz, dt, ap)
            kp_mult = max(sched, tuned)
            if dt is not None and dt > 0.0:
                # Track the step the adaptation really took (the multiplier in use, the self-tuning gain it came
                # from, and the decision that produced them). Steps without a usable dt did not adapt at all, so
                # they are not tracked as adaptation time.
                self._track_step(dt, kp_mult, tuned, action, reason, closing, ap)
        else:
            sched, tuned, kp_mult = 1.0, 1.0, 1.0
            action, reason, closing = "off", "adaptation disabled (plain P controller)", None
            # The tracked record stays empty (nothing adapted), but says WHY it is empty.
            self._track["action"], self._track["reason"] = action, reason
        kp_pos = p.kp_position * kp_mult

        position_control = kp_pos * error
        if current_velocity is not None:
            position_control = position_control - p.kd_velocity * np.asarray(current_velocity, float)
        forward = float(np.clip(position_control[0] / p.max_thrust, -1.0, 1.0))
        lateral = float(np.clip(position_control[1] / p.max_thrust, -1.0, 1.0))
        vertical = float(np.clip(position_control[2] / p.max_thrust + vertical_trim, -1.0, 1.0))

        dx, dy = error[0], error[1]
        desired_yaw = np.arctan2(dy, dx)
        yaw_error = float(np.arctan2(np.sin(desired_yaw - current_yaw), np.cos(desired_yaw - current_yaw)))
        yaw_mult = yaw_gain_multiplier(yaw_error, ap.yaw_near_rad, ap.yaw_far_rad, ap.yaw_gain_min) if ap.enabled else 1.0
        kp_yaw = p.kp_yaw * yaw_mult
        yaw = float(np.clip(kp_yaw * yaw_error / p.max_yaw_torque, -1.0, 1.0))

        self.last_info = {
            "adaptive": bool(ap.enabled),
            "kp_position_nominal": float(p.kp_position), "kp_position_effective": float(kp_pos),
            "kp_position_multiplier": float(kp_mult), "schedule_multiplier": float(sched),
            "self_tuning_gain": float(tuned), "position_error_m": e_horiz,
            "kp_yaw_nominal": float(p.kp_yaw), "kp_yaw_effective": float(kp_yaw),
            "kp_yaw_multiplier": float(yaw_mult), "heading_error_rad": yaw_error,
            "closing_speed_mps": None if closing is None else float(closing),
            # Tracked gain history (a COPY: the snapshot a caller holds must not change under it as the run goes on)
            "tracking": dict(self._track),
        }
        return {"forward": forward, "lateral": lateral, "vertical": vertical, "yaw": yaw}

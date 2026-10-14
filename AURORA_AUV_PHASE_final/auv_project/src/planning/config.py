"""Phase 5 configuration (config.yaml: `planning:` for the global planner, `navigation:` for the rest).

Every value is a SIMULATION parameter, not a calibration of a real vehicle. Validation is strict so a bad
YAML/dashboard override fails loudly instead of producing a silently different behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass, field, fields


def _pick(cls, section: dict | None) -> dict:
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (section or {}).items() if k in known}


@dataclass
class PlannerConfig:
    """Global A* planner (config: planning.*)."""
    connectivity: int = 8                 # 4 | 8
    heuristic: str = "octile"             # octile | euclidean | manhattan (manhattan only admissible for 4)
    allow_diagonal: bool = True           # legacy Week-6 key: false forces 4-connectivity
    heuristic_weight: float = 1.0         # 1.0 = optimal for the cost model; > 1 trades optimality for speed
    unknown_policy: str = "inherit"       # inherit (mapping.costs.unknown_policy) | allow | block
    unknown_cost: float | None = None     # None: use the cost-map value for UNKNOWN cells
    extra_clearance_m: float = 0.5        # planner-level margin ADDED to the mapped inflation radius (localization error allowance)
    start_snap_radius_m: float = 1.5      # a start cell that is not traversable is snapped to the nearest one within this
    goal_snap_radius_m: float = 0.0       # 0: a blocked goal cell is a planning failure (no silent goal change)
    max_expansions: int = 600000          # hard safety stop for A*
    smoothing_enabled: bool = True
    smoothing_min_segment_m: float = 0.0  # reserved: shortcut only if it removes at least one vertex
    simplify_collinear: bool = True

    def __post_init__(self):
        if self.connectivity not in (4, 8):
            raise ValueError("planning.connectivity must be 4 or 8")
        if self.heuristic not in ("octile", "euclidean", "manhattan"):
            raise ValueError("planning.heuristic must be octile, euclidean or manhattan")
        if self.unknown_policy not in ("inherit", "allow", "block"):
            raise ValueError("planning.unknown_policy must be inherit, allow or block")
        if self.heuristic_weight < 1.0:
            raise ValueError("planning.heuristic_weight must be >= 1.0")
        if self.unknown_cost is not None and not self.unknown_cost > 0:
            raise ValueError("planning.unknown_cost must be > 0")
        if self.extra_clearance_m < 0:
            raise ValueError("planning.extra_clearance_m must be >= 0")
        if self.start_snap_radius_m < 0 or self.goal_snap_radius_m < 0:
            raise ValueError("snap radii must be >= 0")
        if self.max_expansions < 1:
            raise ValueError("planning.max_expansions must be >= 1")

    @property
    def effective_connectivity(self) -> int:
        return 8 if (self.connectivity == 8 and self.allow_diagonal) else 4

    @classmethod
    def from_config(cls, cfg: dict) -> "PlannerConfig":
        sec = dict((cfg.get("planning") or {}))
        sm = sec.pop("smoothing", None)
        kw = _pick(cls, sec)
        if isinstance(sm, dict):
            if "enabled" in sm:
                kw["smoothing_enabled"] = bool(sm["enabled"])
        elif isinstance(sm, bool):
            kw["smoothing_enabled"] = sm
        return cls(**kw)


@dataclass
class FollowerConfig:
    """Path follower (config: navigation.follower.*)."""
    cruise_speed_mps: float = 0.9
    min_speed_mps: float = 0.12           # floor while still far from the goal (avoid stalling in drag)
    lookahead_min_m: float = 1.0
    lookahead_gain_s: float = 1.0         # lookahead = min + gain * speed
    goal_slowdown_gain: float = 0.3       # v <= gain * remaining distance (smooth stop at the goal; low gain = early braking)
    kv: float = 22.0                      # N per (m/s) velocity error
    drag_feedforward: float = 2.0         # N per (m/s) of desired speed (== legacy linear drag; feedback absorbs the rest)
    max_thrust_n: float = 10.0            # normalisation of the force command (== simulation.controller.max_thrust)
    max_yaw_torque_nm: float = 2.0
    kp_yaw: float = 1.6
    kd_yaw: float = 4.8
    kp_depth: float = 0.8                 # (m/s) per m of depth error
    ki_depth: float = 0.15                # N per (m*s); anti-windup clamped
    depth_i_limit_n: float = 3.0
    max_vertical_speed_mps: float = 0.4
    heading_slow_start_rad: float = 0.5   # beyond this heading error the vehicle slows so the forward sonar sees its way
    heading_slow_end_rad: float = 1.2
    heading_min_speed_frac: float = 0.25
    station_kp: float = 0.4               # GOAL_REACHED station keeping: (m/s) per m of horizontal goal error
    station_max_speed_mps: float = 0.3
    station_deadband_m: float = 0.1       # no horizontal correction inside this radius (avoids hunting)
    command_frame: str = "world"          # world: legacy dynamics (forces in the world frame) | body: 6dof (wrench in the body frame)

    def __post_init__(self):
        if not self.cruise_speed_mps > 0:
            raise ValueError("navigation.follower.cruise_speed_mps must be > 0")
        if self.command_frame not in ("world", "body"):
            raise ValueError("command_frame must be 'world' or 'body'")
        if self.lookahead_min_m <= 0 or self.max_thrust_n <= 0 or self.max_yaw_torque_nm <= 0:
            raise ValueError("lookahead_min_m, max_thrust_n and max_yaw_torque_nm must be > 0")
        if not 0.0 < self.heading_min_speed_frac <= 1.0:
            raise ValueError("heading_min_speed_frac must be in (0, 1]")

    @classmethod
    def from_config(cls, cfg: dict, command_frame: str | None = None) -> "FollowerConfig":
        nav = (cfg.get("navigation") or {})
        kw = _pick(cls, nav.get("follower"))
        ctl = ((cfg.get("simulation") or {}).get("controller") or {})
        kw.setdefault("max_thrust_n", float(ctl.get("max_thrust", 10.0)))
        kw.setdefault("max_yaw_torque_nm", float(ctl.get("max_yaw_torque", 2.0)))
        phys = ((cfg.get("simulation") or {}).get("physics") or {})
        kw.setdefault("drag_feedforward", float(phys.get("linear_drag", 2.0)))
        if command_frame is not None:
            kw["command_frame"] = command_frame
        return cls(**kw)


@dataclass
class NavigationConfig:
    """Autonomous navigation (config: navigation.*)."""
    enabled: bool = False                 # false -> Phase 4 waypoint demo, bit-for-bit
    auto_start: bool = True               # start planning to the goal at t = 0
    goal_tolerance_m: float = 0.5         # |estimate - goal| below this => GOAL_REACHED
    goal_speed_tolerance_mps: float | None = 0.15  # also require |estimated velocity| below this (None disables)
    # --- replanning -------------------------------------------------------------------------
    validate_interval_s: float = 0.4      # path validity is re-evaluated when the map changed, at most this often
    min_replan_interval_s: float = 1.0    # replans are rate-limited (never every time step)
    max_replans: int = 40
    max_plan_failures: int = 4            # consecutive failed planning attempts before FAILED
    plan_retry_interval_s: float = 1.5
    cost_increase_ratio: float = 0.75     # remaining-path cost grew by more than this fraction ...
    cost_increase_min: float = 30.0       # ... and by at least this many cost-metres => significant map change
    path_check_horizon_m: float = 40.0    # how far ahead the path is checked against the map
    # --- local avoidance ----------------------------------------------------------------------
    avoid_trigger_distance_m: float = 7.0  # blockage nearer than this (path arc length) => local detour; farther => global replan
    local_window_m: float = 14.0          # half-size of the local planning window around the vehicle
    rejoin_margin_m: float = 1.0          # first rejoin candidate this far beyond the blocked section
    rejoin_step_m: float = 1.0
    rejoin_candidates: int = 10
    rejoin_clear_horizon_m: float = 6.0   # the global path must be clear this far beyond the rejoin point
    rejoin_tolerance_m: float = 0.8
    max_avoid_retries: int = 3
    avoid_timeout_s: float = 60.0
    # --- sonar-based reactive speed limit -------------------------------------------------------
    sonar_cone_deg: float = 12.0          # forward cone half-angle used for the clearance estimate
    sonar_slow_range_m: float = 3.5       # below this forward clearance the speed is scaled down ...
    sonar_stop_range_m: float = 0.9       # ... linearly to sonar_min_scale; at/below this the vehicle holds (scale 0)
    sonar_min_scale: float = 0.35         # speed floor above the stop range: the planner's detour may pass closer than
                                          # the slow range by design (inflation radius), so the reactive layer must not veto it
    # --- progress watchdog ------------------------------------------------------------------------
    stuck_timeout_s: float = 25.0         # no progress toward the goal for this long => replan (then FAILED)
    stuck_min_progress_m: float = 0.4
    max_mission_time_s: float = 600.0     # hard bound => FAILED("mission_timeout")
    estimate_timeout_s: float = 2.0       # no valid estimate for this long while navigating => FAILED
    linger_after_terminal_s: float = 2.0  # run loop keeps going this long after GOAL_REACHED/FAILED (interactive goal changes)
    goal_position: list | None = None     # [x, y, z]; None => simulation.goal_position
    follower: FollowerConfig = field(default_factory=FollowerConfig)

    def __post_init__(self):
        for name in ("goal_tolerance_m", "validate_interval_s", "min_replan_interval_s", "local_window_m",
                     "avoid_trigger_distance_m", "stuck_timeout_s", "max_mission_time_s"):
            if not getattr(self, name) > 0:
                raise ValueError(f"navigation.{name} must be > 0")
        if self.max_replans < 0 or self.max_plan_failures < 1:
            raise ValueError("max_replans must be >= 0 and max_plan_failures >= 1")
        if not 0 < self.sonar_stop_range_m < self.sonar_slow_range_m:
            raise ValueError("need 0 < sonar_stop_range_m < sonar_slow_range_m")
        if not 0.0 <= self.sonar_min_scale <= 1.0:
            raise ValueError("sonar_min_scale must be in [0, 1]")
        if self.goal_position is not None and len(self.goal_position) != 3:
            raise ValueError("navigation.goal_position must be [x, y, z]")

    @classmethod
    def from_config(cls, cfg: dict, command_frame: str | None = None) -> "NavigationConfig":
        nav = dict((cfg.get("navigation") or {}))
        nav.pop("follower", None)
        kw = _pick(cls, nav)
        return cls(follower=FollowerConfig.from_config(cfg, command_frame), **kw)

    def resolve_goal(self, cfg: dict) -> list:
        if self.goal_position is not None:
            return [float(v) for v in self.goal_position]
        return [float(v) for v in cfg["simulation"]["goal_position"]]

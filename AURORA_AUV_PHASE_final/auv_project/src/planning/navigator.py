"""AutonomousNavigator: the Phase-5 navigation state machine.

    IDLE -> PLAN -> FOLLOW_PATH <-> AVOID_OBSTACLE
                       |                 |
                       +-----> REPLAN <--+          FOLLOW_PATH/AVOID -> GOAL_REACHED | FAILED

INPUTS of update(): sim time, the EKF EstimatedState, the MappingSystem (cost map + geometry + version) and the
latest SonarMeasurement. That is the complete list: the class has no access to PyBullet, the vehicle body, the
environment description or any truth snapshot (enforced statically by tests/test_navigation_leak.py and
dynamically by a poisoned-truth closed-loop test). Every state transition is caused by one of: the estimate,
the mapped cost-map content, the sonar measurement, elapsed (simulation) time, or an operator command.

OUTPUT: a normalised {forward, lateral, vertical, yaw} control dictionary (the same format the Phase-2
controller feeds AUV.apply_control) plus telemetry.
"""
from __future__ import annotations

import math
import time as _time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from src.planning.config import NavigationConfig, PlannerConfig
from src.planning.global_planner import GlobalPlanner, PlanResult
from src.planning.grid_utils import PlanningGrid
from src.planning.local_avoidance import LocalAvoider, sonar_forward_clearance
from src.planning.path_data import PathData
from src.planning.path_follower import FollowCommand, PathFollower
from src.planning.replanning import PathMonitor


class NavState(str, Enum):
    IDLE = "IDLE"
    PLAN = "PLAN"
    FOLLOW_PATH = "FOLLOW_PATH"
    AVOID_OBSTACLE = "AVOID_OBSTACLE"
    REPLAN = "REPLAN"
    GOAL_REACHED = "GOAL_REACHED"
    FAILED = "FAILED"


ACTIVE_STATES = (NavState.PLAN, NavState.FOLLOW_PATH, NavState.AVOID_OBSTACLE, NavState.REPLAN)
TERMINAL_STATES = (NavState.GOAL_REACHED, NavState.FAILED)


def navigation_prerequisites(cfg: dict) -> str | None:
    """None if the configuration can run autonomous navigation, else the reason it cannot (checked BEFORE a run
    starts, so an impossible configuration fails loudly instead of silently falling back to true-state control)."""
    if not ((cfg.get("sensors") or {}).get("enabled", False)):
        return "navigation needs sensors (sensors.enabled: false)"
    if not ((cfg.get("ekf") or {}).get("enabled", True)):
        return "navigation needs the EKF estimate (ekf.enabled: false)"
    if not (((cfg.get("sensors") or {}).get("sonar") or {}).get("enabled", True)):
        return "navigation needs the sonar-derived map (sensors.sonar.enabled: false)"
    if not ((cfg.get("mapping") or {}).get("enabled", True)):
        return "navigation needs online mapping (mapping.enabled: false)"
    return None


@dataclass
class NavCommand:
    control: dict
    state: NavState
    info: dict = field(default_factory=dict)


class AutonomousNavigator:
    def __init__(self, cfg: dict, command_frame: str = "world"):
        self.cfg = cfg
        self.nav = NavigationConfig.from_config(cfg, command_frame)
        self.pcfg = PlannerConfig.from_config(cfg)
        self.planner = GlobalPlanner(self.pcfg)
        self.follower = PathFollower(self.nav.follower)
        self.avoider = LocalAvoider(self.nav, self.pcfg)
        self.monitor = PathMonitor(self.nav)
        self.reset()

    # ================================================================================== lifecycle
    def reset(self) -> None:
        self.state = NavState.IDLE
        self.prev_state: NavState | None = None
        self.state_since = 0.0
        self.follower.reset_tracking()      # a new mission must not inherit the previous one's tracked gains
        self.goal: np.ndarray | None = None
        self.failure_reason = ""
        self.global_path: PathData | None = None
        self.detour: PathData | None = None
        self.plan_grid: PlanningGrid | None = None
        self.last_plan: PlanResult | None = None
        self.last_detour = None
        self.path_version = 0
        self.history: deque = deque(maxlen=40)
        self.counters = {"plans": 0, "plan_successes": 0, "plan_failures": 0, "replans": 0, "goal_changes": 0,
                         "avoidance_count": 0, "avoid_retries": 0, "throttled_replans": 0, "stuck_events": 0,
                         "estimate_dropouts": 0, "forced_replans": 0}
        self.replan_reasons: dict = {}
        self.plan_times_ms: list = []
        self._consec_plan_fail = 0
        self._force_replan = False
        self._last_plan_time = -1e9
        self._last_validate_time = -1e9
        self._validated_version = None
        self._pending_reason = ""
        self._t_start: float | None = None
        self._last_estimate_time: float | None = None
        self._avoid_started = 0.0
        self._progress_ref = None
        self._grid_cache = (None, None)
        self._speed_scale = 1.0
        self._front_clear: float | None = None
        self._last_control = {"forward": 0.0, "lateral": 0.0, "vertical": 0.0, "yaw": 0.0}
        self._last_info: dict = {}
        self._est_pos = None
        self._goal_distance = None
        self._last_assessment = None
        self._pose_time = None
        self._map_version = None
        self.follower.clear()

    # ================================================================================== commands
    def check_goal(self, goal_xyz, mapping) -> tuple:
        """Validate a goal against the CURRENT map: (ok, detail). A goal outside the map or in an occupied/inflated
        (or policy-blocked unknown) cell is rejected with the real reason; it never gets silently moved."""
        try:
            g = [float(v) for v in goal_xyz]
            if len(g) != 3 or not all(math.isfinite(v) for v in g):
                return False, "goal must be three finite numbers [x, y, z]"
        except (TypeError, ValueError):
            return False, "goal must be three finite numbers [x, y, z]"
        if mapping is None or not getattr(mapping, "enabled", False):
            return False, "mapping unavailable: cannot validate or plan a goal without a map"
        grid = self._grid(mapping)
        r, c = grid.cell_of(g[0], g[1])
        if not grid.in_bounds(r, c):
            ext = grid.geom.extent
            return False, (f"goal ({g[0]:.2f}, {g[1]:.2f}) is outside the map "
                           f"(x {ext[0]:.1f}..{ext[1]:.1f}, y {ext[2]:.1f}..{ext[3]:.1f})")
        if g[2] > 0.0:
            return False, "goal z must be <= 0 (underwater)"
        if not grid.traversable[r, c]:
            why = "unknown space (unknown_policy blocks it)" if grid.unknown[r, c] else "an occupied or inflated cell"
            return False, f"goal cell is in {why}"
        return True, "ok"

    def set_goal(self, goal_xyz, sim_time: float, reason: str = "goal_set", start: bool = True) -> tuple:
        """Set (or change) the goal. A goal change while navigating forces a replan; start=False only stores it."""
        g = np.array([float(v) for v in goal_xyz], float)
        changed = self.goal is not None and not np.allclose(self.goal, g)
        self.goal = g
        self.failure_reason = ""
        if changed:
            self.counters["goal_changes"] += 1
        if self.state in ACTIVE_STATES and changed:
            self._request_replan("goal_changed", sim_time, force=True)
        elif start and self.state in (NavState.IDLE, NavState.GOAL_REACHED, NavState.FAILED):
            self._begin(sim_time)
        return True, f"goal set to [{g[0]:.2f}, {g[1]:.2f}, {g[2]:.2f}]"

    def start(self, sim_time: float) -> tuple:
        if self.goal is None:
            return False, "no goal set"
        if self.state in ACTIVE_STATES:
            return False, f"navigation already active ({self.state.value})"
        self._begin(sim_time)
        return True, "navigation started"

    def cancel(self, sim_time: float) -> tuple:
        if self.state == NavState.IDLE:
            return False, "navigation is already idle"
        self.follower.clear()
        self.global_path = self.detour = None
        self.path_version += 1
        self._enter(NavState.IDLE, sim_time, "cancelled_by_operator")
        return True, "navigation cancelled"

    def request_replan(self, sim_time: float, reason: str = "operator_request") -> tuple:
        if self.state not in (NavState.FOLLOW_PATH, NavState.AVOID_OBSTACLE):
            return False, f"nothing to replan in state {self.state.value}"
        self._request_replan(reason, sim_time, force=True)
        return True, "replan requested"

    def set_unknown_policy(self, policy: str, sim_time: float) -> tuple:
        if policy not in ("inherit", "allow", "block"):
            return False, "policy must be inherit, allow or block"
        self.pcfg.unknown_policy = policy
        self._grid_cache = (None, None)
        if self.state in (NavState.FOLLOW_PATH, NavState.AVOID_OBSTACLE):
            self._request_replan("unknown_policy_changed", sim_time, force=True)
        return True, f"unknown-cell policy set to {policy}"

    # ================================================================================== main entry
    def update(self, sim_time: float, estimate, mapping, sonar=None) -> NavCommand:
        now = float(sim_time)
        est_ok = self._estimate_ok(estimate)
        if est_ok:
            self._last_estimate_time = now
            self._est_pos = np.asarray(estimate.position, float).copy()
            self._pose_time = float(estimate.time)
            if self.goal is not None:
                self._goal_distance = float(np.linalg.norm(self._est_pos - self.goal))
        else:
            self.counters["estimate_dropouts"] += 1
            if self.state in ACTIVE_STATES and self._last_estimate_time is not None \
                    and now - self._last_estimate_time > self.nav.estimate_timeout_s:
                self._fail("estimate_unavailable", now)
            return self._emit(self._zero(), now, {"mode": "no_estimate"})

        if self.state in ACTIVE_STATES:
            if self._t_start is not None and now - self._t_start > self.nav.max_mission_time_s:
                self._fail("mission_timeout", now)
            elif self._goal_reached(estimate):
                self.follower.clear()
                self._enter(NavState.GOAL_REACHED, now, "goal_within_tolerance")

        map_ok = mapping is not None and getattr(mapping, "enabled", False)
        self._map_version = None if mapping is None else int(mapping.version)
        st = self.state
        if st in (NavState.IDLE, NavState.GOAL_REACHED, NavState.FAILED):
            if st == NavState.GOAL_REACHED and self.goal is not None:      # station keeping on the CURRENT goal (estimate only)
                return self._emit(self.follower.station_keep(estimate, self.goal), now)
            return self._emit(self.follower.hold(estimate, None), now)
        if st in (NavState.PLAN, NavState.REPLAN):
            return self._do_plan(now, estimate, mapping if map_ok else None)

        # ---- FOLLOW_PATH / AVOID_OBSTACLE -------------------------------------------------------------------
        self._front_clear = sonar_forward_clearance(sonar, self.nav.sonar_cone_deg)
        self._speed_scale = self._scale_from_clearance(self._front_clear)
        self._watchdog(now, estimate)
        if self.state == NavState.FOLLOW_PATH:
            self._monitor_global(now, estimate, mapping if map_ok else None)
        if self.state == NavState.AVOID_OBSTACLE:
            self._monitor_detour(now, estimate, mapping if map_ok else None)
        if self.state in (NavState.PLAN, NavState.REPLAN):          # monitor decided to replan: do it this tick
            return self._do_plan(now, estimate, mapping if map_ok else None)
        if self.state not in (NavState.FOLLOW_PATH, NavState.AVOID_OBSTACLE):
            return self._emit(self.follower.hold(estimate), now)
        cmd = self.follower.compute(estimate, self._speed_scale)
        return self._emit(cmd, now)

    # ================================================================================== planning
    def _grid(self, mapping) -> PlanningGrid:
        key = (int(mapping.version), self.pcfg.unknown_policy, self.pcfg.unknown_cost, id(mapping))
        if self._grid_cache[0] != key:
            g = PlanningGrid.from_cost_map(mapping.cost_map, mapping.grid.geom, self.pcfg, int(mapping.version))
            self._grid_cache = (key, g)
        return self._grid_cache[1]

    def _do_plan(self, now: float, estimate, mapping) -> NavCommand:
        is_replan = self.state == NavState.REPLAN
        hold = lambda: self._emit(self.follower.hold(estimate), now, {"mode": "planning"})       # noqa: E731
        if self.goal is None:
            self._enter(NavState.IDLE, now, "no_goal")
            return hold()
        # rate limit: a replan is never issued faster than min_replan_interval_s (goal changes / operator are forced)
        if self._consec_plan_fail > 0:
            wait = self.nav.plan_retry_interval_s
        elif is_replan and not self._force_replan:
            wait = self.nav.min_replan_interval_s
        else:
            wait = 0.0
        if now - self._last_plan_time < wait - 1e-9:
            if is_replan and self._consec_plan_fail == 0:
                self.counters["throttled_replans"] += 1
            return hold()
        if mapping is None:
            return self._plan_failed(now, "mapping_unavailable", "mapping disabled or unavailable: no cost map to plan on", hold)
        grid = self._grid(mapping)
        start = np.asarray(estimate.position, float)
        self.counters["plans"] += 1
        self._force_replan = False
        res = self.planner.plan(grid, start[:2], self.goal[:2])
        self._last_plan_time = now
        self.last_plan = res
        self.plan_times_ms.append(res.time_ms)
        if not res.success:
            return self._plan_failed(now, res.status, res.detail, hold)
        self._consec_plan_fail = 0
        self.counters["plan_successes"] += 1
        self.path_version += 1
        self.global_path = PathData.from_xy(res.path_xy, float(start[2]), float(self.goal[2]), kind="global",
                                            version=self.path_version, created_time=now, map_version=grid.map_version)
        self.detour = None
        self.plan_grid = grid
        self.follower.set_path(self.global_path, 0.0)
        self._pending_reason = ""
        self._progress_ref = None
        self._enter(NavState.FOLLOW_PATH, now, "plan_ok" if not is_replan else "replan_ok")
        return self._emit(self.follower.compute(estimate, 1.0), now)

    def _plan_failed(self, now, status, detail, hold):
        self.counters["plan_failures"] += 1
        self._consec_plan_fail += 1
        self._last_plan_time = now
        if self._consec_plan_fail >= self.nav.max_plan_failures:
            self._fail(status, now, detail)
        return hold()

    def _request_replan(self, reason: str, now: float, force: bool = False) -> None:
        self._pending_reason = reason
        self.replan_reasons[reason] = self.replan_reasons.get(reason, 0) + 1
        if force:
            self._force_replan = True
            self.counters["forced_replans"] += 1
        self.counters["replans"] += 1
        if self.counters["replans"] > self.nav.max_replans:
            self._fail("max_replans_exceeded", now)
            return
        self.follower.clear()
        self.detour = None
        self._enter(NavState.REPLAN, now, reason)

    # ================================================================================== monitoring
    def _monitor_global(self, now, estimate, mapping) -> None:
        if mapping is None or self.global_path is None:
            return
        reactive = self._front_clear is not None and self._front_clear < self.nav.sonar_slow_range_m
        changed = self._validated_version != int(mapping.version)
        due = now - self._last_validate_time >= self.nav.validate_interval_s - 1e-9
        if not ((changed and due) or (reactive and due)):
            return
        self._last_validate_time = now
        self._validated_version = int(mapping.version)
        grid = self._grid(mapping)
        a = self.monitor.assess(grid, self.plan_grid, self.global_path, self.follower.s)
        self._last_assessment = a
        if a.blocked:
            if a.blocked_distance_m <= self.nav.avoid_trigger_distance_m:
                self._start_avoidance(now, estimate, grid, f"obstacle_on_path_{a.blocked_distance_m:.1f}m")
            else:
                self._soft_replan("path_blocked", now)
        elif a.significant_change:
            self._soft_replan("map_change", now)

    def _soft_replan(self, reason: str, now: float) -> None:
        if now - self._last_plan_time < self.nav.min_replan_interval_s - 1e-9:
            self.counters["throttled_replans"] += 1            # keep following until the rate limit allows it
            self._validated_version = None                     # ... and re-run the check then: a deferred replan must not be lost
            return
        self._request_replan(reason, now)

    def _start_avoidance(self, now, estimate, grid, reason: str) -> None:
        pos = np.asarray(estimate.position, float)
        d = self.avoider.plan_detour(grid, pos[:2], self.global_path, self.follower.s)
        self.last_detour = d
        if not d.success and d.reason == "path_not_blocked":
            return                              # the block went away between the check and the detour search: keep following
        if not d.success:
            self._request_replan(f"detour_failed:{d.reason}", now, force=True)
            return
        self.counters["avoidance_count"] += 1
        self.path_version += 1
        z0 = float(self.global_path.point(self.follower.s)[2])
        z1 = float(self.global_path.point(d.rejoin_s)[2])
        self._global_s_at_detour = self.follower.s
        self._rejoin_s = d.rejoin_s
        self.detour = PathData.from_xy(d.path_xy, z0, z1, kind="detour", version=self.path_version,
                                       created_time=now, map_version=grid.map_version)
        self._avoid_started = now
        self._avoid_retries = 0
        self._validated_version = int(grid.map_version)
        self.follower.set_path(self.detour, 0.0)
        self._enter(NavState.AVOID_OBSTACLE, now, reason)

    def _monitor_detour(self, now, estimate, mapping) -> None:
        if self.detour is None:
            self._request_replan("detour_lost", now, force=True)
            return
        pos = np.asarray(estimate.position, float)
        end = self.detour.xy[-1]
        if self.follower.remaining_m < self.nav.rejoin_tolerance_m or \
                float(np.linalg.norm(pos[:2] - end)) < self.nav.rejoin_tolerance_m:
            self.path_version += 1
            self.detour = None
            self.follower.set_path(self.global_path, self._rejoin_s)
            self._progress_ref = None
            self._validated_version = None                      # re-check the global path right away
            self._enter(NavState.FOLLOW_PATH, now, "rejoined_global_path")
            return
        if now - self._avoid_started > self.nav.avoid_timeout_s:
            self._request_replan("avoid_timeout", now, force=True)
            return
        if mapping is None:
            return
        changed = self._validated_version != int(mapping.version)
        if changed and now - self._last_validate_time >= self.nav.validate_interval_s - 1e-9:
            self._last_validate_time = now
            self._validated_version = int(mapping.version)
            grid = self._grid(mapping)
            from src.planning.grid_utils import check_path
            if not check_path(grid, self.detour.xy, start_s=self.follower.s).free:
                self._avoid_retries += 1
                self.counters["avoid_retries"] += 1
                if self._avoid_retries > self.nav.max_avoid_retries:
                    self._request_replan("detour_blocked", now, force=True)
                    return
                # retry from the current estimated position; the global path is unchanged
                d = self.avoider.plan_detour(grid, pos[:2], self.global_path, self._global_progress(pos))
                self.last_detour = d
                if not d.success and d.reason == "path_not_blocked":
                    # the obstacle that forced the detour is gone from the map: simply resume the global path
                    self.path_version += 1
                    self.detour = None
                    self.follower.set_path(self.global_path, self._global_progress(pos))
                    self._progress_ref = None
                    self._validated_version = None
                    self._enter(NavState.FOLLOW_PATH, now, "global_path_clear")
                    return
                if not d.success:
                    self._request_replan(f"detour_failed:{d.reason}", now, force=True)
                    return
                self.path_version += 1
                self._rejoin_s = d.rejoin_s
                self.detour = PathData.from_xy(d.path_xy, float(pos[2]), float(self.global_path.point(d.rejoin_s)[2]),
                                               kind="detour", version=self.path_version, created_time=now,
                                               map_version=grid.map_version)
                self.follower.set_path(self.detour, 0.0)

    def _global_progress(self, pos) -> float:
        s, _, _ = self.global_path.project(np.asarray(pos, float)[:2], getattr(self, "_global_s_at_detour", 0.0))
        return max(s, getattr(self, "_global_s_at_detour", 0.0))

    def _watchdog(self, now: float, estimate) -> None:
        if self._goal_distance is None:
            return
        d = self._goal_distance
        if self._progress_ref is None:
            self._progress_ref = (now, d)
            return
        t0, d0 = self._progress_ref
        if d < d0 - self.nav.stuck_min_progress_m:
            self._progress_ref = (now, d)
        elif now - t0 > self.nav.stuck_timeout_s and self.state == NavState.FOLLOW_PATH:
            self.counters["stuck_events"] += 1
            self._progress_ref = (now, d)
            if self.counters["stuck_events"] > 2:
                self._fail("no_progress", now)
            else:
                self._request_replan("no_progress", now, force=True)

    def _scale_from_clearance(self, clear: float | None) -> float:
        if clear is None:
            return 1.0
        lo, hi = self.nav.sonar_stop_range_m, self.nav.sonar_slow_range_m
        if clear <= lo:
            return 0.0
        return float(np.clip((clear - lo) / (hi - lo), self.nav.sonar_min_scale, 1.0))

    # ================================================================================== helpers
    def _estimate_ok(self, est) -> bool:
        if est is None:
            return False
        try:
            return bool(np.all(np.isfinite(est.position)) and np.all(np.isfinite(est.velocity)) and np.isfinite(est.time))
        except Exception:                                             # noqa: BLE001
            return False

    def _goal_reached(self, estimate) -> bool:
        if self.goal is None:
            return False
        if np.linalg.norm(np.asarray(estimate.position, float) - self.goal) >= self.nav.goal_tolerance_m:
            return False
        tol = self.nav.goal_speed_tolerance_mps
        return tol is None or float(np.linalg.norm(estimate.velocity)) <= tol

    def _begin(self, now: float) -> None:
        self.follower.clear()
        self.follower.reset_tracking()      # new navigation mission: start the tracked gain history fresh
        self.global_path = self.detour = self.plan_grid = None
        self.path_version += 1
        self._consec_plan_fail = 0
        self._last_plan_time = -1e9
        self._t_start = now
        self._progress_ref = None
        self._validated_version = None
        self.counters["stuck_events"] = 0
        self._enter(NavState.PLAN, now, "navigation_started")

    def _enter(self, new: NavState, now: float, reason: str) -> None:
        if new == self.state:
            return
        self.prev_state = self.state
        self.history.append({"t": round(float(now), 3), "from": self.state.value, "to": new.value, "reason": reason})
        self.state, self.state_since = new, float(now)
        self._progress_ref = None

    def _fail(self, reason: str, now: float, detail: str = "") -> None:
        self.failure_reason = reason if not detail else f"{reason}: {detail}"
        self.follower.clear()
        self._enter(NavState.FAILED, now, reason)

    @staticmethod
    def _zero() -> FollowCommand:
        return FollowCommand()

    def _emit(self, cmd: FollowCommand, now: float, info: dict | None = None) -> NavCommand:
        ctl = cmd.as_control()
        self._last_control = ctl
        self._last_info = {**cmd.info, **(info or {})}
        return NavCommand(ctl, self.state, self._last_info)

    # ================================================================================== telemetry
    def path_payload(self) -> dict:
        g, d = self.global_path, self.detour
        cur = self.follower.current_waypoint()
        return {"version": self.path_version,
                "global": None if g is None else g.as_points(),
                "global_z": None if g is None else [round(float(v), 3) for v in g.z[[0, -1]]],
                "detour": None if d is None else d.as_points(),
                "progress_s": round(float(self.follower.s), 3),
                "following": None if self.follower.path is None else self.follower.path.kind,
                "waypoint_index": self.follower.waypoint_index if self.follower.path is not None else None,
                "waypoint": None if cur is None else np.round(cur, 3).tolist(),
                "goal": None if self.goal is None else self.goal.tolist()}

    def telemetry(self) -> dict:
        lp, ld = self.last_plan, self.last_detour
        mean_ms = float(np.mean(self.plan_times_ms)) if self.plan_times_ms else None
        a = self._last_assessment
        path = self.follower.path
        return {
            "enabled": True,
            "state": self.state.value, "previous_state": None if self.prev_state is None else self.prev_state.value,
            "state_since": self.state_since, "failure_reason": self.failure_reason or None,
            "goal": None if self.goal is None else self.goal.tolist(),
            "goal_distance_m": self._goal_distance,
            "goal_tolerance_m": self.nav.goal_tolerance_m,
            "estimate_position": None if self._est_pos is None else self._est_pos.tolist(),
            "pose_source": "EKF estimate", "pose_time": self._pose_time, "map_version": self._map_version,
            "planner": {
                "status": "IDLE" if lp is None else ("OK" if lp.success else lp.status.upper()),
                "detail": None if lp is None else lp.detail,
                "plans": self.counters["plans"], "plan_successes": self.counters["plan_successes"],
                "plan_failures": self.counters["plan_failures"],
                "last_planning_time_ms": None if lp is None else lp.time_ms, "mean_planning_time_ms": mean_ms,
                "path_length_m": None if lp is None or not lp.success else lp.length_m,
                "raw_path_length_m": None if lp is None or not lp.success else lp.raw_length_m,
                "path_cost": None if lp is None or not lp.success else lp.cost,
                "expansions": None if lp is None else lp.expansions,
                "smoothed": None if lp is None else lp.smoothed,
                "collision_free": None if lp is None else lp.collision_free,
                "path_vertices": None if lp is None or not lp.success else int(len(lp.path_xy)),
                "start_adjusted_m": None if lp is None else lp.start_adjusted_m,
                "heuristic": None if lp is None else lp.heuristic,
                "connectivity": self.pcfg.effective_connectivity, "unknown_policy": self.pcfg.unknown_policy,
                "unknown_cost_override": self.pcfg.unknown_cost,
                "planned_on_map_version": None if self.plan_grid is None else self.plan_grid.map_version,
            },
            "following": {
                "path_kind": None if path is None else path.kind,
                "waypoint_index": self.follower.waypoint_index if path is not None else None,
                "waypoint_count": None if self.global_path is None else int(self.global_path.n),
                "current_waypoint": None if path is None else np.round(self.follower.current_waypoint(), 3).tolist(),
                "progress_m": float(self.follower.s), "remaining_m": float(self.follower.remaining_m) if path is not None else None,
                "cross_track_m": float(self.follower.cross_track) if path is not None else None,
                "desired_speed_mps": self._last_info.get("desired_speed"),
                "heading_error_rad": self._last_info.get("heading_error_rad"),
                "depth_error_m": self._last_info.get("depth_error_m"),
            },
            "avoidance": {
                "state": "ACTIVE" if self.state == NavState.AVOID_OBSTACLE else "INACTIVE",
                "active": self.state == NavState.AVOID_OBSTACLE,
                "count": self.counters["avoidance_count"], "retries": self.counters["avoid_retries"],
                "last_detour_ok": None if ld is None else ld.success, "last_detour_reason": None if ld is None else ld.reason,
                "detour_length_m": None if ld is None or not ld.success else ld.length_m,
                "rejoin_point": None if ld is None or ld.rejoin_xy is None else list(ld.rejoin_xy),
                "sonar_front_clearance_m": self._front_clear, "speed_scale": self._speed_scale,
                "trigger_distance_m": self.nav.avoid_trigger_distance_m,
            },
            "replanning": {
                "count": self.counters["replans"], "reasons": dict(self.replan_reasons),
                "last_reason": self._pending_reason or (list(self.replan_reasons)[-1] if self.replan_reasons else None),
                "throttled": self.counters["throttled_replans"], "budget": self.nav.max_replans,
                "goal_changes": self.counters["goal_changes"],
                "validations": self.monitor.validations, "blocked_events": self.monitor.blocked_events,
                "map_change_events": self.monitor.change_events,
                "last_assessment": None if a is None else {"blocked": a.blocked, "blocked_distance_m": a.blocked_distance_m,
                                                           "cost_now": a.cost_now, "cost_ref": a.cost_ref,
                                                           "significant_change": a.significant_change},
            },
            "control": dict(self._last_control),
            "history": list(self.history)[-12:],
        }


def authoritative_goal(navigator, fallback) -> tuple:
    """The navigator's CURRENT goal is the single authority (it changes when a dashboard goal command is accepted).
    Everything that displays or evaluates 'the goal' (markers, mission snapshot, final evaluation) follows it.
    Pure function, no PyBullet import: importable and testable anywhere."""
    g = getattr(navigator, "goal", None)
    return tuple(float(v) for v in g) if g is not None else tuple(float(v) for v in fallback)

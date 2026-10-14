"""Phase 5 navigator: state machine, replanning policy, local avoidance, goal / failure states, EKF-only operation.

The navigator is driven with REAL components where it matters (the Phase-4 MappingSystem and cost map, the real
planner, follower and avoider); the 'vehicle' here is just an EstimatedState the test places, so every transition is
caused by exactly the inputs the navigator is allowed to have: estimate, map, sonar, time. No PyBullet."""
import json

import numpy as np
import pytest
import yaml

from src.localization.state import EstimatedState
from src.mapping.map_manager import MappingSystem
from src.planning.grid_utils import PlanningGrid, check_path
from src.planning.local_avoidance import blocked_run, sonar_forward_clearance
from src.planning.navigator import AutonomousNavigator, NavState, navigation_prerequisites
from src.sensors.health import SensorHealth
from src.sensors.models import SonarMeasurement
from src.simulation.commands import apply_command, deep_merge
from src.utils.jsonsafe import jsonsafe
from src.utils.transforms import euler_to_quat

DT = 0.1


def load_cfg(override=None):
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return deep_merge(cfg, {"navigation": {"enabled": True}, **(override or {})})


def est(x, y, z=-5.0, yaw=0.0, t=0.0, v=(0.0, 0.0, 0.0)):
    return EstimatedState(time=t, position=np.array([x, y, z], float), velocity=np.array(v, float),
                          quat=euler_to_quat(0.0, 0.0, yaw), angular_velocity=np.zeros(3), accel_bias=np.zeros(3),
                          gyro_bias=np.zeros(3), covariance=np.eye(15) * 1e-3)


def mark_occupied(mapping, x0, x1, y0, y1, times=12):
    """Put real occupied evidence into the REAL log-odds grid, rebuild the REAL cost map, bump the map version."""
    g = mapping.grid.geom
    xs, ys = np.meshgrid(np.arange(x0, x1, g.resolution / 2), np.arange(y0, y1, g.resolution / 2))
    r, c = g.world_to_grid(xs.ravel(), ys.ravel())
    idx = np.unique(g.flat_index(r, c).astype(np.int64))
    for _ in range(times):                       # repeated hits: the grid is Bayesian, one hit cannot undo free-space evidence
        mapping.grid.apply(np.zeros(0, np.int64), np.zeros(0), idx)
    mapping.version += 1
    mapping.update_cost_map()


def mark_free(mapping, x0, x1, y0, y1, times=3):
    g = mapping.grid.geom
    xs, ys = np.meshgrid(np.arange(x0, x1, g.resolution / 2), np.arange(y0, y1, g.resolution / 2))
    r, c = g.world_to_grid(xs.ravel(), ys.ravel())
    idx = np.unique(g.flat_index(r, c).astype(np.int64))
    for _ in range(times):
        mapping.grid.apply(idx, np.ones(len(idx)), np.zeros(0, np.int64))
    mapping.version += 1
    mapping.update_cost_map()


class Rig:
    def __init__(self, override=None, goal=(30.0, 5.0, -5.0), start=(5.0, 5.0), map_free=True):
        self.cfg = load_cfg(override)
        self.mapping = MappingSystem(self.cfg)
        if map_free:
            mark_free(self.mapping, -4.0, 54.0, -4.0, 54.0)
        self.nav = AutonomousNavigator(self.cfg)
        self.t = 0.0
        self.pos = np.array([start[0], start[1], goal[2]], float)
        self.goal = np.array(goal, float)
        self.nav.set_goal(self.goal, 0.0)

    def step(self, n=1, pos=None, sonar=None, yaw=0.0):
        out = None
        for _ in range(n):
            if pos is not None:
                self.pos = np.array(pos, float)
            out = self.nav.update(self.t, est(*self.pos[:2], self.pos[2], yaw=yaw, t=self.t), self.mapping, sonar)
            self.t += DT
        return out

    def advance_along(self, path_xy, s):
        """Teleport the estimate to arc length s along a polyline (a perfect follower)."""
        from src.planning.grid_utils import cumulative_length, point_at_arc
        p = point_at_arc(path_xy, cumulative_length(path_xy), s)
        self.pos[:2] = p
        return p


# =============================================================================== lifecycle / states
def test_idle_until_a_goal_exists_and_hold_command_is_finite():
    cfg = load_cfg()
    nav = AutonomousNavigator(cfg)
    assert nav.state == NavState.IDLE
    m = MappingSystem(cfg)
    out = nav.update(0.0, est(5, 5), m, None)
    assert nav.state == NavState.IDLE and all(np.isfinite(v) for v in out.control.values())
    ok, why = nav.start(0.0)
    assert not ok and "no goal" in why


def test_plan_then_follow_path_with_the_real_planner_output():
    r = Rig()
    assert r.nav.state == NavState.PLAN
    out = r.step()
    assert r.nav.state == NavState.FOLLOW_PATH and out.state == NavState.FOLLOW_PATH
    g = r.nav.global_path
    assert g is not None and g.length > 24.0
    assert g.xy[0] == pytest.approx([5.0, 5.0]) and g.xy[-1] == pytest.approx([30.0, 5.0])
    assert [h["to"] for h in r.nav.history] == ["PLAN", "FOLLOW_PATH"]
    assert r.nav.last_plan.success and r.nav.last_plan.collision_free
    assert out.control["forward"] > 0.1                                   # commanded toward the goal
    assert r.nav.telemetry()["planner"]["status"] == "OK"


def test_planning_starts_from_the_EKF_estimate():
    """The navigator's only position input is the estimate: moving the estimate moves the path origin 1:1."""
    for start in [(5.0, 5.0), (12.3, 7.7), (40.0, 33.0)]:
        r = Rig(start=start, goal=(45.0, 40.0, -5.0))
        r.step()
        assert r.nav.global_path.xy[0] == pytest.approx(start)
        assert r.nav.telemetry()["estimate_position"][:2] == pytest.approx(start)


def test_goal_reached_is_decided_from_the_estimate_and_tolerance():
    r = Rig()
    r.step()
    r.step(pos=(29.4, 5.0, -5.0))
    assert r.nav.state == NavState.FOLLOW_PATH                            # 0.6 m away > 0.5 m tolerance
    r.step(pos=(29.7, 5.0, -5.0))
    assert r.nav.state == NavState.GOAL_REACHED
    out = r.step(3)
    assert r.nav.state == NavState.GOAL_REACHED and abs(out.control["forward"]) < 0.5
    assert r.nav.telemetry()["goal_distance_m"] == pytest.approx(0.3, abs=1e-6)
    assert [h["to"] for h in r.nav.history][-1] == "GOAL_REACHED"


def test_goal_speed_tolerance_is_honoured():
    r = Rig({"navigation": {"goal_speed_tolerance_mps": 0.1}})
    r.step()
    r.nav.update(r.t, est(30.0, 5.0, -5.0, t=r.t, v=(0.8, 0, 0)), r.mapping, None)
    assert r.nav.state == NavState.FOLLOW_PATH                            # at the goal but still moving fast
    r.nav.update(r.t + DT, est(30.0, 5.0, -5.0, t=r.t + DT, v=(0.02, 0, 0)), r.mapping, None)
    assert r.nav.state == NavState.GOAL_REACHED


def test_goal_validation_uses_the_live_map_and_reports_the_real_reason():
    r = Rig()
    mark_occupied(r.mapping, 40.0, 42.0, 20.0, 22.0)
    ok, why = r.nav.check_goal([100.0, 5.0, -5.0], r.mapping)
    assert not ok and "outside the map" in why
    ok, why = r.nav.check_goal([41.0, 21.0, -5.0], r.mapping)
    assert not ok and "occupied or inflated" in why
    ok, why = r.nav.check_goal([30.0, 5.0, 3.0], r.mapping)
    assert not ok and "underwater" in why
    assert not r.nav.check_goal([float("nan"), 5.0, -5.0], r.mapping)[0]
    assert not r.nav.check_goal("goal", r.mapping)[0]
    assert r.nav.check_goal([30.0, 5.0, -5.0], r.mapping) == (True, "ok")
    r.mapping.set_enabled(False)
    assert not r.nav.check_goal([30.0, 5.0, -5.0], r.mapping)[0]


def test_unreachable_goal_ends_in_FAILED_with_the_planner_reason():
    r = Rig(goal=(30.0, 25.0, -5.0))
    for (a, b, c, d) in [(26.0, 34.0, 28.0, 28.5), (26.0, 34.0, 21.5, 22.0), (26.0, 26.5, 21.5, 28.5), (33.5, 34.0, 21.5, 28.5)]:
        mark_occupied(r.mapping, a, b, c, d)                              # a closed box around the goal
    r.step(300)
    assert r.nav.state == NavState.FAILED
    assert r.nav.failure_reason.startswith("no_path") and "no traversable route" in r.nav.failure_reason
    assert r.nav.counters["plan_failures"] >= r.nav.nav.max_plan_failures
    assert r.nav.telemetry()["planner"]["status"] == "NO_PATH"
    out = r.step()
    assert abs(out.control["forward"]) < 0.6                              # stays put, no runaway


def test_failed_state_can_be_left_by_a_new_reachable_goal():
    r = Rig(goal=(30.0, 25.0, -5.0))
    for (a, b, c, d) in [(26.0, 34.0, 28.0, 28.5), (26.0, 34.0, 21.5, 22.0), (26.0, 26.5, 21.5, 28.5), (33.5, 34.0, 21.5, 28.5)]:
        mark_occupied(r.mapping, a, b, c, d)
    r.step(300)
    assert r.nav.state == NavState.FAILED
    r.nav.set_goal([20.0, 5.0, -5.0], r.t)
    r.step(2)
    assert r.nav.state == NavState.FOLLOW_PATH and r.nav.failure_reason == ""


def test_unknown_policy_block_on_an_unmapped_world_fails_instead_of_guessing():
    r = Rig({"planning": {"unknown_policy": "block"}}, map_free=False)
    r.step(300)
    assert r.nav.state == NavState.FAILED
    assert r.nav.telemetry()["planner"]["unknown_policy"] == "block"


def test_unknown_policy_allow_plans_through_unmapped_space_and_can_change_live():
    r = Rig({"planning": {"unknown_policy": "allow"}}, map_free=False)
    r.step()
    assert r.nav.state == NavState.FOLLOW_PATH
    ok, _ = r.nav.set_unknown_policy("block", r.t)
    assert ok and r.nav.state == NavState.REPLAN
    assert not r.nav.set_unknown_policy("sometimes", r.t)[0]


def test_estimate_dropout_stops_the_vehicle_then_fails_after_the_timeout():
    r = Rig()
    r.step()
    bad = est(float("nan"), 5.0)
    out = None
    t = r.t
    for k in range(int(3.0 / DT)):
        out = r.nav.update(t, bad, r.mapping, None)
        t += DT
        if k == 5:
            assert r.nav.state == NavState.FOLLOW_PATH and out.control == {"forward": 0.0, "lateral": 0.0, "vertical": 0.0, "yaw": 0.0}
    assert r.nav.state == NavState.FAILED and r.nav.failure_reason == "estimate_unavailable"
    assert r.nav.update(t, None, r.mapping, None).control["forward"] == 0.0


def test_mission_timeout_and_replan_budget_fail_cleanly():
    r = Rig({"navigation": {"max_mission_time_s": 5.0}})
    r.step(80)
    assert r.nav.state == NavState.FAILED and r.nav.failure_reason == "mission_timeout"
    r = Rig({"navigation": {"max_replans": 0, "min_replan_interval_s": 0.1}})
    r.step()
    mark_occupied(r.mapping, 17.0, 19.0, 3.0, 7.0)
    r.step(20)
    assert r.nav.state == NavState.FAILED and r.nav.failure_reason == "max_replans_exceeded"


def test_cancel_returns_to_idle_and_manual_replan_requires_an_active_path():
    r = Rig()
    assert not r.nav.request_replan(r.t)[0]                               # PLAN: nothing to replan yet
    r.step()
    assert r.nav.request_replan(r.t)[0] and r.nav.state == NavState.REPLAN
    r.step(2)
    assert r.nav.state == NavState.FOLLOW_PATH and r.nav.replan_reasons.get("operator_request") == 1
    assert r.nav.cancel(r.t)[0] and r.nav.state == NavState.IDLE and r.nav.global_path is None
    assert not r.nav.cancel(r.t)[0]


# =============================================================================== replanning policy
def test_no_blind_replanning_when_nothing_changes():
    r = Rig()
    r.step()
    base_validations = r.nav.monitor.validations
    for k in range(400):                                                   # 40 s with a constant map, vehicle creeping along
        r.advance_along(r.nav.global_path.xy, min(k * 0.02, r.nav.global_path.length - 1.0))
        r.step()
    assert r.nav.counters["plans"] == 1 and r.nav.counters["replans"] == 0
    assert r.nav.monitor.validations - base_validations <= 1               # path validated once per map version, not per tick


def test_validation_only_happens_after_the_map_changes_and_is_rate_limited():
    r = Rig()
    r.step()
    v0 = r.nav.monitor.validations
    for _ in range(5):                                                     # five rapid map-version bumps (new scans)
        mark_free(r.mapping, 0.0, 2.0, 0.0, 2.0, times=1)
        r.step()
    assert r.nav.monitor.validations - v0 <= 2                             # validate_interval_s = 0.4 s of 0.5 s elapsed


def test_newly_mapped_obstacle_far_on_the_path_triggers_one_global_replan():
    r = Rig()
    r.step()
    old = r.nav.global_path.xy.copy()
    mark_occupied(r.mapping, 22.0, 24.0, 3.5, 6.5)                         # 17+ m ahead: beyond the avoidance trigger
    r.step(15)                                                             # > min_replan_interval_s (1.0 s): the check is deferred, not lost
    assert r.nav.counters["replans"] == 1 and r.nav.replan_reasons == {"path_blocked": 1}
    assert r.nav.state == NavState.FOLLOW_PATH and r.nav.avoider is not None and r.nav.counters["avoidance_count"] == 0
    new = r.nav.global_path
    assert not np.array_equal(old, new.xy) and new.xy[0] == pytest.approx(r.pos[:2])
    assert check_path(r.nav.plan_grid, new.xy).free                        # the new path is clear in the CURRENT map
    assert not check_path(r.nav.plan_grid, old).free                       # ... and the old one really was blocked
    r.step(50)
    assert r.nav.counters["replans"] == 1                                  # settled: no further replans


def test_replans_are_rate_limited_and_a_deferred_replan_is_not_lost():
    r = Rig({"navigation": {"min_replan_interval_s": 3.0}})
    r.step()
    mark_occupied(r.mapping, 22.0, 24.0, 3.5, 6.5)
    r.step(10)                                                             # t = 1.1 s < 3 s since the plan
    assert r.nav.counters["replans"] == 0 and r.nav.counters["throttled_replans"] >= 1
    assert r.nav.state == NavState.FOLLOW_PATH                              # still following the old path meanwhile
    r.step(30)                                                             # t = 4.1 s
    assert r.nav.counters["replans"] == 1                                   # the deferred replan DID happen
    g = r.nav.global_path
    q = g.point(0.6 * g.length)
    thr = r.nav.counters["throttled_replans"]
    mark_occupied(r.mapping, q[0] - 1.5, q[0] + 1.5, q[1] - 1.5, q[1] + 1.5)
    r.step(10)
    assert r.nav.counters["replans"] == 1 and r.nav.counters["throttled_replans"] > thr   # held back by the rate limit
    r.step(40)
    assert r.nav.counters["replans"] == 2


def test_goal_change_forces_an_immediate_replan_to_the_new_goal():
    r = Rig({"navigation": {"min_replan_interval_s": 50.0}})                # even a huge interval must not delay a goal change
    r.step(5)
    r.nav.set_goal([20.0, 30.0, -5.0], r.t)
    assert r.nav.state == NavState.REPLAN
    r.step(2)
    assert r.nav.state == NavState.FOLLOW_PATH
    assert r.nav.global_path.xy[-1] == pytest.approx([20.0, 30.0])
    assert r.nav.counters["goal_changes"] == 1 and r.nav.replan_reasons.get("goal_changed") == 1
    r.nav.set_goal([20.0, 30.0, -5.0], r.t)                                 # same goal again: no new replan
    assert r.nav.state == NavState.FOLLOW_PATH


def test_significant_map_change_replans_even_when_the_path_is_not_blocked():
    r = Rig()
    r.step()
    r.mapping.reset()                                                        # operator 'reset map': everything unknown (cost 50 / cell-m)
    r.mapping.version += 1
    r.step(15)
    assert r.nav.replan_reasons.get("map_change") == 1
    assert r.nav.state == NavState.FOLLOW_PATH and r.nav.counters["replans"] == 1


def test_a_free_to_unknown_or_unchanged_cost_does_not_replan():
    r = Rig()
    r.step()
    mark_free(r.mapping, 0.0, 10.0, 0.0, 10.0, times=1)                      # a pure information gain, nothing worse
    r.step(10)
    assert r.nav.counters["replans"] == 0


# =============================================================================== local avoidance
def blocked_ahead(r, ahead_m=3.5, half=0.6):
    g = r.nav.global_path
    from src.planning.grid_utils import cumulative_length, point_at_arc
    p = point_at_arc(g.xy, cumulative_length(g.xy), r.nav.follower.s + ahead_m)
    mark_occupied(r.mapping, p[0] - half, p[0] + half, p[1] - half, p[1] + half)
    return p


def test_obstacle_near_the_path_starts_a_deterministic_detour_that_rejoins_the_global_path():
    r = Rig()
    r.step()
    r.pos[:2] = [12.0, 5.0]
    r.step(2)
    obst = blocked_ahead(r, 4.0)
    r.step(6)
    nav = r.nav
    assert nav.state == NavState.AVOID_OBSTACLE and nav.counters["avoidance_count"] == 1
    assert nav.counters["replans"] == 0                                     # avoidance is local: the global plan is untouched
    d = nav.detour
    assert d is not None and d.kind == "detour" and nav.follower.path is d
    grid = nav._grid(r.mapping)
    assert check_path(grid, d.xy).free                                      # respects occupied + inflated + extra clearance
    assert d.xy[0] == pytest.approx(r.pos[:2], abs=0.3)
    gpath = nav.global_path
    s, cross, _ = gpath.project(d.xy[-1])
    assert cross < 0.05 and s == pytest.approx(nav._rejoin_s, abs=0.05)     # ends ON the global path, beyond the obstacle
    assert s > gpath.project(obst)[0]
    # keep-out distance: sampled detour never closer to the obstacle than inflation + extra clearance
    pts = np.vstack([np.linspace(a, b, 50) for a, b in zip(d.xy[:-1], d.xy[1:])])
    dist = np.min(np.linalg.norm(pts - obst[None, :], axis=1)) - 0.6
    assert dist > 0.6 + 0.5 - 0.3
    # the global path is kept for the return
    assert nav.global_path.version != 0 and nav.telemetry()["avoidance"]["active"]
    # drive the estimate to the end of the detour -> back to FOLLOW_PATH on the global path
    r.step(pos=(*d.xy[-1], -5.0))
    assert nav.state == NavState.FOLLOW_PATH and nav.follower.path is nav.global_path and nav.detour is None
    assert nav.follower.s == pytest.approx(nav._rejoin_s, abs=0.3)
    assert [h["to"] for h in nav.history][-3:] == ["FOLLOW_PATH", "AVOID_OBSTACLE", "FOLLOW_PATH"]
    assert nav.history[-1]["reason"] == "rejoined_global_path"


def test_avoidance_detour_is_identical_for_identical_inputs():
    paths = []
    for _ in range(2):
        r = Rig()
        r.step()
        r.pos[:2] = [12.0, 5.0]
        r.step(2)
        blocked_ahead(r, 4.0)
        r.step(6)
        paths.append(r.nav.detour.xy.copy())
    assert np.array_equal(paths[0], paths[1])


def test_blocked_run_locates_the_obstacle_along_a_polyline():
    r = Rig()
    r.step()
    grid = r.nav._grid(r.mapping)
    poly = np.array([[5.0, 5.0], [30.0, 5.0]])
    assert blocked_run(grid, poly) is None
    mark_occupied(r.mapping, 14.0, 15.0, 4.0, 6.0)
    run = blocked_run(r.nav._grid(r.mapping), poly)
    costs = r.cfg["mapping"]["obstacle"]
    keepout = costs["vehicle_radius_m"] + costs["safety_margin_m"] + r.cfg["planning"]["extra_clearance_m"]   # mapped inflation + planner extra
    s0, s1 = 14.0 - keepout - 5.0, 15.0 + keepout - 5.0                      # obstacle x 14..15, path starts at x = 5
    assert run is not None and s0 - 0.6 < run[0] < run[1] < s1 + 0.6          # the blocked run is the obstacle widened by the keep-out band
    assert run[1] - run[0] > (s1 - s0) - 1.2                                  # ... and covers (almost) all of it, not a sliver


def test_detour_failure_falls_back_to_a_global_replan_that_goes_around():
    r = Rig()
    r.step()
    r.pos[:2] = [12.0, 5.0]
    r.step(2)
    mark_occupied(r.mapping, 15.5, 16.5, -4.9, 40.0)                          # a wall from the map edge, far longer than the local window
    r.step(10)
    assert r.nav.state in (NavState.FOLLOW_PATH, NavState.REPLAN)
    reasons = list(r.nav.replan_reasons)
    assert any(k.startswith("detour_failed") or k == "path_blocked" for k in reasons)
    assert r.nav.counters["replans"] >= 1
    assert check_path(r.nav.plan_grid, r.nav.global_path.xy).free


def test_sonar_forward_clearance_uses_only_hits_in_the_forward_slice():
    az = np.deg2rad(np.linspace(-45, 45, 21))
    el = np.deg2rad(np.array([-15, -7.5, 0, 7.5, 15]))
    A, E = np.meshgrid(az, el)                                                # elevation-major like the sensor
    n = A.size
    rng = np.full(n, 30.0)
    det = np.zeros(n, bool)
    valid = np.ones(n, bool)

    def meas(ranges, detected, ray_valid):
        return SonarMeasurement(sensor="sonar", timestamp=1.0, value=np.asarray(ranges, float), valid=True,
                                health=SensorHealth.HEALTHY, azimuths=A.ravel(), elevations=E.ravel(),
                                detected=detected, max_range=30.0, min_range=0.2, ray_valid=ray_valid)
    assert sonar_forward_clearance(meas(rng, det, valid), 12.0) is None
    k_fwd = int(np.argmin(np.abs(A.ravel()) + np.abs(E.ravel())))             # boresight ray
    r2, d2 = rng.copy(), det.copy()
    r2[k_fwd], d2[k_fwd] = 5.0, True
    assert sonar_forward_clearance(meas(r2, d2, valid), 12.0) == pytest.approx(5.0)
    k_side = int(np.argmin(np.abs(A.ravel() - np.deg2rad(40)) + np.abs(E.ravel())))
    r3, d3 = r2.copy(), d2.copy()
    r3[k_side], d3[k_side] = 1.0, True                                          # a very close hit, but 40 deg off the cone
    assert sonar_forward_clearance(meas(r3, d3, valid), 12.0) == pytest.approx(5.0)
    bad = valid.copy()
    bad[k_fwd] = False                                                           # dropout ray carries no information
    assert sonar_forward_clearance(meas(r2, d2, bad), 12.0) is None
    k_up = int(np.argmin(np.abs(A.ravel()) + np.abs(E.ravel() - np.deg2rad(15))))
    r4, d4 = rng.copy(), det.copy()
    r4[k_up], d4[k_up] = 6.0, True                                               # 6 m * sin(15deg) = 1.55 m above the slice
    assert sonar_forward_clearance(meas(r4, d4, valid), 12.0) is None
    assert sonar_forward_clearance(None, 12.0) is None


def test_reactive_speed_scale_slows_but_never_vetoes_the_planned_detour():
    r = Rig()
    n = r.nav
    assert n._scale_from_clearance(None) == 1.0
    assert n._scale_from_clearance(10.0) == 1.0
    lo, hi, floor = n.nav.sonar_stop_range_m, n.nav.sonar_slow_range_m, n.nav.sonar_min_scale
    mid = n._scale_from_clearance(0.5 * (lo + hi))
    assert floor <= mid < 1.0
    assert n._scale_from_clearance(lo + 0.01) >= floor - 1e-9              # a legitimate close pass is still driven
    assert n._scale_from_clearance(lo) == 0.0                              # only a real hard-stop range holds the vehicle
    vals = [n._scale_from_clearance(c) for c in np.linspace(lo, hi, 20)]
    assert all(b >= a - 1e-12 for a, b in zip(vals[1:], vals[2:]))


# =============================================================================== telemetry / commands
def test_telemetry_is_strict_json_and_contains_every_dashboard_field():
    r = Rig()
    r.step(3)
    mark_occupied(r.mapping, 22.0, 24.0, 3.5, 6.5)
    r.step(15)
    tel = r.nav.telemetry()
    txt = json.dumps(jsonsafe({"navigation": tel, "navigation_path": r.nav.path_payload()}), allow_nan=False)
    assert "NaN" not in txt and "Infinity" not in txt
    for k in ("state", "goal", "goal_distance_m", "planner", "following", "avoidance", "replanning", "history", "pose_source"):
        assert k in tel
    for k in ("status", "path_length_m", "last_planning_time_ms", "plans", "plan_failures", "collision_free", "unknown_policy"):
        assert k in tel["planner"]
    for k in ("current_waypoint", "waypoint_index", "remaining_m", "cross_track_m"):
        assert k in tel["following"]
    assert tel["pose_source"] == "EKF estimate" and tel["replanning"]["count"] >= 1
    pp = r.nav.path_payload()
    assert pp["global"] and pp["goal"] == [30.0, 5.0, -5.0] and pp["waypoint"] is not None


def test_dashboard_commands_act_on_the_real_navigator_and_report_real_reasons():
    r = Rig()
    r.step()
    state = {"sim_time": r.t}
    ok, why = apply_command({"cmd": "nav_set_goal", "goal": [100.0, 0.0, -5.0]}, r.mapping, None, state, r.nav)
    assert not ok and "outside the map" in why and r.nav.counters["goal_changes"] == 0
    ok, why = apply_command({"cmd": "nav_set_goal", "goal": [20.0, 25.0, -5.0]}, r.mapping, None, state, r.nav)
    assert ok and r.nav.state == NavState.REPLAN and list(r.nav.goal) == [20.0, 25.0, -5.0]
    assert apply_command({"cmd": "nav_replan"}, r.mapping, None, state, r.nav)[0] is False       # already replanning
    r.step(2)
    assert apply_command({"cmd": "nav_replan"}, r.mapping, None, state, r.nav)[0]
    assert apply_command({"cmd": "nav_set_unknown_policy", "policy": "block"}, r.mapping, None, state, r.nav)[0]
    assert r.nav.pcfg.unknown_policy == "block"
    assert apply_command({"cmd": "nav_cancel"}, r.mapping, None, state, r.nav)[0] and r.nav.state == NavState.IDLE
    ok, why = apply_command({"cmd": "nav_start"}, r.mapping, None, state, r.nav)
    assert ok and r.nav.state == NavState.PLAN
    ok, why = apply_command({"cmd": "nav_set_goal", "goal": [1, 2, -5]}, r.mapping, None, state, None)
    assert not ok and "not enabled" in why


def test_navigation_prerequisites_are_checked_before_a_run_starts():
    assert navigation_prerequisites(load_cfg()) is None
    for override, word in [({"sensors": {"enabled": False}}, "sensors"), ({"ekf": {"enabled": False}}, "EKF"),
                           ({"sensors": {"sonar": {"enabled": False}}}, "sonar"), ({"mapping": {"enabled": False}}, "mapping")]:
        why = navigation_prerequisites(load_cfg(override))
        assert why and word in why


def test_navigation_is_off_by_default_so_phase4_behaviour_is_unchanged():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        assert yaml.safe_load(f)["navigation"]["enabled"] is False

"""Closed-loop autonomous navigation WITHOUT PyBullet.

REAL here: sensor suite (IMU / DVL / depth with noise), the sonar model (analytic ray cast replaces only the PyBullet
ray query), the 15-state EKF, the Phase-4 occupancy + cost map, and the Phase-5 navigator (planner, follower, avoidance,
replanning). STAND-IN: the rigid body (tests/helpers/closed_loop.NumpyVehicle, same force model as the legacy dynamics).
Obstacle collisions are judged against the generated world's footprints (EVALUATION only). The real-PyBullet versions of
these missions are in tests/test_navigation_pybullet.py. Nothing in this file is a PyBullet result."""
import numpy as np
import pytest

from tests.helpers.closed_loop import run_closed_loop


def pillar(x, y, r=0.8):
    return {"kind": "pillar", "x": x, "y": y, "radius_m": r, "z_top_m": -2.0}


def wall(x, y, hx, hy):
    return {"kind": "wall", "x": x, "y": y, "half_x_m": hx, "half_y_m": hy, "z_top_m": -2.0}


def test_open_water_mission_follows_one_plan_to_the_goal():
    r = run_closed_loop({"environment": {"scenario": "open_terrain"}, "navigation": {"goal_position": [30.0, 5.0, -5.0]}}, max_time=100)
    assert r.final_state == "GOAL_REACHED"
    assert r.states_visited == ["PLAN", "FOLLOW_PATH", "GOAL_REACHED"]
    tel = r.telemetry
    assert tel["planner"]["plans"] == 1 and tel["replanning"]["count"] == 0 and tel["avoidance"]["count"] == 0
    assert tel["planner"]["collision_free"] is True and tel["planner"]["last_planning_time_ms"] > 0
    assert np.linalg.norm(r.est_final - r.goal) < 0.5                         # decided from the estimate
    assert r.truth_goal_distance < 2.0                                       # and the real vehicle is close (EKF drift allowance)
    assert abs(r.truth_final[2] - (-5.0)) < 0.4                              # depth held
    assert r.sim_time < 70.0


def test_unmapped_obstacles_are_discovered_by_sonar_then_avoided_and_the_mission_completes():
    ov = {"environment": {"scenario": "mapping_validation", "explicit_obstacles": [pillar(21, 12.5), pillar(24, 11.2)]},
          "navigation": {"goal_position": [35.0, 28.0, -5.0]}}
    r = run_closed_loop(ov, max_time=150)
    tel = r.telemetry
    assert r.final_state == "GOAL_REACHED"
    assert tel["replanning"]["count"] >= 1 and tel["avoidance"]["count"] >= 1          # both mechanisms were exercised
    for s in ("PLAN", "FOLLOW_PATH", "REPLAN", "AVOID_OBSTACLE", "GOAL_REACHED"):
        assert s in r.states_visited, (s, r.states_visited)
    assert r.collisions == 0 and r.min_clearance_m > 0.5                      # EVALUATION: never touched an obstacle, kept clear of it
    assert r.truth_goal_distance < 2.5
    assert len(r.planned_paths) >= 2                                          # the global path really was replaced after mapping
    # every distinct global path was collision-free on the map version it was planned on
    assert tel["planner"]["collision_free"] is True
    assert all(h["reason"] for h in r.history)


def test_obstacle_that_appears_mid_mission_is_avoided_locally_and_the_global_path_is_rejoined():
    state = {"done": False}

    def inject(step, t, nav, mapping, est, veh):
        if not state["done"] and t >= 9.0:
            state["done"] = True
            est.suite.sensors["sonar"].world.cylinders.append((np.array([17.0, 5.0, -5.0]), 0.5, 10.0))   # EVALUATION world edit
    r = run_closed_loop({"environment": {"scenario": "open_terrain"}, "navigation": {"goal_position": [30.0, 5.0, -5.0]}},
                        max_time=100, pre_step=inject)
    assert r.final_state == "GOAL_REACHED"
    reasons = [h["reason"] for h in r.history]
    assert "rejoined_global_path" in reasons and any(x.startswith("obstacle_on_path") for x in reasons)
    assert r.telemetry["avoidance"]["count"] == 1 and r.telemetry["replanning"]["count"] == 0
    d = np.min(np.linalg.norm(r.truth_path[:, :2] - np.array([17.0, 5.0]), axis=1)) - 0.5
    assert d > 0.5                                                           # passed the real pillar with margin
    assert r.truth_goal_distance < 2.0


def test_goal_change_mid_mission_replans_to_the_new_goal():
    r = run_closed_loop({"environment": {"scenario": "open_terrain"}, "navigation": {"goal_position": [30.0, 5.0, -5.0]}},
                        max_time=100, goal_updates=[(10.0, [20.0, 25.0, -5.0])])
    assert r.final_state == "GOAL_REACHED"
    assert np.linalg.norm(r.est_final[:2] - [20.0, 25.0]) < 0.5 and np.linalg.norm(r.truth_final[:2] - [20.0, 25.0]) < 2.0
    assert r.telemetry["replanning"]["goal_changes"] == 1 and r.telemetry["replanning"]["reasons"].get("goal_changed") == 1


def test_goal_behind_an_unbypassable_wall_ends_in_FAILED_after_real_replanning_attempts():
    dam = [wall(30, 25, 0.5, 40.0)]                                          # spans the whole map: there is no route to the goal
    r = run_closed_loop({"environment": {"scenario": "open_terrain", "explicit_obstacles": dam},
                         "navigation": {"goal_position": [45.0, 20.0, -5.0]}}, max_time=160)
    assert r.final_state == "FAILED"
    assert r.telemetry["failure_reason"].startswith("no_path") and "no traversable route" in r.telemetry["failure_reason"]
    assert r.telemetry["planner"]["status"] == "NO_PATH" and r.telemetry["planner"]["plan_failures"] >= 4
    assert r.telemetry["replanning"]["count"] >= 1                           # it tried: the wall was discovered by sonar first
    assert r.collisions == 0 and r.min_clearance_m > 1.5                     # and it stopped short of the wall instead of ramming it
    assert r.truth_final[0] < 29.0


def test_narrow_corridor_scenario_is_navigated():
    r = run_closed_loop({"environment": {"scenario": "narrow_corridor"}, "navigation": {"goal_position": [35.0, 28.0, -5.0]}}, max_time=150)
    assert r.final_state == "GOAL_REACHED" and r.collisions == 0


def test_the_closed_loop_is_deterministic():
    ov = {"environment": {"scenario": "open_terrain"}, "navigation": {"goal_position": [25.0, 10.0, -5.0]}}
    a, b = run_closed_loop(ov, max_time=80), run_closed_loop(ov, max_time=80)
    assert a.final_state == b.final_state and a.sim_time == b.sim_time
    assert np.array_equal(a.truth_final, b.truth_final) and a.states_visited == b.states_visited


def test_unknown_policy_block_mission_cannot_leave_the_known_area_and_fails_cleanly():
    r = run_closed_loop({"environment": {"scenario": "open_terrain"}, "planning": {"unknown_policy": "block"},
                         "navigation": {"goal_position": [30.0, 5.0, -5.0]}}, max_time=40)
    assert r.final_state == "FAILED" and r.telemetry["planner"]["unknown_policy"] == "block"

"""REAL PyBullet end-to-end autonomous navigation (Phase 5).

AUV (real PyBullet body, real underwater force model, real thrusters) -> real PyBullet sonar ray casts -> EKF estimate ->
online occupancy / cost map -> A* -> path follower -> local avoidance -> replanning -> goal. These tests are SKIPPED if PyBullet
cannot be imported (and reported as such by tests/conftest.py); AURORA_REQUIRE_PYBULLET=1 turns that into a hard failure.

Pass/fail criteria are operational (state machine, estimate-based goal check, physics collision count); the true position is
used only as EVALUATION, never to decide anything."""
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pybullet", reason="pybullet is not installed")

from src.localization.state import EstimatedState                        # noqa: E402
from src.mapping.map_manager import MappingSystem                        # noqa: E402
from src.planning.navigator import AutonomousNavigator                   # noqa: E402
from src.sensors.models import SonarMeasurement, TruthSnapshot           # noqa: E402
from src.simulation.run_simulation import run                            # noqa: E402


def pillar(x, y, r=0.8):
    return {"kind": "pillar", "x": x, "y": y, "radius_m": r, "z_top_m": -2.0}


MISSION = {"environment": {"scenario": "mapping_validation", "explicit_obstacles": [pillar(21, 12.5), pillar(24, 11.2)]},
           "navigation": {"enabled": True, "goal_position": [35.0, 28.0, -5.0], "linger_after_terminal_s": 1.0},
           "simulation": {"max_steps": 14000}, "dashboard": {"realtime_factor": 0.0}}


def mission(override=None):
    from src.simulation.commands import deep_merge
    return run(mode_override="DIRECT", cfg_override=deep_merge(MISSION, override or {}))


@pytest.fixture(scope="module")
def spied_mission():
    """One real mission with a spy on every navigator.update() call (types of everything the navigator ever receives)."""
    seen = {"calls": 0, "bad": [], "times": []}
    orig = AutonomousNavigator.update

    def spy(self, sim_time, estimate, mapping, sonar=None):
        seen["calls"] += 1
        if not isinstance(estimate, EstimatedState):
            seen["bad"].append(("estimate", type(estimate).__name__))
        if not isinstance(mapping, MappingSystem):
            seen["bad"].append(("mapping", type(mapping).__name__))
        if sonar is not None and not isinstance(sonar, SonarMeasurement):
            seen["bad"].append(("sonar", type(sonar).__name__))
        seen["times"].append(float(sim_time))
        return orig(self, sim_time, estimate, mapping, sonar)
    AutonomousNavigator.update = spy
    try:
        result = mission()
    finally:
        AutonomousNavigator.update = orig
    return result, seen


def test_real_pybullet_mission_senses_maps_plans_follows_avoids_replans_and_reaches_the_goal(spied_mission):
    result, seen = spied_mission
    nav = result["navigation"]
    assert nav is not None and nav["enabled"]
    assert nav["state"] == "GOAL_REACHED" and result["goal_reached"] is True and result["final_status"] == "COMPLETED"
    assert nav["goal_distance_m"] < nav["goal_tolerance_m"] + 0.1               # decided from the ESTIMATE
    assert result["collisions"] == 0                                           # real PyBullet contacts
    assert nav["final_true_goal_distance_m"] < 3.0                              # EVALUATION: the real vehicle is near the goal (EKF drift allowance)
    assert nav["planner"]["plans"] >= 1 and nav["planner"]["plan_successes"] >= 1 and nav["planner"]["collision_free"] is True
    assert nav["replanning"]["count"] + nav["avoidance"]["count"] >= 1          # obstacles discovered by sonar forced a reaction
    assert result["mapping"]["mapping"]["update_count"] > 50 and result["mapping"]["mapping"]["occupied_cells"] > 0
    states = [h["to"] for h in nav["state_history"]]
    assert states[:2] == ["PLAN", "FOLLOW_PATH"] and states[-1] == "GOAL_REACHED"


def test_navigator_only_ever_receives_estimate_map_and_sonar(spied_mission):
    result, seen = spied_mission
    assert seen["calls"] > 1000
    assert seen["bad"] == [], seen["bad"][:3]
    assert all(b > a for a, b in zip(seen["times"], seen["times"][1:]))         # one call per simulation step, in order
    assert TruthSnapshot is not None                                           # (imported to document what is NOT passed)


def test_mission_log_shows_the_navigator_drove_the_vehicle(spied_mission):
    result, _ = spied_mission
    df = pd.read_csv(result["log_path"])
    assert {"nav_state", "nav_goal_distance_est", "nav_replans", "nav_avoidance_count"} <= set(df.columns)
    assert df["nav_state"].iloc[0] in ("PLAN", "FOLLOW_PATH") and df["nav_state"].iloc[-1] == "GOAL_REACHED"
    assert "FOLLOW_PATH" in set(df["nav_state"])
    assert df["nav_goal_distance_est"].iloc[-1] < 0.6 and df["nav_goal_distance_est"].iloc[0] > 30.0
    later = df[df["time"] > 5.0] if "time" in df.columns else df.iloc[500:]
    assert (later["z"] + 5.0).abs().max() < 0.8                                 # depth held through the mission (truth, evaluation)
    assert np.isfinite(df[["x", "y", "z"]].to_numpy()).all()


def test_goal_behind_an_unbypassable_wall_fails_cleanly_in_real_pybullet():
    dam = {"kind": "wall", "x": 30, "y": 25, "half_x_m": 0.5, "half_y_m": 40.0, "z_top_m": -2.0}
    result = mission({"environment": {"explicit_obstacles": [dam]}, "navigation": {"goal_position": [45.0, 20.0, -5.0]},
                      "simulation": {"max_steps": 30000}})
    # Time budget 160 s -> 300 s (assertions unchanged). With the compass the pose and map are accurate, so the vehicle
    # explores the 80 m wall honestly (~125 s on the stand-in) before the planner can prove "no route"; with the old
    # drifting yaw the smeared map closed the gaps sooner (~89 s). The mission still ends well inside max_mission_time_s.
    nav = result["navigation"]
    assert nav["state"] == "FAILED" and nav["failure_reason"].startswith("no_path")
    assert result["goal_reached"] is False and result["final_status"] == "ENDED_BEFORE_GOAL"
    assert result["collisions"] == 0


def test_navigation_off_leaves_the_phase4_waypoint_mission_untouched():
    result = run(mode_override="DIRECT", cfg_override={"simulation": {"max_steps": 400}, "dashboard": {"realtime_factor": 0.0}})
    assert result["navigation"] is None and "goal_reached_true_position" in result


def test_navigation_with_a_missing_prerequisite_is_refused_before_the_simulation_starts():
    with pytest.raises(ValueError, match="cannot run"):
        run(mode_override="DIRECT", cfg_override={"navigation": {"enabled": True}, "sensors": {"enabled": False},
                                                  "simulation": {"max_steps": 10}})

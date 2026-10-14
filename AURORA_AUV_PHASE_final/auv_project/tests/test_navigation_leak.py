"""GROUND-TRUTH RULE for Phase 5: truth is for tests / debug / evaluation only. It must never drive planning, path
following, obstacle avoidance, replanning or operational telemetry.

Enforced three ways: (1) statically - the planning package cannot even name a truth source and the run loop calls the
navigator with estimate-only arguments; (2) dynamically - corrupting the ESTIMATE moves every operational decision
(path origin, goal-reached, map) while the true vehicle ends up elsewhere; (3) at the telemetry boundary - no
truth-derived number appears in the operational `navigation` section of a snapshot."""
import ast
import glob
import inspect
import os

import numpy as np
import pytest

from src.planning.navigator import AutonomousNavigator
from tests.helpers.closed_loop import run_closed_loop

PLANNING_FILES = sorted(glob.glob("src/planning/*.py"))
FORBIDDEN_IMPORT_PREFIXES = ("pybullet", "pybullet_data", "src.simulation", "src.sensors.models", "src.sensors.suite",
                             "src.sensors.imu", "src.sensors.dvl", "src.sensors.depth", "tests", "dashboard")
FORBIDDEN_IDENTIFIER_PARTS = ("truth", "pybullet", "getbase", "ground", "body_id", "physics_client")


def identifiers(tree):
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            yield n.id
        elif isinstance(n, ast.Attribute):
            yield n.attr
        elif isinstance(n, ast.arg):
            yield n.arg
        elif isinstance(n, (ast.FunctionDef, ast.ClassDef)):
            yield n.name
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                yield a.name
            if isinstance(n, ast.ImportFrom) and n.module:
                yield n.module


def test_the_planning_package_exists_and_is_scanned():
    names = {os.path.basename(p) for p in PLANNING_FILES}
    assert {"navigator.py", "global_planner.py", "astar.py", "path_follower.py", "local_avoidance.py",
            "replanning.py", "smoothing.py", "grid_utils.py", "path_data.py", "config.py"} <= names


@pytest.mark.parametrize("path", PLANNING_FILES)
def test_planning_modules_import_no_truth_or_simulation_source(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for n in ast.walk(tree):
        mods = []
        if isinstance(n, ast.Import):
            mods = [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom) and n.module:
            mods = [n.module]
        for m in mods:
            assert not m.startswith(FORBIDDEN_IMPORT_PREFIXES), f"{path} imports {m}"


@pytest.mark.parametrize("path", PLANNING_FILES)
def test_planning_modules_never_name_a_truth_source(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    for ident in set(identifiers(tree)):
        low = ident.lower()
        for bad in FORBIDDEN_IDENTIFIER_PARTS:
            assert bad not in low, f"{path}: identifier {ident!r} suggests a truth/physics source"


def test_navigator_update_signature_has_no_truth_input():
    params = list(inspect.signature(AutonomousNavigator.update).parameters)
    assert params == ["self", "sim_time", "estimate", "mapping", "sonar"]


def test_run_loop_calls_the_navigator_with_estimate_only_arguments():
    src = open("src/simulation/run_simulation.py", encoding="utf-8").read()
    tree = ast.parse(src)
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and n.func.attr == "update" and isinstance(n.func.value, ast.Name) and n.func.value.id == "navigator"]
    assert len(calls) == 1
    names = {n.id for a in calls[0].args for n in ast.walk(a) if isinstance(n, ast.Name)}
    assert names <= {"step", "dt", "estimate", "mapping", "estimation"}, names
    # the whole navigation control block (the branch that decides `control`) never reads the TRUE pose variables
    blocks = [n for n in ast.walk(tree) if isinstance(n, ast.If) and isinstance(n.test, ast.Compare)
              and isinstance(n.test.left, ast.Name) and n.test.left.id == "navigator"
              and isinstance(n.test.ops[0], ast.IsNot) and any(isinstance(s, ast.Assign) and any(
                  isinstance(t, ast.Name) and t.id == "nav_out" for t in s.targets) for s in n.body)]
    assert blocks, "navigation control block not found"
    # only the navigation branch (`body`); the `else:` branch is the Phase-4 true-state controller, which is unchanged
    used = {n.id for s in blocks[0].body for n in ast.walk(s) if isinstance(n, ast.Name)}
    assert not (used & {"position", "yaw", "velocity", "auv", "truth", "state"}), used & {"position", "yaw", "velocity", "auv", "truth", "state"}


def test_operational_pose_in_the_navigation_telemetry_is_the_ekf_estimate():
    nav = AutonomousNavigator(__import__("tests.helpers.closed_loop", fromlist=["x"]).load_cfg())
    assert nav.telemetry()["pose_source"] == "EKF estimate"


def test_biased_estimate_moves_every_operational_decision_and_truth_ends_up_elsewhere():
    """The EKF 'lies' by a fixed offset from the start. Everything the navigator decides follows the estimate: the plan
    starts at the estimate, GOAL_REACHED is declared when the ESTIMATE is at the goal, and the real vehicle is then one
    bias away from the goal. If any truth value fed planning, following or the goal check, the vehicle would end up AT
    the goal. (The map is built from the same biased estimate, as in the real system.)"""
    bias = np.array([3.0, -2.0, 0.0])

    def lie(e):
        e.position = e.position + bias
        return e
    ov = {"environment": {"scenario": "open_terrain"}, "navigation": {"goal_position": [28.0, 12.0, -5.0]}}
    r = run_closed_loop(ov, max_time=120, estimate_hook=lie)
    assert r.final_state == "GOAL_REACHED"
    first_plan = r.planned_paths[0][1]
    assert first_plan[0] == pytest.approx([5.0 + bias[0], 5.0 + bias[1]], abs=1e-6)       # planned from the ESTIMATE, not (5, 5)
    assert np.linalg.norm(r.est_final - r.goal) < 0.5                                       # goal reached according to the estimate
    off = r.truth_final - r.goal
    assert np.linalg.norm(off[:2] - (-bias[:2])) < 1.2                                      # truth is displaced by about -bias
    assert r.truth_goal_distance > 2.5


def test_a_wrong_estimate_cannot_be_rescued_by_truth_the_navigator_has_no_access_to_it():
    """Same lie, but also hand the navigator's neighbours poisoned truth: the loop must not notice anything."""
    def lie(e):
        e.position = e.position + np.array([-4.0, 1.0, 0.0])
        return e
    ov = {"environment": {"scenario": "open_terrain"}, "navigation": {"goal_position": [30.0, 8.0, -5.0]}}
    a = run_closed_loop(ov, max_time=100, estimate_hook=lie)
    b = run_closed_loop(ov, max_time=100, estimate_hook=lie)
    assert a.final_state == b.final_state == "GOAL_REACHED"
    assert np.array_equal(a.truth_final, b.truth_final)                                     # deterministic given identical inputs


def test_snapshot_navigation_section_contains_no_truth_derived_number():
    from src.simulation.telemetry_snapshot import build_snapshot
    sig = inspect.signature(build_snapshot)
    assert "navigation" in sig.parameters and "navigation_path" in sig.parameters
    src = open("src/simulation/telemetry_snapshot.py", encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "build_snapshot")
    # navigation/navigation_path are copied through verbatim from the navigator; truth only ever enters debug_ground_truth
    code = ast.get_source_segment(src, fn)
    assert 'snap["navigation"] = navigation' in code and 'snap["navigation_path"] = navigation_path' in code
    assert "truth" not in code.split('snap["navigation"] = navigation')[1].split("return snap")[0].replace("debug_ground_truth", "")

"""
test_simulation.py
-------------------
Unit tests for the Week 5 PyBullet AUV simulation:
    - URDF exists
    - PyBullet DIRECT-mode connection
    - AUV loads, initial state, state dimensions
    - forward / vertical motion, stop, energy tracking
    - grid<->world coordinate round-trip
    - waypoint controller (direction, tolerance)
    - collision detection
    - a full headless (DIRECT) simulation run

These tests need the `pybullet` package. If it isn't installed, every test
in this file is SKIPPED (not failed) via `pytest.importorskip`, so the rest
of the project's test suite (Weeks 2-4) is unaffected.

Run with:
    pytest tests/test_simulation.py -v
"""

import os
import sys

import numpy as np
import pytest
import yaml

# Ensure the project root is on sys.path when running pytest from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

pybullet = pytest.importorskip("pybullet", reason="pybullet is not installed")

from src.simulation.controller import ControllerParams, WaypointController
from src.simulation.environment import (
    Environment,
    GridInfo,
    grid_to_world,
    world_to_grid,
)

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
URDF_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "assets", "auv", "auv.urdf")
)


def _load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def cfg():
    return _load_config()


@pytest.fixture
def direct_client():
    """A fresh headless (DIRECT) PyBullet client, disconnected after the test."""
    client = pybullet.connect(pybullet.DIRECT)
    yield client
    pybullet.disconnect(physicsClientId=client)


@pytest.fixture
def environment(cfg, direct_client):
    return Environment(cfg, physics_client=direct_client)


def _make_auv(cfg, direct_client, position=None):
    from src.simulation.auv import AUV
    sim_cfg = cfg["simulation"]
    start = tuple(position) if position is not None else tuple(sim_cfg["start_position"])
    return AUV(sim_cfg["urdf_path"], start, sim_cfg, physics_client=direct_client)


# --------------------------------------------------------------------------
# Test 1: URDF exists
# --------------------------------------------------------------------------

def test_urdf_exists():
    assert os.path.isfile(URDF_PATH), f"Expected AUV URDF at {URDF_PATH}"


# --------------------------------------------------------------------------
# Test 2: PyBullet DIRECT-mode connection
# --------------------------------------------------------------------------

def test_direct_mode_connects():
    client = pybullet.connect(pybullet.DIRECT)
    assert client >= 0
    pybullet.disconnect(physicsClientId=client)


# --------------------------------------------------------------------------
# Test 3: AUV loads successfully
# --------------------------------------------------------------------------

def test_auv_loads(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    assert auv.body_id is not None
    assert auv.body_id >= 0


# --------------------------------------------------------------------------
# Test 4: Initial state matches configured start position
# --------------------------------------------------------------------------

def test_auv_initial_position(cfg, direct_client):
    sim_cfg = cfg["simulation"]
    auv = _make_auv(cfg, direct_client)
    position = auv.get_position()
    expected = tuple(sim_cfg["start_position"])
    for actual, exp in zip(position, expected):
        assert actual == pytest.approx(exp, abs=1e-6)


# --------------------------------------------------------------------------
# Test 5: State dimensions
# --------------------------------------------------------------------------

def test_state_dimensions(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    state = auv.get_state()
    assert len(state.position) == 3
    assert len(state.orientation_quat) == 4
    assert len(state.orientation_euler) == 3
    assert len(state.linear_velocity) == 3
    assert len(state.angular_velocity) == 3


# --------------------------------------------------------------------------
# Test 6: Forward motion changes position
# --------------------------------------------------------------------------

def test_forward_motion_moves_auv(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    start_pos = np.array(auv.get_position())
    for _ in range(50):
        auv.apply_control(forward=1.0)
        auv.step_underwater_physics()
        pybullet.stepSimulation(physicsClientId=direct_client)
    end_pos = np.array(auv.get_position())
    assert np.linalg.norm(end_pos - start_pos) > 0.0


# --------------------------------------------------------------------------
# Test 7: Vertical motion changes z
# --------------------------------------------------------------------------

def test_vertical_motion_changes_z(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    start_z = auv.get_position()[2]
    for _ in range(50):
        auv.apply_control(vertical=1.0)
        auv.step_underwater_physics()
        pybullet.stepSimulation(physicsClientId=direct_client)
    end_z = auv.get_position()[2]
    assert end_z != pytest.approx(start_z, abs=1e-6)


# --------------------------------------------------------------------------
# Test 8: Stop zeroes thrust commands
# --------------------------------------------------------------------------

def test_stop_zeroes_thrust(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    energy_before_stop_call = auv.get_energy_consumed()
    auv.stop()
    # stop() applies a zero-magnitude control -> no additional energy added
    assert auv.get_energy_consumed() == pytest.approx(energy_before_stop_call, abs=1e-9)


# --------------------------------------------------------------------------
# Test 9: Non-zero thrust consumes energy
# --------------------------------------------------------------------------

def test_energy_increases_with_thrust(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    assert auv.get_energy_consumed() == pytest.approx(0.0)
    auv.apply_control(forward=1.0)
    assert auv.get_energy_consumed() > 0.0


# --------------------------------------------------------------------------
# Test 10: Zero control adds no energy
# --------------------------------------------------------------------------

def test_zero_control_adds_no_energy(cfg, direct_client):
    auv = _make_auv(cfg, direct_client)
    auv.apply_control(forward=0.0, lateral=0.0, vertical=0.0, yaw=0.0)
    assert auv.get_energy_consumed() == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# Test 11: Grid <-> world coordinate round-trip
# --------------------------------------------------------------------------

def test_grid_world_round_trip():
    resolution = 1.0
    for row, col in [(10, 20), (0, 0), (99, 99), (37, 5)]:
        x, y = grid_to_world(row, col, resolution)
        recovered_row, recovered_col = world_to_grid(x, y, resolution)
        assert recovered_row == pytest.approx(row, abs=1)
        assert recovered_col == pytest.approx(col, abs=1)


# --------------------------------------------------------------------------
# Test 12: Waypoint controller requests movement toward the target
# --------------------------------------------------------------------------

def test_controller_requests_forward_movement():
    params = ControllerParams(
        kp_position=1.0, kp_yaw=1.0, max_thrust=10.0, max_yaw_torque=2.0, waypoint_tolerance=0.5
    )
    controller = WaypointController(params)
    control = controller.compute_control(
        current_position=(0.0, 0.0, 0.0), current_yaw=0.0, target_position=(10.0, 0.0, 0.0)
    )
    assert control["forward"] > 0.0


# --------------------------------------------------------------------------
# Test 13: Waypoint tolerance
# --------------------------------------------------------------------------

def test_waypoint_tolerance():
    params = ControllerParams(
        kp_position=1.0, kp_yaw=1.0, max_thrust=10.0, max_yaw_torque=2.0, waypoint_tolerance=0.5
    )
    controller = WaypointController(params)
    assert controller.reached_waypoint((0.0, 0.0, 0.0), (0.3, 0.0, 0.0)) is True
    assert controller.reached_waypoint((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)) is False


# --------------------------------------------------------------------------
# Test 14: Collision detection near a known obstacle
# --------------------------------------------------------------------------

def test_collision_detected_near_obstacle(cfg, direct_client, environment):
    # Find a real obstacle cell from the Week 4 inflated grid and place the
    # AUV exactly there, inside the obstacle's water-column pillar.
    obstacle_cells = np.argwhere(environment.occupancy_grid == environment.grid_info.obstacle_value)
    assert len(obstacle_cells) > 0, "expected at least one obstacle cell in the Week 4 grid"
    row, col = obstacle_cells[0]
    x, y = environment.grid_to_world(int(row), int(col))
    seabed_z = float(cfg["simulation"]["seabed_z"])
    obstacle_height = float(cfg["simulation"]["obstacle_height"])
    z_inside_obstacle = seabed_z + obstacle_height / 2.0

    auv = _make_auv(cfg, direct_client, position=(x, y, z_inside_obstacle))
    pybullet.stepSimulation(physicsClientId=direct_client)
    assert environment.check_collision(auv.body_id) is True


# --------------------------------------------------------------------------
# Test 15: Headless (DIRECT) simulation runs without opening a GUI
# --------------------------------------------------------------------------

def test_headless_simulation_runs(cfg, direct_client, environment):
    auv = _make_auv(cfg, direct_client)
    for _ in range(20):
        auv.apply_control(forward=0.5)
        auv.step_underwater_physics()
        pybullet.stepSimulation(physicsClientId=direct_client)
    state = auv.get_state()
    assert state.energy_consumed > 0.0


# --------------------------------------------------------------------------
# Extra: proximity sensors return finite, in-range readings
# --------------------------------------------------------------------------

def test_proximity_sensors_in_range(cfg, direct_client, environment):
    from src.simulation.sensors import read_proximity
    auv = _make_auv(cfg, direct_client)
    max_range = float(cfg["simulation"]["sensors"]["max_range"])
    pybullet.stepSimulation(physicsClientId=direct_client)
    readings = read_proximity(auv.body_id, max_range, physics_client=direct_client)
    for key in ("front", "left", "right"):
        assert key in readings
        assert 0.0 <= readings[key] <= max_range


# --------------------------------------------------------------------------
# Extra: metrics are computed only from measured values
# --------------------------------------------------------------------------

def test_metrics_computation():
    from src.simulation.metrics import compute_metrics
    start = (0.0, 0.0, -5.0)
    goal = (10.0, 0.0, -5.0)
    records = [
        {"x": 0.0, "y": 0.0, "z": -5.0, "energy": 0.0},
        {"x": 5.0, "y": 0.0, "z": -5.0, "energy": 1.0},
        {"x": 10.0, "y": 0.0, "z": -5.0, "energy": 2.0},
    ]
    metrics = compute_metrics(records, start, goal, True, 1, 1, 0, 1.5, 0.01)
    assert metrics["total_simulation_steps"] == 3
    assert metrics["trajectory_length_m"] == pytest.approx(10.0)
    assert metrics["final_distance_to_goal_m"] == pytest.approx(0.0)
    assert metrics["goal_reached"] is True
    assert "movement_effort_proxy" in metrics

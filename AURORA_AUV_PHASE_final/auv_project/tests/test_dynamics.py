"""
Phase 2 tests: 6-DOF dynamics, buoyancy, drag, current interface.
Pure NumPy -- no PyBullet needed, so these run everywhere.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.dynamics.auv_dynamics import (
    AUVDynamics, AUVParams, AddedMassCompensator, RigidBodyState,
)
from src.dynamics.drag import DragModel, relative_velocity
from src.dynamics.ocean_current import UniformCurrent
from src.utils.transforms import euler_to_quat, quat_to_euler, quat_to_rot

ROOT = Path(__file__).resolve().parent.parent


def _sim_cfg():
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["simulation"]


def _neutral_params(**dyn_overrides):
    """Params with net buoyancy trimmed to exactly zero."""
    cfg = _sim_cfg()
    probe = AUVParams.from_config(cfg)
    cfg["dynamics"]["buoyancy"]["trim_n"] = -probe.buoyancy.net_buoyancy
    cfg["dynamics"].update(dyn_overrides)
    return AUVParams.from_config(cfg)


def _run(dyn, state, tau, seconds, dt=0.01):
    for k in range(int(seconds / dt)):
        state = dyn.step(state, tau(state) if callable(tau) else tau, dt, k * dt)
    return state


# ---------------------------------------------------------------- transforms
def test_euler_quaternion_round_trip():
    for rpy in [(0.1, -0.2, 0.3), (-1.0, 0.5, 2.5), (0, 0, np.pi - 0.01)]:
        assert quat_to_euler(euler_to_quat(*rpy)) == pytest.approx(rpy, abs=1e-9)


def test_rotation_matrix_yaw_90():
    R = quat_to_rot(euler_to_quat(0, 0, np.pi / 2))
    assert R @ np.array([1.0, 0, 0]) == pytest.approx([0, 1, 0], abs=1e-12)


# ---------------------------------------------------------------- buoyancy
def test_legacy_config_gives_slightly_positive_buoyancy():
    b = AUVParams.from_config(_sim_cfg()).buoyancy
    assert b.weight == pytest.approx(196.0)
    assert 0.0 < b.net_buoyancy < 1.0          # legacy ratio 0.068 -> ~ +0.27 N


def test_neutral_vehicle_stays_put():
    params = _neutral_params()
    s = _run(AUVDynamics(params), RigidBodyState.from_euler([1, 2, -5]), np.zeros(6), 10)
    assert s.position == pytest.approx([1, 2, -5], abs=1e-9)
    assert np.allclose(s.nu, 0.0)


def test_positive_buoyancy_rises_and_negative_sinks():
    for trim, sign in [(+2.0, +1), (-2.0, -1)]:
        cfg = _sim_cfg(); cfg["dynamics"]["buoyancy"]["trim_n"] = trim
        p0 = AUVParams.from_config(cfg)
        s = _run(AUVDynamics(p0), RigidBodyState.from_euler([0, 0, -5]), np.zeros(6), 5)
        assert np.sign(s.position[2] + 5) == sign


def test_restoring_moment_opposes_roll():
    dyn = AUVDynamics(_neutral_params())
    R = quat_to_rot(euler_to_quat(0.3, 0, 0))
    w = dyn.params.buoyancy.restoring_wrench_body(R, include_weight=True)
    assert w[3] < 0                              # positive roll -> negative restoring moment
    assert w[:3] == pytest.approx([0, 0, 0], abs=1e-6)   # neutral -> no net force


def test_roll_settles_back_to_level():
    dyn = AUVDynamics(_neutral_params())
    s0 = RigidBodyState.from_euler([0, 0, -5], rpy=(0.3, 0, 0))
    s = _run(dyn, s0, np.zeros(6), 25)
    assert abs(quat_to_euler(s.quat)[0]) < 0.01


def test_pitch_restoring_has_correct_sign():
    dyn = AUVDynamics(_neutral_params())
    R = quat_to_rot(euler_to_quat(0, 0.2, 0))
    assert dyn.params.buoyancy.restoring_wrench_body(R, True)[4] < 0


# ---------------------------------------------------------------- drag / current
def test_drag_opposes_motion_and_is_zero_at_rest():
    d = DragModel()
    assert np.allclose(d.wrench(np.zeros(6)), 0)
    nu = np.array([1.0, -0.5, 0.2, 0.1, -0.1, 0.3])
    assert np.all(d.wrench(nu) * nu <= 0)


def test_quadratic_drag_grows_faster_than_linear():
    d = DragModel(linear=[1] * 6, quadratic=[2] * 6)
    f1 = abs(d.wrench(np.array([1.0, 0, 0, 0, 0, 0]))[0])
    f2 = abs(d.wrench(np.array([2.0, 0, 0, 0, 0, 0]))[0])
    assert f2 / f1 > 2.0


def test_invalid_drag_coefficients_rejected():
    with pytest.raises(ValueError):
        DragModel(linear=[1, 2, 3])
    with pytest.raises(ValueError):
        DragModel(linear=[-1] * 6)


def test_relative_velocity_uses_body_frame():
    R = quat_to_rot(euler_to_quat(0, 0, np.pi / 2))      # body x -> world +y
    # World current +x is body -y (yawed 90 deg), so a vehicle at rest moves at +y relative to the water.
    nu_r = relative_velocity(np.zeros(6), R, np.array([1.0, 0, 0]))
    assert nu_r == pytest.approx([0, 1, 0, 0, 0, 0], abs=1e-12)


def test_no_drag_when_moving_with_the_water():
    dyn = AUVDynamics(_neutral_params(), UniformCurrent([0.5, 0, 0]))
    R = np.eye(3)
    w = dyn.hydro_wrench(np.zeros(3), R, np.array([0.5, 0, 0, 0, 0, 0]))
    assert w == pytest.approx(np.zeros(6), abs=1e-9)


def test_current_pushes_unpowered_vehicle_to_water_speed():
    dyn = AUVDynamics(_neutral_params(), UniformCurrent([0.5, 0, 0]))
    s = _run(dyn, RigidBodyState.from_euler([0, 0, -5]), np.zeros(6), 40)
    assert s.velocity_world[0] == pytest.approx(0.5, rel=0.02)
    assert s.position[0] > 8.0                    # was carried downstream


def test_surge_terminal_velocity_matches_analytic_solution():
    params = _neutral_params()
    lin, quad = params.drag.linear[0], params.drag.quadratic[0]
    T = 20.0
    v_expected = (-lin + np.sqrt(lin ** 2 + 4 * quad * T)) / (2 * quad)   # lin v + quad v^2 = T
    s = _run(AUVDynamics(params), RigidBodyState.from_euler([0, 0, -5]),
             np.array([T, 0, 0, 0, 0, 0]), 30)
    assert s.nu[0] == pytest.approx(v_expected, rel=0.01)


def test_yaw_torque_terminal_rate_and_unit_quaternion():
    params = _neutral_params()
    lin, quad = params.drag.linear[5], params.drag.quadratic[5]
    M = 1.0
    r_expected = (-lin + np.sqrt(lin ** 2 + 4 * quad * M)) / (2 * quad)
    s = _run(AUVDynamics(params), RigidBodyState.from_euler([0, 0, -5]),
             np.array([0, 0, 0, 0, 0, M]), 30)
    assert s.nu[5] == pytest.approx(r_expected, rel=0.02)
    assert np.linalg.norm(s.quat) == pytest.approx(1.0, abs=1e-12)


# ---------------------------------------------------------------- added mass
def test_added_mass_reduces_initial_acceleration():
    dt, F = 0.01, 44.0
    tau = np.array([F, 0, 0, 0, 0, 0])
    with_am = AUVDynamics(_neutral_params(added_mass_enabled=True))
    no_am = AUVDynamics(_neutral_params(added_mass_enabled=False))
    s1 = with_am.step(RigidBodyState.from_euler([0, 0, -5]), tau, dt)
    s2 = no_am.step(RigidBodyState.from_euler([0, 0, -5]), tau, dt)
    assert s1.nu[0] == pytest.approx(dt * F / 22.0, rel=1e-6)    # m + Xudot = 20 + 2
    assert s2.nu[0] == pytest.approx(dt * F / 20.0, rel=1e-6)


def test_lagged_added_mass_compensator_converges_to_target_acceleration():
    M, Ma, F, dt = 20.0, 10.0, 30.0, 0.01          # ratio 0.5 (< 0.8)
    comp = AddedMassCompensator(np.full(6, M), np.full(6, Ma))
    nu = np.zeros(6); accel = None
    for _ in range(60):
        react = comp.reaction(nu, dt)
        a = (np.full(6, F) + react) / M                # what a rigid-body engine would do
        nu = nu + a * dt
        accel = a
    assert accel[0] == pytest.approx(F / (M + Ma), rel=1e-3)


def test_added_mass_compensator_rejects_unstable_ratio():
    with pytest.raises(ValueError):
        AddedMassCompensator(np.full(6, 20.0), np.full(6, 20.0))


# ---------------------------------------------------------------- config
def test_params_load_from_legacy_config_without_dynamics_section():
    cfg = _sim_cfg()
    for k in ("dynamics", "thrusters", "current"):
        cfg.pop(k, None)
    p = AUVParams.from_config(cfg)
    assert p.mass == 20.0
    assert p.drag.linear[0] == cfg["physics"]["linear_drag"]
    assert p.drag.linear[3] == cfg["physics"]["angular_drag"]


def test_rigid_body_coriolis_zero_without_rotation():
    dyn = AUVDynamics(_neutral_params())
    assert np.allclose(dyn.coriolis_rigid_body(np.array([1, 2, 3, 0, 0, 0.0])), 0)

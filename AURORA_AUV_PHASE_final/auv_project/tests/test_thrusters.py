"""
Phase 2 tests: thruster model and allocation (pure NumPy), plus an integrated
thruster -> allocator -> dynamics loop that needs no PyBullet.
"""
import numpy as np
import pytest
import yaml
from pathlib import Path

from src.control.thruster_allocator import ThrusterAllocator
from src.control.thruster_model import (
    ThrusterArray, ThrusterFault, ThrusterSpec, default_layout, specs_from_config,
)
from src.dynamics.auv_dynamics import AUVDynamics, AUVParams, RigidBodyState
from src.dynamics.ocean_current import UniformCurrent

ROOT = Path(__file__).resolve().parent.parent


def _cfg():
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)["simulation"]


# ---------------------------------------------------------------- thruster model
def test_command_is_clipped_to_limits():
    arr = ThrusterArray(default_layout(time_constant=0.0), seed=0)
    f = arr.step(np.full(6, 5.0), 0.01)
    assert f == pytest.approx(np.full(6, 30.0))
    f = arr.step(np.full(6, -5.0), 0.01)
    assert f == pytest.approx(np.full(6, -30.0))


def test_first_order_lag_step_response():
    tau, dt = 0.1, 0.001
    arr = ThrusterArray(default_layout(max_thrust=10.0, time_constant=tau), seed=0)
    cmd = np.array([1.0, 0, 0, 0, 0, 0])
    for _ in range(int(tau / dt)):
        f = arr.step(cmd, dt)
    assert f[0] == pytest.approx(10.0 * (1 - np.exp(-1)), rel=0.02)     # ~63% after one time constant
    assert f[1] == 0.0


def test_noise_is_seeded_and_reproducible():
    def run(seed):
        arr = ThrusterArray(default_layout(noise_std=0.1, time_constant=0.0), seed=seed)
        return np.array([arr.step(np.ones(6), 0.01).copy() for _ in range(5)])
    assert np.array_equal(run(1), run(1))
    assert not np.array_equal(run(1), run(2))


def test_zero_noise_is_deterministic():
    arr = ThrusterArray(default_layout(time_constant=0.0), seed=3)
    assert np.all(arr.step(np.full(6, 0.5), 0.01) == 15.0)


def test_dead_degraded_and_stuck_faults():
    arr = ThrusterArray(default_layout(time_constant=0.0), seed=0)
    arr.set_fault(0, ThrusterFault.DEAD)
    arr.set_fault(1, ThrusterFault.DEGRADED, 0.5)
    arr.set_fault(2, ThrusterFault.STUCK, 12.0)
    f = arr.step(np.ones(6), 0.01)
    assert f[0] == 0.0
    assert f[1] == pytest.approx(15.0)
    assert f[2] == pytest.approx(12.0)
    assert f[3] == pytest.approx(30.0)
    assert arr.health().tolist() == [0.0, 0.5, 0.0, 1.0, 1.0, 1.0]
    arr.clear_faults()
    assert arr.health().tolist() == [1.0] * 6


def test_wrench_equals_allocation_matrix_times_forces():
    specs = default_layout(time_constant=0.0)
    arr = ThrusterArray(specs, seed=0)
    f = arr.step(np.array([0.3, -0.2, 0.5, 0.1, -0.4, 0.6]), 0.01)
    assert arr.wrench() == pytest.approx(ThrusterAllocator(specs).B @ f)


def test_invalid_thruster_specs_rejected():
    with pytest.raises(ValueError):
        ThrusterSpec("bad", [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError):
        ThrusterSpec("bad", [0, 0, 0], [1, 0, 0], max_thrust=-1)
    with pytest.raises(ValueError):
        ThrusterArray([])


def test_wrong_command_shape_rejected():
    with pytest.raises(ValueError):
        ThrusterArray(default_layout()).step(np.zeros(3), 0.01)


def test_layout_from_config():
    specs = specs_from_config(_cfg())
    assert len(specs) == 6 and specs[0].max_thrust == 30.0


# ---------------------------------------------------------------- allocator
def _alloc():
    return ThrusterAllocator(default_layout())


def test_allocation_matrix_has_rank_five_roll_is_unactuated():
    a = _alloc()
    assert a.matrix_rank == 5
    assert np.allclose(a.B[3], 0.0)


@pytest.mark.parametrize("tau", [
    [20, 0, 0, 0, 0, 0], [0, 15, 0, 0, 0, 0], [0, 0, -12, 0, 0, 0],
    [0, 0, 0, 0, 2.0, 0], [0, 0, 0, 0, 0, 3.0], [18, 5, -8, 0, 1.0, 2.0],
])
def test_feasible_wrench_is_met_exactly(tau):
    r = _alloc().allocate(tau)
    assert r.achieved == pytest.approx(tau, abs=1e-6)
    assert not r.saturated
    assert np.all(np.abs(r.commands) <= 1.0 + 1e-9)


def test_zero_wrench_gives_zero_commands():
    assert np.allclose(_alloc().allocate(np.zeros(6)).commands, 0.0)


def test_saturation_respects_limits_and_is_flagged():
    r = _alloc().allocate([200, 0, 0, 0, 0, 0])
    assert r.saturated
    assert np.all(np.abs(r.commands) <= 1.0 + 1e-9)
    assert r.achieved[0] == pytest.approx(60.0, rel=1e-6)       # two aft thrusters x 30 N


def test_unactuated_roll_appears_in_residual():
    r = _alloc().allocate([0, 0, 0, 1.5, 0, 0])
    assert r.residual[3] == pytest.approx(1.5)


def test_reallocation_around_known_failed_thruster_recovers_pure_surge():
    a = _alloc()
    avail = np.array([False, True, True, True, True, True])      # aft_port known dead
    r = a.allocate([20, 0, 0, 0, 0, 0], avail)
    assert r.forces[0] == 0.0
    assert r.achieved == pytest.approx([20, 0, 0, 0, 0, 0], abs=1e-6)   # lateral pair cancels the yaw


def test_both_surge_thrusters_lost_cannot_produce_surge():
    r = _alloc().allocate([20, 0, 0, 0, 0, 0], np.array([False, False, True, True, True, True]))
    assert r.residual[0] == pytest.approx(20.0)


def test_undetected_fault_is_not_compensated():
    """Allocator unaware of a dead thruster -> the vehicle gets a yawing, weaker wrench."""
    specs = default_layout(time_constant=0.0)
    arr = ThrusterArray(specs, seed=0)
    arr.set_fault(0, ThrusterFault.DEAD)
    a = ThrusterAllocator(specs)
    res = a.allocate([20, 0, 0, 0, 0, 0])
    arr.step(res.commands, 0.01)
    w = arr.wrench()
    assert w[0] == pytest.approx(10.0)            # half the surge
    assert abs(w[5]) > 0.5                        # and an unwanted yaw moment


# ---------------------------------------------------------------- integrated loop (no PyBullet)
def _closed_loop(tau_fn, seconds, current=(0, 0, 0), faults=(), known_dead=()):
    cfg = _cfg(); cfg["current"]["base"] = list(current)
    probe = AUVParams.from_config(cfg)
    cfg["dynamics"]["buoyancy"]["trim_n"] = 0.5 - probe.buoyancy.net_buoyancy    # +0.5 N net positive
    dyn = AUVDynamics(AUVParams.from_config(cfg), UniformCurrent(current))
    specs = specs_from_config(cfg)
    arr, alloc = ThrusterArray(specs, seed=0), ThrusterAllocator(specs)
    for i in faults:
        arr.set_fault(i, ThrusterFault.DEAD)
    avail = np.ones(6, bool); avail[list(known_dead)] = False
    s, dt = RigidBodyState.from_euler([0, 0, -5]), 0.01
    for k in range(int(seconds / dt)):
        arr.step(alloc.allocate(tau_fn(s), avail).commands, dt)
        s = dyn.step(s, arr.wrench(), dt, k * dt)
    return s


def test_pipeline_depth_hold_against_buoyancy_and_current():
    target = -8.0
    pd = lambda s: np.array([0, 0, 25.0 * (target - s.position[2]) - 40.0 * s.velocity_world[2], 0, 0, 0])
    s = _closed_loop(pd, 40, current=(0.2, 0.1, 0.0))
    assert s.position[2] == pytest.approx(target, abs=0.05)


def test_pipeline_surge_with_all_thrusters_reaches_expected_speed():
    s = _closed_loop(lambda s: np.array([20.0, 0, 0, 0, 0, 0]), 30)
    assert 1.1 < s.nu[0] < 1.3                    # analytic ~1.21 m/s with the default drag


def test_pipeline_known_dead_thruster_reallocated_keeps_heading():
    s = _closed_loop(lambda s: np.array([15.0, 0, 0, 0, 0, 0]), 15, faults=[0], known_dead=[0])
    assert abs(s.eta[5]) < 0.05 and s.nu[0] > 0.8


def test_pipeline_unknown_dead_thruster_causes_heading_drift():
    s = _closed_loop(lambda s: np.array([15.0, 0, 0, 0, 0, 0]), 15, faults=[0])
    assert abs(s.eta[5]) > 0.3

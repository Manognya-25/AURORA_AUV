"""
Phase 2 tests for the PyBullet-coupled AUV (legacy mode regression + 6dof mode).

Skipped automatically when `pybullet` is not installed. NOTE: these were written
without being able to execute PyBullet in the authoring environment -- run
`python -m pytest tests/test_dynamics_pybullet.py -v` on a machine with PyBullet.
"""
import copy
from pathlib import Path

import numpy as np
import pytest
import yaml

pybullet = pytest.importorskip("pybullet", reason="pybullet is not installed")

from src.control.thruster_model import ThrusterArray, ThrusterFault, specs_from_config
from src.dynamics.auv_dynamics import AUVDynamics, AUVParams, RigidBodyState
from src.dynamics.ocean_current import UniformCurrent
from src.simulation.auv import AUV

ROOT = Path(__file__).resolve().parent.parent
DT = 0.01


def _sim_cfg(model="6dof"):
    with open(ROOT / "configs" / "config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)["simulation"]
    cfg = copy.deepcopy(cfg)
    cfg["dynamics"]["model"] = model
    return cfg


@pytest.fixture
def client():
    c = pybullet.connect(pybullet.DIRECT)
    pybullet.setGravity(0, 0, -9.8, physicsClientId=c)
    pybullet.setTimeStep(DT, physicsClientId=c)
    yield c
    pybullet.disconnect(physicsClientId=c)


def _auv(cfg, client, pos=(5.0, 5.0, -5.0)):
    return AUV(str(ROOT / cfg["urdf_path"]), pos, cfg, physics_client=client)


def _run(auv, client, seconds, **control):
    for _ in range(int(seconds / DT)):
        auv.apply_control(**control, dt=DT)
        auv.step_underwater_physics()
        pybullet.stepSimulation(physicsClientId=client)


def test_default_dynamics_model_is_legacy(client):
    cfg = _sim_cfg("legacy")
    assert _auv(cfg, client).dynamics_model == "legacy"


def test_invalid_model_name_rejected(client):
    cfg = _sim_cfg("warp_drive")
    with pytest.raises(ValueError):
        _auv(cfg, client)


def test_legacy_force_application_does_not_tumble_away_from_origin(client):
    """Regression: forces used to be applied at the world origin, spinning the hull."""
    auv = _auv(_sim_cfg("legacy"), client)            # spawned ~7 m from the origin
    _run(auv, client, 2.0, forward=1.0)
    roll, pitch, _ = auv.get_state_6dof()["eta"][3:]
    assert abs(roll) < 0.1 and abs(pitch) < 0.1


def test_6dof_forward_thrust_is_body_frame_and_level(client):
    auv = _auv(_sim_cfg(), client)
    _run(auv, client, 3.0, forward=1.0)
    s = auv.get_state_6dof()
    assert s["nu"][0] > 0.3                            # moving along body x
    assert abs(s["eta"][3]) < 0.1 and abs(s["eta"][4]) < 0.1


def test_6dof_surge_terminal_velocity_matches_analytic(client):
    cfg = _sim_cfg()
    auv = _auv(cfg, client)
    _run(auv, client, 25.0, forward=1.0)               # 10 N total surge wrench
    d = cfg["dynamics"]["drag"]
    lin, quad, T = d["linear"][0], d["quadratic"][0], cfg["controller"]["max_thrust"]
    expected = (-lin + np.sqrt(lin ** 2 + 4 * quad * T)) / (2 * quad)
    assert auv.get_state_6dof()["nu"][0] == pytest.approx(expected, rel=0.05)


def test_6dof_pybullet_matches_numpy_reference_model(client):
    cfg = _sim_cfg()
    auv = _auv(cfg, client)
    _run(auv, client, 10.0, forward=0.5)

    dyn = AUVDynamics(AUVParams.from_config(cfg))
    arr = ThrusterArray(specs_from_config(cfg), seed=0)
    from src.control.thruster_allocator import ThrusterAllocator
    alloc = ThrusterAllocator(specs_from_config(cfg))
    s = RigidBodyState.from_euler([5.0, 5.0, -5.0])
    for k in range(int(10.0 / DT)):
        arr.step(alloc.allocate([5.0, 0, 0, 0, 0, 0]).commands, DT)
        s = dyn.step(s, arr.wrench(), DT, k * DT)
    travelled_ref = s.position[0] - 5.0
    travelled_pb = auv.get_position()[0] - 5.0
    assert travelled_pb == pytest.approx(travelled_ref, rel=0.05)


def test_6dof_unpowered_vehicle_is_carried_by_current(client):
    auv = _auv(_sim_cfg(), client)
    auv.set_current(UniformCurrent([0.4, 0.0, 0.0]))
    _run(auv, client, 30.0)
    assert auv.get_state_6dof()["rotation"][:, 0] @ np.array(auv.get_linear_velocity()) == pytest.approx(0.4, rel=0.1)


def test_6dof_dead_thruster_yaws_the_vehicle(client):
    auv = _auv(_sim_cfg(), client)
    auv.thrusters.set_fault(0, ThrusterFault.DEAD)     # allocator is not told -> uncompensated
    _run(auv, client, 5.0, forward=1.0)
    assert abs(auv.get_state_6dof()["eta"][5]) > 0.1


def test_6dof_energy_increases_only_with_thrust(client):
    auv = _auv(_sim_cfg(), client)
    _run(auv, client, 1.0)
    idle = auv.get_energy_consumed()
    _run(auv, client, 1.0, forward=1.0)
    assert idle == pytest.approx(0.0, abs=1e-9)
    assert auv.get_energy_consumed() > 0.0

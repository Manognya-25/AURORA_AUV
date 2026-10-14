"""Integration: real PyBullet AUV -> sensors -> EKF, in both dynamics modes. Skipped without PyBullet."""
import copy

import numpy as np
import pytest
import yaml

pb = pytest.importorskip("pybullet")

from src.simulation.auv import AUV                                  # noqa: E402
from src.simulation.estimation import EstimationPipeline            # noqa: E402


def load_cfg(model="legacy", **overrides):
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    cfg["simulation"]["dynamics"]["model"] = model
    for path, val in overrides.items():                              # "sensors.dvl.dropout_probability": 1.0
        d = cfg
        *ks, last = path.split(".")
        for k in ks:
            d = d[k]
        d[last] = val
    return cfg


def drive(cfg, seconds=6.0, control=None, add_wall=False, start=(5.0, 5.0, -5.0)):
    """Step the real simulation under a scripted control; return (log dict, pipeline). No controller feedback."""
    sim = cfg["simulation"]
    client = pb.connect(pb.DIRECT)
    try:
        pb.setGravity(0, 0, sim["gravity"], physicsClientId=client)
        pb.setTimeStep(sim["time_step"], physicsClientId=client)
        if add_wall:
            col = pb.createCollisionShape(pb.GEOM_BOX, halfExtents=(0.5, 20, 20), physicsClientId=client)
            pb.createMultiBody(0, col, basePosition=(start[0] + 12.5, start[1], start[2]), physicsClientId=client)
        auv = AUV(sim["urdf_path"], start, sim, physics_client=client)
        pipe = EstimationPipeline(cfg, auv.body_id, start, physics_client=client)
        dt = sim["time_step"]
        log = {k: [] for k in ("truth_pos", "truth_vel", "est_pos", "est_vel", "imu", "t", "est_state")}
        for step in range(int(seconds / dt)):
            meas, est = pipe.update(step * dt)
            ctl = control(step * dt) if control else {}
            auv.apply_control(**ctl, dt=dt)
            auv.step_underwater_physics()
            pb.stepSimulation(physicsClientId=client)
            log["t"].append(step * dt)
            log["truth_pos"].append(pipe.truth.position.copy())
            log["truth_vel"].append(pipe.truth.velocity_world.copy())
            if est is not None:
                log["est_pos"].append(est.position.copy()); log["est_vel"].append(est.velocity.copy())
                log["est_state"].append(est)
            if "imu" in meas:
                log["imu"].append(meas["imu"].value.copy())
        return {k: np.array(v) if k != "est_state" else v for k, v in log.items()}, pipe
    finally:
        pb.disconnect(client)


FWD = lambda t: {"forward": 0.5}
MODELS = ["legacy", "6dof"]


@pytest.mark.parametrize("model", MODELS)
def test_imu_on_real_pybullet_body_reads_gravity_at_rest(model):
    cfg = load_cfg(model, **{"sensors.imu.accel_noise_std": 0.0, "sensors.imu.gyro_noise_std": 0.0,
                             "sensors.imu.accel_bias": [0, 0, 0], "sensors.imu.gyro_bias": [0, 0, 0]})
    log, _ = drive(cfg, seconds=1.0)
    imu = log["imu"][20:]                                            # skip the first transient samples
    assert np.allclose(imu[:, :3].mean(axis=0), [0, 0, 9.8], atol=0.15)    # neutrally buoyant: only specific force of gravity
    assert np.abs(imu[:, 3:]).max() < 0.05


@pytest.mark.parametrize("model", MODELS)
def test_imu_sees_real_acceleration_when_thrusting(model):
    cfg = load_cfg(model, **{"sensors.imu.accel_noise_std": 0.0, "sensors.imu.gyro_noise_std": 0.0,
                             "sensors.imu.accel_bias": [0, 0, 0], "sensors.imu.gyro_bias": [0, 0, 0]})
    log, _ = drive(cfg, seconds=1.0, control=lambda t: {"forward": 1.0})
    assert log["imu"][5:40, 0].mean() > 0.3                         # forward specific force from real PyBullet motion
    assert np.linalg.norm(log["truth_vel"][-1]) > 0.1


@pytest.mark.parametrize("model", MODELS)
def test_ekf_tracks_real_vehicle_in_both_dynamics_modes(model):
    log, pipe = drive(load_cfg(model), seconds=8.0, control=FWD)
    err = np.linalg.norm(log["est_pos"] - log["truth_pos"], axis=1)
    verr = np.linalg.norm(log["est_vel"] - log["truth_vel"], axis=1)
    assert np.all(np.isfinite(err)) and err.max() < 1.0
    assert verr[-200:].mean() < 0.15
    assert pipe.ekf.stats["dvl_updates"] > 50 and pipe.ekf.stats["depth_updates"] > 50
    assert np.linalg.norm(log["truth_pos"][-1] - log["truth_pos"][0]) > 0.5   # it really moved


def test_sensors_and_ekf_do_not_perturb_the_physics():
    """Truth must be identical with sensors on vs off: sensing is observation only."""
    on, _ = drive(load_cfg("6dof"), seconds=3.0, control=FWD)
    off_cfg = load_cfg("6dof", **{"sensors.enabled": False})
    off, pipe = drive(off_cfg, seconds=3.0, control=FWD)
    assert pipe.ekf is None and len(off["est_pos"]) == 0
    assert np.array_equal(on["truth_pos"], off["truth_pos"])


def test_estimation_is_deterministic_for_fixed_seeds():
    a, _ = drive(load_cfg("legacy"), seconds=3.0, control=FWD)
    b, _ = drive(load_cfg("legacy"), seconds=3.0, control=FWD)
    assert np.array_equal(a["est_pos"], b["est_pos"]) and np.array_equal(a["imu"], b["imu"])
    c, _ = drive(load_cfg("legacy", **{"sensors.imu.seed": 999}), seconds=3.0, control=FWD)
    assert not np.array_equal(a["imu"], c["imu"])


def test_total_dvl_dropout_in_live_loop_keeps_estimating_without_truth_substitution():
    cfg = load_cfg("legacy", **{"sensors.dvl.dropout_probability": 1.0})
    log, pipe = drive(cfg, seconds=6.0, control=FWD)
    assert pipe.ekf.stats["dvl_updates"] == 0 and pipe.ekf.stats["rejected_invalid"] > 20
    assert np.all(np.isfinite(log["est_pos"]))
    # depth aiding still holds z; x/y are pure dead reckoning and drift from truth more than the aided run
    aided, _ = drive(load_cfg("legacy"), seconds=6.0, control=FWD)
    err_dr = np.linalg.norm(log["est_vel"][-1] - log["truth_vel"][-1])
    err_ad = np.linalg.norm(aided["est_vel"][-1] - aided["truth_vel"][-1])
    assert err_dr > err_ad
    assert abs(log["est_pos"][-1, 2] - log["truth_pos"][-1, 2]) < 0.2
    assert pipe.ekf.state.velocity_std.max() > aided["est_state"][-1].velocity_std.max()   # and it knows it is less sure


def test_depth_failure_in_live_loop_degrades_z_gracefully():
    cfg = load_cfg("legacy", **{"sensors.depth.dropout_probability": 1.0})
    log, pipe = drive(cfg, seconds=4.0, control=FWD)
    assert pipe.ekf.stats["depth_updates"] == 0 and np.all(np.isfinite(log["est_pos"]))


def test_sonar_in_live_loop_sees_real_wall():
    cfg = load_cfg("legacy", **{"sensors.sonar.dropout_probability": 0.0, "sensors.sonar.range_noise_std_m": 0.0})
    _, pipe = drive(cfg, seconds=0.5, add_wall=True)
    s = pipe.latest.get("sonar")                                    # sonar is 5 Hz: not present on every step
    assert s is not None and s.detected.any()
    center = 2 * 21 + 10
    assert np.isclose(s.value[center], 12.0 - 0.8 - 0.0, atol=0.15)        # wall face 12 m ahead of CoG


def test_sonar_disabled_and_individual_sensors_can_be_switched_off():
    cfg = load_cfg("legacy", **{"sensors.sonar.enabled": False, "sensors.dvl.enabled": False})
    log, pipe = drive(cfg, seconds=2.0, control=FWD)
    assert "sonar" not in pipe.suite.sensors and "dvl" not in pipe.suite.sensors
    assert pipe.ekf.stats["dvl_updates"] == 0 and np.all(np.isfinite(log["est_pos"]))


def test_ekf_disabled_but_sensors_enabled():
    cfg = load_cfg("legacy", **{"ekf.enabled": False})
    log, pipe = drive(cfg, seconds=1.0)
    assert pipe.ekf is None and len(log["est_pos"]) == 0     # no estimator...
    assert len(log["imu"]) > 0 and "imu" in pipe.latest      # ...but sensors still produce measurements

"""Sensor tests that need no PyBullet (IMU, DVL, depth, health). Sonar tests: test_sonar_pybullet.py."""
import numpy as np
import pytest

from src.sensors import (DVL, IMU, DVLConfig, DepthConfig, DepthSensor, IMUConfig, SensorHealth, TruthSnapshot)
from src.utils.transforms import euler_to_quat


def truth(t=0.0, pos=(0, 0, -5), rpy=(0, 0, 0), v=(0, 0, 0), w=(0, 0, 0), a=None):
    return TruthSnapshot(t, pos, euler_to_quat(*rpy), v, w, a)


def quiet_imu(**kw):
    base = dict(accel_noise_std=0, gyro_noise_std=0, accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0],
                accel_bias_walk_std=0, gyro_bias_walk_std=0)
    base.update(kw)
    return IMU(IMUConfig(**base))


# ------------------------------------------------------------------ IMU
def test_imu_shape_and_metadata():
    m = IMU().measure(truth())
    assert m.value.shape == (6,) and m.valid and m.health == SensorHealth.HEALTHY
    assert m.covariance.shape == (6, 6) and np.all(np.diag(m.covariance) > 0)


def test_imu_at_rest_measures_specific_force_up_not_zero():
    f = quiet_imu().measure(truth(a=[0, 0, 0])).value
    assert np.allclose(f[:3], [0, 0, 9.8]) and np.allclose(f[3:], 0)


def test_imu_gravity_rotates_into_body_frame():
    f = quiet_imu().measure(truth(rpy=(0, np.deg2rad(30), 0), a=[0, 0, 0])).value[:3]
    assert np.allclose(f, [-9.8 * np.sin(np.deg2rad(30)), 0, 9.8 * np.cos(np.deg2rad(30))])   # nose-up pitch
    f = quiet_imu().measure(truth(rpy=(np.deg2rad(30), 0, 0), a=[0, 0, 0])).value[:3]
    assert np.allclose(f, [0, 9.8 * np.sin(np.deg2rad(30)), 9.8 * np.cos(np.deg2rad(30))])


def test_imu_accel_derived_from_motion_by_finite_difference():
    imu = quiet_imu()
    imu.measure(truth(0.0, v=(0, 0, 0)))
    f = imu.measure(truth(0.1, v=(0.2, 0, 0))).value[:3]       # a_w = 2 m/s^2 along +x, level
    assert np.allclose(f, [2.0, 0, 9.8])


def test_imu_yawed_body_sees_world_acceleration_in_body_axes():
    imu = quiet_imu()
    yaw = np.pi / 2                                             # body x points to world +y
    imu.measure(truth(0.0, rpy=(0, 0, yaw)))
    f = imu.measure(truth(0.1, rpy=(0, 0, yaw), v=(0, 0.1, 0))).value[:3]   # world accel +y = body +x
    assert np.allclose(f, [1.0, 0, 9.8], atol=1e-9)


def test_imu_gyro_is_body_frame_rate():
    w = quiet_imu().measure(truth(rpy=(0, 0, np.pi / 2), w=(0, 0, 0.3))).value[3:]
    assert np.allclose(w, [0, 0, 0.3])
    w = quiet_imu().measure(truth(rpy=(0, 0, np.pi / 2), w=(1.0, 0, 0))).value[3:]   # world x = body -y
    assert np.allclose(w, [0, -1.0, 0], atol=1e-9)


def test_imu_noise_statistics():
    imu = IMU(IMUConfig(accel_noise_std=0.1, gyro_noise_std=0.01, accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0],
                        accel_bias_walk_std=0, gyro_bias_walk_std=0, seed=1))
    vals = np.array([imu.measure(truth(0.01 * i, a=[0, 0, 0])).value for i in range(4000)])
    assert np.isclose(vals[:, 0].std(), 0.1, rtol=0.08) and np.isclose(vals[:, 3].std(), 0.01, rtol=0.08)
    assert abs(vals[:, 0].mean()) < 0.01 and np.isclose(vals[:, 2].mean(), 9.8, atol=0.01)


def test_imu_constant_bias_appears_in_mean():
    imu = quiet_imu(accel_bias=[0.3, -0.2, 0.1], gyro_bias=[0.01, 0.02, -0.03])
    m = imu.measure(truth(a=[0, 0, 0])).value
    assert np.allclose(m, [0.3, -0.2, 9.9, 0.01, 0.02, -0.03])


def test_imu_deterministic_seed_and_different_seeds_differ():
    def run(seed):
        imu = IMU(IMUConfig(seed=seed))
        return np.array([imu.measure(truth(0.01 * i)).value for i in range(50)])
    assert np.array_equal(run(7), run(7))
    assert not np.allclose(run(7), run(8))


def test_imu_bias_drift_random_walk():
    imu = IMU(IMUConfig(accel_noise_std=0, gyro_noise_std=0, accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0],
                        accel_bias_walk_std=0.01, gyro_bias_walk_std=0.001, seed=3))
    b0 = imu.accel_bias.value.copy()
    for i in range(2000):
        imu.measure(truth(0.01 * i, a=[0, 0, 0]))
    assert np.linalg.norm(imu.accel_bias.value - b0) > 1e-3         # it moved
    # walk magnitude ~ std*sqrt(T): over 20 s expect ~0.045 per axis (loose bound over 3 axes)
    assert np.linalg.norm(imu.accel_bias.value) < 0.01 * np.sqrt(20) * 6
    still = IMU(IMUConfig(accel_bias_walk_std=0, gyro_bias_walk_std=0, accel_bias=[0.1, 0, 0]))
    for i in range(100):
        still.measure(truth(0.01 * i))
    assert np.allclose(still.accel_bias.value, [0.1, 0, 0])


def test_imu_degraded_scales_noise_and_failed_is_invalid_zeros():
    imu = IMU(IMUConfig(accel_noise_std=0.1, accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0],
                        accel_bias_walk_std=0, gyro_bias_walk_std=0, degraded_noise_scale=5.0, seed=2))
    imu.inject_fault("degraded")
    vals = np.array([imu.measure(truth(0.01 * i, a=[0, 0, 0])).value[0] for i in range(3000)])
    assert np.isclose(vals.std(), 0.5, rtol=0.1)
    m = imu.measure(truth(40.0))
    assert m.health == SensorHealth.DEGRADED and np.isclose(m.covariance[0, 0], 0.25)
    imu.inject_fault("failed")
    m = imu.measure(truth(41.0))
    # FAILED = a real sample every cycle, but stuck at 0.0 on every channel (invalid, so the EKF rejects it)
    assert not m.valid and m.health == SensorHealth.FAILED and np.all(m.value == 0.0)
    assert imu.health_monitor.zeroed is True and imu.health_monitor.stopped is False


# ------------------------------------------------------------------ DVL
def test_dvl_measures_body_frame_velocity_relative_to_seabed():
    dvl = DVL(DVLConfig(noise_std=0, dropout_probability=0))
    m = dvl.measure(truth(rpy=(0, 0, np.pi / 2), v=(0, 1.0, 0)))   # moving world +y, heading +y -> body +x
    assert m.valid and m.value.shape == (3,) and np.allclose(m.value, [1.0, 0, 0], atol=1e-9)


def test_dvl_noise_deterministic_and_statistics():
    cfg = DVLConfig(noise_std=0.05, dropout_probability=0, seed=5)
    a, b = DVL(cfg), DVL(cfg)
    va = np.array([a.measure(truth(0.1 * i, v=(1, 0, 0))).value for i in range(3000)])
    vb = np.array([b.measure(truth(0.1 * i, v=(1, 0, 0))).value for i in range(3000)])
    assert np.array_equal(va, vb)
    assert np.isclose(va[:, 1].std(), 0.05, rtol=0.08) and np.isclose(va[:, 0].mean(), 1.0, atol=0.005)


def test_dvl_dropout_rate_and_nan_not_truth():
    dvl = DVL(DVLConfig(dropout_probability=0.3, seed=9))
    ms = [dvl.measure(truth(0.1 * i, v=(1, 0, 0))) for i in range(3000)]
    frac = np.mean([not m.valid for m in ms])
    assert 0.25 < frac < 0.35
    for m in ms:
        if not m.valid:
            assert np.all(np.isnan(m.value))               # never substituted with ground truth


def test_dvl_no_bottom_lock_outside_altitude_window():
    dvl = DVL(DVLConfig(dropout_probability=0, seabed_z=-15, max_altitude_m=30, min_altitude_m=0.3))
    assert dvl.measure(truth(pos=(0, 0, -5))).valid
    assert not dvl.measure(truth(pos=(0, 0, -14.9))).valid       # 0.1 m above seabed < min
    d2 = DVL(DVLConfig(dropout_probability=0, seabed_z=-100, max_altitude_m=30))
    assert not d2.measure(truth(pos=(0, 0, -5))).valid           # 95 m above seabed > max


def test_dvl_health_transitions_with_consecutive_dropouts_and_recovers():
    dvl = DVL(DVLConfig(dropout_probability=1.0))
    healths = [dvl.measure(truth(0.1 * i)).health for i in range(60)]
    assert healths[0] == SensorHealth.HEALTHY and SensorHealth.DEGRADED in healths and healths[-1] == SensorHealth.FAILED
    dvl.cfg.dropout_probability = 0.0
    assert dvl.measure(truth(10.0)).health == SensorHealth.HEALTHY


def test_dvl_not_affected_by_ocean_current_concept():
    # Seabed-relative: depends only on vehicle ground velocity, not on any water velocity input.
    dvl = DVL(DVLConfig(noise_std=0, dropout_probability=0))
    assert np.allclose(dvl.measure(truth(v=(0.4, 0, 0))).value, [0.4, 0, 0])


# ------------------------------------------------------------------ Depth
def test_depth_positive_down_and_shape():
    s = DepthSensor(DepthConfig(noise_std=0, bias=0, dropout_probability=0))
    m = s.measure(truth(pos=(3, 4, -7.5)))
    assert m.valid and m.value.shape == (1,) and np.isclose(m.value[0], 7.5)


def test_depth_bias_and_noise_statistics_and_determinism():
    cfg = DepthConfig(noise_std=0.05, bias=0.3, dropout_probability=0, seed=11)
    a, b = DepthSensor(cfg), DepthSensor(cfg)
    va = np.array([a.measure(truth(0.1 * i, pos=(0, 0, -5))).value[0] for i in range(3000)])
    vb = np.array([b.measure(truth(0.1 * i, pos=(0, 0, -5))).value[0] for i in range(3000)])
    assert np.array_equal(va, vb)
    assert np.isclose(va.mean(), 5.3, atol=0.005) and np.isclose(va.std(), 0.05, rtol=0.08)


def test_depth_dropout_and_failure():
    s = DepthSensor(DepthConfig(dropout_probability=0.4, seed=1))
    ms = [s.measure(truth(0.1 * i)) for i in range(2000)]
    assert 0.35 < np.mean([not m.valid for m in ms]) < 0.45
    assert all(np.isnan(m.value[0]) for m in ms if not m.valid)
    s.inject_fault("failed")
    m = s.measure(truth(300.0))
    assert not m.valid and m.health == SensorHealth.FAILED


def test_sensor_streams_are_independent_of_dropout_pattern():
    # Noise realisation of valid samples must not depend on the dropout probability (fixed draw count).
    from src.sensors.health import HealthMonitor
    a = DepthSensor(DepthConfig(noise_std=0.1, dropout_probability=0.0, seed=4))
    b = DepthSensor(DepthConfig(noise_std=0.1, dropout_probability=0.5, seed=4))
    b.health_monitor = HealthMonitor(10_000, 10_000)   # isolate the RNG stream from auto-escalation
    for i in range(200):
        ma, mb = a.measure(truth(i * 0.1)), b.measure(truth(i * 0.1))
        if mb.valid:
            assert ma.value[0] == mb.value[0]


def test_auto_failed_sensor_recovers_but_injected_failure_does_not():
    dvl = DVL(DVLConfig(dropout_probability=1.0))
    for i in range(60):
        dvl.measure(truth(0.1 * i))
    assert dvl.health_monitor.health == SensorHealth.FAILED          # auto status only
    dvl.cfg.dropout_probability = 0.0
    assert dvl.measure(truth(10.0)).valid                             # recovered
    dvl.inject_fault("failed")
    assert not dvl.measure(truth(11.0)).valid
    dvl.inject_fault(None)
    assert dvl.measure(truth(12.0)).valid

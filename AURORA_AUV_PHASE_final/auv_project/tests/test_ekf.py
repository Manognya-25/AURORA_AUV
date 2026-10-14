"""EKF tests (pure NumPy, no PyBullet)."""
import re
from pathlib import Path

import numpy as np
import pytest

from src.localization import EKF, EKFConfig
from src.sensors import (DVL, IMU, DVLConfig, DepthConfig, DepthSensor, IMUConfig, Measurement, SensorHealth,
                         TruthSnapshot)
from src.utils.transforms import euler_to_quat, quat_to_euler, quat_to_rot

G = 9.8


def perfect_imu():
    return IMU(IMUConfig(accel_noise_std=0, gyro_noise_std=0, accel_bias=[0, 0, 0], gyro_bias=[0, 0, 0],
                         accel_bias_walk_std=0, gyro_bias_walk_std=0))


def snap(t, pos, rpy=(0, 0, 0), v=(0, 0, 0), w_world=(0, 0, 0), a=(0, 0, 0)):
    return TruthSnapshot(t, pos, euler_to_quat(*rpy), v, w_world, a)


def meas(sensor, value, t, valid=True, health=SensorHealth.HEALTHY, cov=None):
    return Measurement(sensor, t, np.asarray(value, float), valid, health, cov)


def ekf0(**kw):
    cfg = EKFConfig(**kw.pop("cfg", {}))
    return EKF(cfg, **kw)


# ------------------------------------------------------------------ prediction
def test_stationary_perfect_imu_stays_put():
    e, imu = ekf0(position=[1, 2, -5]), perfect_imu()
    for i in range(1, 501):
        e.predict(imu.measure(snap(0.01 * i, [1, 2, -5])))
    assert np.allclose(e.p, [1, 2, -5], atol=1e-9) and np.allclose(e.v, 0, atol=1e-9)


def test_constant_acceleration_integrates_to_kinematics():
    e, imu = ekf0(position=[0, 0, -5]), perfect_imu()
    for i in range(1, 201):                                        # a = 0.5 m/s^2 along +x for 2 s
        t = 0.01 * i
        e.predict(imu.measure(snap(t, [0.25 * t * t, 0, -5], v=[0.5 * t, 0, 0], a=[0.5, 0, 0])))
    assert np.isclose(e.v[0], 1.0, atol=1e-6) and np.isclose(e.p[0], 1.0, atol=1e-3)


def test_prediction_respects_orientation_and_gravity_when_pitched_and_yawed():
    rpy = (0.1, -0.2, 0.7)
    e = ekf0(position=[0, 0, -5], quat=euler_to_quat(*rpy))
    imu = perfect_imu()
    for i in range(1, 301):
        e.predict(imu.measure(snap(0.01 * i, [0, 0, -5], rpy=rpy)))   # at rest in a tilted attitude
    assert np.allclose(e.v, 0, atol=1e-8) and np.allclose(e.p, [0, 0, -5], atol=1e-8)


def test_world_acceleration_in_yawed_body_goes_the_right_way():
    yaw = np.pi / 2                                                # body x = world +y
    e = ekf0(position=[0, 0, -5], quat=euler_to_quat(0, 0, yaw))
    imu = perfect_imu()
    for i in range(1, 101):
        t = 0.01 * i
        e.predict(imu.measure(snap(t, [0, 0.5 * t * t, -5], rpy=(0, 0, yaw), v=[0, t, 0], a=[0, 1.0, 0])))
    assert np.isclose(e.v[1], 1.0, atol=1e-6) and abs(e.v[0]) < 1e-8


def test_angular_velocity_integrates_attitude():
    e, imu = ekf0(), perfect_imu()
    for i in range(1, 201):
        t = 0.01 * i
        e.predict(imu.measure(snap(t, [0, 0, 0], rpy=(0, 0, 0.5 * t), w_world=[0, 0, 0.5])))
    # attitude passed to the sensor changes with t; check the EKF's own integrated yaw: 0.5 rad/s * 2 s
    assert np.isclose(quat_to_euler(e.q)[2], 1.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(e.q), 1.0)


def test_predict_ignores_non_advancing_time_and_handles_big_gap_by_coasting():
    e, imu = ekf0(position=[0, 0, -5], velocity=[1, 0, 0]), perfect_imu()
    assert not e.predict(imu.measure(snap(0.0, [0, 0, -5])))        # dt = 0
    assert not e.predict(imu.measure(snap(2.0, [0, 0, -5])))        # gap > max_dt -> coast
    assert np.isclose(e.p[0], 2.0) and e.stats["imu_missing"] == 1 and e.time == 2.0


# ------------------------------------------------------------------ covariance
def check_cov(P):
    assert P.shape == (15, 15) and np.all(np.isfinite(P))
    assert np.allclose(P, P.T, atol=1e-12)
    assert np.linalg.eigvalsh(P).min() > -1e-10


def test_covariance_valid_and_grows_without_measurements():
    e, imu = ekf0(), perfect_imu()
    p0 = np.trace(e.P[:3, :3])
    for i in range(1, 1001):
        e.predict(imu.measure(snap(0.01 * i, [0, 0, 0])))
        if i % 100 == 0:
            check_cov(e.P)
    assert np.trace(e.P[:3, :3]) > p0 and np.trace(e.P[3:6, 3:6]) > 3 * 0.1 ** 2


def test_covariance_shrinks_after_updates_and_stays_valid():
    e, imu = ekf0(position=[0, 0, -5]), perfect_imu()
    for i in range(1, 51):
        e.predict(imu.measure(snap(0.01 * i, [0, 0, -5])))
    before = e.P.copy()
    t = e.time
    assert e.update_dvl(meas("dvl", [0, 0, 0], t))
    assert e.update_depth(meas("depth", [5.0], t))
    check_cov(e.P)
    assert np.trace(e.P[3:6, 3:6]) < np.trace(before[3:6, 3:6])
    assert e.P[2, 2] < before[2, 2]


# ------------------------------------------------------------------ updates
def test_dvl_update_pulls_velocity_toward_measurement_body_frame():
    yaw = np.pi / 2
    e = ekf0(velocity=[0, 0, 0], quat=euler_to_quat(0, 0, yaw), cfg=dict(init_velocity_std=1.0))
    e.update_dvl(meas("dvl", [1.0, 0, 0], 0.0))                    # body x = world +y -> world v_y ~ +1
    assert e.v[1] > 0.9 and abs(e.v[0]) < 0.05
    assert e.stats["dvl_updates"] == 1


def test_depth_update_pulls_z_and_leaves_xy_alone():
    e = ekf0(position=[3, 4, -5], cfg=dict(init_position_std=1.0))
    e.update_depth(meas("depth", [6.0], 0.0))                      # depth 6 -> z = -6
    assert -6.0 <= e.p[2] < -5.7 and np.isclose(e.p[0], 3, atol=1e-9) and np.isclose(e.p[1], 4, atol=1e-9)


def test_dvl_jacobian_matches_numerical_derivative():
    e = ekf0(velocity=[0.8, -0.3, 0.1], quat=euler_to_quat(0.2, -0.3, 1.1))
    from src.utils.transforms import quat_to_rot, skew, quat_multiply, quat_normalize
    R, v = quat_to_rot(e.q), e.v
    h = lambda dth, dv: quat_to_rot(quat_multiply(e.q, quat_normalize(np.r_[0.5 * dth, 1.0]))).T @ (v + dv)
    eps = 1e-6
    H = np.zeros((3, 6))
    for k in range(3):
        d = np.zeros(3); d[k] = eps
        H[:, k] = (h(np.zeros(3), d) - h(np.zeros(3), -d)) / (2 * eps)
        H[:, 3 + k] = (h(d, np.zeros(3)) - h(-d, np.zeros(3))) / (2 * eps)
    assert np.allclose(H[:, :3], R.T, atol=1e-6)
    assert np.allclose(H[:, 3:], skew(R.T @ v), atol=1e-5)


def test_outlier_dvl_is_gated_out():
    e = ekf0(velocity=[0, 0, 0])
    assert not e.update_dvl(meas("dvl", [5.0, 5.0, 5.0], 0.0))
    assert e.stats["rejected_gate"] == 1 and np.allclose(e.v, 0)
    e2 = ekf0(velocity=[0, 0, 0], cfg=dict(gate_enabled=False))
    assert e2.update_dvl(meas("dvl", [5.0, 5.0, 5.0], 0.0))


# ------------------------------------------------------------------ missing / dropped measurements
def test_invalid_and_missing_measurements_do_not_change_state():
    e = ekf0(position=[0, 0, -5], velocity=[0.3, 0, 0])
    p, v, P = e.p.copy(), e.v.copy(), e.P.copy()
    assert not e.update_dvl(None) and not e.update_depth(None)
    assert not e.update_dvl(meas("dvl", [np.nan] * 3, 0.0, valid=False))
    assert not e.update_depth(meas("depth", [np.nan], 0.0, valid=False))
    assert not e.update_dvl(meas("dvl", [1, 0, 0], 0.0, health=SensorHealth.FAILED))
    assert np.array_equal(p, e.p) and np.array_equal(v, e.v) and np.array_equal(P, e.P)
    assert e.stats["rejected_invalid"] == 3


def test_stale_measurement_rejected():
    e = ekf0()
    e.coast(1.0)
    assert not e.update_depth(meas("depth", [0.0], 0.5))
    assert e.stats["rejected_stale"] == 1


def test_degraded_sensor_is_trusted_less():
    a, b = ekf0(cfg=dict(init_velocity_std=1.0)), ekf0(cfg=dict(init_velocity_std=1.0))
    a.update_dvl(meas("dvl", [0.5, 0, 0], 0.0))
    b.update_dvl(meas("dvl", [0.5, 0, 0], 0.0, health=SensorHealth.DEGRADED))
    assert a.v[0] > b.v[0] > 0


def test_imu_failure_coasts_with_growing_uncertainty_and_no_nans():
    e = ekf0(position=[0, 0, -5], velocity=[1, 0, 0])
    imu = perfect_imu()
    imu.inject_fault("failed")
    p0 = np.trace(e.P[:3, :3])
    for i in range(1, 201):
        e.predict(imu.measure(snap(0.01 * i, [0, 0, -5])))
    assert np.isclose(e.p[0], 2.0, atol=1e-9) and np.all(np.isfinite(e.P)) and np.trace(e.P[:3, :3]) > p0
    assert e.stats["imu_missing"] == 200 and e.stats["predictions"] == 0
    check_cov(e.P)


# ------------------------------------------------------------------ closed-loop scenario
def circle_scenario(dvl_dropout=0.0, dvl_blackout=None, seed=0, steps=3000, ekf_cfg=None, init_err=(0.4, -0.3, 0.8)):
    """Level circular motion, r=6 m, 1 m/s, depth 5 m. Returns (ekf, truth_final, errors over time)."""
    r, speed, depth, dt = 6.0, 1.0, 5.0, 0.01
    om = speed / r
    imu = IMU(IMUConfig(seed=seed + 1))
    dvl = DVL(DVLConfig(dropout_probability=dvl_dropout, seed=seed + 2, seabed_z=-30, max_altitude_m=60))
    dep = DepthSensor(DepthConfig(dropout_probability=0.0, seed=seed + 3))
    pos0 = np.array([r, 0.0, -depth])
    # Honest initial uncertainty: the scenario starts ~0.5 m/s and ~0.5 m off, so declare sigma accordingly.
    # (The deliberately overconfident case is covered by test_gate_cannot_deadlock_an_overconfident_filter.)
    base = dict(init_velocity_std=0.6, init_position_std=1.0)
    base.update(ekf_cfg or {})
    cfg = EKFConfig(**base)
    e = EKF(cfg, position=pos0 + np.array(init_err), quat=euler_to_quat(0, 0, np.pi / 2), velocity=[0.5, 0.2, 0.0])
    errs = []
    for i in range(1, steps + 1):
        t = i * dt
        th = om * t
        pos = np.array([r * np.cos(th), r * np.sin(th), -depth])
        vel = np.array([-speed * np.sin(th), speed * np.cos(th), 0.0])
        acc = np.array([-speed * om * np.cos(th), -speed * om * np.sin(th), 0.0])
        truth = snap(t, pos, rpy=(0, 0, th + np.pi / 2), v=vel, w_world=[0, 0, om], a=acc)
        e.predict(imu.measure(truth))
        if i % 10 == 0:
            m = dvl.measure(truth)
            if dvl_blackout and dvl_blackout[0] <= t < dvl_blackout[1]:
                m = meas("dvl", [np.nan] * 3, t, valid=False)
            e.update_dvl(m)
            e.update_depth(dep.measure(truth))
        Rt, Re = truth.rotation, quat_to_rot(e.q)
        yaw_err = abs(np.arctan2(np.sin(quat_to_euler(e.q)[2] - (th + np.pi / 2)), np.cos(quat_to_euler(e.q)[2] - (th + np.pi / 2))))
        errs.append((np.linalg.norm(e.p - pos), np.linalg.norm(e.v - vel), abs(e.p[2] - pos[2]),
                     np.linalg.norm(Re.T @ e.v - Rt.T @ vel), yaw_err))
    return e, truth, np.array(errs)


def test_convergence_velocity_and_depth_with_dvl_and_depth_aiding():
    e, truth, errs = circle_scenario()
    assert errs[0, 1] > 0.3 and errs[0, 2] > 0.5                    # started wrong
    assert errs[-300:, 1].mean() < 0.06                             # velocity converged
    assert errs[-300:, 2].mean() < 0.05                             # depth converged
    assert np.all(np.isfinite(e.P)) and abs(np.linalg.norm(e.q) - 1) < 1e-9
    check_cov(e.P)


def test_horizontal_position_is_dead_reckoned_and_bounded_but_not_corrected():
    # No absolute xy reference exists in this sensor set -> initial xy offset persists (honest limitation).
    e, truth, errs = circle_scenario(init_err=(0.4, -0.3, 0.0))
    assert errs[-1, 0] < 1.5                                        # no catastrophic drift over 30 s
    assert errs[-1, 0] > 0.2                                        # ...but the 0.5 m initial error is NOT removed
    assert e.state.position_std[:2].min() > 0.1                     # and the filter knows it (covariance stays open)


def test_dvl_aiding_reduces_error_versus_imu_only():
    aided, _, ea = circle_scenario(seed=5, steps=2000)
    # IMU-only: same scenario but DVL updates always invalid
    imu_only, _, eb = circle_scenario(seed=5, steps=2000, dvl_blackout=(0, 100))
    assert ea[-1, 1] < eb[-1, 1] and ea[-300:, 1].mean() < 0.5 * eb[-300:, 1].mean()


def test_dvl_blackout_keeps_filter_running_and_body_velocity_recovers():
    """During a 10 s DVL blackout the filter runs prediction-only and its error grows (boundedly). When the
    DVL returns, the quantity the DVL actually measures -- body-frame velocity -- recovers immediately.
    World-frame velocity does NOT fully recover: with no compass, yaw is unobservable and world velocity
    error ~ speed * yaw error. The filter must stay CONSISTENT about that (error within its own 3 sigma)."""
    e, _, errs = circle_scenario(dvl_blackout=(10.0, 20.0))
    assert e.stats["rejected_invalid"] >= 90 and np.all(np.isfinite(errs)) and np.all(np.isfinite(e.P))
    pre, end_blackout, post = errs[900:990, 3].mean(), errs[1900:1990, 3].mean(), errs[-300:, 3].mean()
    assert end_blackout > 3 * pre                         # prediction-only error really did grow...
    assert end_blackout < 0.5                             # ...but stayed bounded
    assert post < 0.03 and post < 0.25 * end_blackout     # body-frame velocity recovered once DVL returned
    # honest uncertainty: world-velocity error after recovery is explained by yaw error (+ stays within 3 sigma)
    assert errs[-300:, 1].max() < 3 * np.linalg.norm(e.state.velocity_std) + 0.05
    assert np.allclose(errs[-300:, 1].mean(), 1.0 * errs[-300:, 4].mean(), atol=0.08)   # v_err ~ speed * yaw_err
    check_cov(e.P)


def test_random_dvl_dropouts_do_not_break_filter():
    e, _, errs = circle_scenario(dvl_dropout=0.5, seed=8)
    assert e.stats["dvl_updates"] > 100 and e.stats["rejected_invalid"] > 100
    assert errs[-300:, 1].mean() < 0.1
    check_cov(e.P)


def test_covariance_is_consistent_with_actual_velocity_error():
    # NEES-style sanity: over a run, true velocity error should mostly sit inside ~3 sigma.
    e, _, errs = circle_scenario(seed=11)
    sigma = np.linalg.norm(e.state.velocity_std)
    assert errs[-1, 1] < 3 * sigma + 0.02


def test_ekf_is_deterministic():
    a, _, ea = circle_scenario(seed=21, steps=800)
    b, _, eb = circle_scenario(seed=21, steps=800)
    assert np.array_equal(a.p, b.p) and np.array_equal(a.P, b.P) and np.array_equal(ea, eb)
    c, _, _ = circle_scenario(seed=22, steps=800)
    assert not np.array_equal(a.p, c.p)


def test_estimated_state_object_and_config_loader():
    import yaml
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    ec = EKFConfig.from_config(cfg)
    assert ec.enabled and ec.gravity == 9.8 and ec.dvl_noise_std == cfg["ekf"]["dvl_noise_std"]
    s = EKF(ec, position=[1, 2, -3]).state
    assert s.depth == 3.0 and s.euler.shape == (3,) and s.covariance.shape == (15, 15) and s.velocity_body.shape == (3,)


# ------------------------------------------------------------------ no ground truth in the estimator
def test_localization_package_never_touches_pybullet_or_ground_truth():
    banned = re.compile(r"pybullet|get_state|getBasePosition|getBaseVelocity|TruthSnapshot|snapshot_from|src\.simulation")
    for f in Path("src/localization").glob("*.py"):
        code = "\n".join(l.split("#")[0] for l in f.read_text().splitlines())
        # strip docstrings so prose such as "no ground truth" does not trip the scan
        code = re.sub(r'""".*?"""', "", code, flags=re.S)
        assert not banned.search(code), f"{f} references simulator ground truth"


def test_gate_cannot_deadlock_an_overconfident_filter():
    """Filter told sigma_v=0.05 but actually 0.8 m/s off (16 sigma). A naive chi-square gate rejects
    every DVL sample forever; the filter must recognise persistent rejection and recover."""
    e, _, errs = circle_scenario(init_err=(0, 0, 0), steps=2000,
                                 ekf_cfg=dict(init_velocity_std=0.05))
    assert errs[0, 1] > 0.5                                    # genuinely started far off
    assert errs[-300:, 1].mean() < 0.1, f"stuck at {errs[-300:, 1].mean():.2f} m/s"
    assert e.stats["dvl_updates"] > 50
    assert np.abs(quat_to_euler(e.q)[:2]).max() < np.deg2rad(3)     # and did not wreck roll/pitch doing so


def test_phase3_controller_does_not_consume_the_estimate():
    """Phase 3 keeps the controller on TRUE state. Fails if someone wires the estimate in early."""
    ctl = Path("src/simulation/controller.py").read_text()
    assert not re.search(r"localization|EKF|estimat", ctl)
    run_src = Path("src/simulation/run_simulation.py").read_text()
    call = re.search(r"controller\.compute_control\(([^)]*)\)", run_src).group(1)
    assert "estimate" not in call and "est" not in call.replace("position", "").replace("target", "")
    assert "state" not in call or "position" in call            # args are position / yaw / target, from auv.get_state()

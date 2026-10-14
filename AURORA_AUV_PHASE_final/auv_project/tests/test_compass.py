"""Compass (absolute heading reference) + EKF heading update.

Root cause of the ~1.4 m physical goal error of the Phase 5 mission (EKF believed 0.3 m): yaw was pure gyro dead
reckoning, so every DVL velocity was rotated by a drifting heading. These tests use the REAL sensor models and the REAL
EKF on a scripted weaving trajectory (no PyBullet); the truth only enters through the TruthSnapshot like everywhere else."""
import numpy as np
import yaml

from src.localization.ekf import EKF, EKFConfig
from src.sensors.compass import Compass, CompassConfig
from src.sensors.models import TruthSnapshot
from src.simulation.estimation import EstimationPipeline


def _cfg(compass: bool, gyro_bias_z: float):
    import pathlib
    cfg = yaml.safe_load((pathlib.Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())
    cfg["sensors"]["imu"]["gyro_bias"] = [0.001, -0.001, gyro_bias_z]
    cfg["sensors"]["compass"]["enabled"] = compass
    cfg["sensors"]["sonar"]["enabled"] = False                      # no PyBullet in this test
    return cfg


def _weave(compass: bool, gyro_bias_z: float, T=60.0):
    est = EstimationPipeline(_cfg(compass, gyro_bias_z), 0, [5, 5, -5], start_quat=(0, 0, np.sin(0.3), np.cos(0.3)))
    dt, p, yaw, errs = 0.01, np.array([5.0, 5.0, -5.0]), 0.6, []
    for i in range(int(T / dt)):
        t = i * dt
        wz = 0.25 * np.sin(2 * np.pi * t / 12.0)
        yaw += wz * dt
        v = 0.8 * np.array([np.cos(yaw), np.sin(yaw), 0.0])
        p = p + v * dt
        truth = TruthSnapshot(time=t, position=p.copy(), quat=np.array([0, 0, np.sin(yaw / 2), np.cos(yaw / 2)]),
                              velocity_world=v, angular_velocity_world=np.array([0, 0, wz]),
                              acceleration_world=np.array([-0.8 * np.sin(yaw) * wz, 0.8 * np.cos(yaw) * wz, 0.0]))
        _, e = est.update_from_truth(truth)
        errs.append(float(np.linalg.norm(e.position - p)))
    yerr = float(np.degrees(np.arctan2(np.sin(e.euler[2] - yaw), np.cos(e.euler[2] - yaw))))
    return errs[-1], yerr, e.status


def test_compass_measurement_is_noisy_biased_wrapped_and_deterministic():
    truth = TruthSnapshot(0.0, [0, 0, -5], (0, 0, np.sin(np.pi / 2 * 0.99), np.cos(np.pi / 2 * 0.99)), [0, 0, 0], [0, 0, 0])
    a = [Compass(CompassConfig(seed=3)).measure(truth).value[0] for _ in range(1)]
    b = [Compass(CompassConfig(seed=3)).measure(truth).value[0] for _ in range(1)]
    assert a == b                                                   # same seed -> same stream
    c = Compass(CompassConfig(seed=3, noise_std=0.0, bias=0.0, dropout_probability=0.0)).measure(truth).value[0]
    assert abs(c - np.pi * 0.99) < 1e-9 or abs(abs(c) - np.pi * 0.99) < 1e-9 or abs(c - (np.pi * 0.99 - 2 * np.pi)) < 1e-9
    m = Compass(CompassConfig(seed=3, dropout_probability=1.0)).measure(truth)
    assert not m.valid and np.isnan(m.value[0])                     # dropout: NaN, never truth
    for s in range(20):
        assert -np.pi <= Compass(CompassConfig(seed=s)).measure(truth).value[0] <= np.pi


def test_ekf_heading_update_pulls_yaw_to_the_measurement_and_shrinks_yaw_variance():
    ekf = EKF(EKFConfig(), position=(0, 0, -5), quat=(0, 0, 0, 1))
    from src.sensors.models import Measurement
    from src.sensors.health import SensorHealth
    var0 = ekf.P[8, 8]
    for k in range(20):
        ekf.coast(0.1 * (k + 1))
        ekf.update_compass(Measurement("compass", ekf.time, np.array([0.2]), True, SensorHealth.HEALTHY, np.array([[4e-4]])))
    assert abs(ekf.state.euler[2] - 0.2) < 0.02
    assert ekf.P[8, 8] < var0 and ekf.stats["compass_updates"] > 0


def test_compass_bounds_the_dead_reckoning_error_that_gyro_only_yaw_cannot():
    err_off, yaw_off, _ = _weave(False, 0.005)
    err_on, yaw_on, status = _weave(True, 0.005)
    assert abs(yaw_off) > 3.0                                        # gyro-only yaw drifts (degrees)
    assert abs(yaw_on) < 1.0 and status["compass_updates"] > 200
    assert err_off > 1.0 and err_on < 0.6 and err_on < 0.3 * err_off


def test_compass_is_opt_in_and_does_not_change_the_other_sensors():
    on, off = _cfg(True, 0.0005), _cfg(False, 0.0005)
    from src.sensors.suite import SensorSuite
    assert "compass" in SensorSuite(on).sensors and "compass" not in SensorSuite(off).sensors
    del on["sensors"]["compass"]
    assert "compass" not in SensorSuite(on).sensors                  # absent section -> old behaviour


def test_closed_loop_mission_with_compass_ends_physically_close_to_the_goal():
    """Navigation-logic stand-in (NOT PyBullet): real sensors + EKF + map + planner; only the rigid body is a stand-in."""
    from tests.helpers.closed_loop import run_closed_loop
    def pillar(x, y, r=0.8):
        return {"kind": "pillar", "x": x, "y": y, "radius_m": r, "z_top_m": -2.0}
    ov = {"environment": {"scenario": "mapping_validation", "explicit_obstacles": [pillar(21, 12.5), pillar(24, 11.2)]},
          "navigation": {"goal_position": [35.0, 28.0, -5.0]}, "sensors": {"compass": {"enabled": True}}}
    r = run_closed_loop(ov, max_time=150)
    assert r.final_state == "GOAL_REACHED" and r.collisions == 0
    assert np.linalg.norm(r.est_final - r.goal) < 0.5          # decided from the ESTIMATE
    assert r.truth_goal_distance < 0.6                          # EVALUATION ONLY: the vehicle is physically at the goal


def test_unbypassable_wall_still_ends_in_FAILED_no_path_with_the_compass_without_collisions():
    """Stand-in (NOT PyBullet) twin of test_navigation_pybullet's dam test, compass ON: accurate mapping makes the
    exploration of the wall slower than with drifting yaw, but it must still end cleanly in FAILED/no_path."""
    from tests.helpers.closed_loop import run_closed_loop
    dam = {"kind": "wall", "x": 30, "y": 25, "half_x_m": 0.5, "half_y_m": 40.0, "z_top_m": -2.0}
    r = run_closed_loop({"environment": {"scenario": "mapping_validation", "explicit_obstacles": [dam]},
                         "navigation": {"goal_position": [45.0, 20.0, -5.0]}, "sensors": {"compass": {"enabled": True}}},
                        max_time=300)
    assert r.final_state == "FAILED" and r.telemetry["failure_reason"].startswith("no_path")
    assert r.collisions == 0 and r.sim_time < 300.0

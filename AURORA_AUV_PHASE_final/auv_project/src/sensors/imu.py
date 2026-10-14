"""Simulated 3-axis IMU (accelerometer + gyroscope).

Frames / units
  world: +Z up. Gravity vector g_w = [0, 0, -g] (g = 9.8 m/s^2 from simulation.gravity).
  body = sensor frame (IMU at the centre of gravity, aligned with the body axes): X fwd, Y left, Z up.
  accelerometer [m/s^2]: SPECIFIC FORCE  f_b = R^T (a_w - g_w) + b_a + n_a
      a_w = true world-frame kinematic acceleration. At rest and level f_b = [0, 0, +g].
  gyroscope [rad/s]: omega_b = R^T omega_w + b_g + n_g   (true body angular rate).
  Noise n ~ N(0, std^2) PER SAMPLE (not a continuous noise density). Bias b follows a
  random walk b += N(0, walk_std^2 * dt), walk_std in [unit / sqrt(s)].
True acceleration comes from the simulated motion: finite difference of the world
velocity between consecutive samples (unless the snapshot supplies it). The first
sample has no previous velocity, so a_w = 0 there. Lever arm / centripetal effects are
not modelled (IMU at CoG).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sensors.health import HealthMonitor, SensorHealth
from src.sensors.models import Measurement, TruthSnapshot
from src.sensors.noise import RandomWalkBias, make_rng


def _vec3(v) -> np.ndarray:
    return np.broadcast_to(np.asarray(v, float), (3,)).copy()


@dataclass
class IMUConfig:
    accel_noise_std: float = 0.05            # m/s^2 per sample
    gyro_noise_std: float = 0.005            # rad/s per sample
    accel_bias: list = field(default_factory=lambda: [0.02, -0.02, 0.01])   # m/s^2 initial
    gyro_bias: list = field(default_factory=lambda: [0.001, -0.001, 0.0005])  # rad/s initial
    accel_bias_walk_std: float = 1e-4        # m/s^2 / sqrt(s)
    gyro_bias_walk_std: float = 1e-5         # rad/s / sqrt(s)
    degraded_noise_scale: float = 5.0        # noise multiplier when health == DEGRADED
    gravity: float = 9.8
    seed: int = 42

    @classmethod
    def from_config(cls, d: dict | None, gravity: float = 9.8) -> "IMUConfig":
        d = dict(d or {})
        d.pop("enabled", None); d.pop("rate_hz", None)
        d.setdefault("gravity", gravity)
        return cls(**d)


class IMU:
    name = "imu"

    def __init__(self, config: IMUConfig | None = None):
        self.cfg = config or IMUConfig()
        self.rng = make_rng(self.cfg.seed)
        self.accel_bias = RandomWalkBias(_vec3(self.cfg.accel_bias), self.cfg.accel_bias_walk_std, self.rng)
        self.gyro_bias = RandomWalkBias(_vec3(self.cfg.gyro_bias), self.cfg.gyro_bias_walk_std, self.rng)
        self.health_monitor = HealthMonitor()
        self._prev_time: float | None = None
        self._prev_vel: np.ndarray | None = None

    def inject_fault(self, health) -> None:
        """'degraded' -> noise x degraded_noise_scale; 'failed' -> no valid output; None clears."""
        self.health_monitor.inject(health)

    def measure(self, truth: TruthSnapshot) -> Measurement:
        cfg = self.cfg
        dt = 0.0 if self._prev_time is None else truth.time - self._prev_time
        # --- true specific force from simulated motion
        if truth.acceleration_world is not None:
            a_w = truth.acceleration_world
        elif self._prev_vel is not None and dt > 0:
            a_w = (truth.velocity_world - self._prev_vel) / dt
        else:
            a_w = np.zeros(3)
        self._prev_time, self._prev_vel = truth.time, truth.velocity_world.copy()
        R = truth.rotation
        f_true = R.T @ (a_w - np.array([0.0, 0.0, -cfg.gravity]))
        w_true = truth.angular_velocity_body

        # --- error model (all variates drawn every call -> deterministic stream)
        ba = self.accel_bias.step(dt)
        bg = self.gyro_bias.step(dt)
        n_a = self.rng.standard_normal(3)
        n_g = self.rng.standard_normal(3)
        scale = cfg.degraded_noise_scale if self.health_monitor.fault_degraded else 1.0
        sa, sg = cfg.accel_noise_std * scale, cfg.gyro_noise_std * scale

        failed = self.health_monitor.fault_failed
        valid = not failed
        if failed:
            # FAILED = stuck at zero: the sensor still outputs a sample at its own rate, but that sample is
            # 0.0 on every axis and never changes. valid=False keeps the EKF from fusing a stuck sensor.
            value = np.zeros(6)
        else:
            value = np.concatenate([f_true + ba + sa * n_a, w_true + bg + sg * n_g])
        cov = np.diag([sa ** 2] * 3 + [sg ** 2] * 3)
        self.health_monitor.record(valid)
        return Measurement(self.name, truth.time, value, valid, self.health_monitor.health, cov)

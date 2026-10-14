"""Simulated magnetic compass / AHRS heading (yaw) sensor.

The only absolute heading reference on the vehicle. Without it yaw is pure gyro dead reckoning (unobservable
bias + drift) and every DVL velocity is rotated by the wrong heading, so horizontal position error grows with
distance travelled (a 1 deg heading error costs ~1.7 % of the distance travelled).

    heading_meas = wrap( yaw_true + bias + n ),   n ~ N(0, noise_std^2)      [rad, world frame, 0 = +X, CCW]

Like every simulated sensor it is derived from the TruthSnapshot plus an explicit error model (constant bias, white
noise, dropouts); the EKF only ever sees the noisy Measurement, never the truth. Invalid samples carry NaN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sensors.health import HealthMonitor
from src.sensors.models import Measurement, TruthSnapshot
from src.sensors.noise import make_rng


@dataclass
class CompassConfig:
    noise_std: float = 0.02             # rad per sample (~1.1 deg)
    bias: float = 0.005                 # rad constant (~0.3 deg residual after calibration; NOT known to the EKF)
    dropout_probability: float = 0.02
    degraded_noise_scale: float = 5.0
    seed: int = 46

    @classmethod
    def from_config(cls, d: dict | None) -> "CompassConfig":
        d = dict(d or {}); d.pop("enabled", None); d.pop("rate_hz", None)
        return cls(**d)


class Compass:
    name = "compass"

    def __init__(self, config: CompassConfig | None = None):
        self.cfg = config or CompassConfig()
        self.rng = make_rng(self.cfg.seed)
        self.health_monitor = HealthMonitor()

    def inject_fault(self, health) -> None:
        self.health_monitor.inject(health)

    def measure(self, truth: TruthSnapshot) -> Measurement:
        cfg = self.cfg
        u = self.rng.random()
        n = self.rng.standard_normal()
        std = cfg.noise_std * (cfg.degraded_noise_scale if self.health_monitor.fault_degraded else 1.0)
        failed = self.health_monitor.fault_failed
        valid = (u >= cfg.dropout_probability) and not failed
        R = truth.rotation
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        heading = float(np.arctan2(np.sin(yaw + cfg.bias + std * n), np.cos(yaw + cfg.bias + std * n)))
        if valid:
            value = np.array([heading])
        elif failed:
            value = np.zeros(1)               # FAILED = stuck at zero: 0 rad on every sample, never changing
        else:
            value = np.array([np.nan])        # dropout: no output at all this sample
        self.health_monitor.record(valid)
        return Measurement(self.name, truth.time, value, bool(valid), self.health_monitor.health,
                           np.array([[std ** 2]]))

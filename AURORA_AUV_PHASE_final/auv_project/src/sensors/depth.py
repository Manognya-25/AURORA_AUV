"""Simulated depth / pressure sensor.

Depth is POSITIVE DOWN, measured from the surface z = 0 in the world frame (world +Z up):
    depth_meas = -z + bias + n,  n ~ N(0, noise_std^2)   [m]
(The sensor is at the CoG; atmospheric pressure/water density are folded into metres.)
Invalid on random dropout or failure (value NaN, never truth).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sensors.health import HealthMonitor, SensorHealth
from src.sensors.models import Measurement, TruthSnapshot
from src.sensors.noise import make_rng


@dataclass
class DepthConfig:
    noise_std: float = 0.02            # m
    bias: float = 0.0                  # m (constant)
    dropout_probability: float = 0.01
    degraded_noise_scale: float = 5.0
    seed: int = 42

    @classmethod
    def from_config(cls, d: dict | None) -> "DepthConfig":
        d = dict(d or {}); d.pop("enabled", None); d.pop("rate_hz", None)
        return cls(**d)


class DepthSensor:
    name = "depth"

    def __init__(self, config: DepthConfig | None = None):
        self.cfg = config or DepthConfig()
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
        if valid:
            value = np.array([-truth.position[2] + cfg.bias + std * n])
        elif failed:
            value = np.zeros(1)               # FAILED = stuck at zero: 0.0 m on every sample, never changing
        else:
            value = np.array([np.nan])        # dropout: no output at all this sample
        self.health_monitor.record(valid)
        return Measurement(self.name, truth.time, value, bool(valid), self.health_monitor.health,
                           np.array([[std ** 2]]))

"""Simulated Doppler Velocity Log (bottom-lock).

Measures vehicle velocity RELATIVE TO THE SEABED (ground velocity, not water-relative,
so ocean current does not appear), expressed in the BODY frame (X fwd, Y left, Z up):
    v_meas = (1 + scale_error) * R^T v_world + n,  n ~ N(0, noise_std^2 I)   [m/s]
Bottom lock requires altitude above the seabed in [min_altitude_m, max_altitude_m]
(altitude = z - seabed_z; the seabed is modelled as the plane z = seabed_z, matching
simulation.seabed_z). Samples are invalid on random dropout (dropout_probability), out
of altitude range, or sensor failure. Invalid samples carry NaN values, NOT truth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.sensors.health import HealthMonitor, SensorHealth
from src.sensors.models import Measurement, TruthSnapshot
from src.sensors.noise import make_rng


@dataclass
class DVLConfig:
    noise_std: float = 0.02            # m/s per axis
    scale_error: float = 0.0           # fractional
    dropout_probability: float = 0.02
    min_altitude_m: float = 0.3
    max_altitude_m: float = 30.0
    seabed_z: float = -15.0            # world z of the seabed plane
    degraded_noise_scale: float = 4.0
    seed: int = 42

    @classmethod
    def from_config(cls, d: dict | None, seabed_z: float | None = None) -> "DVLConfig":
        d = dict(d or {}); d.pop("enabled", None); d.pop("rate_hz", None)
        if seabed_z is not None:
            d.setdefault("seabed_z", seabed_z)
        return cls(**d)


class DVL:
    name = "dvl"

    def __init__(self, config: DVLConfig | None = None):
        self.cfg = config or DVLConfig()
        self.rng = make_rng(self.cfg.seed)
        self.health_monitor = HealthMonitor()

    def inject_fault(self, health) -> None:
        self.health_monitor.inject(health)

    def measure(self, truth: TruthSnapshot) -> Measurement:
        cfg = self.cfg
        u = self.rng.random()                 # always drawn
        n = self.rng.standard_normal(3)
        scale = cfg.degraded_noise_scale if self.health_monitor.fault_degraded else 1.0
        std = cfg.noise_std * scale
        altitude = float(truth.position[2] - cfg.seabed_z)
        locked = cfg.min_altitude_m <= altitude <= cfg.max_altitude_m
        failed = self.health_monitor.fault_failed
        valid = (u >= cfg.dropout_probability) and locked and not failed
        if valid:
            value = (1.0 + cfg.scale_error) * truth.velocity_body + std * n
        elif failed:
            # FAILED = stuck at zero: a real sample, but it reads 0.0 m/s on every axis and never changes.
            value = np.zeros(3)
        else:
            value = np.full(3, np.nan)        # dropout / out of altitude range: no output at all this sample
        self.health_monitor.record(valid)
        return Measurement(self.name, truth.time, value, bool(valid), self.health_monitor.health,
                           np.eye(3) * std ** 2)

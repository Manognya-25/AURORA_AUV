"""Forward-looking multi-ray sonar using real PyBullet ray casting.

NOT acoustic physics: each beam is one geometric ray (no beam pattern, multipath,
attenuation or sound-speed effects). Range comes from an actual p.rayTestBatch
intersection with the simulated world (seabed + obstacle bodies).

Mounting (body frame: X fwd, Y left, Z up): origin `mount_position_m`
(default [0.8, 0, 0], just ahead of the 0.75 m nose so rays never start inside the hull),
orientation `mount_rpy_deg` relative to the body (default zeros = looking along +X).
Ray coordinates (sensor frame, same axes as body): azimuth az is positive toward +Y (left,
CCW about +Z), elevation el positive toward +Z (up):
    d_s = [cos(el) cos(az), cos(el) sin(az), sin(el)]
Angles are spread evenly over +/- FOV/2 (a single ray sits at 0). Ray index = i_el*n_az + i_az.
World ray: start = p + R (mount), end = start + R R_ms d_s * max_range; the reported range
is hit_fraction * max_range, measured from the sensor origin.

Per ray, in fixed draw order: dropout u1 (< dropout_probability -> no return), detection u2
(>= detection_probability -> missed), noise n ~ N(0, range_noise_std^2) added to true range.
Hits closer than min_range, or hitting the AUV's own body, give no return. Not detected ->
range = max_range, detected = False, point = NaN. Output: SonarMeasurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sensors.health import HealthMonitor, SensorHealth
from src.sensors.models import SonarMeasurement, TruthSnapshot
from src.sensors.noise import make_rng
from src.utils.transforms import euler_to_quat, quat_to_rot


@dataclass
class SonarConfig:
    enabled: bool = True
    horizontal_fov_deg: float = 90.0
    vertical_fov_deg: float = 30.0
    horizontal_rays: int = 21
    vertical_rays: int = 5
    max_range_m: float = 30.0
    min_range_m: float = 0.2
    range_noise_std_m: float = 0.05
    dropout_probability: float = 0.05
    detection_probability: float = 0.98
    mount_position_m: list = field(default_factory=lambda: [0.8, 0.0, 0.0])
    mount_rpy_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    seed: int = 42

    @classmethod
    def from_config(cls, d: dict | None) -> "SonarConfig":
        d = dict(d or {}); d.pop("rate_hz", None)
        return cls(**d)


class Sonar:
    name = "sonar"

    def __init__(self, body_id: int, physics_client: int = 0, config: SonarConfig | None = None):
        self.cfg = config or SonarConfig()
        c = self.cfg
        if c.horizontal_rays < 1 or c.vertical_rays < 1:
            raise ValueError("ray counts must be >= 1")
        if not 0 <= c.min_range_m < c.max_range_m:
            raise ValueError("need 0 <= min_range_m < max_range_m")
        self.body_id, self.client = int(body_id), int(physics_client)
        self.rng = make_rng(c.seed)
        self.health_monitor = HealthMonitor(degrade_after=3, fail_after=30)
        az = self._angles(c.horizontal_fov_deg, c.horizontal_rays)
        el = self._angles(c.vertical_fov_deg, c.vertical_rays)
        EL, AZ = np.meshgrid(el, az, indexing="ij")          # elevation-major
        self.azimuths, self.elevations = AZ.ravel(), EL.ravel()
        self.dirs_sensor = np.stack([np.cos(self.elevations) * np.cos(self.azimuths),
                                     np.cos(self.elevations) * np.sin(self.azimuths),
                                     np.sin(self.elevations)], axis=1)
        self.R_ms = quat_to_rot(euler_to_quat(*np.deg2rad(c.mount_rpy_deg)))
        self.mount = np.asarray(c.mount_position_m, float)

    @staticmethod
    def _angles(fov_deg: float, n: int) -> np.ndarray:
        half = np.deg2rad(fov_deg) / 2.0
        return np.zeros(1) if n == 1 else np.linspace(-half, half, n)

    @property
    def n_rays(self) -> int:
        return len(self.azimuths)

    def inject_fault(self, health) -> None:
        self.health_monitor.inject(health)

    def _ray_ranges(self, start: np.ndarray, dirs_w: np.ndarray) -> np.ndarray:
        """True geometric range per ray (inf = no hit within max range) from a REAL PyBullet
        rayTestBatch. Factored out of measure() so a test can substitute another ray caster."""
        import pybullet as p                      # lazy: sensors package stays importable without it
        c, n = self.cfg, self.n_rays
        starts = np.repeat(start[None, :], n, axis=0)
        results = p.rayTestBatch(starts.tolist(), (starts + dirs_w * c.max_range_m).tolist(),
                                 physicsClientId=self.client)
        true_range = np.full(n, np.inf)
        for i, r in enumerate(results):
            if r[0] != -1 and r[0] != self.body_id:        # r[0] = hit body id, r[2] = hit fraction
                true_range[i] = r[2] * c.max_range_m
        return true_range

    def measure(self, truth: TruthSnapshot) -> SonarMeasurement:
        c, n = self.cfg, self.n_rays
        R = truth.rotation
        start = truth.position + R @ self.mount
        dirs_w = (R @ self.R_ms @ self.dirs_sensor.T).T
        true_range = self._ray_ranges(start, dirs_w)

        u_drop = self.rng.random(n)
        u_det = self.rng.random(n)
        noise = self.rng.standard_normal(n)
        # FAILED: the scan is still produced at its own rate but carries nothing - every ray is invalid
        # (ray_valid False) and the hit count is 0. A range of 0.0 would mean "obstacle at the transducer",
        # so a failed scan reports no hits instead of zeros; the dashboard shows 0 valid rays / 0 hits for it.
        failed = self.health_monitor.fault_failed
        hit = np.isfinite(true_range)
        noisy = true_range + c.range_noise_std_m * noise
        detected = (hit & (u_drop >= c.dropout_probability) & (u_det < c.detection_probability)
                    & (noisy >= c.min_range_m) & (true_range < c.max_range_m) & (not failed))
        ranges = np.where(detected, np.minimum(noisy, c.max_range_m), c.max_range_m)
        points = np.where(detected[:, None],
                          ranges[:, None] * (self.R_ms @ self.dirs_sensor.T).T + self.mount, np.nan)
        # Phase 4: per-ray validity. A ray carries information unless it was dropped, the sensor is
        # failed, or a hit was too close to resolve (min-range blanking). A valid ray without a
        # detection is a genuine "nothing here" (no-return); an invalid ray says nothing at all.
        blanked = hit & (noisy < c.min_range_m)
        ray_valid = (u_drop >= c.dropout_probability) & (not failed) & ~blanked
        scan_ok = (not failed) and float(np.mean(u_drop >= c.dropout_probability)) >= 0.5
        self.health_monitor.record(scan_ok)
        return SonarMeasurement(
            sensor=self.name, timestamp=truth.time, value=ranges, valid=bool(scan_ok),
            health=self.health_monitor.health, covariance=np.eye(n) * c.range_noise_std_m ** 2,
            azimuths=self.azimuths.copy(), elevations=self.elevations.copy(), detected=detected,
            points_sensor=points, max_range=c.max_range_m, min_range=c.min_range_m,
            ray_valid=ray_valid, mount_position=self.mount.copy(), mount_rotation=self.R_ms.copy())

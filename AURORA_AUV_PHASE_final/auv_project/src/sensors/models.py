"""Shared data types for sensors.

Frames: world +Z up (depth = -z); body/sensor X forward, Y left, Z up;
quaternion (x, y, z, w) body->world (PyBullet order).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.sensors.health import SensorHealth
from src.utils.transforms import quat_to_rot


@dataclass
class TruthSnapshot:
    """Ground truth handed to the sensors (the ONLY place truth enters the sensor path).

    position, velocity_world, angular_velocity_world: world frame (m, m/s, rad/s).
    quat: (x, y, z, w) body->world. acceleration_world: optional true linear
    acceleration (m/s^2, world, EXCLUDING gravity). If None the IMU derives it by
    finite-differencing velocity_world between samples.
    """
    time: float
    position: np.ndarray
    quat: np.ndarray
    velocity_world: np.ndarray
    angular_velocity_world: np.ndarray
    acceleration_world: np.ndarray | None = None

    def __post_init__(self):
        self.position = np.asarray(self.position, float)
        self.quat = np.asarray(self.quat, float)
        self.velocity_world = np.asarray(self.velocity_world, float)
        self.angular_velocity_world = np.asarray(self.angular_velocity_world, float)
        if self.acceleration_world is not None:
            self.acceleration_world = np.asarray(self.acceleration_world, float)

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rot(self.quat)

    @property
    def velocity_body(self) -> np.ndarray:
        return self.rotation.T @ self.velocity_world

    @property
    def angular_velocity_body(self) -> np.ndarray:
        return self.rotation.T @ self.angular_velocity_world


@dataclass
class Measurement:
    """One sensor reading. value is NaN-filled when valid is False (never truth)."""
    sensor: str
    timestamp: float
    value: np.ndarray
    valid: bool
    health: SensorHealth
    covariance: np.ndarray | None = None   # measurement-noise covariance (same order as value)


@dataclass
class SonarMeasurement(Measurement):
    """value = ranges (N,) in metres (max_range where nothing detected).

    Rays are ordered elevation-major: index = i_elev * n_az + i_az.
    """
    azimuths: np.ndarray = field(default_factory=lambda: np.zeros(0))     # rad, +left (CCW about +Z)
    elevations: np.ndarray = field(default_factory=lambda: np.zeros(0))   # rad, +up
    detected: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    points_sensor: np.ndarray = field(default_factory=lambda: np.zeros((0, 3)))  # NaN where not detected
    max_range: float = 0.0
    min_range: float = 0.0
    # Phase 4 additions (defaults keep every Phase 3 constructor call valid):
    ray_valid: np.ndarray | None = None          # (N,) bool: ray produced usable information (not dropped /
                                                 # blanked / failed). None => derived from `valid`.
    mount_position: np.ndarray | None = None     # sonar origin in the BODY frame (m)
    mount_rotation: np.ndarray | None = None     # 3x3 sensor->body rotation (the mounting extrinsics)

    def ray_status(self) -> np.ndarray:
        """Per-ray status: 1 = detected hit, 0 = valid ray with no return (free-space evidence only),
        -1 = invalid ray (dropout / min-range blanking / sensor failed: NO information)."""
        n = len(self.azimuths)
        valid = np.full(n, bool(self.valid)) if self.ray_valid is None else np.asarray(self.ray_valid, bool)
        det = np.asarray(self.detected, bool) if len(self.detected) == n else np.zeros(n, bool)
        status = np.where(valid, np.where(det, 1, 0), -1).astype(np.int8)
        return status

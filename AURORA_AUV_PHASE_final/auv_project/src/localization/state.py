"""Estimated-state container.

Frames (same as Phase 2): world +Z up (depth = -z); body X fwd, Y left, Z up; quaternion
(x, y, z, w) body->world. Error-state ordering of `covariance` (15x15):
    [dp(0:3) world m | dv(3:6) world m/s | dtheta(6:9) body-frame small rotation rad |
     dba(9:12) m/s^2 | dbg(12:15) rad/s]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.utils.transforms import quat_to_euler, quat_to_rot

P, V, TH, BA, BG = slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 12), slice(12, 15)


@dataclass
class EstimatedState:
    time: float
    position: np.ndarray                 # world [x, y, z] m
    velocity: np.ndarray                 # world m/s
    quat: np.ndarray                     # (x, y, z, w) body->world
    angular_velocity: np.ndarray         # body rad/s (last gyro reading minus estimated bias)
    accel_bias: np.ndarray
    gyro_bias: np.ndarray
    covariance: np.ndarray               # 15x15
    status: dict = field(default_factory=dict)

    @property
    def euler(self) -> np.ndarray:
        """(roll, pitch, yaw) rad."""
        return quat_to_euler(self.quat)

    @property
    def rotation(self) -> np.ndarray:
        return quat_to_rot(self.quat)

    @property
    def velocity_body(self) -> np.ndarray:
        return self.rotation.T @ self.velocity

    @property
    def depth(self) -> float:
        return float(-self.position[2])

    @property
    def position_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance)[P])

    @property
    def velocity_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance)[V])

    @property
    def attitude_std(self) -> np.ndarray:
        return np.sqrt(np.diag(self.covariance)[TH])

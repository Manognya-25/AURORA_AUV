"""
transforms.py
-------------
Small NumPy rotation / quaternion helpers shared by the dynamics, control and
(later) estimation modules. No PyBullet dependency, so everything that uses
these can be unit-tested headlessly.

Conventions (match PyBullet so values can be exchanged without conversion):
  * Quaternion order is (x, y, z, w).
  * Euler angles are (roll, pitch, yaw) = intrinsic Z-Y-X, i.e.
    R = Rz(yaw) @ Ry(pitch) @ Rx(roll), mapping BODY -> WORLD vectors.
  * World frame: +Z up (depth is negative z, as in configs/config.yaml).
"""

from __future__ import annotations

import numpy as np


def wrap_angle(angle: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def skew(v: np.ndarray) -> np.ndarray:
    """Return the 3x3 skew-symmetric matrix S such that S @ a == cross(v, a)."""
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Return q scaled to unit norm; raises ValueError for a zero quaternion."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("cannot normalize a zero quaternion")
    return q / n


def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product a * b for (x, y, z, w) quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """Rotation matrix R (body -> world) from an (x, y, z, w) quaternion."""
    x, y, z, w = quat_normalize(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_to_euler(q: np.ndarray) -> np.ndarray:
    """(roll, pitch, yaw) from an (x, y, z, w) quaternion (pitch clipped at +/-90 deg)."""
    x, y, z, w = quat_normalize(q)
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def euler_to_quat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """(x, y, z, w) quaternion from (roll, pitch, yaw)."""
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def quat_integrate(q: np.ndarray, omega_body: np.ndarray, dt: float) -> np.ndarray:
    """Advance q by body-frame angular velocity omega_body [rad/s] over dt (exact for constant omega)."""
    angle = np.linalg.norm(omega_body) * dt
    if angle < 1e-12:
        dq = np.array([*(0.5 * dt * np.asarray(omega_body)), 1.0])
    else:
        axis = np.asarray(omega_body) / np.linalg.norm(omega_body)
        dq = np.array([*(axis * np.sin(angle / 2)), np.cos(angle / 2)])
    return quat_normalize(quat_multiply(q, dq))

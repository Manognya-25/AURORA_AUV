"""
drag.py
-------
Simplified linear + quadratic hydrodynamic damping in all 6 DOF.

    D(nu_r) nu_r = (D_lin + D_quad * |nu_r|) * nu_r          (element-wise, body frame)
    wrench_drag  = - D(nu_r) nu_r

nu_r = nu - nu_current is the velocity RELATIVE TO THE WATER; only the linear
part of the current is modelled (water angular velocity is taken as zero).

SIMPLIFICATION: diagonal damping, no cross-coupling, no lift/Munk moment, no
added-mass Coriolis. Coefficients are illustrative, not identified from a real
vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def relative_velocity(nu: np.ndarray, R_wb: np.ndarray, current_world: np.ndarray) -> np.ndarray:
    """
    nu_r = nu - [R^T v_current ; 0]  (body frame, length 6).

    Equivalent to ``relative_velocity = vehicle_velocity - water_velocity``
    expressed in the body frame.
    """
    nu_r = np.asarray(nu, dtype=float).copy()
    nu_r[:3] -= R_wb.T @ np.asarray(current_world, dtype=float)
    return nu_r


@dataclass
class DragModel:
    """Diagonal linear + quadratic damping, ordered [surge, sway, heave, roll, pitch, yaw]."""
    linear: np.ndarray = field(default_factory=lambda: np.array([2.0, 2.0, 2.0, 0.5, 0.5, 0.5]))
    quadratic: np.ndarray = field(default_factory=lambda: np.zeros(6))

    def __post_init__(self) -> None:
        self.linear = np.asarray(self.linear, dtype=float)
        self.quadratic = np.asarray(self.quadratic, dtype=float)
        if self.linear.shape != (6,) or self.quadratic.shape != (6,):
            raise ValueError("drag coefficient vectors must have length 6")
        if np.any(self.linear < 0) or np.any(self.quadratic < 0):
            raise ValueError("drag coefficients must be non-negative")

    def wrench(self, nu_r: np.ndarray) -> np.ndarray:
        """Body-frame damping wrench (opposes the relative velocity)."""
        nu_r = np.asarray(nu_r, dtype=float)
        return -(self.linear + self.quadratic * np.abs(nu_r)) * nu_r

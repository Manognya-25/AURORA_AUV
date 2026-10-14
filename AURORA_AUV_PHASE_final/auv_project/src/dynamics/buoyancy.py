"""
buoyancy.py
-----------
Simplified buoyancy / restoring-force model for the AUV.

    F_buoyancy = rho * g * V * ratio  (+ trim)           [world +Z, N]
    restoring moment = r_b x R^T [0, 0, F_buoyancy]       [body frame, N*m]

``r_b`` is the centre-of-buoyancy offset from the centre of gravity (body
frame). A CoB ABOVE the CoG (positive z) makes roll/pitch passively stable.

``ratio`` is the legacy ``simulation.physics.buoyancy_ratio`` calibration
factor, kept for backward compatibility (see configs/config.yaml for why it
exists: the 20 kg model body has the displaced volume of a much heavier solid
cylinder). ``trim_n`` adds a net buoyancy offset (N, + = floats up) on top so
the vehicle can be made slightly positive/negative.

SIMPLIFICATION: buoyancy is constant (no depth/compressibility/temperature).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class BuoyancyModel:
    """Constant buoyancy force with an optional centre-of-buoyancy offset."""
    mass: float
    volume: float
    water_density: float = 1000.0
    gravity: float = 9.8                       # magnitude, m/s^2
    buoyancy_ratio: float = 1.0
    trim_n: float = 0.0
    cob_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def weight(self) -> float:
        """Weight m*g in Newtons."""
        return self.mass * self.gravity

    @property
    def buoyancy_force(self) -> float:
        """Upward buoyancy force magnitude in Newtons."""
        return self.water_density * self.gravity * self.volume * self.buoyancy_ratio + self.trim_n

    @property
    def net_buoyancy(self) -> float:
        """buoyancy - weight in Newtons (> 0 means the vehicle floats up)."""
        return self.buoyancy_force - self.weight

    def restoring_wrench_body(self, R_wb: np.ndarray, include_weight: bool = True) -> np.ndarray:
        """
        Body-frame wrench [Fx, Fy, Fz, Mx, My, Mz] from buoyancy (and weight).

        Input:
            R_wb: 3x3 body->world rotation matrix.
            include_weight: False when the physics engine already applies
                gravity (PyBullet path); True for the standalone integrator.
        Weight acts at the CoG (no moment); buoyancy acts at the CoB offset,
        which produces the restoring moment.
        """
        up_body = R_wb.T @ np.array([0.0, 0.0, 1.0])
        f_b = self.buoyancy_force * up_body
        force = f_b - (self.weight * up_body if include_weight else 0.0)
        moment = np.cross(self.cob_offset, f_b)
        return np.concatenate([force, moment])

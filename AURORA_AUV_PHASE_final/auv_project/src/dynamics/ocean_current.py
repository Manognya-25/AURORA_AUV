"""
ocean_current.py
----------------
Water-velocity interface used by the drag model.

Phase 2 provides only a time-invariant, spatially-uniform current. The
spatially/temporally varying field (x, y, z, t dependence, disturbances) is
Phase 6 and will implement the same ``velocity(position, t)`` interface, so
the dynamics code will not need to change.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class CurrentField(Protocol):
    """Anything that can report the water velocity (world frame, m/s) at a place and time."""

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray: ...


class UniformCurrent:
    """Constant world-frame current vector."""

    def __init__(self, velocity=(0.0, 0.0, 0.0)):
        self._v = np.asarray(velocity, dtype=float).reshape(3)

    def velocity(self, position: np.ndarray, t: float) -> np.ndarray:
        return self._v.copy()

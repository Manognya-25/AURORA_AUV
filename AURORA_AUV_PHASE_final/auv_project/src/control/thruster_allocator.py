"""
thruster_allocator.py
---------------------
Maps a desired body-frame wrench tau = [Fx,Fy,Fz,Mx,My,Mz] to individual
normalised thruster commands.

    tau = B u_N,   B[:, i] = [d_i ; r_i x d_i]  (wrench per Newton of thruster i)

Solved as a bounded least-squares problem
    min ||B u - tau||^2   s.t. |u_i| <= max_thrust_i
(scipy ``lsq_linear``), with a pseudo-inverse fast path when the unconstrained
solution is already within limits. Thrusters marked unavailable (e.g. after
fault detection) are removed from the problem, so the remaining ones are
re-allocated to cover as much of the wrench as they can.

Wrench components with no actuator (roll here) show up in ``residual``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from src.control.thruster_model import ThrusterSpec


@dataclass
class AllocationResult:
    commands: np.ndarray       # normalised per-thruster commands, in [-1, 1] scale
    forces: np.ndarray         # commanded thrust per thruster, N
    achieved: np.ndarray       # B @ forces, the wrench actually requested from the array
    residual: np.ndarray       # desired - achieved
    saturated: bool            # True if any thruster is at its limit and the wrench is not met


class ThrusterAllocator:
    """Configurable allocation matrix + bounded least-squares solver."""

    def __init__(self, specs: list[ThrusterSpec]):
        self.specs = specs
        self.B = np.column_stack([s.wrench_column for s in specs])     # (6, n)
        self.max_thrust = np.array([s.max_thrust for s in specs])
        self._pinv_cache: dict[tuple, np.ndarray] = {}

    @property
    def matrix_rank(self) -> int:
        return int(np.linalg.matrix_rank(self.B))

    def allocate(self, tau: np.ndarray, available: np.ndarray | None = None) -> AllocationResult:
        """
        Input: tau (6,) desired body wrench; available (n,) bool mask of usable thrusters.
        Output: AllocationResult. Never raises on infeasible wrenches -- it returns
                the best bounded least-squares fit and flags ``saturated``.
        """
        tau = np.asarray(tau, dtype=float)
        n = len(self.specs)
        mask = np.ones(n, bool) if available is None else np.asarray(available, bool)
        forces = np.zeros(n)
        idx = np.flatnonzero(mask)
        if idx.size:
            B = self.B[:, idx]
            key = tuple(idx)
            if key not in self._pinv_cache:
                self._pinv_cache[key] = np.linalg.pinv(B)
            u = self._pinv_cache[key] @ tau
            limit = self.max_thrust[idx]
            if np.any(np.abs(u) > limit):
                u = lsq_linear(B, tau, bounds=(-limit, limit), method="bvls").x
            forces[idx] = u
        achieved = self.B @ forces
        residual = tau - achieved
        scale = np.maximum(np.linalg.norm(tau), 1e-9)
        saturated = bool(np.any(np.abs(forces) >= self.max_thrust - 1e-6)
                         and np.linalg.norm(residual) > 1e-6 * scale + 1e-9)
        return AllocationResult(forces / self.max_thrust, forces, achieved, residual, saturated)

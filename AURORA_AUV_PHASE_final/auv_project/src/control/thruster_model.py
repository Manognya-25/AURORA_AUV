"""
thruster_model.py
-----------------
Logical (simulated) thruster array.

The current URDF is a single rigid hull with NO separate thruster links, so
these thrusters exist only in the control/dynamics layer: each has a position
and a thrust axis in the body frame, used to build the allocation matrix.
Nothing is rendered for them.

Per-thruster behaviour:
    command u in [cmd_min, cmd_max]  (normalised, 1.0 = max_thrust)
    -> clipped -> first-order lag (time constant tau) -> *max_thrust
    -> optional multiplicative Gaussian noise (seeded)
    -> failure modes: OK | DEGRADED (effectiveness<1) | DEAD (0 N) | STUCK (frozen at a force)

Thrust is linear in the command (documented simplification; real propellers
are roughly quadratic in rpm and asymmetric fwd/rev).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class ThrusterFault(str, Enum):
    OK = "ok"
    DEGRADED = "degraded"
    DEAD = "dead"
    STUCK = "stuck"


@dataclass
class ThrusterSpec:
    """Static description of one thruster (body frame; +x fwd, +y left, +z up)."""
    name: str
    position: np.ndarray        # m, relative to the CoG
    direction: np.ndarray       # thrust axis (normalised on construction)
    max_thrust: float = 30.0    # N, magnitude in either direction
    cmd_min: float = -1.0
    cmd_max: float = 1.0
    time_constant: float = 0.1  # s, first-order response lag (0 = instantaneous)
    noise_std: float = 0.0      # fraction of commanded thrust (1-sigma)

    def __post_init__(self) -> None:
        self.position = np.asarray(self.position, dtype=float)
        d = np.asarray(self.direction, dtype=float)
        n = np.linalg.norm(d)
        if n < 1e-9:
            raise ValueError(f"thruster {self.name}: zero direction vector")
        self.direction = d / n
        if self.max_thrust <= 0 or self.cmd_min > self.cmd_max:
            raise ValueError(f"thruster {self.name}: invalid limits")

    @property
    def wrench_column(self) -> np.ndarray:
        """6-vector [d ; r x d]: wrench produced by +1 N of thrust."""
        return np.concatenate([self.direction, np.cross(self.position, self.direction)])


def default_layout(length: float = 1.5, max_thrust: float = 30.0, time_constant: float = 0.1,
                   noise_std: float = 0.0) -> list[ThrusterSpec]:
    """
    Six-thruster layout scaled to the hull length:
      T0/T1 aft-port/starboard, along +x   -> surge, yaw (differential)
      T2/T3 fore/aft vertical, along +z    -> heave, pitch (differential)
      T4/T5 fore/aft lateral, along +y     -> sway, yaw (differential)
    Roll has no actuator: it is passively stabilised by the CoB-above-CoG
    restoring moment. Rank of the allocation matrix is therefore 5.
    """
    L = length
    kw = dict(max_thrust=max_thrust, time_constant=time_constant, noise_std=noise_std)
    return [
        ThrusterSpec("aft_port",   [-0.47 * L, +0.10, 0.0], [1, 0, 0], **kw),
        ThrusterSpec("aft_stbd",   [-0.47 * L, -0.10, 0.0], [1, 0, 0], **kw),
        ThrusterSpec("vert_fore",  [+0.30 * L, 0.0, 0.0],   [0, 0, 1], **kw),
        ThrusterSpec("vert_aft",   [-0.30 * L, 0.0, 0.0],   [0, 0, 1], **kw),
        ThrusterSpec("lat_fore",   [+0.30 * L, 0.0, 0.0],   [0, 1, 0], **kw),
        ThrusterSpec("lat_aft",    [-0.30 * L, 0.0, 0.0],   [0, 1, 0], **kw),
    ]


def specs_from_config(sim_config: dict) -> list[ThrusterSpec]:
    """
    Build thruster specs from cfg["simulation"]["thrusters"] (optional section).
    Either ``units: [{name, position, direction, max_thrust?}, ...]`` or the
    default 6-thruster layout with ``max_thrust`` / ``time_constant`` / ``noise_std``.
    """
    cfg = sim_config.get("thrusters", {}) or {}
    common = dict(max_thrust=float(cfg.get("max_thrust", 30.0)),
                  time_constant=float(cfg.get("time_constant", 0.1)),
                  noise_std=float(cfg.get("noise_std", 0.0)))
    if cfg.get("units"):
        return [ThrusterSpec(u["name"], u["position"], u["direction"],
                             **{**common, **{k: u[k] for k in ("max_thrust", "time_constant", "noise_std") if k in u}})
                for u in cfg["units"]]
    return default_layout(float(sim_config["auv"]["length"]), **common)


class ThrusterArray:
    """Stateful set of thrusters: commands in, actual forces out."""

    def __init__(self, specs: list[ThrusterSpec], seed: int | None = 0):
        if not specs:
            raise ValueError("at least one thruster required")
        self.specs = specs
        self.n = len(specs)
        self._rng = np.random.default_rng(seed)
        self.max_thrust = np.array([s.max_thrust for s in specs])
        self.fault = [ThrusterFault.OK] * self.n
        self.effectiveness = np.ones(self.n)
        self._stuck_force = np.zeros(self.n)
        self.force = np.zeros(self.n)            # actual current thrust per thruster, N
        self.last_command = np.zeros(self.n)     # last clipped normalised command

    def reset(self) -> None:
        self.force[:] = 0.0
        self.last_command[:] = 0.0

    # -- failure injection ---------------------------------------------
    def set_fault(self, index: int, fault: ThrusterFault, value: float = 0.0) -> None:
        """
        DEGRADED: ``value`` = remaining effectiveness in [0,1].
        STUCK: ``value`` = frozen thrust in N. DEAD/OK ignore ``value``.
        """
        if not 0 <= index < self.n:
            raise IndexError(index)
        self.fault[index] = fault
        self.effectiveness[index] = float(np.clip(value, 0.0, 1.0)) if fault == ThrusterFault.DEGRADED else 1.0
        if fault == ThrusterFault.STUCK:
            self._stuck_force[index] = float(value)

    def clear_faults(self) -> None:
        self.fault = [ThrusterFault.OK] * self.n
        self.effectiveness[:] = 1.0

    def health(self) -> np.ndarray:
        """Per-thruster health in [0,1] (1 = healthy, 0 = dead/stuck)."""
        h = self.effectiveness.copy()
        for i, f in enumerate(self.fault):
            if f in (ThrusterFault.DEAD, ThrusterFault.STUCK):
                h[i] = 0.0
        return h

    # -- simulation step -----------------------------------------------
    def step(self, commands: np.ndarray, dt: float) -> np.ndarray:
        """Apply normalised commands for dt seconds; returns actual forces (N)."""
        u = np.asarray(commands, dtype=float)
        if u.shape != (self.n,):
            raise ValueError(f"expected {self.n} commands, got shape {u.shape}")
        lo = np.array([s.cmd_min for s in self.specs]); hi = np.array([s.cmd_max for s in self.specs])
        u = np.clip(u, lo, hi)
        self.last_command = u
        target = u * self.max_thrust
        for i, s in enumerate(self.specs):
            tgt = target[i]
            if s.noise_std > 0:
                tgt *= 1.0 + s.noise_std * self._rng.standard_normal()
            alpha = 1.0 if s.time_constant <= 0 else dt / (s.time_constant + dt)
            self.force[i] += alpha * (tgt - self.force[i])
            if self.fault[i] == ThrusterFault.DEAD:
                self.force[i] = 0.0
            elif self.fault[i] == ThrusterFault.STUCK:
                self.force[i] = self._stuck_force[i]
        return self.force * self.effectiveness

    def wrench(self) -> np.ndarray:
        """Body-frame wrench [Fx,Fy,Fz,Mx,My,Mz] currently produced."""
        actual = self.force * self.effectiveness
        return sum(f * s.wrench_column for f, s in zip(actual, self.specs))

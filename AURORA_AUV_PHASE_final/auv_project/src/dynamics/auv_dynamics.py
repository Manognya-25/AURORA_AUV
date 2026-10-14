"""
auv_dynamics.py
---------------
Simplified 6-DOF underwater-vehicle dynamics (NumPy only, no PyBullet).

State
    eta = [x, y, z, roll, pitch, yaw]   world pose (z up; depth = -z)
    nu  = [u, v, w, p, q, r]            BODY-frame linear/angular velocity

Equation of motion (body frame, centre of gravity at the body origin):

    (M_RB + M_A) nu_dot + C_RB(nu) nu = tau_thrust + tau_restoring(eta) + tau_drag(nu_r)

    M_RB   = diag(m, m, m, Ix, Iy, Iz)           rigid-body mass/inertia
    M_A    = diag(added mass)                    optional, diagonal, constant
    C_RB   = rigid-body Coriolis/centripetal      [m w x v ; w x (I w)]
    tau_restoring = gravity + buoyancy (see buoyancy.py), incl. CoB offset
    tau_drag      = -(D_lin + D_quad |nu_r|) nu_r, nu_r = nu - R^T v_current

Two ways to use it:
  1. ``AUVDynamics.step`` integrates the model itself (semi-implicit Euler +
     quaternion attitude). Used for headless unit tests and as the reference
     the PyBullet-coupled path is compared against.
  2. ``AUVDynamics.hydro_wrench`` only returns the hydrodynamic/restoring
     wrench; ``AUV`` (auv.py) applies it to the PyBullet rigid body, which
     does the integration and collision handling. PyBullet cannot represent
     added mass, so ``AddedMassCompensator`` reproduces it with a one-step-lag
     reaction force.

NOT a high-fidelity hydrodynamic model: diagonal added mass, no added-mass
Coriolis, no lift or cross-flow drag coupling, no waves, illustrative
coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.dynamics.buoyancy import BuoyancyModel
from src.dynamics.drag import DragModel, relative_velocity
from src.dynamics.ocean_current import CurrentField, UniformCurrent
from src.utils.transforms import (
    euler_to_quat, quat_integrate, quat_to_euler, quat_to_rot,
)

_DEFAULT_ADDED_MASS = [2.0, 10.0, 10.0, 0.1, 2.0, 2.0]   # kg (x3), kg*m^2 (x3); illustrative
_DEFAULT_LINEAR_DRAG = [2.0, 2.0, 2.0, 0.5, 0.5, 0.5]    # == legacy linear_drag / angular_drag
_DEFAULT_QUAD_DRAG = [12.0, 150.0, 150.0, 0.2, 1.0, 1.0]  # illustrative, not identified from hardware


@dataclass
class AUVParams:
    """All parameters of the simplified 6-DOF model."""
    mass: float
    inertia: np.ndarray                      # diag (Ix, Iy, Iz), kg*m^2
    added_mass: np.ndarray                   # diag, length 6
    added_mass_enabled: bool
    buoyancy: BuoyancyModel
    drag: DragModel

    @classmethod
    def from_config(cls, sim_config: dict) -> "AUVParams":
        """
        Build from cfg["simulation"]. Works with the legacy config alone
        (``auv``, ``physics``, ``gravity`` keys) and picks up the optional
        ``dynamics`` section when present.
        """
        auv, phys = sim_config["auv"], sim_config["physics"]
        dyn = sim_config.get("dynamics", {}) or {}
        mass, radius, length = float(auv["mass"]), float(auv["radius"]), float(auv["length"])
        volume = np.pi * radius ** 2 * length

        # Solid-cylinder inertia (matches assets/auv/auv.urdf) unless overridden.
        default_inertia = [mass * radius ** 2 / 2, mass * (3 * radius ** 2 + length ** 2) / 12,
                           mass * (3 * radius ** 2 + length ** 2) / 12]
        inertia = np.asarray(dyn.get("inertia_diag", default_inertia), dtype=float)

        buoy_cfg = dyn.get("buoyancy", {}) or {}
        buoyancy = BuoyancyModel(
            mass=mass, volume=volume,
            water_density=float(phys["water_density"]),
            gravity=abs(float(sim_config["gravity"])),
            buoyancy_ratio=float(phys["buoyancy_ratio"]),
            trim_n=float(buoy_cfg.get("trim_n", 0.0)),
            cob_offset=np.asarray(buoy_cfg.get("cob_offset_m", [0.0, 0.0, 0.03]), dtype=float),
        )

        drag_cfg = dyn.get("drag", {}) or {}
        default_lin = [float(phys["linear_drag"])] * 3 + [float(phys["angular_drag"])] * 3
        drag = DragModel(
            linear=np.asarray(drag_cfg.get("linear", default_lin), dtype=float),
            quadratic=np.asarray(drag_cfg.get("quadratic", _DEFAULT_QUAD_DRAG), dtype=float),
        )
        return cls(
            mass=mass, inertia=inertia,
            added_mass=np.asarray(dyn.get("added_mass_diag", _DEFAULT_ADDED_MASS), dtype=float),
            added_mass_enabled=bool(dyn.get("added_mass_enabled", True)),
            buoyancy=buoyancy, drag=drag,
        )

    @property
    def rigid_body_mass(self) -> np.ndarray:
        """Diagonal of M_RB (length 6)."""
        return np.array([self.mass] * 3 + list(self.inertia))

    @property
    def effective_mass(self) -> np.ndarray:
        """Diagonal of M_RB + M_A (M_A only if enabled)."""
        return self.rigid_body_mass + (self.added_mass if self.added_mass_enabled else 0.0)


@dataclass
class RigidBodyState:
    """Pose (position + unit quaternion) and body-frame velocity nu."""
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    quat: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0, 1.0]))
    nu: np.ndarray = field(default_factory=lambda: np.zeros(6))

    @property
    def eta(self) -> np.ndarray:
        """[x, y, z, roll, pitch, yaw] for display / logging."""
        return np.concatenate([self.position, quat_to_euler(self.quat)])

    @property
    def rotation(self) -> np.ndarray:
        """Body->world rotation matrix."""
        return quat_to_rot(self.quat)

    @property
    def velocity_world(self) -> np.ndarray:
        """Linear velocity in the world frame."""
        return self.rotation @ self.nu[:3]

    @classmethod
    def from_euler(cls, position, rpy=(0.0, 0.0, 0.0)) -> "RigidBodyState":
        return cls(np.asarray(position, float), euler_to_quat(*rpy), np.zeros(6))


class AUVDynamics:
    """Simplified 6-DOF model: forces, and an optional self-contained integrator."""

    def __init__(self, params: AUVParams, current: CurrentField | None = None):
        self.params = params
        self.current: CurrentField = current if current is not None else UniformCurrent()

    # -- force terms ---------------------------------------------------
    def hydro_wrench(self, position: np.ndarray, R_wb: np.ndarray, nu: np.ndarray,
                     t: float = 0.0, include_weight: bool = True) -> np.ndarray:
        """
        Body-frame wrench from buoyancy/weight restoring + drag relative to the
        water (current). Does NOT include thrust or added mass.
        """
        v_water = self.current.velocity(np.asarray(position, float), t)
        nu_r = relative_velocity(nu, R_wb, v_water)
        return (self.params.buoyancy.restoring_wrench_body(R_wb, include_weight)
                + self.params.drag.wrench(nu_r))

    def coriolis_rigid_body(self, nu: np.ndarray) -> np.ndarray:
        """C_RB(nu) nu = [m w x v ; w x (I w)] (CoG at body origin)."""
        v, w = nu[:3], nu[3:]
        return np.concatenate([self.params.mass * np.cross(w, v),
                               np.cross(w, self.params.inertia * w)])

    # -- reference integrator -----------------------------------------
    def step(self, state: RigidBodyState, tau_thrust: np.ndarray, dt: float,
             t: float = 0.0) -> RigidBodyState:
        """
        Advance the state by dt with semi-implicit Euler (velocity first, then
        pose with the new velocity). ``tau_thrust`` is the body-frame wrench.
        """
        R = state.rotation
        tau = (np.asarray(tau_thrust, float)
               + self.hydro_wrench(state.position, R, state.nu, t, include_weight=True)
               - self.coriolis_rigid_body(state.nu))
        nu_new = state.nu + dt * tau / self.params.effective_mass
        pos_new = state.position + dt * (R @ nu_new[:3])
        quat_new = quat_integrate(state.quat, nu_new[3:], dt)
        return RigidBodyState(pos_new, quat_new, nu_new)


class AddedMassCompensator:
    """
    Reproduces diagonal added mass inside a rigid-body engine that only knows
    M_RB. Engine solves M a = F_ext; we want (M + M_A) a = F_true, so apply
    F_ext = F_true - M_A * a_prev with a_prev the previous step's measured
    acceleration. Converges to a = F/(M+M_A) with a decaying oscillation when
    M_A/M < 1 on every axis (enforced: < max_ratio).
    """

    def __init__(self, rigid_mass: np.ndarray, added_mass: np.ndarray,
                 max_ratio: float = 0.8, max_accel: float = 50.0):
        ratio = np.asarray(added_mass, float) / np.asarray(rigid_mass, float)
        if np.any(ratio >= max_ratio):
            raise ValueError(
                f"added mass / rigid mass must be < {max_ratio} for the lagged PyBullet "
                f"compensation to stay stable (got {np.round(ratio, 2).tolist()})")
        self._added = np.asarray(added_mass, float)
        self._max_accel = float(max_accel)
        self._prev_nu: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_nu = None

    def reaction(self, nu: np.ndarray, dt: float) -> np.ndarray:
        """Body-frame wrench -M_A * a_prev; zero on the first call."""
        nu = np.asarray(nu, float)
        if self._prev_nu is None or dt <= 0:
            accel = np.zeros(6)
        else:
            accel = np.clip((nu - self._prev_nu) / dt, -self._max_accel, self._max_accel)
        self._prev_nu = nu.copy()
        return -self._added * accel

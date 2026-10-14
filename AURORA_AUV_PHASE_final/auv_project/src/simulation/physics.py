"""
physics.py
-----------
Simplified underwater physics forces applied to the AUV rigid body every
simulation step. PyBullet only provides dry rigid-body dynamics (gravity +
contact forces) and has no built-in model of buoyancy or fluid drag, so
both are added here explicitly.

Pipeline implemented here:

    AUV velocity/orientation (read each step from PyBullet)
                       |
                       v
      compute_buoyancy_force()        -> constant upward force
      compute_linear_drag_force()     -> opposes linear velocity
      compute_angular_drag_torque()   -> opposes angular velocity
                       |
                       v
      step_physics()  -> applies all forces/torques to the PyBullet body
                          for one timestep via p.applyExternalForce/Torque

LIMITATIONS (see docs/simulation.md, section 9, for the full list): this
is NOT a hydrodynamics simulation. There is no added mass, no lift, no
vortex shedding, no current field, and drag is a linear (not quadratic)
approximation. It is deliberately simplified to give Weeks 6-9 a
consistent, controllable environment rather than a physically exact one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass
class PhysicsParams:
    """Simplified underwater physics parameters, derived from config + AUV geometry."""
    gravity: float
    water_density: float
    buoyancy_ratio: float
    linear_drag: float
    angular_drag: float
    mass: float
    volume: float  # displaced volume in m^3, derived from the AUV's cylinder geometry

    @classmethod
    def from_config(cls, sim_config: dict) -> "PhysicsParams":
        """
        Input:
            sim_config: cfg["simulation"] dict, needs "gravity" plus the
                        "auv" (mass, length, radius) and "physics"
                        (water_density, buoyancy_ratio, linear_drag,
                        angular_drag) sub-sections.
        Processing:
            Approximates the AUV's displaced volume as a solid cylinder of
            the configured length/radius (matches the URDF geometry in
            assets/auv/auv.urdf -- keep the two in sync if either changes).
        Output:
            PhysicsParams ready for compute_buoyancy_force / step_physics.
        """
        auv_cfg = sim_config["auv"]
        phys_cfg = sim_config["physics"]
        radius = float(auv_cfg["radius"])
        length = float(auv_cfg["length"])
        volume = np.pi * radius**2 * length  # solid cylinder, matches the URDF hull
        return cls(
            gravity=float(sim_config["gravity"]),
            water_density=float(phys_cfg["water_density"]),
            buoyancy_ratio=float(phys_cfg["buoyancy_ratio"]),
            linear_drag=float(phys_cfg["linear_drag"]),
            angular_drag=float(phys_cfg["angular_drag"]),
            mass=float(auv_cfg["mass"]),
            volume=volume,
        )


def compute_buoyancy_force(params: PhysicsParams) -> tuple:
    """
    Compute the (constant, per-step) buoyancy force.

    Input:
        params: PhysicsParams (mass, volume, water_density, buoyancy_ratio, gravity).
    Processing:
        F_b = water_density * |gravity| * volume * buoyancy_ratio, applied
        straight up (+z world axis). buoyancy_ratio == 1.0 is calibrated so
        F_b ~= weight (mass * |gravity|), i.e. approximately neutral
        buoyancy rather than the AUV sinking or floating away (STEP 13).
    Output:
        (fx, fy, fz) world-frame force in Newtons.
    """
    magnitude = params.water_density * abs(params.gravity) * params.volume * params.buoyancy_ratio
    return (0.0, 0.0, magnitude)


def compute_linear_drag_force(linear_velocity: tuple, linear_drag: float) -> tuple:
    """
    Input:
        linear_velocity: (vx, vy, vz) m/s, current body velocity.
        linear_drag: drag coefficient k (config: simulation.physics.linear_drag).
    Processing:
        Simplified LINEAR drag: F_drag = -k * v per axis (not the more
        physically accurate quadratic v|v| drag -- documented simplification).
    Output:
        (fx, fy, fz) world-frame drag force in Newtons, opposing motion.
    """
    return tuple(-linear_drag * v for v in linear_velocity)


def compute_angular_drag_torque(angular_velocity: tuple, angular_drag: float) -> tuple:
    """
    Input:
        angular_velocity: (wx, wy, wz) rad/s.
        angular_drag: angular damping coefficient (config: simulation.physics.angular_drag).
    Processing:
        Analogous linear damping torque opposing angular velocity: T = -k * w.
    Output:
        (tx, ty, tz) world-frame damping torque in N*m.
    """
    return tuple(-angular_drag * w for w in angular_velocity)


def step_physics(body_id: int, params: PhysicsParams, physics_client: int = 0) -> dict:
    """
    Apply one timestep of simplified underwater forces to a PyBullet body.

    Input:
        body_id: PyBullet body unique id (the AUV).
        params: PhysicsParams (see from_config).
        physics_client: PyBullet client id to operate on.
    Processing:
        Reads current linear/angular velocity via p.getBaseVelocity(),
        computes buoyancy + linear drag + angular drag, and applies them
        with p.applyExternalForce / p.applyExternalTorque (link index -1,
        WORLD_FRAME). PyBullet's own gravity (set globally via
        p.setGravity) already supplies the weight force, so only buoyancy
        and drag are added here -- adding gravity again would double it.
    Output:
        dict with the applied "buoyancy_force", "drag_force", "drag_torque"
        (metres/Newtons; returned for logging/debugging, not required by
        the caller).
    """
    linear_vel, angular_vel = p.getBaseVelocity(body_id, physicsClientId=physics_client)

    buoyancy = compute_buoyancy_force(params)
    drag_force = compute_linear_drag_force(linear_vel, params.linear_drag)
    drag_torque = compute_angular_drag_torque(angular_vel, params.angular_drag)

    total_force = tuple(b + d for b, d in zip(buoyancy, drag_force))
    # BUG FIX (Phase 1): with WORLD_FRAME, PyBullet interprets the position
    # argument as a WORLD coordinate. The original [0, 0, 0] applied the force
    # at the world origin, creating a large spurious torque about the AUV's
    # centre of mass (the saved Week 5 log shows the hull tumbling). Apply at the COM.
    com, _ = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
    p.applyExternalForce(
        body_id, -1, total_force, com, p.WORLD_FRAME, physicsClientId=physics_client
    )
    p.applyExternalTorque(
        body_id, -1, drag_torque, p.WORLD_FRAME, physicsClientId=physics_client
    )

    return {"buoyancy_force": buoyancy, "drag_force": drag_force, "drag_torque": drag_torque}

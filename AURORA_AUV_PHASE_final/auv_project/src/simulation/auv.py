"""
auv.py
------
The AUV itself: loads assets/auv/auv.urdf into PyBullet, exposes its state
(position/orientation/velocity/energy), and translates high-level control
commands (forward/lateral/vertical/yaw) into forces via the simplified
physics model in physics.py.

Pipeline implemented here:

    assets/auv/auv.urdf -> p.loadURDF() -> PyBullet body
                                                |
                                                v
                     AUV.get_state()  <-->  AUV.apply_control()
                                                |
                                                v
                          AUV.step_underwater_physics()  (physics.py, every step)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pybullet as p

from src.control.thruster_allocator import ThrusterAllocator
from src.control.thruster_model import ThrusterArray, specs_from_config
from src.dynamics.auv_dynamics import AUVDynamics, AUVParams, AddedMassCompensator
from src.dynamics.ocean_current import CurrentField, UniformCurrent
from src.simulation.physics import PhysicsParams, step_physics
from src.utils.transforms import quat_to_rot


@dataclass
class ControlInput:
    """One normalized thruster command, each component intended to lie in [-1, 1]."""
    forward: float = 0.0
    lateral: float = 0.0
    vertical: float = 0.0
    yaw: float = 0.0

    def clamp(self, limit: float = 1.0) -> "ControlInput":
        """Clamp every component to [-limit, limit]."""
        return ControlInput(
            forward=float(np.clip(self.forward, -limit, limit)),
            lateral=float(np.clip(self.lateral, -limit, limit)),
            vertical=float(np.clip(self.vertical, -limit, limit)),
            yaw=float(np.clip(self.yaw, -limit, limit)),
        )


@dataclass
class AUVState:
    """Snapshot of the AUV's full state at one simulation step (STEP 11)."""
    position: tuple            # (x, y, z)
    orientation_quat: tuple    # (x, y, z, w)
    orientation_euler: tuple   # (roll, pitch, yaw)
    linear_velocity: tuple     # (vx, vy, vz)
    angular_velocity: tuple    # (wx, wy, wz)
    energy_consumed: float
    remaining_battery: float

    def as_dict(self) -> dict:
        return {
            "position": self.position,
            "orientation": self.orientation_euler,
            "linear_velocity": self.linear_velocity,
            "angular_velocity": self.angular_velocity,
            "energy": self.energy_consumed,
        }


class AUV:
    """Simplified underwater vehicle: URDF body + thruster model + energy tracking."""

    def __init__(self, urdf_path: str, start_position: tuple, sim_config: dict, physics_client: int = 0):
        """
        Input:
            urdf_path: path to assets/auv/auv.urdf (config: simulation.urdf_path).
            start_position: (x, y, z) world position to spawn/reset at.
            sim_config: cfg["simulation"] dict (needs "time_step" plus the
                        "auv", "physics", "energy", "controller" sub-sections).
            physics_client: PyBullet client id this AUV lives in.
        Processing:
            Loads the URDF at `start_position`, builds PhysicsParams from
            config, reads energy/thrust-limit parameters from config, then
            calls reset() to zero velocity and energy.
        Output:
            None (constructs the AUV instance; access via .body_id, etc.).
        """
        self._client = physics_client
        self.urdf_path = urdf_path
        self.start_position = tuple(start_position)
        self.sim_config = sim_config
        self.physics_params = PhysicsParams.from_config(sim_config)

        energy_cfg = sim_config["energy"]
        self.thrust_coefficient = float(energy_cfg["thrust_coefficient"])
        self.battery_capacity = float(energy_cfg["battery_capacity"])

        controller_cfg = sim_config["controller"]
        self.max_thrust = float(controller_cfg["max_thrust"])
        self.max_yaw_torque = float(controller_cfg["max_yaw_torque"])

        self.body_id = p.loadURDF(
            self.urdf_path, basePosition=self.start_position, physicsClientId=self._client
        )
        self._energy_consumed = 0.0

        # Dynamics model selection (simulation.dynamics.model):
        #   "legacy" (default) -> original world-frame force + linear drag behaviour
        #   "6dof"             -> body-frame wrench -> thruster allocation -> thruster
        #                         lag/faults -> 6-DOF hydrodynamics (see src/dynamics)
        dyn_cfg = sim_config.get("dynamics", {}) or {}
        self.dynamics_model = str(dyn_cfg.get("model", "legacy")).lower()
        if self.dynamics_model not in ("legacy", "6dof"):
            raise ValueError(f"simulation.dynamics.model must be 'legacy' or '6dof', got {self.dynamics_model!r}")
        self._sim_time = 0.0
        self._thrust_wrench = np.zeros(6)
        self.dynamics = self.thrusters = self.allocator = self._added_mass = None
        self.thruster_available = None
        self.last_allocation = None
        if self.dynamics_model == "6dof":
            self._init_6dof(sim_config)
        self.reset()

    def _init_6dof(self, sim_config: dict) -> None:
        """Create the 6-DOF dynamics, thruster array and allocator (opt-in mode)."""
        params = AUVParams.from_config(sim_config)
        base = (sim_config.get("current", {}) or {}).get("base", [0.0, 0.0, 0.0])
        self.dynamics = AUVDynamics(params, UniformCurrent(base))
        specs = specs_from_config(sim_config)
        seed = int((sim_config.get("thrusters", {}) or {}).get("seed", 0))
        self.thrusters = ThrusterArray(specs, seed=seed)
        self.allocator = ThrusterAllocator(specs)
        # Which thrusters the allocator believes are usable. Starts all-True; fault
        # DETECTION (Phase 8) is what should clear entries -- an injected fault the
        # allocator has not been told about is deliberately NOT compensated.
        self.thruster_available = np.ones(self.thrusters.n, bool)
        if params.added_mass_enabled:
            self._added_mass = AddedMassCompensator(params.rigid_body_mass, params.added_mass)
        # PyBullet applies its own default damping (0.04) to every body; disable it
        # so the DragModel is the only source of damping in this mode.
        p.changeDynamics(self.body_id, -1, linearDamping=0.0, angularDamping=0.0,
                         physicsClientId=self._client)

    def set_current(self, current: CurrentField) -> None:
        """Replace the water-velocity field (6dof mode). Phase 6 supplies richer fields."""
        if self.dynamics is None:
            raise RuntimeError("set_current() requires simulation.dynamics.model = '6dof'")
        self.dynamics.current = current

    def reset(self, position: tuple | None = None) -> None:
        """
        Reset the AUV to a given (or its original) position with zero velocity/energy.

        Input: position: optional (x, y, z) override; defaults to self.start_position.
        Processing: p.resetBasePositionAndOrientation + p.resetBaseVelocity(0,0),
                    then zeroes the internal energy counter.
        Output: None.
        """
        pos = position if position is not None else self.start_position
        p.resetBasePositionAndOrientation(
            self.body_id, pos, [0, 0, 0, 1], physicsClientId=self._client
        )
        p.resetBaseVelocity(self.body_id, [0, 0, 0], [0, 0, 0], physicsClientId=self._client)
        self._energy_consumed = 0.0
        self._sim_time = 0.0
        self._thrust_wrench = np.zeros(6)
        if self.thrusters is not None:
            self.thrusters.reset()
        if self._added_mass is not None:
            self._added_mass.reset()

    def get_position(self) -> tuple:
        pos, _ = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._client)
        return pos

    def get_orientation(self) -> dict:
        """Returns {"quaternion": (x, y, z, w), "euler": (roll, pitch, yaw)}."""
        _, quat = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._client)
        euler = p.getEulerFromQuaternion(quat)
        return {"quaternion": quat, "euler": euler}

    def get_linear_velocity(self) -> tuple:
        lin, _ = p.getBaseVelocity(self.body_id, physicsClientId=self._client)
        return lin

    def get_angular_velocity(self) -> tuple:
        _, ang = p.getBaseVelocity(self.body_id, physicsClientId=self._client)
        return ang

    def get_energy_consumed(self) -> float:
        return self._energy_consumed

    def get_remaining_battery(self) -> float:
        """remaining = battery_capacity - energy_consumed, floored at 0 (STEP 16)."""
        return max(self.battery_capacity - self._energy_consumed, 0.0)

    def get_state_6dof(self) -> dict:
        """
        Ground-truth 6-DOF state from PyBullet.
        Returns {"eta": [x,y,z,roll,pitch,yaw], "nu": [u,v,w,p,q,r] (body frame),
        "rotation": 3x3 body->world}. Ground truth: for evaluation/logging only once
        the estimator exists (Phase 3), not for control.
        """
        pos, quat = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self._client)
        lin_w, ang_w = p.getBaseVelocity(self.body_id, physicsClientId=self._client)
        R = quat_to_rot(np.array(quat))
        nu = np.concatenate([R.T @ np.array(lin_w), R.T @ np.array(ang_w)])
        eta = np.concatenate([pos, p.getEulerFromQuaternion(quat)])
        return {"eta": eta, "nu": nu, "rotation": R}

    def get_state(self) -> AUVState:
        """Bundle position/orientation/velocity/energy into one AUVState snapshot."""
        orient = self.get_orientation()
        return AUVState(
            position=self.get_position(),
            orientation_quat=orient["quaternion"],
            orientation_euler=orient["euler"],
            linear_velocity=self.get_linear_velocity(),
            angular_velocity=self.get_angular_velocity(),
            energy_consumed=self._energy_consumed,
            remaining_battery=self.get_remaining_battery(),
        )

    def apply_control(
        self, forward: float = 0.0, lateral: float = 0.0, vertical: float = 0.0,
        yaw: float = 0.0, dt: float | None = None,
    ) -> None:
        """
        Convert a normalized thruster command into world-frame forces/torque
        and apply/queue it for the next simulation step, updating energy.

        Input:
            forward, lateral, vertical, yaw: control inputs, each intended
                to lie in [-1, 1] (STEP 12; out-of-range values are clamped).
            dt: timestep used for the energy integral; defaults to
                config.simulation.time_step.
        Processing:
            1. Clamp inputs to [-1, 1] (ControlInput.clamp).
            2. Scale forward/lateral/vertical by max_thrust (N) and yaw by
               max_yaw_torque (N*m).
            3. Apply the (forward, lateral, vertical) thrust vector directly
               in the WORLD frame (forward == world +X, lateral == world +Y).
               This matches WaypointController.compute_control(), which
               computes its "forward"/"lateral" commands from the raw
               world-frame X/Y position error (see controller.py's
               docstring) rather than a body-frame error -- so the force
               applied here must NOT be re-rotated by the AUV's current
               orientation, or the two frames disagree and the AUV can spin
               without making net progress toward the target. (Bug fixed:
               an earlier version rotated this vector by orientation, which
               is only correct for a body-frame controller.)
            4. p.applyExternalForce / applyExternalTorque, queued for the
               next p.stepSimulation() call.
            5. Energy model (STEP 15):
               energy += dt * thrust_coefficient * (|forward|+|lateral|+|vertical|+|yaw|)
               -- zero control therefore adds zero energy (STEP 10 test).
        Output:
            None.
        """
        control = ControlInput(forward, lateral, vertical, yaw).clamp(1.0)
        dt = dt if dt is not None else float(self.sim_config["time_step"])

        if self.dynamics_model == "6dof":
            # In 6dof mode the normalised command is a BODY-frame wrench request
            # (same scaling as legacy: max_thrust N per axis, max_yaw_torque N*m).
            self.apply_wrench(np.array([
                control.forward * self.max_thrust, control.lateral * self.max_thrust,
                control.vertical * self.max_thrust, 0.0, 0.0, control.yaw * self.max_yaw_torque,
            ]), dt)
            return

        world_force = (
            control.forward * self.max_thrust,
            control.lateral * self.max_thrust,
            control.vertical * self.max_thrust,
        )
        torque = (0.0, 0.0, control.yaw * self.max_yaw_torque)

        # BUG FIX (Phase 1): apply at the COM; WORLD_FRAME positions are world coordinates.
        p.applyExternalForce(
            self.body_id, -1, world_force, self.get_position(), p.WORLD_FRAME,
            physicsClientId=self._client,
        )
        p.applyExternalTorque(
            self.body_id, -1, torque, p.WORLD_FRAME, physicsClientId=self._client
        )

        self._energy_consumed += dt * self.thrust_coefficient * (
            abs(control.forward) + abs(control.lateral) + abs(control.vertical) + abs(control.yaw)
        )

    def apply_wrench(self, tau_body: np.ndarray, dt: float | None = None) -> None:
        """
        6dof mode: desired BODY-frame wrench [Fx,Fy,Fz,Mx,My,Mz] -> thruster
        allocation -> thruster dynamics/faults. The resulting thrust is applied to
        the rigid body in step_underwater_physics(). Call once per physics step.
        """
        if self.dynamics_model != "6dof":
            raise RuntimeError("apply_wrench() requires simulation.dynamics.model = '6dof'")
        dt = dt if dt is not None else float(self.sim_config["time_step"])
        self.last_allocation = self.allocator.allocate(tau_body, self.thruster_available)
        forces = self.thrusters.step(self.last_allocation.commands, dt)
        self._thrust_wrench = self.thrusters.wrench()
        # Energy proxy: sum of |actual thrust| / max thrust (not a battery model; see Phase 8).
        self._energy_consumed += dt * self.thrust_coefficient * float(
            np.sum(np.abs(forces) / self.thrusters.max_thrust))

    def step_underwater_physics(self) -> dict:
        """
        Apply this step's underwater forces.
        legacy: simplified buoyancy + linear drag (physics.step_physics).
        6dof:   thrust wrench + restoring + drag relative to the current (+ lagged
                added-mass reaction), all converted to world frame and applied at the COM.
        """
        if self.dynamics_model != "6dof":
            return step_physics(self.body_id, self.physics_params, physics_client=self._client)

        dt = float(self.sim_config["time_step"])
        s = self.get_state_6dof()
        R, nu = s["rotation"], s["nu"]
        hydro = self.dynamics.hydro_wrench(s["eta"][:3], R, nu, self._sim_time, include_weight=False)
        wrench = hydro + self._thrust_wrench
        if self._added_mass is not None:
            wrench = wrench + self._added_mass.reaction(nu, dt)
        com = tuple(s["eta"][:3])
        p.applyExternalForce(self.body_id, -1, tuple(R @ wrench[:3]), com, p.WORLD_FRAME,
                             physicsClientId=self._client)
        p.applyExternalTorque(self.body_id, -1, tuple(R @ wrench[3:]), p.WORLD_FRAME,
                              physicsClientId=self._client)
        self._sim_time += dt
        return {"hydro_wrench": hydro, "thrust_wrench": self._thrust_wrench.copy(),
                "net_buoyancy_n": self.dynamics.params.buoyancy.net_buoyancy}

    def stop(self) -> None:
        """Zero thrust for this step (STEP 17; still subject to buoyancy/drag/momentum)."""
        self.apply_control(0.0, 0.0, 0.0, 0.0)

    def move_forward(self, magnitude: float = 1.0) -> None:
        self.apply_control(forward=magnitude)

    def move_backward(self, magnitude: float = 1.0) -> None:
        self.apply_control(forward=-magnitude)

    def move_left(self, magnitude: float = 1.0) -> None:
        self.apply_control(lateral=-magnitude)

    def move_right(self, magnitude: float = 1.0) -> None:
        self.apply_control(lateral=magnitude)

    def move_up(self, magnitude: float = 1.0) -> None:
        self.apply_control(vertical=magnitude)

    def move_down(self, magnitude: float = 1.0) -> None:
        self.apply_control(vertical=-magnitude)

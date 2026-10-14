# Dynamics (Phase 2) -- simplified 6-DOF model

`eta = [x,y,z,roll,pitch,yaw]`, `nu = [u,v,w,p,q,r]` (body frame).
`(M_RB + M_A) nu_dot + C_RB nu = tau_thrust + tau_restoring + tau_drag(nu_r)`, `nu_r = nu - R^T v_current`.

* `src/dynamics/buoyancy.py` -- constant buoyancy + centre-of-buoyancy offset (passive roll/pitch stability).
* `src/dynamics/drag.py` -- diagonal linear + quadratic drag on velocity relative to the water.
* `src/dynamics/ocean_current.py` -- `CurrentField` interface; only a uniform constant current so far (spatial/temporal field = Phase 6).
* `src/dynamics/auv_dynamics.py` -- parameters, force terms, a self-contained NumPy integrator (tests/reference), and
  `AddedMassCompensator` (PyBullet cannot do added mass; reproduced by a one-step-lag reaction force, stable while M_A/M < 0.8).
* `src/control/thruster_model.py` -- 6 logical thrusters (URDF has none): limits, first-order lag, seeded noise, DEAD/DEGRADED/STUCK faults.
* `src/control/thruster_allocator.py` -- `tau = B u`, bounded least squares, re-allocation around thrusters known to be failed.
  Rank 5: roll has no actuator.

Modes: `simulation.dynamics.model: legacy` (default, original behaviour + COM bug fix) or `6dof`
(`AUV.apply_control` becomes a body-frame wrench request -> allocator -> thrusters -> hydrodynamics).

Limitations: illustrative coefficients; diagonal added mass without Coriolis; no lift/cross-coupling/waves; thrust linear in command;
constant buoyancy; 6dof energy is a thrust-sum proxy, not a battery model.

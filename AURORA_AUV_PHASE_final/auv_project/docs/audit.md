# Repository audit and phase plan (Phase 1)

## Architecture as found
`data_processing` (bathymetry load/clean/synthetic) -> `terrain` (slope, roughness, curvature, synthetic labels, Random Forest)
-> `mapping` (occupancy grid, inflation, cost map) -> `simulation` (PyBullet world from the grid, URDF hull, P controller,
3-ray proximity sensor, metrics) -> `dashboard` (Flask; PyBullet runs in a child process, telemetry via JSONL).

## Works
Bathymetry/terrain/mapping pipeline and tests; RF classifier (`models/terrain_classifier.pkl`); PyBullet world, AUV, P waypoint
following; Flask dashboard process architecture (core simulation already survives dashboard restarts).

## Missing
A* (`src/planning/` is empty), EKF, noisy IMU/DVL/depth sensors, multi-ray sonar, PID, thrusters, ocean current,
dynamic obstacles, battery model, state machine, CNN, GRU, experiment runner, most dashboard tabs.

## Problems found
1. **Force-application bug (fixed in Phase 1).** `applyExternalForce(..., [0,0,0], WORLD_FRAME)` applies the force at the world
   origin; PyBullet treats that argument as a world coordinate, producing a large spurious torque. The saved Week 5 log
   shows roll up to 3.1 rad and pitch up to 1.57 rad: the "successful" mission was flown by a tumbling hull.
2. PyBullet's default body damping (0.04) silently stacked on the custom drag (neutralised in `6dof` mode).
3. Controller commands are world-frame forces (not body-frame); kept for legacy mode.
4. `controller_backup.py` was byte-identical to `controller.py` (removed). No `.gitignore`; `__pycache__` shipped.
5. `buoyancy_ratio: 0.068` is a calibration hack: a 20 kg body with the displaced volume of a ~294 kg solid cylinder.
6. `pytest` could not find `src` without `python -m pytest`; fixed via `pytest.ini`.
7. No A*: docs describe it but only config keys exist.

## Phase plan
1 audit/cleanup (done) | 2 6-DOF + thrusters (done) | 3 sensors + EKF | 4 occupancy/cost/A* | 5 local avoidance + replanning |
6 dynamic env + currents | 7 PID + state machine | 8 battery/failures/return-home | 9 CNN | 10 GRU | 11 missions | 12 dashboard | 13 experiments/docs

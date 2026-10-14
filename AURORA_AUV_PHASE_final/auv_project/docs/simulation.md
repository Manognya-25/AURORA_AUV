# Week 5 — PyBullet AUV Simulation

This document describes the simplified physics-based simulation built in
Week 5: a PyBullet world that consumes Week 4's occupancy grid and cost
map, an AUV with a basic thruster/energy model, and a proportional
waypoint controller. It does **not** perform path planning — that begins
in Week 6.

```text
                 WEEK 4
        Environmental Representation
                    |
       occupancy_grid_inflated.npy
       navigation_cost_map.npy
                    |
                    v
             PyBullet Environment
                    |
           +--------+--------+
           v                 v
         Seabed           Obstacles
           |                 |
           +--------+--------+
                    v
                   AUV
                    |
           +--------+---------+
           v        v         v
        Sensors   Physics   Energy
           |        |         |
           +--------+---------+
                    v
              AUV State
                    |
                    v
        Basic Waypoint Control
                    |
                    v
                 WEEK 6
              A* Planning
```

## 1. Purpose

PyBullet provides real rigid-body physics (mass, inertia, contacts,
gravity) essentially for free, which is what Weeks 6–9's navigation
algorithms need to be tested against something more realistic than a
pure kinematic grid-walk. It's open source, has a Python API, and needs
no additional simulator licenses or external services.

## 2. AUV model

`assets/auv/auv.urdf` defines a single rigid-body link: a cylinder hull
(mass 20 kg, length 1.5 m, radius 0.25 m — matching
`configs/config.yaml: simulation.auv`). URDF has no native capsule
primitive, so a plain cylinder stands in for the "capsule or cylinder"
called for in the spec; a true rounded-hull capsule would need a mesh,
which is unnecessary fidelity here. The cylinder is rotated 90° about Y
so its long axis lies along the link's local X axis — "forward" thrust
then points along the vehicle's nose, as expected for a torpedo-shaped
AUV. Inertia values follow the closed-form solid-cylinder formulas
(`Ixx = m·r²/2` about the roll/long axis, `Iyy = Izz = m·(3r²+L²)/12`
about pitch/yaw) — see the comment block in the URDF file itself.

**Keep the URDF and config in sync.** URDF can't read YAML, so
`simulation.auv.{mass,length,radius}` in `config.yaml` and the
`<mass>`/`<cylinder>`/`<inertia>` values in `auv.urdf` are two copies of
the same numbers. If you change one, update the other (and re-derive the
inertia and the buoyancy calibration in section 3).

## 3. Physics (`src/simulation/physics.py`)

PyBullet gives dry rigid-body dynamics (gravity + contacts) but has no
built-in fluid model, so three forces are added manually every step:

- **Gravity** — set globally via `p.setGravity(0, 0, simulation.gravity)`;
  not computed in `physics.py`, PyBullet applies it automatically.
- **Buoyancy** — `F_b = water_density · |g| · displaced_volume ·
  buoyancy_ratio`, applied straight up. The AUV's displaced volume is
  approximated as a solid cylinder matching the URDF geometry
  (`π·r²·L ≈ 0.2945 m³`). At `buoyancy_ratio = 1.0` this produces
  ~2886 N — about **14.7× the AUV's 196 N weight**, which would send it
  rocketing to the surface. `buoyancy_ratio` is therefore calibrated to
  `0.068` (`weight / (water_density · |g| · volume)`) so buoyancy ≈
  weight, i.e. approximately neutral buoyancy. **This value must be
  recomputed if `auv.mass`/`length`/`radius` change** — see the comment
  in `config.yaml`.
- **Drag** — simplified *linear* drag, `F = -k·v` per linear/angular
  velocity component (`physics.linear_drag` / `angular_drag`), not the
  physically more accurate quadratic `v|v|` drag.

All three are applied together each step via `step_physics()`
(`p.applyExternalForce`/`applyExternalTorque`, link index -1, world
frame).

## 4. Thrusters (`src/simulation/auv.py: AUV.apply_control`)

Control inputs are four normalized values in `[-1, 1]`: `forward`,
`lateral`, `vertical`, `yaw`. They're scaled by `controller.max_thrust`
(N) and `controller.max_yaw_torque` (N·m), rotated from the AUV's body
frame into the world frame using its current orientation quaternion (so
"forward" always means "forward relative to where the AUV currently
points"), and applied via `p.applyExternalForce`/`applyExternalTorque`.
Convenience wrappers (`move_forward`, `move_backward`, `move_left`,
`move_right`, `move_up`, `move_down`, `stop`) call `apply_control` with
one axis set.

## 5. Energy model

Every control action adds to a running energy counter:

```text
energy += dt · thrust_coefficient · (|forward| + |lateral| + |vertical| + |yaw|)
```

`thrust_coefficient` (`simulation.energy.thrust_coefficient`) is a
configurable constant, not calibrated battery chemistry — the goal is a
**consistent metric for comparing navigation strategies** in later weeks,
not a real energy budget. `AUV.get_remaining_battery()` returns
`max(battery_capacity - energy_consumed, 0)`, floored at zero.

## 6. Environment (`src/simulation/environment.py`)

**Seabed** — one static box sized to the full mapped area
(`rows·resolution × cols·resolution`), positioned so its top face sits at
`simulation.seabed_z`. A bounded box is used instead of PyBullet's
infinite `p.GEOM_PLANE` so the seabed matches the actual surveyed extent.

**Obstacles** — built directly from Week 4's
`occupancy_grid_inflated.npy` (the *inflated* grid, so the AUV's safety
margin is already physical geometry, not recomputed here). Since the 2D
occupancy grid has no per-cell height, each obstacle cell is extruded as
a **full water-column pillar** from the seabed up to just below the
surface (`simulation.obstacle_height`) — a standard simplification for
2D-grid-based marine obstacle avoidance, and consistent with what the
Week 6 planner will assume when it reads the same grid.

Spawning one PyBullet body per obstacle *cell* doesn't scale (thousands
of bodies on a dense 100×100 grid). `_merge_obstacle_runs()` instead
does row-wise run-length encoding, merging horizontally-contiguous
obstacle cells into single boxes — on the current project's data this
collapses ~2,600 obstacle cells into under 300 bodies (about a 9×
reduction), while staying simple. A full 2D (quadtree/rectangle) merge
would do better and is noted as future work.

**Coordinate system** — grid column → world X, grid row → world Y, both
scaled by `resolution` (metres/cell): `world_x = col·resolution`,
`world_y = row·resolution`. World Z is a separate, independent vertical
axis (the AUV cruises at a fixed operating depth given by
`start_position`/`goal_position`, not tied to the terrain depth at each
cell). `grid_to_world()`/`world_to_grid()` implement and invert this
mapping; round-trip correctness is covered by `test_grid_world_round_trip`.

**Bounds/safety** — `Bounds` (STEP 25) is derived from the grid extent
plus `simulation.bounds_margin`, with Z bounded by `[seabed_z, 0.0]`
(surface). If the AUV leaves these bounds, the run loop calls
`auv.stop()`.

## 7. Week 4 integration

Nothing from Week 4 is recomputed. `Environment.__init__` loads, as-is:

- `data/processed/mapping_metadata.json` — grid dimensions, resolution,
  and the free/obstacle/unknown occupancy codes (never hard-coded).
- `data/processed/occupancy_grid_inflated.npy` — becomes the physical
  obstacle geometry (section 6).
- `data/processed/navigation_cost_map.npy` — loaded onto
  `Environment.cost_map` and exposed, but **not used for control in
  Week 5**. It's there for Week 6's A* planner to read.

**Unknown cells** (`occupancy_values.unknown = 2`): for this project no
obstacle is spawned for unknown cells (unknown means "not yet observed,"
not "known obstacle"). Week 4 classified the entire surveyed grid, so no
cell is actually unknown yet in the current data; this becomes relevant
once frontier exploration (Week 7) starts building the map incrementally.

## 8. Controller (`src/simulation/controller.py`)

A proportional (P) controller, not PID or MPC. Two implementations share the same
interface: the plain `WaypointController` and the Week-9 `AdaptivePController`
(`src/simulation/adaptive_controller.py`), which is selected whenever
`simulation.controller.adaptive.enabled` is true (the default).

- **Position**: `error = target - current`; `control = kp_position ·
  error`, normalized into `[-1, 1]` by dividing by `max_thrust`.
- **Yaw**: `desired_yaw = atan2(target_y - y, target_x - x)`; the yaw
  error is wrapped into `[-π, π]` via `atan2(sin(Δ), cos(Δ))` so the
  controller always turns the short way, then scaled by `kp_yaw /
  max_yaw_torque`.
- **Simplification**: forward/lateral commands use the world-frame X/Y
  error directly rather than rotating it into the AUV's body frame using
  its current heading. This keeps the controller a pure P-controller as
  specified; a heading-aware, body-frame controller is a natural
  candidate for Week 9's "adaptive controller."
- `reached_waypoint()` returns true once Euclidean distance to the
  target is under `controller.waypoint_tolerance` (default 0.5 m).
- **Terminal approach to the GOAL** (the final waypoint): the mission only
  completes once the vehicle is within `controller.goal_tolerance`
  (default 0.05 m) of the goal, so a finished run really ends on the goal
  x/y/z the dashboard/ config asked for instead of stopping anywhere in
  the 0.5 m waypoint radius. That last phase adds two things the
  mid-mission law does not have, both bounded by
  `controller.goal_settle_timeout_s` (default 60 s of simulated time,
  after which the run completes anyway and the residual is reported):
  velocity feedback `-kd_velocity · v` (`controller.kd_velocity`, so the
  AUV settles instead of hunting around the goal) and a feed-forward
  `vertical_trim` that cancels the deliberately slightly positive legacy
  buoyancy (see section 3), because a P-only law would otherwise sit at a
  constant depth offset. Waypoint following itself stays P-only.

### 8.1 Adaptive proportional control (Week 9)

`AdaptivePController` does not change the *law* -- the command is still
`kp_effective * error / max_thrust`, in the same world frame and the same
`[-1, 1]` convention -- it changes how `kp_effective` is chosen. That keeps
the vehicle, dynamics and command path untouched, and makes the change
verifiable against the plain controller.

- **Gain schedule (no state).** The gain is a function of the horizontal
  error: `kp_position` at/above `adaptive.far_m`, rising linearly to
  `kp_position * adaptive.gain_boost` at/below `adaptive.near_m`. Mid-mission
  (far from the target) the command is therefore identical to the plain P
  law; only the final approach gets the extra authority that overcomes drag
  and the positive legacy buoyancy, so the AUV settles *on* the set-point
  instead of stopping short of it.
- **Self-tuning (state, needs `dt`).** The controller watches its own closing
  speed. While stalled (closing speed below `adaptive.stall_progress_mps`)
  the gain ramps up at `adaptive.adapt_rate_per_s`; while tracking well
  (closing speed at/above `adaptive.healthy_progress_mps`, or already inside
  `adaptive.settle_deadband_m`) it relaxes back at `adaptive.relax_rate_per_s`.
  It is clamped to `[1, adaptive.gain_boost]` in both directions, so it can
  never wind up without bound and the gain sequence is reproducible. A jump
  larger than `adaptive.target_change_jump_m` is treated as a new target and
  re-seeds the history rather than adapting on a phantom error change.
- **Heading.** The yaw gain is scheduled the other way round: full `kp_yaw`
  when nearly aligned (`adaptive.yaw_near_rad`), floored at
  `adaptive.yaw_gain_min * kp_yaw` for large heading errors
  (`adaptive.yaw_far_rad`), so the vehicle does not spin while the horizontal
  error is still being reduced.
- The effective gains of the last command are reported in the telemetry
  `controller.adaptive` block (`kp_position_effective`,
  `kp_position_multiplier`, `schedule_multiplier`, `self_tuning_gain`,
  `kp_yaw_effective`), so the adaptation can be watched live rather than
  inferred. With `adaptive.enabled: false` the two controllers are identical.
- **Gain tracking.** The controller also keeps a cumulative record of its own
  adaptation (`controller.adaptive.tracking`, a fresh copy per command): which
  way the self-tuning pushed on that step (`action`) and the measured quantity
  that chose it (`reason`, e.g. *stalled (closing 0.000 m/s)*), the closing
  speed, the simulated time spent ramping / relaxing / holding, boosted (gain
  above nominal) and at the clamped cap, the tracked peak and mean active
  multiplier (with the adaptation time the peak was reached at), and the
  number of adaptation episodes (times the gain started ramping again) and
  re-seeds. Every counter is updated in the very branch that moved the gain,
  so the tracked direction can never disagree with the gain it describes; a
  step without a usable `dt` does not adapt and is not counted. The record is
  reset with `reset_adaptation()` and exported as the `ctrl_track_*` columns.

**Waypoint sequencing** (`run_simulation.py`): the AUV is driven through
`simulation.waypoints` in order (START → each waypoint → GOAL, the goal
is always appended as the final waypoint if not already last). This is
still just a hand-picked list, not planning — each entry was manually
checked against `occupancy_grid_inflated.npy` to confirm it falls in free
space (see `configs/config.yaml` comments). Week 6's A* replaces this
list with a computed one.

## 8b. Sensors (`src/simulation/sensors.py`)

Three single-ray proximity readings — front (+X body frame), left (+Y),
right (−Y) — via `p.rayTest()`, each returning the distance to the
nearest obstacle or `sensors.max_range` if nothing is hit. This is **not**
a sonar cone/beam model; it demonstrates that the AUV can obtain basic
environmental information (Week 5 spec, STEP 8), and is logged every step
in `results/logs/week5_simulation_log.csv` for later weeks (e.g. reactive
local avoidance) to build on if useful.

## 9. Metrics (`src/simulation/metrics.py`)

Only measured or clearly-labeled-as-a-proxy values are reported — nothing
is fabricated:

- `total_simulation_steps`, `simulation_duration_seconds_simulated`
  (steps × dt) and `simulation_wall_time_seconds` (both reported, clearly
  distinguished).
- `trajectory_length_m` — the real measured path length (sum of
  consecutive-step distances), not a straight-line estimate.
- `final_position`, `final_distance_to_goal_m`, `waypoints_reached` /
  `total_waypoints`, `goal_reached`, `collision_count`.
- `movement_effort_proxy` — since Week 5 has no calibrated thrust/energy
  model, this integral of `|control components|` is explicitly labeled
  (in the JSON output itself) as **not a physical energy measurement**,
  only a relative proxy for comparing runs. Per the Week 5 spec: "if
  energy/thrust is not physically modeled, do not claim real energy
  consumption."

Saved to `results/logs/week5_simulation_metrics.json`, alongside
`week5_simulation_log.csv` (full per-step log) and `week5_trajectory.npy`
(just the x/y/z columns, for quick reloading).

## 10. Limitations

This is a teaching/demonstration simulation, not a hydrodynamics engine.
Explicitly out of scope for Week 5:

- Simplified hydrodynamics (no added mass, lift, or vortex shedding)
- Linear, not quadratic, drag
- Simplified, uncalibrated buoyancy (tuned only for approximate neutral buoyancy)
- No water currents
- No sensor noise or realistic sensor models
- No dynamic (moving) obstacles
- No real AUV hardware in the loop
- No PID/MPC control (proportional only; the Week-9 adaptive layer schedules and self-tunes the *proportional* gain, it is not a PID or MPC law)
- No path planning (single manually-specified waypoint only)

## 11. Future work

- **Week 6** — A* path planning over `navigation_cost_map.npy`, feeding a
  sequence of waypoints to this same `WaypointController`.
- **Week 7** — Frontier exploration; `unknown` cells become meaningful
  once the map is built incrementally rather than fully known upfront.
- **Week 8** — Reinforcement-learning agent as an alternative to A*/frontier control.
- **Week 9** — DONE for the adaptive layer: `AdaptivePController` (gain scheduling + bounded self-tuning) replaces the plain P law by default (section 8.1). A full PID or MPC law remains future work.
- Sensor simulation (sonar/camera models, noise).
- Dynamic obstacles and replanning.
- A true 2D rectangle/quadtree merge for obstacle geometry (fewer, larger
  bodies than the current row-wise run-length approach).

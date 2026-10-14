# Phase 5 — Autonomous navigation and path planning

Everything here is **simulation**. Numbers are illustrative parameters, not a calibrated vehicle.

```
 PyBullet AUV ─► sensors (IMU/DVL/depth/sonar) ─► 15-state EKF ──► EstimatedState ─┐
                                   │                                              │
                                   └─► sonar ─► occupancy + cost map (Phase 4) ───┤
                                                                                  ▼
                              AutonomousNavigator.update(sim_time, estimate, mapping, sonar)
                                   │  PLAN ─► A* (global_planner) ─► smoothing ─► PathData
                                   │  FOLLOW_PATH ─► PathFollower (lookahead + velocity loop, depth, yaw)
                                   │  AVOID_OBSTACLE ─► LocalAvoider detour that rejoins the global path
                                   │  REPLAN ─► (rate-limited, only when the map invalidates the path)
                                   ▼
                     {forward, lateral, vertical, yaw} ─► AUV.apply_control (unchanged Phase 2 dynamics/thrusters)
```

## Ground-truth rule
`AutonomousNavigator.update()` has exactly four inputs: simulation time, the **EKF estimate**, the **mapping system**
(cost map, built from the estimate and the sonar) and the latest **sonar measurement**. `src/planning/` imports nothing from
the simulation, PyBullet or the sensors' truth path (`tests/test_navigation_leak.py` enforces this with an AST scan, a
signature check, a check of the call site in `run_simulation.py`, and a closed-loop run with a deliberately *wrong* estimate).
Truth appears only in `debug_ground_truth` (labelled) and in evaluation fields (`goal_reached_true_position`,
`final_true_goal_distance_m`, collision counts).

## Modules (`src/planning/`)
| file | role |
|---|---|
| `config.py` | `planning:` / `navigation:` config dataclasses with validation |
| `grid_utils.py` | `PlanningGrid` (traversable mask + cost from the Phase-4 cost map), world↔grid via the mapping module's own `GridGeometry`, exact supercover path collision checker |
| `astar.py` | A* (4/8-connected, octile/euclid/manhattan, no corner cutting, cost-weighted, deterministic) |
| `smoothing.py` | collinear simplification + cost-aware line-of-sight shortcutting (only removes vertices, re-verified) |
| `global_planner.py` | start (EKF) → goal; unknown policy; start snapping; independent collision check; timing |
| `path_data.py`, `path_follower.py` | arc-length path with depth profile; lookahead + velocity controller using only the estimate |
| `local_avoidance.py` | deterministic detour on a local window that rejoins the global path; sonar forward-clearance |
| `replanning.py` | path validity (blocked / significant cost change) monitor |
| `navigator.py` | the state machine, telemetry, operator commands |

## Planning details
* Cost of a move = step length × mean cell cost. The cost map already carries inflation and clearance costs; occupied and
  inflated cells are **never** traversable.
* `planning.extra_clearance_m` (default 0.5 m) widens the keep-out band beyond the mapped inflation radius, as an allowance
  for estimator drift (see limitations).
* Unknown cells: `planning.unknown_policy` = `inherit` (use `mapping.costs.unknown_policy`), `allow`, or `block`.
  A goal in a blocked cell is a planning failure; it is never moved silently.
* Start in a non-traversable cell → snapped to the nearest traversable cell within `start_snap_radius_m`, else `start_blocked`.
* Smoothing never adds or moves a vertex; every new segment is verified on the same mask and the result is re-checked.

## Replanning policy (never every time step)
The path is re-examined only when `mapping.version` changed and at least `validate_interval_s` has passed.
* **blocked** within `avoid_trigger_distance_m` → local detour (`AVOID_OBSTACLE`); farther away → global replan.
* **significant cost change** (remaining path cost grew by `cost_increase_ratio` and `cost_increase_min`) → replan.
* **goal change / operator request / unknown-policy change** → forced replan.
* Replans are rate-limited (`min_replan_interval_s`) and budgeted (`max_replans`); a deferred replan is retried, not lost.
* Failures: `max_plan_failures` consecutive planning failures, no progress for `stuck_timeout_s` (repeated), mission timeout,
  estimate unavailable for `estimate_timeout_s` → `FAILED` with the reason.

## Running it
```
# config.yaml: navigation.enabled: true     (default false: Phase 4 waypoint mission, unchanged)
python tools/run_phase5_mission.py            # real PyBullet, headless, prints the mission report
python tools/run_phase5_mission.py --gui      # PyBullet window; green = global path, orange = detour
python dashboard.py                           # "Autonomous navigation: on" under Next start, then Start
```
`environment.explicit_obstacles: [...]` (same spec as the scenario-internal explicit obstacles) adds fixed obstacles to any scenario.

## Gains the follower is applying (telemetry)
Autonomous navigation runs have no adaptive P controller, so the follower reports its own gains every command as
`controller.follower` in the snapshot (built by `PathFollower.gain_report()`, nothing inferred on the dashboard):

* the **loop gains** in the units they are applied in — `kv` (N per m/s), `kp_yaw` / `kd_yaw` (N·m per rad, N·m per
  rad/s), `kp_depth` (m/s per m), `ki_depth` (N per m·s), `station_kp` (m/s per m), `cruise_speed_mps`;
* the live **speed schedule**: the commanded speed as a fraction of cruise (`speed_factor`) and the real factors that
  produced it — distance to the goal (`goal_factor`), heading error (`heading_factor`) and the sonar/avoidance scale
  (`sonar_factor`) — plus the live `desired_speed_mps`, `lookahead_m`, `cross_track_m`, `heading_error_rad`,
  `depth_error_m`, the clamped `depth_integral_n` and the normalised `command`;
* the **tracked schedule history** (`tracking`): commands and simulated time seen, time in each mode
  (following / holding / station-keeping) and in each speed state (full speed / slowed / braked), the now / mean /
  slowest speed factor and the slowest heading and sonar factors, plus the slow-down and brake EPISODES (rising edges,
  so one long slow section counts once). It is counted in the follower, step by step, and reset only when a NEW
  navigation mission begins (`_begin` / `reset`) — a replan, which merely swaps the path, does not clear it.

This is the block the dashboard's gain panel and both gain charts read (see `docs/dashboard.md`), and the CSV records
it as the `follow_*` columns.

## Dashboard / API
`GET /api/navigation`, `/api/planner`, `/api/path` (read the live snapshot). `POST /api/navigation/goal|start|cancel|replan|unknown_policy`
act on the running navigator through the command queue and are acknowledged in `command_log`. Goals are validated against
the live map (outside the map, underwater violation, occupied/inflated cell → 422 with the reason).

## Known limitations
* **EKF drift is the dominant error.** There is no absolute position fix; yaw is unobservable. After ~60 s of manoeuvring
  the estimate is 1–2 m off truth. The map and the vehicle are both in the estimate frame, so local geometry stays consistent,
  but a wall that was mapped earlier can be 1+ m displaced from where it physically is when the vehicle comes back to it.
  The keep-out margin (mapped inflation 0.6 m + 0.5 m) covers ~1 m. In a stand-in run with a closed box of walls around the
  goal, the vehicle sidled along an obstacle outside the ±45° sonar field of view with 1.7 m of drift and clipped it
  (19 evaluation-collision samples). Long wall-hugging missions can therefore collide.
* Sonar is forward-looking (±45° azimuth), so obstacles beside the vehicle are not seen.
* The planner is optimistic about unknown space (default policy), so it replans as obstacles are discovered; long unbroken
  walls cost many replans (budget 40).
* A* is pure Python: ~10–150 ms per plan on the 240×240 map. Planning runs inside the simulation step (one hitch per replan).
* 6-DOF dynamics: the follower converts the world-frame force to the body frame with the estimated yaw (unit-tested), but a
  closed-loop 6-DOF mission has not been verified.
* Depth is a linear profile from start to goal depth; there is no 3-D planning around obstacles above/below the sonar slice.
* The run loop ends `linger_after_terminal_s` (2 s) after `GOAL_REACHED`/`FAILED`; raise it to keep a dashboard session
  alive for interactive goal changes.

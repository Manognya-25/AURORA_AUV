# Online mapping (Phase 4)

`SENSE -> ESTIMATE -> PERCEIVE -> MAP -> COST MAP -> TELEMETRY`

The mapper (`src/mapping/map_manager.MappingSystem`) is handed exactly three things per call:
`process(sim_time, sonar_measurement, ekf_estimate)`. It never receives PyBullet, the ground-truth
snapshot, the environment description or obstacle lists. See "Ground-truth isolation" below.

## Frames and conventions (identical to Phases 2/3)
| Item | Convention |
|---|---|
| world / map frame | right-handed, +X forward at yaw 0, +Y left of +X, **+Z up**, depth = -z. The map frame *is* the world frame; the grid is a horizontal slice indexed by world (x, y). |
| body frame | X forward, Y left, Z up; quaternion (x, y, z, w) body -> world |
| sonar frame | body axes rotated by the mounting rotation, offset by the mount position (`sensors.sonar.mount_*`) |
| yaw | rotation about +Z, 0 = body X along world +X, positive counter-clockwise |
| grid | `array[row, col]`, col <-> X, row <-> Y (row grows with +Y: **plot with origin='lower'**) |
| origin | world (x, y) of the lower-left **corner** of cell (0, 0) (`mapping.origin_x_m/origin_y_m`) |
| cell centre | +0.5 cell; `world->grid` uses `floor()` |
| default grid | 60 m x 60 m at 0.25 m, origin (-5, -5) -> 240 x 240 cells |

Per-ray frame chain: `d_world = R_est @ R_mount @ d_sensor`, `origin = p_est + R_est @ mount_position`,
`point(t) = origin + t * d_world`. `R_est`, `p_est` come from the **EKF estimate**.

## Occupancy representation
Log-odds per cell, `0` = no information. Parameters (`mapping:` in `configs/config.yaml`, all simulation
parameters, not calibrations of real hardware):

| key | default | meaning |
|---|---|---|
| `log_odds_occupied` | +1.0 | added at a sonar hit cell |
| `log_odds_free` | -0.5 | added to cells a valid ray passed through |
| `log_odds_min/max` | -4 / +4 | clamp (P ~ 0.018 / 0.982) so cells never saturate and stay revisable |
| `occupied_threshold` | 0.7 | P >= this -> OCCUPIED |
| `free_threshold` | 0.3 | P <= this -> FREE |
| otherwise | | UNKNOWN |

A new map is entirely UNKNOWN. A single free observation (-0.5, P = 0.38) is not yet FREE; two are
(-1.0, P = 0.27). Conflicting evidence cancels and cells can recover (tests: repeated free, repeated
occupied, conflicting, unknown initial state).

## Sonar -> map semantics (documented contract, each row is a test)
| situation | effect |
|---|---|
| healthy detected hit inside the slice | cells before the hit FREE, hit cell OCCUPIED |
| cells behind a hit | untouched (UNKNOWN) |
| valid ray, no detection / max range | cells along the ray up to `min(sensor max range, max_update_range_m)` get weaker FREE evidence (`no_return_free_weight`); nothing beyond; **no obstacle invented** |
| dropped ray / min-range blanked ray | no update at all (not free, not occupied) |
| scan invalid or sensor FAILED | whole scan ignored, counted in `scans_rejected`, map unchanged |
| hit beyond `max_update_range_m` | treated as no-return up to that range |
| hit outside the horizontal slice (`slice_half_height_m` around the *estimated* z, e.g. seabed seen by a downward ray) | free only while inside the slice; no obstacle |
| non-finite pose, or estimate older than `max_pose_age_s` | scan rejected with a reason |
| faster than `update_rate_hz` / duplicate scan | skipped, counted |

Sonar noise, dropout and detection probability come from the Phase 3 sensor model; the mapper consumes
the measurement *after* that model and never replaces it with perfect data. The sonar measurement now
also carries `ray_valid` (per-ray validity), and its mounting extrinsics (`mount_position`,
`mount_rotation`) so the mapper uses the real mounting transform.

## Unknown space
Never treated as free. The cost map has its own UNKNOWN class with a configurable policy
(`mapping.costs.unknown_policy`: `cost` = traversable at `costs.unknown`; `blocked` = non-traversable).

## Terrain
An *optional* a-priori terrain-risk layer (`mapping.terrain_risk.enabled`, default off) is built from the
existing bathymetry + Random Forest classifier. It is a pre-dive chart, kept separate from sensor
occupancy, only adds cost to non-obstacle cells, and the obstacle class risk must stay 0 so the prior
cannot smuggle in obstacle locations. The Random Forest model is unchanged.

## Telemetry
`mapping`: enabled, map_resolution, map_size, map_origin, known/unknown/free/occupied cells,
coverage_percent, occupied_percent, last_update_time, update_count, estimated_map_x/y,
mapped_obstacle_clusters, scans_received/rejected/skipped. `sonar_mapping`: valid_returns, no_returns,
dropouts, rays_processed, cells_updated. `timing`: map and cost-map update times. All are computed from
the grid/mapper at publish time; none are stored or simulated.

## Ground-truth isolation (what is enforced)
* `tests/test_mapping_leak.py` AST-scans the online mapping modules: no import of PyBullet,
  `src.simulation`, sensor or EKF modules, and no identifier mentioning truth / world spec / ray tests.
* A subprocess test proves importing the mapper does not load PyBullet or `src.simulation`.
* A strict-estimate test proves the mapper reads only `time`, `position`, `rotation` from the estimate.
* Replay test: identical estimate + measurements with different ground truth -> identical map.
* Operational dashboard sections (`estimate`, `sensors`, `sonar`, `mapping`, `cost_map`) are byte-identical
  for different ground truth; truth appears only in `debug_ground_truth` (labelled) and the
  `environment` block (labelled GROUND TRUTH ENVIRONMENT).
* `src/mapping/map_plots.py` (post-run figures) is the only mapping-package file that accepts a truth
  path; it is read-only, labelled DEBUG ONLY, and a test proves nothing operational imports it.
* The Phase 3 controller still uses true state (unchanged, by design). That is simulation control, not
  mapping.

## Limits
2-D horizontal slice only (no 3-D map); the sonar model has no multipath/beam-pattern effects; the map
frame is fixed to the world frame and the EKF is not corrected by the map (no SLAM), so EKF drift
smears the map over long runs.

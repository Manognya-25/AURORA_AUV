# Procedural underwater environment (Phase 4)

`environment.scenario` selects the world. `legacy_week4` (default) is the exact Phase 1-3 world built from
`data/processed/*.npy`. Procedural scenarios are built by `src/simulation/world_generator.py` (pure NumPy,
seeded `numpy.random.Generator`, no global RNG) and `src/simulation/world_builder.py` (PyBullet).

| scenario | content |
|---|---|
| `open_terrain` | flat seabed, no obstacles |
| `sparse_obstacles` | gentle relief, a few pillars/walls, low boulders |
| `dense_obstacles` | many obstacles, mission corridor kept clear |
| `narrow_corridor` | two parallel walls along the mission line |
| `terrain_variation` | strong relief (ridges, depressions, slopes), almost no obstacles |
| `mixed_terrain_obstacles` | relief + pillars, walls, reef towers, a wreck-like structure |
| `mapping_validation` | fixed pillar/wall in sonar range of the default start, for mapping validation |

* Seabed: a height field sampled on a grid (rolling terrain, ridges, depressions, slopes, roughness), built as a PyBullet triangle mesh (`GEOM_MESH`)
  with collision *and* visual shapes. It is kept at least `min_seabed_clearance_m` below the shallowest
  mission z, so it can never rise into the mission depth.
* Obstacles: pillars, reef towers (several cylinders), walls, wreck hull + tower, boulders; each has
  collision and visual geometry, position, size and yaw.
* **Obstacle count = number of PyBullet bodies.** `environment.obstacles.count` (dashboard *Next start ->
  Obstacle count*, `run_visual_demo.py --obstacles`) is the exact number of obstacle bodies
  `world_builder.build_world()` creates: 10 / 30 / 53 requests 10 / 30 / 53 bodies, in every procedural
  scenario. A scenario's own fixed structures (the `mapping_validation` pillar/wall, the `narrow_corridor`
  walls) count towards that total, so the random placement fills the rest; a multi-body structure (reef
  tower, wreck) only fills the slots that are left. `Environment.describe()` reports
  `obstacle_count` (what was built), `obstacles_requested`, `obstacles_fixed`, `obstacles_placed` and
  `obstacle_kinds`, and the dashboard Environment panel shows *requested / actual*.
  On `legacy_week4` (world from the Week-4 map) an explicit count is **rejected** with a 422 instead of
  being silently ignored, and the field is disabled in the UI with that reason.
* Determinism: same scenario + seed + config -> identical world; different seed -> different world;
  global NumPy random state is untouched (all tested).
* Safety: nothing is placed on the start pose or within clearance of the mission polyline; structures do
  not overlap each other. Impossible configurations raise a clear error (and `POST
  /api/environment/configure` returns it as a 422).
* Overrides: `environment.seabed.*`, `environment.obstacles.{enabled,count,min_size_m,max_size_m,clearance_m,...}`.
  With `count > 0` on a scenario whose own mix is empty (`open_terrain`, `narrow_corridor`,
  `mapping_validation`) a representative default mix is used, so an explicit count always works.
* Sensors act on this geometry: sonar via `rayTestBatch`, DVL altitude via the seabed ray, depth from z.
  Nothing is detected from the generator's data.

`Environment.describe()` reports generation metadata (scenario, seed, terrain type, seabed z range,
**the number of collision bodies actually built**, requested/placed/fixed obstacle counts, structures).
It is **ground-truth environment information**, labelled as such in the dashboard, and never reaches the mapper.

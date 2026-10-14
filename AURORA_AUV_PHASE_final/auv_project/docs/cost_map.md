# Navigation cost map (Phase 4)

Built by `src/mapping/online_cost_map.build_cost_map` from the occupancy *state* (UNKNOWN/FREE/OCCUPIED)
and never writes back to it: **occupancy = sensor belief, cost map = navigation representation.**
No planner consumes it yet (Phase 5).

| class | rule | cost (default) | traversable |
|---|---|---|---|
| OCCUPIED | state OCCUPIED | `occupied` = 1000 | no |
| INFLATED | free/unknown cell within `inflation_radius_m` of an occupied cell | `inflated` = 500 | no |
| UNKNOWN | state UNKNOWN | `unknown` = 50 (+ clearance) | `unknown_policy`: `cost` yes / `blocked` no |
| FREE | otherwise | `free` = 1 + clearance + terrain risk | yes |

`inflation_radius_m = obstacle.vehicle_radius_m + obstacle.safety_margin_m` (0.35 + 0.25 = 0.60 m).
The hull is 1.5 m x 0.5 m; a circular footprint is a documented approximation. Inflation is an exact
Euclidean distance transform, so the inflated region is a disc of the metric radius (tested at 0.25 m
cells: 2 cells, not 3).

Clearance cost, added to FREE **and UNKNOWN** cells beyond the inflation radius (an unexplored cell next to an obstacle is
more costly than one far away), deterministic and monotonic:
`clearance_weight * max(0, 1 - (d - r_infl) / clearance_decay_m)` (defaults 10 and 1.5 m).
Terrain risk (optional) adds `terrain_risk.weight * risk` to free cells only.

Config: `mapping.costs.*`, `mapping.obstacle.*`, `mapping.terrain_risk.*`; ordering of costs is validated.
Telemetry (`cost_map`): min/max cost, unknown/occupied/inflated/traversable/high-cost counts,
inflation radius, unknown policy, whether terrain risk is used. The cost map is recomputed every
`cost_update_every_n` map updates with a distance transform (not per simulation tick).

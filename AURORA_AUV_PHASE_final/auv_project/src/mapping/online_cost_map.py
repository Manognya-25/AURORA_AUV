"""Navigation cost map derived from the online occupancy grid (+ optional terrain-risk prior).

    occupancy state (sensor belief)  ->  distance field  ->  inflation + clearance  ->  cost
The occupancy grid is never modified; this is a separate, derived representation (the planner-facing
one). Per-cell cost, in priority order:
    OCCUPIED                      occupied_cost                       not traversable
    within inflation radius       inflated_cost                       not traversable
    UNKNOWN                       unknown_cost (+clearance)           traversable iff unknown_policy=='cost'
    FREE                          free_cost    (+clearance)           traversable
  + terrain_weight * terrain_risk   (non-occupied cells, only if terrain_risk.enabled)
Phase 5 will consume `cost` + `traversable`; nothing in Phase 4 plans on it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mapping.inflation import clearance_cost, inflation_masks, obstacle_distance_m
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, CostMapConfig

CLASS_FREE, CLASS_UNKNOWN, CLASS_OCCUPIED, CLASS_INFLATED = 0, 1, 2, 3


@dataclass
class CostMapResult:
    cost: np.ndarray            # float32 (rows, cols)
    traversable: np.ndarray     # bool
    cls: np.ndarray             # uint8: CLASS_* (planner-facing semantic layer)
    distance_m: np.ndarray      # float32, distance to nearest occupied cell (inf = none)
    stats: dict


def build_cost_map(state: np.ndarray, resolution: float, cfg: CostMapConfig,
                   terrain_risk: np.ndarray | None = None) -> CostMapResult:
    occupied = state == OCCUPIED
    unknown = state == UNKNOWN
    dist = obstacle_distance_m(occupied, resolution)
    inflated, _ = inflation_masks(dist, occupied, cfg.inflation_radius_m)

    cost = np.where(unknown, cfg.unknown_cost, cfg.free_cost).astype(np.float32)
    cost += clearance_cost(dist, cfg.inflation_radius_m, cfg.clearance_decay_m, cfg.clearance_weight)
    if cfg.terrain_enabled and terrain_risk is not None:
        cost += (cfg.terrain_weight * np.clip(terrain_risk, 0.0, 1.0)).astype(np.float32)
    cost[inflated] = cfg.inflated_cost
    cost[occupied] = cfg.occupied_cost

    traversable = ~(occupied | inflated)
    if cfg.unknown_policy == "blocked":
        traversable &= ~unknown
    cls = np.full(state.shape, CLASS_FREE, np.uint8)
    cls[unknown] = CLASS_UNKNOWN
    cls[inflated] = CLASS_INFLATED
    cls[occupied] = CLASS_OCCUPIED

    stats = {
        "min_cost": float(cost.min()), "max_cost": float(cost.max()),
        "unknown_count": int(unknown.sum()), "occupied_count": int(occupied.sum()),
        "inflated_count": int(inflated.sum()), "traversable_count": int(traversable.sum()),
        "high_cost_count": int(np.sum((cost >= cfg.high_cost_threshold) & ~occupied & ~inflated)),
        "inflation_radius_m": cfg.inflation_radius_m, "unknown_policy": cfg.unknown_policy,
        "terrain_risk_used": bool(cfg.terrain_enabled and terrain_risk is not None),
    }
    return CostMapResult(cost, traversable, cls, dist, stats)

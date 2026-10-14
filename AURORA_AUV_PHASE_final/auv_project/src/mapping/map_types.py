"""Phase 4 online-mapping shared types + configuration.

Cell states (the semantic contract every consumer, including a future planner, relies on):
    UNKNOWN  (-1)  never observed, or evidence still too weak (probability between the thresholds)
    FREE     ( 0)  P(occupied) <= free_threshold
    OCCUPIED (100) P(occupied) >= occupied_threshold
These values follow the ROS OccupancyGrid convention in spirit (unknown/free/occupied) but are
stored as int8 here. The offline Week-4 grid (src/mapping/occupancy_grid.py: free=0, obstacle=1,
unknown=2) is a different, unrelated encoding and is left untouched.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UNKNOWN = -1
FREE = 0
OCCUPIED = 100


def _logit(p: float) -> float:
    return float(np.log(p / (1.0 - p)))


@dataclass
class MapConfig:
    """Occupancy-grid + sonar-mapper parameters (config: mapping.*). Simulation parameters, not
    calibrated values for any real vehicle or sonar."""
    enabled: bool = True
    update_rate_hz: float = 5.0
    # --- geometry (map frame == world frame; see coordinate_utils.GridGeometry) ---
    resolution_m: float = 0.25
    width_m: float = 60.0              # extent along world X
    height_m: float = 60.0             # extent along world Y
    origin_x_m: float = -5.0           # world X of the grid's lower-left CORNER
    origin_y_m: float = -5.0
    # --- probabilistic model ---
    occupied_threshold: float = 0.7
    free_threshold: float = 0.3
    log_odds_occupied: float = 1.0     # added to a cell on a sonar hit
    log_odds_free: float = -0.5        # added to a cell a ray passed through
    log_odds_min: float = -4.0         # clamp (P ~ 0.018): a cell can always be revised
    log_odds_max: float = 4.0          # clamp (P ~ 0.982)
    # --- sonar integration ---
    max_update_range_m: float = 30.0   # rays are never integrated beyond this distance
    slice_half_height_m: float = 1.0   # 2-D slice: only rays inside |z - z_vehicle| <= this count
    no_return_free_weight: float = 0.5 # weight of free-space evidence from a valid no-return ray
    max_pose_age_s: float = 0.1        # reject a scan if the pose estimate is older/newer than this
    # --- online cost map ---
    cost_update_every_n: int = 1       # recompute the cost map every N map updates
    # --- optional sonar-mount fallback (only used if a measurement carries no extrinsics) ---
    sonar_mount_position_m: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    sonar_mount_rpy_deg: list = field(default_factory=lambda: [0.0, 0.0, 0.0])

    def __post_init__(self):
        if not self.resolution_m > 0:
            raise ValueError("mapping.resolution_m must be > 0")
        if not (self.width_m > 0 and self.height_m > 0):
            raise ValueError("mapping.width_m/height_m must be > 0")
        if not 0.0 < self.free_threshold < self.occupied_threshold < 1.0:
            raise ValueError("need 0 < free_threshold < occupied_threshold < 1")
        if not (self.log_odds_free < 0 < self.log_odds_occupied):
            raise ValueError("need log_odds_free < 0 < log_odds_occupied")
        if not self.log_odds_min < 0 < self.log_odds_max:
            raise ValueError("need log_odds_min < 0 < log_odds_max")
        # thresholds must be reachable inside the clamp or cells could never become FREE/OCCUPIED
        if _logit(self.occupied_threshold) > self.log_odds_max or _logit(self.free_threshold) < self.log_odds_min:
            raise ValueError("log-odds clamp range does not reach the occupied/free thresholds")
        if self.max_update_range_m <= 0 or self.slice_half_height_m <= 0:
            raise ValueError("max_update_range_m and slice_half_height_m must be > 0")
        if not 0.0 <= self.no_return_free_weight <= 1.0:
            raise ValueError("no_return_free_weight must be in [0, 1]")
        if self.update_rate_hz <= 0 or self.cost_update_every_n < 1:
            raise ValueError("update_rate_hz must be > 0 and cost_update_every_n >= 1")

    @classmethod
    def from_config(cls, cfg: dict) -> "MapConfig":
        """cfg = FULL project config. Reads mapping.* (unknown/legacy Week-4 keys are ignored)."""
        m = dict((cfg.get("mapping") or {}))
        known = {f for f in cls.__dataclass_fields__}
        kw = {k: v for k, v in m.items() if k in known}
        # fall back to the sonar mounting in `sensors.sonar` if present (never required)
        so = ((cfg.get("sensors") or {}).get("sonar") or {})
        kw.setdefault("sonar_mount_position_m", list(so.get("mount_position_m", [0.0, 0.0, 0.0])))
        kw.setdefault("sonar_mount_rpy_deg", list(so.get("mount_rpy_deg", [0.0, 0.0, 0.0])))
        kw.setdefault("max_update_range_m", float(so.get("max_range_m", 30.0)))
        return cls(**kw)


@dataclass
class CostMapConfig:
    """Navigation cost-map parameters (config: mapping.costs / mapping.obstacle / mapping.terrain_risk)."""
    free_cost: float = 1.0
    unknown_cost: float = 50.0
    occupied_cost: float = 1000.0
    inflated_cost: float = 500.0       # cells inside the inflation radius (non-traversable)
    unknown_policy: str = "cost"       # "cost": traversable at unknown_cost | "blocked": not traversable
    clearance_weight: float = 10.0     # extra cost at the inflation boundary, decaying to 0
    clearance_decay_m: float = 1.5     # distance beyond the inflation radius over which it decays
    vehicle_radius_m: float = 0.35
    safety_margin_m: float = 0.25
    high_cost_threshold: float = 20.0
    terrain_enabled: bool = False
    terrain_weight: float = 5.0
    terrain_class_risk: dict = field(default_factory=lambda: {0: 0.0, 1: 0.4, 2: 0.8, 3: 0.0})

    def __post_init__(self):
        if self.unknown_policy not in ("cost", "blocked"):
            raise ValueError("unknown_policy must be 'cost' or 'blocked'")
        if self.vehicle_radius_m < 0 or self.safety_margin_m < 0 or self.clearance_decay_m <= 0:
            raise ValueError("vehicle_radius_m/safety_margin_m must be >= 0 and clearance_decay_m > 0")
        if not (0 <= self.free_cost <= self.unknown_cost <= self.inflated_cost <= self.occupied_cost):
            raise ValueError("need 0 <= free <= unknown <= inflated <= occupied cost")

    @property
    def inflation_radius_m(self) -> float:
        return float(self.vehicle_radius_m + self.safety_margin_m)

    @classmethod
    def from_config(cls, cfg: dict) -> "CostMapConfig":
        m = (cfg.get("mapping") or {})
        costs = dict(m.get("costs") or {})
        obst = dict(m.get("obstacle") or {})
        terr = dict(m.get("terrain_risk") or {})
        kw = {}
        for src_key, dst in (("free", "free_cost"), ("unknown", "unknown_cost"), ("occupied", "occupied_cost"),
                             ("inflated", "inflated_cost"), ("clearance_weight", "clearance_weight"),
                             ("clearance_decay_m", "clearance_decay_m"), ("unknown_policy", "unknown_policy"),
                             ("high_cost_threshold", "high_cost_threshold")):
            if src_key in costs:
                kw[dst] = costs[src_key]
        for k in ("vehicle_radius_m", "safety_margin_m"):
            if k in obst:
                kw[k] = float(obst[k])
        if "enabled" in terr:
            kw["terrain_enabled"] = bool(terr["enabled"])
        if "weight" in terr:
            kw["terrain_weight"] = float(terr["weight"])
        if "class_risk" in terr:
            kw["terrain_class_risk"] = {int(k): float(v) for k, v in terr["class_risk"].items()}
        return cls(**kw)

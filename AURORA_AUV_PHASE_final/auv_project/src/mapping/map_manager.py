"""MappingSystem: the online mapping pipeline owner (grid + sonar mapper + cost map + telemetry).

Data flow (the only inputs):   SonarMeasurement  +  EKF EstimatedState  ->  grid  ->  cost map.
No PyBullet / truth import exists in this module; tests/test_mapping_leak.py enforces that statically.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import label

from src.mapping.log_odds_grid import LogOddsGrid
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, CostMapConfig, MapConfig
from src.mapping.online_cost_map import CLASS_INFLATED, CLASS_OCCUPIED, CostMapResult, build_cost_map
from src.mapping.sonar_mapper import ScanStats, SonarMapper


def rle_encode(a: np.ndarray) -> list:
    """Flat run-length encoding [v0, n0, v1, n1, ...] of a 2-D uint8 array (row-major)."""
    flat = np.ascontiguousarray(a).reshape(-1)
    if flat.size == 0:
        return []
    change = np.flatnonzero(np.diff(flat)) + 1
    starts = np.r_[0, change]
    lengths = np.diff(np.r_[starts, flat.size])
    return np.stack([flat[starts].astype(np.int64), lengths], axis=1).reshape(-1).tolist()


def rle_decode(runs: list, shape: tuple) -> np.ndarray:
    vals, lens = np.asarray(runs[0::2], np.int64), np.asarray(runs[1::2], np.int64)
    return np.repeat(vals, lens).astype(np.uint8).reshape(shape)


class MappingSystem:
    def __init__(self, cfg: dict, terrain_risk: np.ndarray | None = None):
        """cfg: FULL project config. terrain_risk: optional risk raster already on the map grid."""
        self.cfg = MapConfig.from_config(cfg)
        self.cost_cfg = CostMapConfig.from_config(cfg)
        self.enabled = bool(self.cfg.enabled)
        self.grid = LogOddsGrid(self.cfg)
        self.mapper = SonarMapper(self.cfg)
        if terrain_risk is not None and terrain_risk.shape != self.grid.geom.shape:
            raise ValueError("terrain_risk must already be on the map grid")
        self.terrain_risk = terrain_risk
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Back to the all-UNKNOWN prior; all counters/timestamps cleared. Keeps enabled flag + config."""
        self.grid.reset()
        self._state = self.grid.state()
        self._cost: CostMapResult = build_cost_map(self._state, self.grid.geom.resolution, self.cost_cfg,
                                                   self.terrain_risk)
        self.stats = {"map_updates": 0, "scans_received": 0, "scans_processed": 0, "scans_rejected": 0,
                      "scans_skipped_rate": 0, "scans_rejected_stale_pose": 0,
                      "rays_processed": 0, "valid_returns": 0, "no_returns": 0, "dropouts": 0,
                      "cells_updated": 0, "hits_mapped": 0}
        self.last_update_time: float | None = None
        self.last_scan = ScanStats()
        self.last_reject_reason = ""
        self.last_pose_xy: tuple | None = None
        self._last_sonar_ts = -np.inf
        self._last_map_time = -np.inf
        self._since_cost = 0
        self.timing = {"last_map_ms": None, "last_cost_ms": None, "mean_map_ms": None, "mean_cost_ms": None,
                       "_n_map": 0, "_n_cost": 0}
        self.version = getattr(self, "version", 0) + 1   # monotonic, also across resets (dashboard cache key)

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = bool(enabled)

    # ------------------------------------------------------------------ main entry
    def process(self, sim_time: float, sonar, estimate) -> bool:
        """Offer the newest sonar measurement (or None) and the current EKF estimate.
        Returns True iff the grid was updated this call. Never raises on bad sensor data."""
        if not self.enabled or sonar is None or estimate is None:
            return False
        if sonar.timestamp <= self._last_sonar_ts:          # same scan already seen
            return False
        self._last_sonar_ts = sonar.timestamp
        self.stats["scans_received"] += 1
        if sim_time - self._last_map_time < 1.0 / self.cfg.update_rate_hz - 1e-9:
            self.stats["scans_skipped_rate"] += 1
            return False
        if abs(float(estimate.time) - float(sonar.timestamp)) > self.cfg.max_pose_age_s:
            self.stats["scans_rejected"] += 1
            self.stats["scans_rejected_stale_pose"] += 1
            self.last_reject_reason = "stale_pose"
            return False

        t0 = time.perf_counter()
        scan = self.mapper.integrate(self.grid, sonar, estimate.position, estimate.rotation)
        self.last_scan = scan
        self.stats["rays_processed"] += scan.rays_total
        self.stats["dropouts"] += scan.rays_invalid
        if not scan.processed:
            self.stats["scans_rejected"] += 1
            self.last_reject_reason = scan.reason
            return False
        self.last_reject_reason = ""
        self._last_map_time = sim_time
        self.last_update_time = float(sim_time)
        self.last_pose_xy = (float(estimate.position[0]), float(estimate.position[1]))
        self.stats["scans_processed"] += 1
        self.stats["map_updates"] += 1
        self.stats["valid_returns"] += scan.rays_hit
        self.stats["no_returns"] += scan.rays_no_return
        self.stats["cells_updated"] += scan.cells_updated
        self.stats["hits_mapped"] += scan.hits_mapped
        self._record("map", time.perf_counter() - t0)
        self.version += 1

        self._since_cost += 1
        if self._since_cost >= self.cfg.cost_update_every_n:
            self.update_cost_map()
        return True

    def update_cost_map(self) -> None:
        t0 = time.perf_counter()
        self._state = self.grid.state()
        self._cost = build_cost_map(self._state, self.grid.geom.resolution, self.cost_cfg, self.terrain_risk)
        self._since_cost = 0
        self._record("cost", time.perf_counter() - t0)

    def _record(self, which: str, seconds: float) -> None:
        ms, t = seconds * 1e3, self.timing
        t[f"last_{which}_ms"] = ms
        t[f"_n_{which}"] += 1
        prev = t[f"mean_{which}_ms"] or 0.0
        t[f"mean_{which}_ms"] = prev + (ms - prev) / t[f"_n_{which}"]

    # ------------------------------------------------------------------ outputs
    @property
    def cost_map(self) -> CostMapResult:
        return self._cost

    def state_grid(self) -> np.ndarray:
        return self._state if self._state is not None else self.grid.state()

    def telemetry(self) -> dict:
        """Scalar telemetry computed from the live grid / cost map (no hardcoded values)."""
        state = self.grid.state()
        c = self.grid.counts(state)
        g = self.grid.geom
        s = self.stats
        cs = self._cost.stats
        return {
            "mapping": {
                "enabled": self.enabled, "map_resolution": g.resolution, "map_size": [g.rows, g.cols],
                "map_origin": [g.origin_x, g.origin_y], "map_extent_m": list(g.extent),
                "known_cells": c["known_cells"], "unknown_cells": c["unknown_cells"],
                "free_cells": c["free_cells"], "occupied_cells": c["occupied_cells"],
                "coverage_percent": c["coverage_percent"], "occupied_percent": c["occupied_percent"],
                "last_update_time": self.last_update_time, "update_count": s["map_updates"],
                "estimated_map_x": None if self.last_pose_xy is None else self.last_pose_xy[0],
                "estimated_map_y": None if self.last_pose_xy is None else self.last_pose_xy[1],
                "mapped_obstacle_clusters": int(label(state == OCCUPIED, structure=np.ones((3, 3)))[1]),
                "last_reject_reason": self.last_reject_reason, "version": self.version,
                "scans_received": s["scans_received"], "scans_rejected": s["scans_rejected"],
                "scans_skipped_rate": s["scans_skipped_rate"],
            },
            "cost_map": {k: cs[k] for k in ("min_cost", "max_cost", "unknown_count", "occupied_count",
                                            "inflated_count", "traversable_count", "high_cost_count",
                                            "inflation_radius_m", "unknown_policy", "terrain_risk_used")},
            "sonar_mapping": {
                "valid_returns": s["valid_returns"], "no_returns": s["no_returns"], "dropouts": s["dropouts"],
                "rays_processed": s["rays_processed"], "cells_updated": s["cells_updated"],
                "hits_mapped": s["hits_mapped"],
                "last_scan": {"rays_total": self.last_scan.rays_total, "valid": self.last_scan.rays_valid,
                              "invalid": self.last_scan.rays_invalid, "hits": self.last_scan.rays_hit,
                              "no_return": self.last_scan.rays_no_return,
                              "hits_mapped": self.last_scan.hits_mapped,
                              "min_valid_range": self.last_scan.min_valid_range,
                              "max_valid_range": self.last_scan.max_valid_range},
            },
            "timing": {k: v for k, v in self.timing.items() if not k.startswith("_")},
        }

    def occupancy_payload(self) -> dict:
        """Compact RLE-encoded occupancy raster: 0 unknown, 1 free, 2 occupied (row-major, row<->y)."""
        s = self.grid.state()
        code = np.zeros(s.shape, np.uint8)
        code[s == FREE] = 1
        code[s == OCCUPIED] = 2
        g = self.grid.geom
        return {"shape": [g.rows, g.cols], "resolution": g.resolution, "origin": [g.origin_x, g.origin_y],
                "encoding": "rle[value,run]", "legend": {"0": "unknown", "1": "free", "2": "occupied"},
                "data": rle_encode(code), "version": self.version, "time": self.last_update_time}

    def costmap_payload(self) -> dict:
        """RLE classes (0 free, 1 unknown, 2 occupied, 3 inflated) + cost level 0..15 (cost / cap)."""
        cm = self._cost
        cap = max(2.0 * float(self.cost_cfg.high_cost_threshold), 1.0)
        level = np.clip(np.floor(np.minimum(cm.cost, cap) / cap * 15), 0, 15).astype(np.uint8)
        level[(cm.cls == CLASS_INFLATED) | (cm.cls == CLASS_OCCUPIED)] = 15
        g = self.grid.geom
        return {"shape": [g.rows, g.cols], "resolution": g.resolution, "origin": [g.origin_x, g.origin_y],
                "encoding": "rle[value,run]",
                "legend": {"classes": {"0": "free", "1": "unknown", "2": "occupied", "3": "inflated"},
                           "level": "min(cost, 2*high_cost_threshold) / (2*high_cost_threshold) * 15; shading only, class layer is authoritative"},
                "classes": rle_encode(cm.cls), "level": rle_encode(level),
                "stats": cm.stats, "version": self.version}

    def export_arrays(self) -> dict:
        """Raw arrays for debug export (np.savez)."""
        return {"occupancy_state": self.grid.state(), "log_odds": self.grid.log_odds,
                "probability": self.grid.probability(), "cost": self._cost.cost,
                "traversable": self._cost.traversable, "cost_class": self._cost.cls,
                "origin": np.array([self.grid.geom.origin_x, self.grid.geom.origin_y]),
                "resolution": np.array([self.grid.geom.resolution])}

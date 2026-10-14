"""Global planner: EKF-estimated start -> goal over the CURRENT cost map.

INPUTS (nothing else): a PlanningGrid built from the mapping module's cost map, the start position (the
caller passes the EKF ESTIMATE), the goal, and a PlannerConfig. There is no truth or simulation import in
this package; `tests/test_navigation_leak.py` enforces that statically.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.planning.astar import astar
from src.planning.config import PlannerConfig
from src.planning.grid_utils import PlanningGrid, check_path, path_cost, polyline_length
from src.planning.smoothing import simplify_collinear, smooth_path


@dataclass
class PlanResult:
    success: bool
    status: str                                  # OK | start_out_of_bounds | goal_out_of_bounds | start_blocked |
                                                 # goal_blocked | no_path | expansion_limit | path_check_failed
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    raw_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    length_m: float = 0.0
    raw_length_m: float = 0.0
    cost: float = float("inf")
    expansions: int = 0
    time_ms: float = 0.0
    smoothed: bool = False
    collision_free: bool = False                 # verdict of the INDEPENDENT checker on the returned path
    start_cell: tuple | None = None
    goal_cell: tuple | None = None
    start_adjusted_m: float = 0.0                # how far the start was moved to reach a traversable cell
    goal_adjusted_m: float = 0.0
    heuristic: str = ""
    connectivity: int = 8
    unknown_policy: str = ""
    detail: str = ""

    def summary(self) -> dict:
        return {"success": self.success, "status": self.status, "length_m": self.length_m,
                "raw_length_m": self.raw_length_m, "cost": None if not np.isfinite(self.cost) else self.cost,
                "expansions": self.expansions, "time_ms": self.time_ms, "smoothed": self.smoothed,
                "collision_free": self.collision_free, "vertices": int(len(self.path_xy)),
                "raw_vertices": int(len(self.raw_xy)), "start_adjusted_m": self.start_adjusted_m,
                "goal_adjusted_m": self.goal_adjusted_m, "heuristic": self.heuristic,
                "connectivity": self.connectivity, "unknown_policy": self.unknown_policy, "detail": self.detail}


class GlobalPlanner:
    def __init__(self, cfg: PlannerConfig | None = None):
        self.cfg = cfg or PlannerConfig()

    def plan(self, grid: PlanningGrid, start_xy, goal_xy, smooth: bool | None = None) -> PlanResult:
        t0 = time.perf_counter()
        cfg = self.cfg
        smooth = cfg.smoothing_enabled if smooth is None else smooth
        conn = cfg.effective_connectivity
        res = PlanResult(False, "no_path", connectivity=conn, heuristic=cfg.heuristic)
        res.unknown_policy = cfg.unknown_policy
        sx, sy = float(start_xy[0]), float(start_xy[1])
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        if not (np.isfinite(sx) and np.isfinite(sy) and np.isfinite(gx) and np.isfinite(gy)):
            res.status, res.detail = "start_out_of_bounds", "non-finite start/goal"
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res
        sr, sc = grid.cell_of(sx, sy)
        gr, gc = grid.cell_of(gx, gy)
        if not grid.in_bounds(sr, sc):
            res.status, res.detail = "start_out_of_bounds", f"start ({sx:.2f},{sy:.2f}) is outside the map"
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res
        if not grid.in_bounds(gr, gc):
            res.status, res.detail = "goal_out_of_bounds", f"goal ({gx:.2f},{gy:.2f}) is outside the map"
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res
        snapped_s = grid.nearest_traversable(sr, sc, cfg.start_snap_radius_m)
        if snapped_s is None:
            res.status, res.detail = "start_blocked", "start cell not traversable and no traversable cell within snap radius"
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res
        snapped_g = grid.nearest_traversable(gr, gc, cfg.goal_snap_radius_m)
        if snapped_g is None:
            why = "unknown" if grid.unknown[gr, gc] and not grid.traversable[gr, gc] else "occupied/inflated"
            res.status, res.detail = "goal_blocked", f"goal cell is not traversable ({why})"
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res
        s_cell, g_cell = (snapped_s[0], snapped_s[1]), (snapped_g[0], snapped_g[1])
        res.start_cell, res.goal_cell = s_cell, g_cell
        res.start_adjusted_m, res.goal_adjusted_m = snapped_s[2], snapped_g[2]

        a = astar(grid, s_cell, g_cell, conn, cfg.heuristic, cfg.heuristic_weight, cfg.max_expansions)
        res.expansions = a.expansions
        res.heuristic = a.heuristic_used
        if not a.found:
            res.status = a.reason or "no_path"
            res.detail = {"no_path": "no traversable route from start to goal in the current map",
                          "expansion_limit": "A* expansion limit reached"}.get(res.status, res.status)
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res

        centers = np.array([grid.center_of(r, c) for r, c in a.cells], float)
        # exact endpoints when the start/goal were not moved (a segment from a point inside a traversable cell to
        # the neighbouring cell centre can clip a corner the cell-centre path did not, so it is verified below)
        exact = centers.copy()
        if res.start_adjusted_m == 0.0:
            exact[0] = (sx, sy)
        if res.goal_adjusted_m == 0.0:
            exact[-1] = (gx, gy)
        raw = simplify_collinear(exact) if cfg.simplify_collinear else exact
        if not check_path(grid, raw).free:
            exact, raw = centers, (simplify_collinear(centers) if cfg.simplify_collinear else centers)
        res.raw_xy, res.raw_length_m = raw, polyline_length(raw)

        path = raw
        if smooth and len(raw) > 2:
            sm = smooth_path(grid, raw)
            if check_path(grid, sm).free:
                path = sm
        res.smoothed = bool(len(path) < len(raw))
        rep = check_path(grid, path)
        if not rep.free and not np.array_equal(path, raw):
            path, rep = raw, check_path(grid, raw)
            res.smoothed = False
        if not rep.free:
            res.status, res.detail = "path_check_failed", "independent collision check rejected the planned path"
            res.time_ms = (time.perf_counter() - t0) * 1e3
            return res
        res.path_xy, res.length_m = np.asarray(path, float), polyline_length(path)
        res.cost = path_cost(grid, path)
        res.collision_free = True
        res.success, res.status = True, "OK"
        res.time_ms = (time.perf_counter() - t0) * 1e3
        return res

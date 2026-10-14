"""Path validity monitoring: decides WHEN a path must be replaced (never 'every time step').

A global path is re-examined only when the map actually changed (new mapping.version) and at most every
`validate_interval_s`. Two independent triggers:
  * BLOCKED : some cell on the remaining path (within `path_check_horizon_m`) is no longer traversable in the
              current cost map (newly mapped obstacle, inflation growth, policy change);
  * COST    : the remaining path now costs much more than the SAME geometry cost when it was planned
              (cost_increase_ratio and cost_increase_min), i.e. the map changed significantly around it.
Rate limiting (min_replan_interval_s) and a replan budget (max_replans) live in the navigator state machine.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.planning.config import NavigationConfig
from src.planning.grid_utils import PlanningGrid, check_path, path_cost
from src.planning.path_data import PathData


@dataclass
class PathAssessment:
    blocked: bool = False
    blocked_distance_m: float | None = None      # path distance AHEAD of the vehicle to the first blocked cell
    blocked_cell: tuple | None = None
    cost_now: float | None = None
    cost_ref: float | None = None
    cost_increase: float | None = None
    significant_change: bool = False
    checked_length_m: float = 0.0


class PathMonitor:
    def __init__(self, cfg: NavigationConfig):
        self.cfg = cfg
        self.validations = 0
        self.blocked_events = 0
        self.change_events = 0

    def assess(self, grid: PlanningGrid, plan_grid: PlanningGrid | None, path: PathData, s: float) -> PathAssessment:
        cfg = self.cfg
        self.validations += 1
        a = PathAssessment()
        horizon_end = min(path.length, s + cfg.path_check_horizon_m)
        seg = path.segment_xy(s, horizon_end)
        a.checked_length_m = float(horizon_end - s)
        rep = check_path(grid, seg)
        if not rep.free:
            a.blocked = True
            a.blocked_distance_m = float(rep.first_blocked_distance_m or 0.0)
            a.blocked_cell = rep.first_blocked_cell
            self.blocked_events += 1
            return a
        if plan_grid is not None:
            a.cost_now = path_cost(grid, seg)
            a.cost_ref = path_cost(plan_grid, seg)
            a.cost_increase = a.cost_now - a.cost_ref
            if a.cost_increase > cfg.cost_increase_min and a.cost_now > a.cost_ref * (1.0 + cfg.cost_increase_ratio):
                a.significant_change = True
                self.change_events += 1
        return a

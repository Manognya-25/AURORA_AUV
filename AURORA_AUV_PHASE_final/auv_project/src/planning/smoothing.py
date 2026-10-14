"""Collision-safe path smoothing.

Two steps, both of which can only ever REMOVE vertices (never add or move one):
  1. collinear simplification  - drops vertices that do not change the direction;
  2. line-of-sight shortcutting - replaces a run of vertices by one straight segment if and only if the whole
     segment lies on traversable cells (exact supercover test, grid_utils.segment_is_free) AND does not cost
     more than the sub-path it replaces (cost-aware: a shortcut through the clearance band or unknown space
     is rejected if it is more expensive).
Because the result is a subsequence of vertices of an already-safe path and every new segment is verified on
the same traversable mask the planner used, the smoothed path is collision-safe by construction; the planner
then re-verifies it independently (check_path) and falls back to the unsmoothed path if that ever fails.
"""
from __future__ import annotations

import numpy as np

from src.planning.grid_utils import PlanningGrid, path_cost, segment_is_free


def simplify_collinear(xy: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    xy = np.asarray(xy, float)
    if len(xy) <= 2:
        return xy.copy()
    keep = [0]
    for i in range(1, len(xy) - 1):
        a, b, c = xy[keep[-1]], xy[i], xy[i + 1]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) > tol:
            keep.append(i)
    keep.append(len(xy) - 1)
    return xy[keep]


def smooth_path(grid: PlanningGrid, xy: np.ndarray, cost_tolerance: float = 1.02) -> np.ndarray:
    """Greedy farthest-visible shortcutting (see module docstring)."""
    xy = np.asarray(xy, float)
    n = len(xy)
    if n <= 2:
        return xy.copy()
    out = [xy[0]]
    i = 0
    while i < n - 1:
        j = n - 1
        chosen = i + 1
        while j > i + 1:
            if segment_is_free(grid, xy[i], xy[j]):
                direct = path_cost(grid, np.stack([xy[i], xy[j]]))
                replaced = path_cost(grid, xy[i:j + 1])
                if direct <= replaced * cost_tolerance + 1e-9:
                    chosen = j
                    break
            j -= 1
        out.append(xy[chosen])
        i = chosen
    return np.asarray(out, float)

"""Planning grid (traversability + cost on the mapping geometry), path collision checking, transforms.

The planner works on exactly the arrays the Phase-4 cost map already produces (`CostMapResult`): the
traversable mask (occupied and inflated cells are never traversable) and the per-cell cost. World <-> grid
conversion is the mapping module's own `GridGeometry` (floor() convention, cell centres at +0.5 cell), so a
planned path lives in the same world frame as the map: no second coordinate convention is introduced.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt

from src.mapping.coordinate_utils import GridGeometry, trace_segment
from src.mapping.online_cost_map import CLASS_UNKNOWN


@dataclass
class PlanningGrid:
    geom: GridGeometry
    traversable: np.ndarray      # bool (rows, cols)
    cost: np.ndarray             # float64 (rows, cols), > 0
    unknown: np.ndarray          # bool: UNKNOWN cells (informational + policy)
    map_version: int = -1

    # ----------------------------------------------------------------------------- construction
    @classmethod
    def from_cost_map(cls, cost_map, geom: GridGeometry, planner_cfg=None, map_version: int = -1) -> "PlanningGrid":
        """cost_map: CostMapResult. planner_cfg: PlannerConfig (unknown policy / cost override)."""
        unknown = np.asarray(cost_map.cls) == CLASS_UNKNOWN
        trav = np.asarray(cost_map.traversable, bool).copy()
        cost = np.asarray(cost_map.cost, np.float64).copy()
        if planner_cfg is not None:
            pol = planner_cfg.unknown_policy
            if pol == "block":
                trav &= ~unknown
            elif pol == "allow":
                # UNKNOWN cells become traversable; occupied / inflated cells are not UNKNOWN so stay blocked
                trav |= unknown & ~_hard_blocked(cost_map)
            if planner_cfg.unknown_cost is not None:
                cost[unknown] = float(planner_cfg.unknown_cost)
        if planner_cfg is not None and planner_cfg.extra_clearance_m > 0 and hasattr(cost_map, "distance_m"):
            # Localization-uncertainty margin: the EKF has no absolute position fix, so the map and the vehicle can be
            # offset by ~1 m after a minute. Cells within (inflation radius + extra) of a mapped obstacle are not planned through.
            infl = float((getattr(cost_map, "stats", None) or {}).get("inflation_radius_m", 0.0))
            trav &= ~(np.asarray(cost_map.distance_m) <= infl + planner_cfg.extra_clearance_m)
        cost = np.maximum(cost, 1e-3)
        return cls(geom, trav, cost, unknown, int(map_version))

    # ----------------------------------------------------------------------------- queries
    @property
    def shape(self) -> tuple:
        return self.geom.shape

    @property
    def resolution(self) -> float:
        return self.geom.resolution

    def cell_of(self, x: float, y: float) -> tuple:
        r, c = self.geom.world_to_grid(np.array([x]), np.array([y]))
        return int(r[0]), int(c[0])

    def center_of(self, row: int, col: int) -> tuple:
        x, y = self.geom.grid_to_world(np.array([row]), np.array([col]))
        return float(x[0]), float(y[0])

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.geom.rows and 0 <= col < self.geom.cols

    def is_traversable_xy(self, x: float, y: float) -> bool:
        r, c = self.cell_of(x, y)
        return self.in_bounds(r, c) and bool(self.traversable[r, c])

    def min_traversable_cost(self) -> float:
        if not self.traversable.any():
            return 1.0
        return float(self.cost[self.traversable].min())

    def nearest_traversable(self, row: int, col: int, max_radius_m: float):
        """Nearest traversable cell to (row, col) within max_radius_m, or None. Deterministic."""
        if self.in_bounds(row, col) and self.traversable[row, col]:
            return row, col, 0.0
        if max_radius_m <= 0 or not self.traversable.any():
            return None
        n = int(np.ceil(max_radius_m / self.geom.resolution)) + 1
        r0, r1 = max(0, row - n), min(self.geom.rows, row + n + 1)
        c0, c1 = max(0, col - n), min(self.geom.cols, col + n + 1)
        if r0 >= r1 or c0 >= c1:
            return None
        sub = self.traversable[r0:r1, c0:c1]
        if not sub.any():
            return None
        dist, (ir, ic) = distance_transform_edt(~sub, return_indices=True)
        rr, cc = row - r0, col - c0
        rr, cc = int(np.clip(rr, 0, sub.shape[0] - 1)), int(np.clip(cc, 0, sub.shape[1] - 1))
        d_m = float(dist[rr, cc]) * self.geom.resolution
        if d_m > max_radius_m + 1e-9:
            return None
        return int(ir[rr, cc]) + r0, int(ic[rr, cc]) + c0, d_m

    def crop(self, cx: float, cy: float, half_m: float) -> tuple:
        """Sub-grid around world (cx, cy). Returns (PlanningGrid, row0, col0). Geometry origin is shifted so
        world coordinates are identical in the sub-grid."""
        g = self.geom
        r, c = self.cell_of(cx, cy)
        n = int(np.ceil(half_m / g.resolution))
        r0, r1 = max(0, r - n), min(g.rows, r + n + 1)
        c0, c1 = max(0, c - n), min(g.cols, c + n + 1)
        r1, c1 = max(r1, r0 + 1), max(c1, c0 + 1)
        sub = GridGeometry(g.resolution, r1 - r0, c1 - c0, g.origin_x + c0 * g.resolution, g.origin_y + r0 * g.resolution)
        return (PlanningGrid(sub, self.traversable[r0:r1, c0:c1], self.cost[r0:r1, c0:c1],
                             self.unknown[r0:r1, c0:c1], self.map_version), r0, c0)


def _hard_blocked(cost_map) -> np.ndarray:
    from src.mapping.online_cost_map import CLASS_INFLATED, CLASS_OCCUPIED
    cls = np.asarray(cost_map.cls)
    return (cls == CLASS_OCCUPIED) | (cls == CLASS_INFLATED)


# ============================================================================== path geometry
def polyline_length(xy: np.ndarray) -> float:
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1)))


def cumulative_length(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, float)
    if len(xy) == 0:
        return np.zeros(0)
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])


def segment_cells(grid: PlanningGrid, a, b):
    """In-bounds cells (rows, cols) visited by the segment a->b (exact supercover), in order."""
    return trace_segment(grid.geom, float(a[0]), float(a[1]), float(b[0]), float(b[1]))


def segment_is_free(grid: PlanningGrid, a, b) -> bool:
    """True iff the WHOLE segment a->b lies in-bounds on traversable cells (supercover: no cell is skipped)."""
    for p in (a, b):
        r, c = grid.cell_of(p[0], p[1])
        if not grid.in_bounds(r, c):
            return False
    rows, cols = segment_cells(grid, a, b)
    if len(rows) == 0:
        return False
    return bool(grid.traversable[rows, cols].all())


@dataclass
class CollisionReport:
    free: bool
    first_blocked_segment: int | None = None
    first_blocked_cell: tuple | None = None
    first_blocked_distance_m: float | None = None     # arc length from the polyline start
    blocked_segments: int = 0


def check_path(grid: PlanningGrid, xy: np.ndarray, start_s: float = 0.0) -> CollisionReport:
    """Independent path collision check against the traversable mask (used to validate plans and live paths).

    start_s: arc length from the polyline start at which checking begins (the vehicle's progress). Cells before
    that are ignored, so a vehicle that is already past a segment does not re-trigger on it."""
    xy = np.asarray(xy, float)
    if len(xy) == 0:
        return CollisionReport(False, None, None, None, 0)
    if len(xy) == 1:
        r, c = grid.cell_of(*xy[0])
        ok = grid.in_bounds(r, c) and bool(grid.traversable[r, c])
        return CollisionReport(ok, None if ok else 0, None if ok else (r, c), None if ok else 0.0, 0 if ok else 1)
    cum = cumulative_length(xy)
    first = None
    nblocked = 0
    for i in range(len(xy) - 1):
        if cum[i + 1] <= start_s:
            continue
        a, b = xy[i], xy[i + 1]
        if cum[i] < start_s:                                   # clip the first partially-travelled segment
            t = (start_s - cum[i]) / max(cum[i + 1] - cum[i], 1e-12)
            a = a + t * (b - a)
        ra, ca = grid.cell_of(*a)
        rb, cb = grid.cell_of(*b)
        if not (grid.in_bounds(ra, ca) and grid.in_bounds(rb, cb)):
            bad_cell, dist = (ra, ca) if not grid.in_bounds(ra, ca) else (rb, cb), float(max(cum[i], start_s))
        else:
            rows, cols = segment_cells(grid, a, b)
            bad = ~grid.traversable[rows, cols] if len(rows) else np.array([True])
            if not bad.any():
                continue
            k = int(np.argmax(bad))
            bad_cell = (int(rows[k]), int(cols[k])) if len(rows) else (ra, ca)
            cx, cy = grid.center_of(*bad_cell)
            d = b - a
            L = float(np.linalg.norm(d))
            tproj = 0.0 if L < 1e-12 else float(np.clip(np.dot(np.array([cx, cy]) - a, d) / (L * L), 0.0, 1.0))
            dist = float(max(cum[i], start_s) + tproj * L)
        nblocked += 1
        if first is None:
            first = (i, bad_cell, dist)
    if first is None:
        return CollisionReport(True)
    return CollisionReport(False, first[0], first[1], first[2], nblocked)


def path_cost(grid: PlanningGrid, xy: np.ndarray, step_m: float | None = None) -> float:
    """Integral of the cell cost along the polyline (cost x metres), sampled every step_m (default res/2)."""
    xy = np.asarray(xy, float)
    if len(xy) < 2:
        return 0.0
    step = step_m or 0.5 * grid.resolution
    total = 0.0
    for a, b in zip(xy[:-1], xy[1:]):
        L = float(np.linalg.norm(b - a))
        if L < 1e-12:
            continue
        n = max(1, int(np.ceil(L / step)))
        t = (np.arange(n) + 0.5) / n
        pts = a[None, :] + t[:, None] * (b - a)[None, :]
        r, c = grid.geom.world_to_grid(pts[:, 0], pts[:, 1])
        ok = grid.geom.in_bounds(r, c)
        cc = np.where(ok, grid.cost[np.clip(r, 0, grid.geom.rows - 1), np.clip(c, 0, grid.geom.cols - 1)], 1e3)
        total += float(cc.sum()) * (L / n)
    return total


def point_at_arc(xy: np.ndarray, cum: np.ndarray, s: float) -> np.ndarray:
    """Point on the polyline at arc length s (clamped to [0, total])."""
    if len(xy) == 1:
        return np.asarray(xy[0], float)
    s = float(np.clip(s, 0.0, cum[-1]))
    i = int(np.searchsorted(cum, s, side="right") - 1)
    i = min(max(i, 0), len(xy) - 2)
    seg = cum[i + 1] - cum[i]
    t = 0.0 if seg < 1e-12 else (s - cum[i]) / seg
    return xy[i] + t * (xy[i + 1] - xy[i])


def project_onto_polyline(xy: np.ndarray, cum: np.ndarray, p, s_min: float = 0.0, s_max: float | None = None):
    """Closest point of the polyline to p restricted to arc range [s_min, s_max].
    Returns (s, cross_track_distance, closest_point)."""
    p = np.asarray(p, float)
    if len(xy) == 1:
        return 0.0, float(np.linalg.norm(p - xy[0])), np.asarray(xy[0], float)
    s_max = cum[-1] if s_max is None else min(s_max, cum[-1])
    best = (s_min, np.inf, point_at_arc(xy, cum, s_min))
    for i in range(len(xy) - 1):
        if cum[i + 1] < s_min or cum[i] > s_max:
            continue
        a, b = xy[i], xy[i + 1]
        d = b - a
        L2 = float(np.dot(d, d))
        t = 0.0 if L2 < 1e-18 else float(np.clip(np.dot(p - a, d) / L2, 0.0, 1.0))
        s = cum[i] + t * np.sqrt(L2)
        s = float(np.clip(s, s_min, s_max))
        q = point_at_arc(xy, cum, s)
        dist = float(np.linalg.norm(p - q))
        if dist < best[1] - 1e-12:
            best = (s, dist, q)
    return best

"""Deterministic local obstacle avoidance: a bounded detour that REJOINS the global path.

Triggered by the navigator when the upcoming global path is blocked in the CURRENT cost map (a newly mapped
obstacle, or inflation that grew over the path) close to the vehicle. Method (no randomness):
  1. take the remaining global path from the vehicle's projection;
  2. find the first blocked run along it and its end;
  3. choose rejoin candidates beyond that run, closest first (step rejoin_step_m), each required to be
     traversable, inside the local window, and followed by a clear stretch of the global path
     (rejoin_clear_horizon_m);
  4. run the SAME A* (same cost map, same inflation, same unknown policy) on a local window around the vehicle
     from the ESTIMATED position to the candidate; first success wins;
  5. smooth with the collision-safe line-of-sight shortcut and verify with the independent path checker.
The detour never leaves the traversable mask, so it respects inflated obstacles by construction; the vehicle
returns to the global path at the rejoin point (arc length `rejoin_s` on the global path).

`sonar_forward_clearance` is the reactive part: the shortest valid sonar hit inside a narrow forward cone
and the 2-D slice, from the measurement itself (ranges + beam angles). It carries no world coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.planning.astar import astar
from src.planning.config import NavigationConfig, PlannerConfig
from src.planning.grid_utils import PlanningGrid, check_path, polyline_length, segment_is_free
from src.planning.path_data import PathData
from src.planning.smoothing import simplify_collinear, smooth_path


@dataclass
class DetourResult:
    success: bool
    reason: str = ""
    path_xy: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    rejoin_s: float = 0.0                         # arc length on the GLOBAL path where the detour rejoins it
    rejoin_xy: tuple | None = None
    length_m: float = 0.0
    blocked_from_m: float | None = None           # arc length (from the vehicle) of the blockage start / end
    blocked_to_m: float | None = None
    candidates_tried: int = 0
    expansions: int = 0
    time_ms: float = 0.0


def blocked_run(grid: PlanningGrid, poly: np.ndarray, step: float | None = None):
    """First blocked run along polyline poly: (s_start, s_end) arc lengths from poly[0], or None.
    Sampled at the grid resolution / 2 (a 1-cell obstacle cannot be skipped)."""
    poly = np.asarray(poly, float)
    if len(poly) == 0:
        return None
    step = step or 0.5 * grid.resolution
    cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(poly, axis=0), axis=1))]) if len(poly) > 1 else np.zeros(1)
    total = float(cum[-1])
    ss = np.unique(np.concatenate([np.arange(0.0, total, step), [total]]))
    pts = np.array([np.interp(ss, cum, poly[:, 0]), np.interp(ss, cum, poly[:, 1])]).T if len(poly) > 1 else poly
    r, c = grid.geom.world_to_grid(pts[:, 0], pts[:, 1])
    ok_b = grid.geom.in_bounds(r, c)
    free = np.zeros(len(ss), bool)
    free[ok_b] = grid.traversable[r[ok_b], c[ok_b]]
    # the segment-level check is authoritative; point sampling is only used to LOCATE the run
    bad = ~free
    if not bad.any():
        return None
    i0 = int(np.argmax(bad))
    j = i0
    while j + 1 < len(bad) and bad[j + 1]:
        j += 1
    return float(ss[i0]), float(ss[j])


class LocalAvoider:
    def __init__(self, nav_cfg: NavigationConfig, planner_cfg: PlannerConfig):
        self.nav, self.pcfg = nav_cfg, planner_cfg

    def plan_detour(self, grid: PlanningGrid, pos_xy, global_path: PathData, global_s: float) -> DetourResult:
        import time
        t0 = time.perf_counter()
        nav = self.nav
        out = DetourResult(False)
        pos = np.asarray(pos_xy, float)
        remaining = global_path.remaining_xy(global_s)                 # starts at the projection point
        run = blocked_run(grid, np.vstack([pos[None, :], remaining[1:]]) if len(remaining) > 1 else remaining)
        if run is None:
            out.reason = "path_not_blocked"
            out.time_ms = (time.perf_counter() - t0) * 1e3
            return out
        out.blocked_from_m, out.blocked_to_m = run
        # arc length (on the global path) where the blocked run ends
        d0 = float(np.linalg.norm(remaining[0] - pos))
        s_block_end_global = global_s + max(run[1] - d0, 0.0)
        sub, r0, c0 = grid.crop(pos[0], pos[1], nav.local_window_m)
        goal_end = global_path.length
        res = grid.resolution
        for k in range(nav.rejoin_candidates):
            s_c = min(s_block_end_global + nav.rejoin_margin_m + k * nav.rejoin_step_m, goal_end)
            cand = global_path.point(s_c)[:2]
            out.candidates_tried += 1
            if not sub.is_traversable_xy(cand[0], cand[1]):
                if s_c >= goal_end:
                    break
                continue
            # a clear stretch of the global path must follow the rejoin point
            if s_c < goal_end and not check_path(grid, global_path.segment_xy(s_c, s_c + nav.rejoin_clear_horizon_m)).free:
                continue
            sr, sc_ = sub.cell_of(pos[0], pos[1])
            snapped = sub.nearest_traversable(sr, sc_, self.pcfg.start_snap_radius_m)
            gr, gc = sub.cell_of(cand[0], cand[1])
            if snapped is None or not sub.in_bounds(gr, gc):
                out.reason = "start_blocked" if snapped is None else "rejoin_outside_window"
                if snapped is None:
                    break
                continue
            a = astar(sub, (snapped[0], snapped[1]), (gr, gc), self.pcfg.effective_connectivity, self.pcfg.heuristic,
                      1.0, self.pcfg.max_expansions)
            out.expansions += a.expansions
            if not a.found:
                out.reason = "no_local_path"
                if s_c >= goal_end:
                    break
                continue
            centers = np.array([sub.center_of(r, c) for r, c in a.cells], float)
            xy = centers.copy()
            if snapped[2] == 0.0:
                xy[0] = pos
            xy[-1] = cand
            xy = simplify_collinear(xy)
            if not check_path(sub, xy).free:
                xy = simplify_collinear(centers)
            if self.pcfg.smoothing_enabled and len(xy) > 2:
                sm = smooth_path(sub, xy)
                if check_path(sub, sm).free:
                    xy = sm
            if not check_path(sub, xy).free:
                out.reason = "detour_check_failed"
                continue
            out.success, out.reason = True, ""
            out.path_xy, out.rejoin_s, out.rejoin_xy = xy, float(s_c), (float(cand[0]), float(cand[1]))
            out.length_m = polyline_length(xy)
            break
        if not out.success and not out.reason:
            out.reason = "no_rejoin_candidate"
        out.time_ms = (time.perf_counter() - t0) * 1e3
        return out


def sonar_forward_clearance(sonar, cone_deg: float, slice_half_height_m: float = 1.0) -> float | None:
    """Shortest valid sonar HIT in a forward cone (|azimuth| <= cone) that lies in the horizontal slice
    (|range * sin(elevation)| <= slice_half_height). None if no such hit / no valid scan. Uses the measurement only."""
    if sonar is None or not getattr(sonar, "valid", False):
        return None
    try:
        status = np.asarray(sonar.ray_status())
        rng = np.asarray(sonar.value, float)
        az, el = np.asarray(sonar.azimuths, float), np.asarray(sonar.elevations, float)
    except Exception:                                        # noqa: BLE001 - never let a malformed scan stop navigation
        return None
    sel = (status == 1) & np.isfinite(rng) & (np.abs(az) <= np.deg2rad(cone_deg)) \
        & (np.abs(rng * np.sin(el)) <= slice_half_height_m)
    if not sel.any():
        return None
    return float(np.min(rng[sel] * np.cos(el[sel])))

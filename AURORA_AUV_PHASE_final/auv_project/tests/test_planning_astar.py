"""Phase 5 global planner: A* correctness, world<->grid transforms, obstacle / unknown handling, path collision
checking and smoothing. Pure NumPy (no PyBullet): runs everywhere."""
import heapq
import math

import numpy as np
import pytest

from src.mapping.coordinate_utils import GridGeometry
from src.mapping.log_odds_grid import LogOddsGrid
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, CostMapConfig, MapConfig
from src.mapping.online_cost_map import build_cost_map
from src.planning.astar import astar
from src.planning.config import PlannerConfig
from src.planning.global_planner import GlobalPlanner
from src.planning.grid_utils import (PlanningGrid, check_path, path_cost, polyline_length, segment_is_free)
from src.planning.smoothing import simplify_collinear, smooth_path


def geom(rows=40, cols=40, res=1.0, ox=0.0, oy=0.0):
    return GridGeometry(res, rows, cols, ox, oy)


def grid_from_blocked(blocked=(), rows=40, cols=40, res=1.0, ox=0.0, oy=0.0, cost=None, unknown=None):
    g = geom(rows, cols, res, ox, oy)
    trav = np.ones(g.shape, bool)
    for r, c in blocked:
        trav[r, c] = False
    cst = np.ones(g.shape) if cost is None else np.asarray(cost, float)
    unk = np.zeros(g.shape, bool) if unknown is None else unknown
    return PlanningGrid(g, trav, cst, unk)


def real_cost_map(state, res=0.25, **cost_kw):
    """The Phase-4 cost map built from an occupancy STATE raster, exactly as the online mapper does."""
    return build_cost_map(state, res, CostMapConfig(**cost_kw))


def state_grid(rows, cols, fill=FREE):
    return np.full((rows, cols), fill, np.int8)


# =============================================================================== world <-> grid transforms
def test_world_grid_roundtrip_on_shifted_fine_grid():
    g = geom(rows=240, cols=240, res=0.25, ox=-5.0, oy=-5.0)
    pg = PlanningGrid(g, np.ones(g.shape, bool), np.ones(g.shape), np.zeros(g.shape, bool))
    for r, c in [(0, 0), (10, 37), (239, 239), (120, 5)]:
        x, y = pg.center_of(r, c)
        assert pg.cell_of(x, y) == (r, c)
        assert x == pytest.approx(-5.0 + (c + 0.5) * 0.25) and y == pytest.approx(-5.0 + (r + 0.5) * 0.25)


def test_planner_cell_lookup_is_the_mapping_modules_own_convention():
    g = geom(rows=60, cols=80, res=0.25, ox=-5.0, oy=-5.0)
    pg = PlanningGrid(g, np.ones(g.shape, bool), np.ones(g.shape), np.zeros(g.shape, bool))
    for x, y in [(-5.0, -5.0), (0.0, 0.0), (3.1, 7.9), (14.999, 9.999)]:
        r, c = g.world_to_grid(np.array([x]), np.array([y]))
        assert pg.cell_of(x, y) == (int(r[0]), int(c[0]))
    # a point exactly on a cell boundary belongs to the cell with the larger index (floor convention)
    assert pg.cell_of(-4.75, -4.75) == (1, 1)


def test_crop_keeps_world_coordinates_identical():
    g = geom(rows=100, cols=100, res=0.5, ox=-10.0, oy=-10.0)
    pg = PlanningGrid(g, np.ones(g.shape, bool), np.ones(g.shape), np.zeros(g.shape, bool))
    sub, r0, c0 = pg.crop(3.0, 4.0, 6.0)
    for x, y in [(3.0, 4.0), (0.0, 1.0), (8.9, 9.9)]:
        rs, cs = sub.cell_of(x, y)
        rf, cf = pg.cell_of(x, y)
        assert (rs + r0, cs + c0) == (rf, cf)
        assert sub.center_of(rs, cs) == pytest.approx(pg.center_of(rf, cf))


def test_path_is_returned_in_world_metres_at_the_real_resolution():
    # same physical problem on a 1.0 m and a 0.25 m grid -> the same world-metre path length
    lens = []
    for res in (1.0, 0.25):
        n = int(40 / res)
        g = geom(rows=n, cols=n, res=res, ox=-5.0, oy=-5.0)
        pg = PlanningGrid(g, np.ones(g.shape, bool), np.ones(g.shape), np.zeros(g.shape, bool))
        r = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0)).plan(pg, (0.0, 0.0), (15.0, 20.0))
        assert r.success
        lens.append(r.length_m)
        assert r.path_xy[0] == pytest.approx([0.0, 0.0]) and r.path_xy[-1] == pytest.approx([15.0, 20.0])
    assert lens[0] == pytest.approx(25.0, abs=1e-6) and lens[1] == pytest.approx(25.0, abs=1e-6)


# =============================================================================== A* correctness
def dijkstra_cost(grid, start, goal, connectivity):
    """Independent reference: plain Dijkstra with the SAME move-cost definition (no heuristic, no shared code)."""
    rows, cols = grid.shape
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)] + ([(1, 1), (1, -1), (-1, -1), (-1, 1)] if connectivity == 8 else [])
    dist = {start: 0.0}
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, math.inf):
            continue
        if u == goal:
            return d
        for dr, dc in moves:
            v = (u[0] + dr, u[1] + dc)
            if not (0 <= v[0] < rows and 0 <= v[1] < cols) or not grid.traversable[v]:
                continue
            if dr and dc and not (grid.traversable[u[0], v[1]] and grid.traversable[v[0], u[1]]):
                continue
            step = math.hypot(dr, dc) * grid.resolution * 0.5 * (grid.cost[u] + grid.cost[v])
            if d + step < dist.get(v, math.inf):
                dist[v] = d + step
                heapq.heappush(pq, (d + step, v))
    return math.inf


@pytest.mark.parametrize("conn", [4, 8])
def test_astar_cost_equals_independent_dijkstra_on_random_weighted_grids(conn):
    rng = np.random.default_rng(1234 + conn)
    checked = 0
    for trial in range(40):
        rows, cols = 18, 22
        trav = rng.random((rows, cols)) > 0.28
        cost = 1.0 + 9.0 * rng.random((rows, cols))
        g = geom(rows, cols, 0.5)
        pg = PlanningGrid(g, trav, cost, np.zeros(g.shape, bool))
        free = np.argwhere(trav)
        if len(free) < 2:
            continue
        s, e = tuple(free[rng.integers(len(free))]), tuple(free[rng.integers(len(free))])
        ref = dijkstra_cost(pg, s, e, conn)
        res = astar(pg, s, e, connectivity=conn, heuristic="octile")
        if math.isinf(ref):
            assert not res.found and res.reason == "no_path"
        else:
            assert res.found
            assert res.cost == pytest.approx(ref, rel=1e-9)     # optimal, not merely feasible
            assert res.cells[0] == s and res.cells[-1] == e
            checked += 1
    assert checked >= 15


def test_astar_path_cells_are_all_traversable_and_adjacent():
    rng = np.random.default_rng(5)
    trav = rng.random((30, 30)) > 0.25
    trav[0, 0] = trav[29, 29] = True
    g = geom(30, 30)
    pg = PlanningGrid(g, trav, np.ones(g.shape), np.zeros(g.shape, bool))
    res = astar(pg, (0, 0), (29, 29), connectivity=8)
    if res.found:
        for (r, c), (r2, c2) in zip(res.cells[:-1], res.cells[1:]):
            assert trav[r, c] and trav[r2, c2]
            assert max(abs(r - r2), abs(c - c2)) == 1
            if r != r2 and c != c2:                              # diagonal: no corner cutting
                assert trav[r, c2] and trav[r2, c]


def test_astar_open_grid_straight_and_octile_lengths():
    pg = grid_from_blocked()
    res = astar(pg, (0, 0), (0, 10), connectivity=8)
    assert res.found and res.cost == pytest.approx(10.0)
    res = astar(pg, (0, 0), (10, 10), connectivity=8)
    assert res.cost == pytest.approx(10 * math.sqrt(2))
    res4 = astar(pg, (0, 0), (10, 10), connectivity=4, heuristic="manhattan")
    assert res4.cost == pytest.approx(20.0)


def test_astar_does_not_cut_a_diagonal_corner():
    # two blocked cells touching at a corner: the diagonal between them must not be used
    pg = grid_from_blocked(blocked=[(5, 6), (6, 5)], rows=12, cols=12)
    res = astar(pg, (5, 5), (6, 6), connectivity=8)
    assert res.found
    assert (6, 6) in res.cells and len(res.cells) > 2        # had to go around, not straight through the gap


def test_astar_prefers_a_longer_cheap_route_over_a_short_expensive_one():
    cost = np.ones((20, 20))
    cost[:, 8:12] = 40.0                                     # expensive band
    cost[18:, 8:12] = 1.0                                    # cheap crossing far south
    g = geom(20, 20)
    pg = PlanningGrid(g, np.ones(g.shape, bool), cost, np.zeros(g.shape, bool))
    res = astar(pg, (2, 2), (2, 17), connectivity=8)
    assert res.found and max(r for r, _ in res.cells) >= 17    # detoured to the cheap crossing
    straight = sum(0.5 * (cost[2, c] + cost[2, c + 1]) for c in range(2, 17))
    assert res.cost < straight


def test_astar_failure_reasons():
    pg = grid_from_blocked(blocked=[(5, 5)], rows=10, cols=10)
    assert astar(pg, (5, 5), (0, 0)).reason == "start_blocked"
    assert astar(pg, (0, 0), (5, 5)).reason == "goal_blocked"
    assert astar(pg, (0, 0), (50, 50)).reason == "out_of_bounds"
    wall = grid_from_blocked(blocked=[(r, 5) for r in range(10)], rows=10, cols=10)
    res = astar(wall, (0, 0), (0, 9))
    assert not res.found and res.reason == "no_path"
    lim = astar(grid_from_blocked(rows=60, cols=60), (0, 0), (59, 59), max_expansions=5)
    assert not lim.found and lim.reason == "expansion_limit"


def test_astar_is_deterministic():
    rng = np.random.default_rng(9)
    trav = rng.random((25, 25)) > 0.2
    trav[0, 0] = trav[24, 24] = True
    g = geom(25, 25)
    pg = PlanningGrid(g, trav, 1 + 3 * rng.random((25, 25)), np.zeros(g.shape, bool))
    a, b = astar(pg, (0, 0), (24, 24)), astar(pg, (0, 0), (24, 24))
    assert a.cells == b.cells and a.cost == b.cost


def test_manhattan_is_replaced_by_octile_with_8_connectivity_so_it_stays_admissible():
    pg = grid_from_blocked()
    res = astar(pg, (0, 0), (10, 10), connectivity=8, heuristic="manhattan")
    assert res.heuristic_used == "octile" and res.cost == pytest.approx(10 * math.sqrt(2))


# =============================================================================== occupied / unknown handling (real cost map)
def test_occupied_and_inflated_cells_are_never_planned_through():
    state = state_grid(120, 120)
    state[40:80, 55:60] = OCCUPIED                                   # a wall segment, 1.25 m thick at 0.25 m resolution
    cm = real_cost_map(state, 0.25)
    g = GridGeometry(0.25, 120, 120, 0.0, 0.0)
    pg = PlanningGrid.from_cost_map(cm, g, PlannerConfig(extra_clearance_m=0.0))
    res = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0)).plan(pg, (2.0, 15.0), (28.0, 15.0))
    assert res.success and res.collision_free
    assert check_path(pg, res.path_xy).free
    # sample the path finely: every sample is outside occupied cells AND outside the inflation radius
    pts = np.vstack([np.linspace(a, b, 60) for a, b in zip(res.path_xy[:-1], res.path_xy[1:])])
    r, c = g.world_to_grid(pts[:, 0], pts[:, 1])
    assert not (state[r, c] == OCCUPIED).any()
    assert (cm.distance_m[r, c] > cm.stats["inflation_radius_m"] - 1e-6).all()
    assert res.length_m > 26.0                                       # it had to go around the wall


def test_extra_clearance_widens_the_keep_out_band():
    state = state_grid(100, 100)
    state[45:55, 45:55] = OCCUPIED
    cm = real_cost_map(state, 0.25)
    g = GridGeometry(0.25, 100, 100, 0.0, 0.0)
    base = PlanningGrid.from_cost_map(cm, g, PlannerConfig(extra_clearance_m=0.0))
    wide = PlanningGrid.from_cost_map(cm, g, PlannerConfig(extra_clearance_m=0.5))
    assert wide.traversable.sum() < base.traversable.sum()
    assert not (wide.traversable & (cm.distance_m <= cm.stats["inflation_radius_m"] + 0.5)).any()
    assert (base.traversable == cm.traversable).all()                # extra = 0 reproduces the Phase-4 mask exactly


def unknown_world():
    """Left half FREE, right half UNKNOWN, a FREE detour row at the top so a known-only route exists."""
    state = state_grid(80, 160, UNKNOWN)
    state[:, :80] = FREE
    state[70:80, :] = FREE                                            # known corridor along the top edge
    return state


def test_unknown_policy_inherit_follows_the_cost_map_setting():
    state = unknown_world()
    g = GridGeometry(0.25, 80, 160, 0.0, 0.0)
    cm_cost = real_cost_map(state, 0.25, unknown_policy="cost")
    cm_block = real_cost_map(state, 0.25, unknown_policy="blocked")
    a = PlanningGrid.from_cost_map(cm_cost, g, PlannerConfig(unknown_policy="inherit", extra_clearance_m=0.0))
    b = PlanningGrid.from_cost_map(cm_block, g, PlannerConfig(unknown_policy="inherit", extra_clearance_m=0.0))
    assert a.traversable[:, 100].all() and not b.traversable[:70, 100].any()


def test_unknown_policy_block_forces_the_known_route_and_allow_cuts_through_unknown():
    state = unknown_world()
    g = GridGeometry(0.25, 80, 160, 0.0, 0.0)
    cm = real_cost_map(state, 0.25, unknown_policy="cost", unknown_cost=2.0, free_cost=1.0, inflated_cost=500.0,
                       occupied_cost=1000.0, clearance_weight=0.0)
    start, goal = (2.0, 10.0), (36.0, 10.0)                           # goal is in the UNKNOWN half
    # allow: unknown cells are traversable, the straight path is cheap (cost 2) -> goes straight
    pa = PlanningGrid.from_cost_map(cm, g, PlannerConfig(unknown_policy="allow", extra_clearance_m=0.0))
    ra = GlobalPlanner(PlannerConfig(unknown_policy="allow", extra_clearance_m=0.0)).plan(pa, start, goal)
    assert ra.success and ra.length_m == pytest.approx(34.0, abs=0.3)
    # block: the goal itself is unknown -> planning fails with the real reason (the goal is never silently moved)
    pb = PlanningGrid.from_cost_map(cm, g, PlannerConfig(unknown_policy="block", extra_clearance_m=0.0))
    rb = GlobalPlanner(PlannerConfig(unknown_policy="block", extra_clearance_m=0.0)).plan(pb, start, goal)
    assert not rb.success and rb.status == "goal_blocked" and "unknown" in rb.detail
    # block with a KNOWN goal: the path may only use known cells (here: the top corridor is the only link)
    known_goal = (36.0, 18.0)
    rb2 = GlobalPlanner(PlannerConfig(unknown_policy="block", extra_clearance_m=0.0)).plan(pb, start, known_goal)
    assert rb2.success
    pts = np.vstack([np.linspace(a, b, 80) for a, b in zip(rb2.path_xy[:-1], rb2.path_xy[1:])])
    r, c = g.world_to_grid(pts[:, 0], pts[:, 1])
    assert not (state[r, c] == UNKNOWN).any()


def test_unknown_cost_override_changes_the_route_choice():
    state = unknown_world()
    state[:, 100:] = UNKNOWN
    g = GridGeometry(0.25, 80, 160, 0.0, 0.0)
    cm = real_cost_map(state, 0.25, unknown_cost=2.0, clearance_weight=0.0)
    start, goal = (2.0, 10.0), (38.0, 18.4)
    cheap = PlanningGrid.from_cost_map(cm, g, PlannerConfig(unknown_cost=2.0, extra_clearance_m=0.0))
    dear = PlanningGrid.from_cost_map(cm, g, PlannerConfig(unknown_cost=500.0, extra_clearance_m=0.0))
    pc = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0)).plan(cheap, start, goal)
    pd = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0)).plan(dear, start, goal)
    assert pc.success and pd.success
    assert pd.cost > pc.cost                                          # same geometry is priced by the override


def test_start_in_an_inflated_cell_is_snapped_only_within_the_configured_radius():
    state = state_grid(80, 80)
    state[38:42, 38:42] = OCCUPIED
    cm = real_cost_map(state, 0.25)
    g = GridGeometry(0.25, 80, 80, 0.0, 0.0)
    pg = PlanningGrid.from_cost_map(cm, g, PlannerConfig(extra_clearance_m=0.0))
    start = (10.0, 10.0)                                              # in the middle of the obstacle
    ok = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0, start_snap_radius_m=3.0)).plan(pg, start, (18.0, 18.0))
    assert ok.success and ok.start_adjusted_m > 0
    far = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0, start_snap_radius_m=0.2)).plan(pg, start, (18.0, 18.0))
    assert not far.success and far.status == "start_blocked"


def test_planner_reports_out_of_bounds_and_blocked_goal():
    pg = grid_from_blocked(blocked=[(10, 10)], rows=20, cols=20)
    p = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0))
    assert p.plan(pg, (1.5, 1.5), (100.0, 5.0)).status == "goal_out_of_bounds"
    assert p.plan(pg, (-5.0, 1.5), (5.0, 5.0)).status == "start_out_of_bounds"
    assert p.plan(pg, (1.5, 1.5), (10.5, 10.5)).status == "goal_blocked"
    assert p.plan(pg, (float("nan"), 1.0), (5.0, 5.0)).success is False


# =============================================================================== path collision checking
def test_check_path_catches_an_obstacle_BETWEEN_two_free_vertices():
    pg = grid_from_blocked(blocked=[(10, 10)], rows=20, cols=20)
    path = np.array([[2.5, 10.5], [18.5, 10.5]])                     # both end points free, the segment crosses cell (10, 10)
    rep = check_path(pg, path)
    assert not rep.free and rep.first_blocked_cell == (10, 10)
    assert rep.first_blocked_distance_m == pytest.approx(8.0, abs=1.0)
    assert not segment_is_free(pg, path[0], path[1])
    assert segment_is_free(pg, np.array([2.5, 12.5]), np.array([18.5, 12.5]))


def test_check_path_start_s_ignores_what_the_vehicle_already_passed():
    pg = grid_from_blocked(blocked=[(5, 5)], rows=20, cols=20)
    path = np.array([[0.5, 5.5], [18.5, 5.5]])
    assert not check_path(pg, path).free
    assert check_path(pg, path, start_s=7.0).free                      # past cell (5, 5): only free cells remain
    assert not check_path(pg, path, start_s=2.0).free


def test_check_path_out_of_bounds_is_a_collision():
    pg = grid_from_blocked(rows=10, cols=10)
    assert not check_path(pg, np.array([[1.0, 1.0], [30.0, 1.0]])).free
    assert check_path(pg, np.array([[1.0, 1.0], [8.0, 8.0]])).free


def test_every_planned_path_passes_the_independent_checker_on_random_maps():
    rng = np.random.default_rng(77)
    planned = 0
    for _ in range(25):
        rows = cols = 40
        trav = rng.random((rows, cols)) > 0.3
        g = geom(rows, cols, 0.5)
        pg = PlanningGrid(g, trav, 1.0 + 4 * rng.random((rows, cols)), np.zeros(g.shape, bool))
        free = np.argwhere(trav)
        s, e = free[rng.integers(len(free))], free[rng.integers(len(free))]
        sx, sy = pg.center_of(*s)
        ex, ey = pg.center_of(*e)
        res = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0)).plan(pg, (sx, sy), (ex, ey))
        if res.success:
            planned += 1
            assert res.collision_free and check_path(pg, res.path_xy).free
            assert check_path(pg, res.raw_xy).free
    assert planned >= 8


# =============================================================================== smoothing
def test_smoothing_removes_staircase_vertices_in_free_space():
    pg = grid_from_blocked(rows=30, cols=30)
    res = astar(pg, (1, 1), (25, 12), connectivity=4)                 # a 4-connected staircase
    centers = np.array([pg.center_of(r, c) for r, c in res.cells])
    simp = simplify_collinear(centers)
    sm = smooth_path(pg, simp)
    assert len(sm) == 2 and polyline_length(sm) < polyline_length(centers)
    assert sm[0] == pytest.approx(centers[0]) and sm[-1] == pytest.approx(centers[-1])


def test_smoothing_never_cuts_through_an_obstacle():
    blocked = [(r, 12) for r in range(0, 20)]
    pg = grid_from_blocked(blocked=blocked, rows=30, cols=30)
    res = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0)).plan(pg, (2.5, 5.5), (24.5, 5.5))
    assert res.success and res.smoothed
    assert check_path(pg, res.path_xy).free
    assert len(res.path_xy) >= 3                                       # the corner around the wall tip survives
    assert res.length_m <= res.raw_length_m + 1e-9
    # the straight shortcut start->goal really would have hit the wall
    assert not segment_is_free(pg, res.path_xy[0], res.path_xy[-1])


def test_smoothing_property_safe_and_not_longer_on_random_maps():
    rng = np.random.default_rng(31)
    shortened = 0
    for _ in range(30):
        trav = rng.random((40, 40)) > 0.25
        g = geom(40, 40, 0.5)
        pg = PlanningGrid(g, trav, np.ones(g.shape), np.zeros(g.shape, bool))
        free = np.argwhere(trav)
        s, e = free[rng.integers(len(free))], free[rng.integers(len(free))]
        res = astar(pg, tuple(s), tuple(e), connectivity=8)
        if not res.found or len(res.cells) < 3:
            continue
        pts = np.array([pg.center_of(r, c) for r, c in res.cells])
        sm = smooth_path(pg, simplify_collinear(pts))
        assert check_path(pg, sm).free
        assert polyline_length(sm) <= polyline_length(pts) + 1e-9
        shortened += int(polyline_length(sm) < polyline_length(pts) - 1e-6)
    assert shortened >= 5


def test_cost_aware_smoothing_rejects_a_shortcut_through_a_costly_band():
    cost = np.ones((30, 30))
    cost[8:22, 10:20] = 60.0                                           # traversable but very expensive region
    g = geom(30, 30)
    pg = PlanningGrid(g, np.ones(g.shape, bool), cost, np.zeros(g.shape, bool))
    around = np.array([[1.5, 15.5], [1.5, 24.5], [28.5, 24.5], [28.5, 15.5]])
    sm = smooth_path(pg, around)
    assert path_cost(pg, sm) <= path_cost(pg, around) * 1.02 + 1e-6    # never accepted a more expensive shortcut
    assert len(sm) > 2                                                  # it kept the way around the band


def test_smoothing_can_be_disabled():
    pg = grid_from_blocked(blocked=[(r, 12) for r in range(0, 20)], rows=30, cols=30)
    a = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0, smoothing_enabled=False)).plan(pg, (2.5, 5.5), (24.5, 5.5))
    b = GlobalPlanner(PlannerConfig(extra_clearance_m=0.0, smoothing_enabled=True)).plan(pg, (2.5, 5.5), (24.5, 5.5))
    assert a.success and b.success and not a.smoothed and b.smoothed
    assert b.length_m <= a.length_m + 1e-9


def test_planner_config_validation_and_legacy_keys():
    with pytest.raises(ValueError):
        PlannerConfig(connectivity=6)
    with pytest.raises(ValueError):
        PlannerConfig(unknown_policy="maybe")
    with pytest.raises(ValueError):
        PlannerConfig(heuristic_weight=0.5)
    legacy = PlannerConfig.from_config({"planning": {"connectivity": 8, "allow_diagonal": False, "heuristic": "manhattan"}})
    assert legacy.effective_connectivity == 4                          # the Week-6 placeholder keys keep their meaning
    assert PlannerConfig.from_config({"planning": {"smoothing": {"enabled": False}}}).smoothing_enabled is False


def test_planning_runs_on_a_real_online_map_with_inflation_and_unknown_space():
    """Planning on a map produced by the real LogOddsGrid + cost map (not a hand-made mask)."""
    cfg = {"mapping": {"resolution_m": 0.25, "width_m": 30.0, "height_m": 30.0, "origin_x_m": 0.0, "origin_y_m": 0.0}}
    mc = MapConfig.from_config(cfg)
    lg = LogOddsGrid(mc)
    rows, cols = np.meshgrid(np.arange(40, 80), np.arange(58, 62), indexing="ij")
    lg.apply(np.zeros(0, np.int64), np.zeros(0), lg.geom.flat_index(rows, cols).ravel().astype(np.int64))
    state = lg.state()
    assert (state == OCCUPIED).sum() == 160 and (state == UNKNOWN).sum() > 0
    cm = build_cost_map(state, mc.resolution_m, CostMapConfig())
    pg = PlanningGrid.from_cost_map(cm, lg.geom, PlannerConfig())
    res = GlobalPlanner(PlannerConfig()).plan(pg, (3.0, 15.0), (27.0, 15.0))
    assert res.success and res.collision_free
    assert res.length_m > 24.0 and res.expansions > 0 and res.time_ms > 0

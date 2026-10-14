"""Occupancy grid core: coordinates, exact ray traversal, log-odds behaviour, config validation."""
import numpy as np
import pytest

from src.mapping.coordinate_utils import GridGeometry, trace_segment
from src.mapping.log_odds_grid import LogOddsGrid
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, MapConfig


def geom(res=0.5, w=10.0, h=8.0, ox=-2.0, oy=1.0):
    return GridGeometry.from_extent(res, w, h, ox, oy)


# ---------------------------------------------------------------- coordinate conversion
def test_geometry_shape_and_extent():
    g = geom()
    assert g.shape == (16, 20)          # rows <-> Y (8 m / 0.5), cols <-> X (10 m / 0.5)
    assert g.extent == (-2.0, 8.0, 1.0, 9.0)


def test_world_to_grid_known_points():
    g = geom()
    assert tuple(int(v) for v in g.world_to_grid(-2.0, 1.0)) == (0, 0)          # lower-left corner -> cell (0,0)
    assert tuple(int(v) for v in g.world_to_grid(-1.76, 1.24)) == (0, 0)        # still inside cell (0,0)
    assert tuple(int(v) for v in g.world_to_grid(-1.5, 1.5)) == (1, 1)          # boundary belongs to larger index
    assert tuple(int(v) for v in g.world_to_grid(0.0, 3.0)) == (4, 4)
    row, col = g.world_to_grid(2.3, 7.9)
    assert (int(row), int(col)) == (13, 8)


def test_row_follows_y_and_col_follows_x():
    g = geom()
    r0, c0 = g.world_to_grid(0.0, 2.0)
    r1, c1 = g.world_to_grid(1.0, 2.0)                  # +X -> column increases only
    r2, c2 = g.world_to_grid(0.0, 3.0)                  # +Y -> row increases only
    assert (int(r1), int(c1)) == (int(r0), int(c0) + 2)
    assert (int(r2), int(c2)) == (int(r0) + 2, int(c0))


def test_grid_to_world_is_cell_centre_and_roundtrips():
    g = geom()
    x, y = g.grid_to_world(0, 0)
    assert (float(x), float(y)) == pytest.approx((-1.75, 1.25))
    for row in (0, 5, 15):
        for col in (0, 7, 19):
            x, y = g.grid_to_world(row, col)
            r, c = g.world_to_grid(x, y)
            assert (int(r), int(c)) == (row, col)


def test_world_to_grid_roundtrip_within_half_cell():
    g = geom()
    rng = np.random.default_rng(0)
    x, y = rng.uniform(-2, 8, 200), rng.uniform(1, 9, 200)
    r, c = g.world_to_grid(x, y)
    xc, yc = g.grid_to_world(r, c)
    assert np.all(np.abs(xc - x) <= g.resolution / 2 + 1e-9) and np.all(np.abs(yc - y) <= g.resolution / 2 + 1e-9)


def test_in_bounds():
    g = geom()
    assert bool(g.in_bounds(0, 0)) and bool(g.in_bounds(15, 19))
    assert not bool(g.in_bounds(-1, 0)) and not bool(g.in_bounds(16, 0)) and not bool(g.in_bounds(0, 20))
    r, c = g.world_to_grid(8.5, 5.0)                     # beyond +X edge
    assert not bool(g.in_bounds(r, c))


# ---------------------------------------------------------------- ray traversal
def test_trace_horizontal_and_diagonal():
    g = GridGeometry.from_extent(1.0, 10, 10, 0, 0)
    r, c = trace_segment(g, 0.5, 0.5, 3.5, 0.5)
    assert list(zip(r, c)) == [(0, 0), (0, 1), (0, 2), (0, 3)]
    r, c = trace_segment(g, 0.5, 0.5, 3.5, 3.5)
    assert list(zip(r, c)) == [(0, 0), (1, 1), (2, 2), (3, 3)]


def test_trace_negative_direction_and_order():
    g = GridGeometry.from_extent(1.0, 10, 10, 0, 0)
    r, c = trace_segment(g, 4.5, 2.5, 1.5, 2.5)
    assert list(zip(r, c)) == [(2, 4), (2, 3), (2, 2), (2, 1)]      # start cell first, end cell last


def test_trace_visits_every_crossed_cell_and_no_duplicates():
    g = GridGeometry.from_extent(0.25, 30, 30, -5, -5)
    rng = np.random.default_rng(3)
    for _ in range(100):
        p0, p1 = rng.uniform(-4, 24, 2), rng.uniform(-4, 24, 2)
        r, c = trace_segment(g, p0[0], p0[1], p1[0], p1[1])
        cells = list(zip(r.tolist(), c.tolist()))
        assert len(cells) == len(set(cells))
        # dense sampling must only ever land in traversed cells, and the end points must be included
        t = np.linspace(0, 1, 2000)
        sr, sc = g.world_to_grid(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
        assert set(zip(sr.tolist(), sc.tolist())) <= set(cells)
        assert cells[0] == (int(g.world_to_grid(*p0)[0]), int(g.world_to_grid(*p0)[1]))
        assert cells[-1] == (int(g.world_to_grid(*p1)[0]), int(g.world_to_grid(*p1)[1]))


def test_trace_clips_to_bounds_and_zero_length():
    g = GridGeometry.from_extent(1.0, 5, 5, 0, 0)
    r, c = trace_segment(g, 2.5, 2.5, 9.5, 2.5)
    assert list(zip(r, c)) == [(2, 2), (2, 3), (2, 4)]              # nothing beyond the edge
    r, c = trace_segment(g, 2.2, 2.2, 2.2, 2.2)
    assert list(zip(r, c)) == [(2, 2)]
    r, c = trace_segment(g, 20, 20, 25, 25)
    assert len(r) == 0


# ---------------------------------------------------------------- log-odds behaviour
def small_grid(**kw):
    return LogOddsGrid(MapConfig(width_m=5, height_m=5, resolution_m=0.5, origin_x_m=0, origin_y_m=0, **kw))


def test_initial_grid_is_entirely_unknown():
    g = small_grid()
    s = g.state()
    assert np.all(s == UNKNOWN)
    assert np.allclose(g.probability(), 0.5)
    c = g.counts()
    assert c["known_cells"] == 0 and c["unknown_cells"] == c["total_cells"] and c["coverage_percent"] == 0.0
    assert not g.observed.any()


def test_single_free_observation_is_not_yet_free_but_repeated_ones_are():
    g = small_grid()
    idx = np.array([12])
    g.apply(idx, np.ones(1), np.zeros(0, np.int64))
    assert g.state().reshape(-1)[12] == UNKNOWN                       # weak evidence stays unknown
    assert g.observed.reshape(-1)[12]                                   # ...but it HAS been observed
    for _ in range(3):
        g.apply(idx, np.ones(1), np.zeros(0, np.int64))
    assert g.state().reshape(-1)[12] == FREE
    assert g.probability().reshape(-1)[12] < g.cfg.free_threshold


def test_occupied_after_repeated_hits_and_probability_rises():
    g = small_grid()
    idx = np.array([30])
    p_prev = 0.5
    for _ in range(3):
        g.apply(np.zeros(0, np.int64), np.zeros(0), idx)
        p = float(g.probability().reshape(-1)[30])
        assert p > p_prev
        p_prev = p
    assert g.state().reshape(-1)[30] == OCCUPIED


def test_log_odds_is_clamped_and_probability_never_saturates():
    g = small_grid()
    idx = np.array([5])
    for _ in range(500):
        g.apply(np.zeros(0, np.int64), np.zeros(0), idx)
    assert float(g.log_odds.reshape(-1)[5]) == pytest.approx(g.cfg.log_odds_max)
    assert float(g.probability().reshape(-1)[5]) < 1.0
    for _ in range(500):
        g.apply(idx, np.ones(1), np.zeros(0, np.int64))
    assert float(g.log_odds.reshape(-1)[5]) == pytest.approx(g.cfg.log_odds_min)
    assert float(g.probability().reshape(-1)[5]) > 0.0


def test_conflicting_observations_cancel_and_cell_is_recoverable():
    g = small_grid()
    idx = np.array([7])
    g.apply(np.zeros(0, np.int64), np.zeros(0), idx)                    # +1.0
    g.apply(idx, np.ones(1), np.zeros(0, np.int64))                      # -0.5
    g.apply(idx, np.ones(1), np.zeros(0, np.int64))                      # -0.5
    assert float(g.log_odds.reshape(-1)[7]) == pytest.approx(0.0, abs=1e-6)
    assert g.state().reshape(-1)[7] == UNKNOWN                           # evidence exactly balanced
    for _ in range(20):                                                  # a saturated obstacle can still be cleared
        g.apply(np.zeros(0, np.int64), np.zeros(0), idx)
    for _ in range(40):
        g.apply(idx, np.ones(1), np.zeros(0, np.int64))
    assert g.state().reshape(-1)[7] == FREE


def test_free_weight_scales_the_update():
    g = small_grid()
    g.apply(np.array([1]), np.array([0.5]), np.zeros(0, np.int64))
    assert float(g.log_odds.reshape(-1)[1]) == pytest.approx(0.5 * g.cfg.log_odds_free)


def test_reset_restores_unknown():
    g = small_grid()
    g.apply(np.array([1, 2]), np.ones(2), np.array([3]))
    g.reset()
    assert np.all(g.state() == UNKNOWN) and g.update_count == 0 and not g.observed.any()


# ---------------------------------------------------------------- config validation
@pytest.mark.parametrize("kw", [
    dict(resolution_m=0.0), dict(width_m=-1.0), dict(free_threshold=0.8, occupied_threshold=0.7),
    dict(log_odds_free=0.1), dict(log_odds_occupied=-1.0), dict(log_odds_max=0.5),   # clamp can't reach 0.7
    dict(max_update_range_m=0.0), dict(no_return_free_weight=1.5), dict(update_rate_hz=0.0),
])
def test_invalid_map_config_is_rejected(kw):
    with pytest.raises(ValueError):
        MapConfig(**kw)


def test_map_config_from_project_config_ignores_legacy_week4_keys():
    cfg = {"mapping": {"occupancy_values": {"free": 0, "obstacle": 1, "unknown": 2}, "resolution_m": 0.5,
                       "unknown_cell_cost": 50}, "sensors": {"sonar": {"max_range_m": 20.0}}}
    mc = MapConfig.from_config(cfg)
    assert mc.resolution_m == 0.5 and mc.max_update_range_m == 20.0

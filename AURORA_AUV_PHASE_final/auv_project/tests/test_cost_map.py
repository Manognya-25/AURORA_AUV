"""Navigation cost map: derived from the occupancy state, inflation, clearance, unknown policy, terrain risk."""
import numpy as np
import pytest

from src.mapping.inflation import clearance_cost, inflation_masks, obstacle_distance_m
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, CostMapConfig
from src.mapping.online_cost_map import CLASS_FREE, CLASS_INFLATED, CLASS_OCCUPIED, CLASS_UNKNOWN, build_cost_map

RES = 0.25


def world(n=60):
    s = np.full((n, n), FREE, np.int8)
    return s


def with_block(s, r, c):
    s = s.copy(); s[r, c] = OCCUPIED; return s


def test_inflation_radius_is_vehicle_radius_plus_margin_and_configurable():
    assert CostMapConfig(vehicle_radius_m=0.35, safety_margin_m=0.25).inflation_radius_m == pytest.approx(0.60)
    assert CostMapConfig(vehicle_radius_m=0.5, safety_margin_m=0.0).inflation_radius_m == pytest.approx(0.50)
    with pytest.raises(ValueError):
        CostMapConfig(vehicle_radius_m=-1.0)


def test_inflated_region_is_a_disc_of_the_configured_metric_radius():
    s = with_block(world(), 30, 30)
    cfg = CostMapConfig(vehicle_radius_m=0.35, safety_margin_m=0.25)
    cm = build_cost_map(s, RES, cfg)
    rr, cc = np.mgrid[0:60, 0:60]
    d = np.hypot(rr - 30, cc - 30) * RES
    expect = (d <= 0.60 + 1e-6) & ~((rr == 30) & (cc == 30))
    assert (cm.cls == CLASS_INFLATED).tolist() == expect.tolist()
    assert cm.stats["inflated_count"] == int(expect.sum())
    # radius 0.6 m at 0.25 m cells: 2 cells straight, not 3
    assert cm.cls[30, 32] == CLASS_INFLATED and cm.cls[30, 33] != CLASS_INFLATED


def test_larger_margin_inflates_more_cells():
    s = with_block(world(), 30, 30)
    small = build_cost_map(s, RES, CostMapConfig(safety_margin_m=0.0)).stats["inflated_count"]
    large = build_cost_map(s, RES, CostMapConfig(safety_margin_m=1.0)).stats["inflated_count"]
    assert large > small > 0


def test_occupied_inflated_and_traversability_semantics():
    cm = build_cost_map(with_block(world(), 30, 30), RES, CostMapConfig())
    assert cm.cls[30, 30] == CLASS_OCCUPIED and cm.cost[30, 30] == 1000 and not cm.traversable[30, 30]
    assert cm.cls[30, 31] == CLASS_INFLATED and cm.cost[30, 31] == 500 and not cm.traversable[30, 31]
    assert cm.traversable[5, 5] and cm.cost[5, 5] == pytest.approx(1.0)


def test_clearance_cost_decreases_monotonically_with_distance_and_is_zero_far_away():
    s = with_block(world(80), 40, 5)
    cm = build_cost_map(s, RES, CostMapConfig())
    row = cm.cost[40, 5:70]
    outside = row[3:]                                   # beyond the inflation radius (0.6 m = 2.4 cells)
    assert (np.diff(outside) <= 1e-6).all()
    assert outside[0] > outside[4] > outside[-1]
    assert cm.cost[40, 69] == pytest.approx(1.0)        # far cell: pure free cost
    d = obstacle_distance_m(s == OCCUPIED, RES)
    assert d[40, 5] == 0 and d[40, 9] == pytest.approx(1.0)


def test_clearance_weight_is_configurable():
    s = with_block(world(), 30, 30)
    a = build_cost_map(s, RES, CostMapConfig(clearance_weight=0.0)).cost[30, 34]
    b = build_cost_map(s, RES, CostMapConfig(clearance_weight=40.0)).cost[30, 34]
    assert b > a


def test_clearance_cost_function_is_deterministic():
    d = np.linspace(0, 5, 30)
    assert np.array_equal(clearance_cost(d, 0.6, 1.5, 10.0), clearance_cost(d, 0.6, 1.5, 10.0))
    assert clearance_cost(np.array([np.inf]), 0.6, 1.5, 10.0)[0] == 0.0


def test_unknown_cost_policy_cost_vs_blocked():
    s = world(); s[10:20, 10:20] = UNKNOWN
    cost_pol = build_cost_map(s, RES, CostMapConfig(unknown_policy="cost", unknown_cost=50))
    blocked = build_cost_map(s, RES, CostMapConfig(unknown_policy="blocked", unknown_cost=50))
    assert cost_pol.cls[15, 15] == CLASS_UNKNOWN and cost_pol.cost[15, 15] == 50 and cost_pol.traversable[15, 15]
    assert blocked.cls[15, 15] == CLASS_UNKNOWN and not blocked.traversable[15, 15]
    assert cost_pol.cost[15, 15] > cost_pol.cost[40, 40]        # unknown is never treated as cheap/safe-as-free
    with pytest.raises(ValueError):
        CostMapConfig(unknown_policy="optimistic")


def test_fresh_all_unknown_map_has_no_obstacles_and_no_inflation():
    cm = build_cost_map(np.full((20, 20), UNKNOWN, np.int8), RES, CostMapConfig())
    assert cm.stats["occupied_count"] == 0 and cm.stats["inflated_count"] == 0 and cm.stats["unknown_count"] == 400
    assert np.isinf(cm.distance_m).all()


def test_cost_map_does_not_modify_the_occupancy_state():
    s = with_block(world(), 30, 30)
    before = s.copy()
    build_cost_map(s, RES, CostMapConfig())
    assert np.array_equal(s, before)


def test_cost_map_changes_when_the_occupancy_changes():
    a = build_cost_map(world(), RES, CostMapConfig())
    b = build_cost_map(with_block(world(), 30, 30), RES, CostMapConfig())
    assert not np.array_equal(a.cost, b.cost) and a.stats["inflated_count"] == 0 < b.stats["inflated_count"]


def test_terrain_risk_adds_cost_only_when_enabled_and_never_to_occupied():
    s = with_block(world(), 30, 30)
    risk = np.zeros((60, 60), np.float32); risk[5:10, 5:10] = 1.0
    off = build_cost_map(s, RES, CostMapConfig(terrain_enabled=False), risk)
    on = build_cost_map(s, RES, CostMapConfig(terrain_enabled=True, terrain_weight=5.0), risk)
    assert off.cost[7, 7] == pytest.approx(1.0) and on.cost[7, 7] == pytest.approx(6.0)
    assert on.cost[30, 30] == 1000 and on.stats["terrain_risk_used"] and not off.stats["terrain_risk_used"]


def test_cost_ordering_config_is_validated():
    with pytest.raises(ValueError):
        CostMapConfig(free_cost=10, unknown_cost=5)


def test_terrain_prior_uses_real_random_forest_and_bathymetry_and_keeps_obstacle_risk_zero():
    import yaml
    from src.mapping.coordinate_utils import GridGeometry
    from src.mapping.terrain_prior import classes_from_depth, resample_to_geometry, risk_from_classes
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    depth = np.load("data/processed/processed_bathymetry.npy")
    classes = classes_from_depth(depth, 1.0, cfg["terrain"]["model_path"])
    assert classes.shape == depth.shape and set(np.unique(classes)) <= {0, 1, 2, 3}
    risk_cfg = {int(k): float(v) for k, v in cfg["mapping"]["terrain_risk"]["class_risk"].items()}
    assert risk_cfg[3] == 0.0                                          # prior cannot smuggle obstacle locations in
    risk = risk_from_classes(classes, risk_cfg)
    g = GridGeometry.from_extent(0.25, 60, 60, -5, -5)
    r = resample_to_geometry(risk, (0.0, 0.0), 1.0, g)
    assert r.shape == g.shape and r.max() <= 0.8 and (r[:20, :] == 0).all()    # y < 0 m is outside the 0..100 m raster


def test_mapping_system_cost_map_is_built_from_its_own_occupancy():
    import yaml
    from src.mapping.map_manager import MappingSystem
    ms = MappingSystem(yaml.safe_load(open("configs/config.yaml", encoding="utf-8")))
    assert ms.cost_map.stats["occupied_count"] == 0
    # hand the grid evidence directly (as the sonar mapper would) and rebuild
    g = ms.grid
    idx = np.array([int(g.geom.flat_index(*g.geom.world_to_grid(10.0, 10.0)))])
    for _ in range(5):
        g.apply(np.zeros(0, np.int64), np.zeros(0), idx)
    ms.update_cost_map()
    assert ms.cost_map.stats["occupied_count"] == 1 and ms.cost_map.stats["inflated_count"] > 0

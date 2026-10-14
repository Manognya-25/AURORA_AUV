"""run_visual_demo.py: the long demo route must be clear of obstacles in every scenario and inside the map."""
import numpy as np
import yaml

import run_visual_demo
from src.mapping.map_types import MapConfig
from src.simulation.commands import deep_merge
from src.simulation.world_generator import SCENARIOS, _dist_to_polyline, generate_world

def long_cfg(scenario, seed):
    """The config the launcher really produces for this scenario (same route choice, same overrides)."""
    with open("configs/config.yaml", encoding="utf-8") as f:
        base = yaml.safe_load(f)
    over = {"environment": {"scenario": scenario, "seed": seed}}
    route = run_visual_demo.demo_route(scenario, "long")
    if route is not None:
        over["simulation"] = {"waypoints": route[1:], "goal_position": route[-1]}
        over["mapping"] = {"width_m": 110.0, "height_m": 110.0}
    return deep_merge(base, over)


def test_long_route_is_clear_of_obstacles_in_every_scenario_and_seed():
    checked = 0
    for scenario in SCENARIOS:
        for seed in (1, 3, 7):
            cfg = long_cfg(scenario, seed)
            env = cfg["environment"]
            spec = generate_world(env, cfg["simulation"], (int(env["grid_size"]["rows"]), int(env["grid_size"]["cols"]),
                                                          float(env["resolution"])))   # raises if anything blocks the route
            route = run_visual_demo.demo_route(scenario, "long")
            if route is not None:
                assert [list(map(float, p)) for p in spec.mission_points][-1] == [float(v) for v in route[-1]]   # saw the route
            poly = np.array([p[:2] for p in spec.mission_points], float)
            for o in spec.obstacles:
                pts = np.asarray(o.footprint, float)
                assert _dist_to_polyline(pts, poly).min() > o.footprint_radius + 0.5, (scenario, seed, o.kind)
                checked += 1
            assert spec.heights.max() <= min(p[2] for p in spec.mission_points) - 1.0 + 1e-9
    assert checked > 100


def test_each_scenario_gets_a_compatible_route():
    assert run_visual_demo.demo_route("mapping_validation") is None and run_visual_demo.demo_route("sparse_obstacles", "default") is None
    assert run_visual_demo.demo_route("narrow_corridor") == run_visual_demo.STRAIGHT_ROUTE
    assert run_visual_demo.demo_route("mixed_terrain_obstacles") == run_visual_demo.LONG_ROUTE
    d1 = np.subtract(run_visual_demo.STRAIGHT_ROUTE[1], run_visual_demo.STRAIGHT_ROUTE[0])[:2]
    d2 = np.subtract(run_visual_demo.STRAIGHT_ROUTE[2], run_visual_demo.STRAIGHT_ROUTE[1])[:2]
    assert abs(d1[0] * d2[1] - d1[1] * d2[0]) < 1e-9                          # collinear segments


def test_enlarged_map_covers_every_waypoint_of_the_long_route():
    mc = MapConfig.from_config(long_cfg("sparse_obstacles", 3))
    for x, y, _ in run_visual_demo.LONG_ROUTE:
        assert mc.origin_x_m <= x < mc.origin_x_m + mc.width_m and mc.origin_y_m <= y < mc.origin_y_m + mc.height_m


def test_launcher_lists_scenarios_and_reports_missing_pybullet_cleanly(capsys=None):
    assert run_visual_demo.main(["--list"]) == 0
    try:
        import pybullet  # noqa: F401
    except ImportError:
        assert run_visual_demo.main([]) == 2                 # clear failure code, no traceback


def test_demo_override_enables_the_underwater_look_camera_controls_and_the_exact_obstacle_count():
    """The demo drives the SAME real run(): its override carries the underwater theme, the orbit/zoom side-panel
    buttons and an exact obstacle-body count, and the world generated from it really has that many bodies."""
    vis = {"camera_follow": True, "underwater_theme": True, "camera": {"mode": "follow", "orbit_zoom_buttons": True}}
    over = run_visual_demo.build_override("dense_obstacles", 7, "long", vis, None, 30)
    assert over["environment"]["scenario"] == "dense_obstacles"
    assert over["environment"]["obstacles"] == {"count": 30}
    assert over["visualization"]["underwater_theme"] is True
    assert over["visualization"]["camera"]["orbit_zoom_buttons"] is True
    cfg = deep_merge(yaml.safe_load(open("configs/config.yaml", encoding="utf-8")), over)
    env, sim = cfg["environment"], cfg["simulation"]
    spec = generate_world(env, sim, (int(env["grid_size"]["rows"]), int(env["grid_size"]["cols"]),
                                     float(env["resolution"])))
    assert len(spec.obstacles) == 30                                          # exactly what --obstacles asked for
    assert [list(map(float, p)) for p in spec.mission_points][-1] == [float(v) for v in run_visual_demo.LONG_ROUTE[-1]]
    for o in spec.obstacles:                                                  # and the demo route is still clear
        pts = np.asarray(o.footprint, float)
        poly = np.array([p[:2] for p in spec.mission_points])
        assert _dist_to_polyline(pts, poly).min() > o.footprint_radius + 0.5


def test_obstacle_flag_is_refused_instead_of_ignored_on_the_week4_world():
    assert run_visual_demo.main(["--obstacles", "10", "--scenario", "legacy_week4"]) == 2
    assert run_visual_demo.main(["--obstacles", "999"]) == 2
    assert run_visual_demo.main(["--obstacles", "-1"]) == 2

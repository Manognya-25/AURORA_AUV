"""Procedural environment: determinism, scenarios, start safety, geometry fidelity, sensor interaction."""
import numpy as np
import pytest
import yaml

from src.simulation.world_generator import ALL_SCENARIOS, SCENARIOS, generate_world
from tests.helpers.analytic_world import world_from_spec

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
SIM = CFG["simulation"]
GRID = (100, 100, 1.0)


def gen(scenario, seed=42, **over):
    env = {"scenario": scenario, "seed": seed, **over}
    return generate_world(env, SIM, GRID)


def same(a, b):
    sa, sb = a.signature(), b.signature()
    return (np.array_equal(sa["heights"], sb["heights"]) and sa["obstacles"] == sb["obstacles"])


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_same_seed_gives_identical_world(scenario):
    assert same(gen(scenario, 42), gen(scenario, 42))


@pytest.mark.parametrize("scenario", ["sparse_obstacles", "dense_obstacles", "terrain_variation", "mixed_terrain_obstacles"])
def test_different_seed_gives_different_world(scenario):
    a, b = gen(scenario, 1), gen(scenario, 2)
    assert not same(a, b)
    assert not np.array_equal(a.heights, b.heights) or a.obstacles != b.obstacles


def test_obstacle_positions_differ_between_seeds_and_counts_match_request():
    a, b = gen("sparse_obstacles", 1), gen("sparse_obstacles", 2)
    assert [(o.x, o.y) for o in a.obstacles] != [(o.x, o.y) for o in b.obstacles]
    assert len(a.obstacles) == len(b.obstacles) == SCENARIOS["sparse_obstacles"]["obstacles"]["count"]


def test_generation_does_not_touch_global_numpy_random_state():
    np.random.seed(123)
    expect = np.random.rand(3)
    np.random.seed(123)
    gen("dense_obstacles", 7)
    assert np.array_equal(np.random.rand(3), expect)


def test_scenarios_are_meaningfully_different():
    counts = {n: len(gen(n).obstacles) for n in SCENARIOS}
    assert counts["open_terrain"] == 0 and counts["dense_obstacles"] > counts["sparse_obstacles"] > 0
    assert counts["narrow_corridor"] == 2                                       # two corridor walls
    assert gen("open_terrain").is_flat and not gen("terrain_variation").is_flat
    assert np.ptp(gen("terrain_variation").heights) > np.ptp(gen("sparse_obstacles").heights)
    kinds = gen("mixed_terrain_obstacles", 5).summary()["obstacle_kinds"]
    assert len(kinds) >= 3


def test_terrain_has_slopes_ridges_depressions_and_roughness():
    h = gen("terrain_variation", 3).heights
    gy, gx = np.gradient(h)
    assert np.hypot(gx, gy).max() > 0.1                                         # real slopes
    assert h.max() - np.median(h) > 0.5 and np.median(h) - h.min() > 0.5        # crests and depressions
    rough = gen("terrain_variation", 3, seabed={"roughness_m": 0.4}).heights
    smooth = gen("terrain_variation", 3, seabed={"roughness_m": 0.0}).heights
    assert np.abs(np.diff(rough, 2, axis=0)).mean() > np.abs(np.diff(smooth, 2, axis=0)).mean()


def test_seabed_always_stays_below_the_mission_depth():
    for n in SCENARIOS:
        for seed in (1, 2, 3):
            s = gen(n, seed)
            assert s.heights.max() <= min(p[2] for p in s.mission_points) - 1.0 + 1e-9


def test_no_obstacle_spawns_on_the_start_pose_or_blocks_the_mission_path():
    from src.simulation.world_generator import _dist_to_polyline
    checked = 0
    for scenario in ("sparse_obstacles", "dense_obstacles", "mixed_terrain_obstacles", "terrain_variation"):
        for seed in range(1, 7):
            s = gen(scenario, seed)
            start = np.array(s.mission_points[0][:2])
            poly = np.array([p[:2] for p in s.mission_points])
            for o in s.obstacles:
                pts = np.asarray(o.footprint, float)
                assert np.linalg.norm(pts - start, axis=1).min() > o.footprint_radius + 1.0, (scenario, seed, o.kind)
                assert _dist_to_polyline(pts, poly).min() > o.footprint_radius + 1.0, (scenario, seed, o.kind)
                checked += 1
    assert checked > 100                      # the loop really examined many placed obstacles


def test_placement_never_overlaps_other_structures():
    s = gen("dense_obstacles", 9)
    for i, a in enumerate(s.obstacles):
        for b in s.obstacles[i + 1:]:
            if a.group == b.group:
                continue
            pa, pb = np.asarray(a.footprint, float), np.asarray(b.footprint, float)
            d = np.linalg.norm(pa[:, None] - pb[None], axis=2).min()
            assert d >= a.footprint_radius + b.footprint_radius


def test_invalid_environment_configuration_is_rejected():
    with pytest.raises(ValueError):
        generate_world({"scenario": "nope"}, SIM, GRID)
    with pytest.raises(ValueError):
        generate_world({"scenario": "sparse_obstacles", "obstacles": {"min_size_m": 3.0, "max_size_m": 1.0}}, SIM, GRID)
    with pytest.raises(ValueError):
        generate_world({"scenario": "sparse_obstacles", "seabed": {"type": "swamp"}}, SIM, GRID)
    with pytest.raises(ValueError):
        generate_world({"scenario": "open_terrain", "seabed": {"base_z_m": -4.0}}, SIM, GRID)   # seabed above the mission


def test_obstacle_count_override_is_honoured():
    assert len(gen("sparse_obstacles", 3, obstacles={"count": 3}).obstacles) == 3
    assert len(gen("sparse_obstacles", 3, obstacles={"count": 0}).obstacles) == 0
    assert len(gen("sparse_obstacles", 3, obstacles={"enabled": False}).obstacles) == 0


@pytest.mark.parametrize("count", [10, 30, 53])
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_exactly_that_many_obstacle_bodies_are_generated(scenario, count):
    """`obstacles.count` == the number of PyBullet bodies build_world() creates, in every procedural scenario
    (including the ones with scenario-fixed structures, which count towards the total)."""
    for seed in (42, 5):
        s = gen(scenario, seed, obstacles={"count": count})
        assert len(s.obstacles) == count, (scenario, count, seed)
        assert s.summary()["obstacle_count"] == count
        assert s.meta["obstacles_requested"] == count
        assert s.meta["obstacles_fixed"] + s.meta["obstacles_placed"] == count
        assert len({o.group for o in s.obstacles}) >= 1


def test_requested_count_world_is_deterministic_for_the_seed():
    a = gen("dense_obstacles", 5, obstacles={"count": 30})
    b = gen("dense_obstacles", 5, obstacles={"count": 30})
    c = gen("dense_obstacles", 6, obstacles={"count": 30})
    assert same(a, b)                                                     # same seed + count -> identical world
    assert not same(a, c)                                                 # another seed -> a different world
    assert same(a, gen("dense_obstacles", 5, obstacles={"count": 30}))


# poses spread over the arena, so the count is compared where the obstacles actually are
COUNT_POSES = [(5, 5, -5), (20, 20, -5), (40, 40, -5), (60, 60, -5),
               (75, 75, -5), (30, 60, -5), (60, 30, -5), (50, 50, -5)]


def test_the_obstacle_count_changes_what_the_sonar_sees_and_therefore_what_the_mapper_can_map():
    """The count control is not cosmetic: it changes the REAL geometry the sonar ray-casts (the mapper's only
    input is that scan + the EKF estimate), so more obstacles means strictly more detections."""
    for seed in (3, 7):
        def hits(count):
            world = world_from_spec(gen("sparse_obstacles", seed, obstacles={"count": count}))
            return sum(int(sonar_ranges(world, p).detected.sum()) for p in COUNT_POSES)
        empty, few, many = hits(0), hits(10), hits(53)
        assert empty == 0 and 0 < few < many, (seed, empty, few, many)


def test_every_obstacle_kind_has_collision_and_visual_builders():
    import inspect
    from src.simulation import world_builder
    src = inspect.getsource(world_builder.build_obstacle)
    for kind in ("pillar", "reef_tower", "wall", "wreck_hull", "wreck_tower", "boulder"):
        assert f'"{kind}"' in src
    assert src.count("createCollisionShape") >= 3 and src.count("createVisualShape") >= 3
    assert "createCollisionShape" in inspect.getsource(world_builder.build_seabed)
    assert "createVisualShape" in inspect.getsource(world_builder.build_seabed)


# ------------------------------------------------------------------ sensors act on the generated geometry
def sonar_ranges(world, pos=(5.0, 5.0, -5.0)):
    from src.sensors.models import TruthSnapshot
    from src.sensors.sonar import SonarConfig
    from tests.helpers.analytic_sonar import AnalyticSonar
    cfg = SonarConfig(horizontal_rays=21, vertical_rays=3, range_noise_std_m=0.0, dropout_probability=0.0,
                      detection_probability=1.0)
    t = TruthSnapshot(0.0, np.array(pos, float), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3))
    return AnalyticSonar(world, cfg).measure(t)


def test_generated_obstacle_changes_the_sonar_measurement():
    with_obs = world_from_spec(gen("mapping_validation"))
    without = world_from_spec(gen("mapping_validation", obstacles={"enabled": False}))
    a, b = sonar_ranges(with_obs), sonar_ranges(without)
    assert a.detected.sum() > b.detected.sum()
    hit = np.flatnonzero(a.detected & ~b.detected)
    assert len(hit) and (a.value[hit] < 30.0).all()


def test_mapping_validation_scenario_places_a_pillar_within_sonar_range_of_the_start():
    s = gen("mapping_validation")
    p = [o for o in s.obstacles if o.kind == "pillar"][0]
    start = SIM["start_position"]
    assert np.hypot(p.x - start[0], p.y - start[1]) < CFG["sensors"]["sonar"]["max_range_m"]


def test_terrain_changes_what_a_downward_ray_sees():
    flat = world_from_spec(gen("open_terrain"))
    hilly_spec = gen("terrain_variation", 3)
    hilly = world_from_spec(hilly_spec)
    o = np.array([30.0, 30.0, -5.0]); d = np.array([[0.0, 0.0, -1.0]])
    r_flat = flat.ray_caster(o[None], o[None] + d * 30)[0]
    r_hill = hilly.ray_caster(o[None], o[None] + d * 30)[0]
    assert r_flat == pytest.approx(o[2] - SIM["seabed_z"], abs=0.2)           # 10 m of water below the vehicle
    assert r_hill == pytest.approx(o[2] - hilly_spec.seabed_height(30.0, 30.0), abs=0.25)
    assert abs(r_hill - r_flat) > 0.05

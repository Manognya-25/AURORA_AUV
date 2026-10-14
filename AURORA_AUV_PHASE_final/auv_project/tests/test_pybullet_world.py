"""REAL PyBullet integration for the Phase 4 procedural environment, sonar and mission.

Every scenario is built in real PyBullet and verified: collision + visual geometry, determinism, ray-by-ray
agreement with the generator's own geometry, real Sonar interaction, and a real mission run through the
simulation loop. Skipped (and reported as skipped) ONLY if PyBullet is not importable; set
AURORA_REQUIRE_PYBULLET=1 to make a missing PyBullet a hard failure (see tests/conftest.py)."""
import numpy as np
import pytest

pb = pytest.importorskip("pybullet")

from src.mapping.map_types import OCCUPIED, UNKNOWN                      # noqa: E402
from src.sensors import Sonar, SonarConfig                                # noqa: E402
from src.simulation.auv import AUV                                        # noqa: E402
from src.simulation.run_simulation import run                             # noqa: E402
from src.simulation.truth import snapshot_from_pybullet                   # noqa: E402
from tests.helpers import pybullet_checks as chk                          # noqa: E402


def with_world(scenario, seed=42, **over):
    client, cfg, env = chk.build(pb, scenario, seed, **over)
    return client, cfg, env


def test_every_scenario_builds_real_collision_and_visual_geometry():
    for scenario in chk.PROCEDURAL:
        client, cfg, env = with_world(scenario, 7)
        try:
            g = chk.geometry_report(pb, client, env)
            assert g["bodies"] == g["expected_bodies"], (scenario, g)
            assert g["missing_collision"] == [] and g["missing_visual"] == [], (scenario, g)
            assert g["seabed_is_mesh"], scenario
            assert g["max_position_error_m"] < 1e-5, (scenario, g["max_position_error_m"])
            if scenario != "open_terrain":
                assert g["obstacles"] > 0, scenario
        finally:
            pb.disconnect(client)


def test_scenarios_differ_from_each_other_in_real_pybullet():
    counts, relief = {}, {}
    for scenario in chk.PROCEDURAL:
        client, cfg, env = with_world(scenario, 7)
        try:
            counts[scenario] = len(env.obstacle_ids)
            relief[scenario] = float(np.ptp(env.world_spec.heights))
        finally:
            pb.disconnect(client)
    assert counts["open_terrain"] == 0
    assert counts["dense_obstacles"] > counts["sparse_obstacles"] > 0
    assert counts["narrow_corridor"] >= 2
    assert relief["terrain_variation"] > relief["open_terrain"] + 0.5


def test_generation_is_deterministic_in_real_pybullet_and_the_seed_matters():
    for scenario in chk.SEED_SENSITIVE:
        layouts = []
        for seed in (11, 11, 12):
            client, cfg, env = with_world(scenario, seed)
            try:
                layouts.append((chk.geometry_report(pb, client, env)["positions"], env.world_spec.heights.copy()))
            finally:
                pb.disconnect(client)
        assert layouts[0][0] == layouts[1][0] and np.array_equal(layouts[0][1], layouts[1][1]), scenario
        assert layouts[0][0] != layouts[2][0] or not np.array_equal(layouts[0][1], layouts[2][1]), scenario


def test_real_pybullet_rays_agree_with_the_generated_geometry_in_every_scenario():
    report = {}
    for scenario in chk.PROCEDURAL:
        client, cfg, env = with_world(scenario, 7)
        try:
            report[scenario] = r = chk.ray_agreement(pb, client, env, cfg)
        finally:
            pb.disconnect(client)
        assert r["hit_agreement"] >= 0.95, (scenario, r)                  # same hit / no-hit decision
        assert r["range_within_0.5m"] >= 0.90, (scenario, r)              # and the same distance
        if r["analytic_hits"] > 0:
            assert r["pybullet_hits"] > 0, (scenario, r)
    assert report["mapping_validation"]["pybullet_hits"] > 0               # obstacles really are in the rays' way


def test_seabed_mesh_matches_the_generated_height_function():
    for scenario in ("terrain_variation", "mixed_terrain_obstacles", "sparse_obstacles"):
        client, cfg, env = with_world(scenario, 5)
        try:
            r = chk.downward_ray_vs_seabed(pb, client, env)
            assert r["samples"] >= 2 and r["max_abs_error_m"] < 0.6, (scenario, r)
        finally:
            pb.disconnect(client)


def test_real_sonar_sees_the_generated_pillar_and_not_an_empty_world():
    client, cfg, env = with_world("mapping_validation")
    try:
        auv = AUV(cfg["simulation"]["urdf_path"], (5.0, 5.0, -5.0), cfg["simulation"], physics_client=client)
        sc = SonarConfig(horizontal_rays=61, vertical_rays=3, horizontal_fov_deg=90.0, vertical_fov_deg=4.0,
                         range_noise_std_m=0.0, dropout_probability=0.0, detection_probability=1.0, max_range_m=30.0)
        m = Sonar(auv.body_id, client, sc).measure(snapshot_from_pybullet(auv.body_id, 0.0, client))
        assert m.valid and int(m.detected.sum()) > 0
        pillar = [o for o in env.world_spec.obstacles if o.kind == "pillar"][0]
        origin = np.array([5.8, 5.0])                                        # 0.8 m mount offset along +x at yaw 0
        expect = float(np.hypot(pillar.x - origin[0], pillar.y - origin[1]) - pillar.radius)
        nearest = float(m.value[m.detected].min())
        assert expect - 0.1 <= nearest <= expect + 0.8, (nearest, expect)    # closest hit ~ distance to the pillar face
    finally:
        pb.disconnect(client)
    client2, cfg2, env2 = with_world("open_terrain")
    try:
        auv2 = AUV(cfg2["simulation"]["urdf_path"], (5.0, 5.0, -5.0), cfg2["simulation"], physics_client=client2)
        m2 = Sonar(auv2.body_id, client2, sc).measure(snapshot_from_pybullet(auv2.body_id, 0.0, client2))
        assert int(m2.detected.sum()) < int(m.detected.sum())                # no obstacles -> fewer / no detections
    finally:
        pb.disconnect(client2)


def mission(scenario, seed=7, max_steps=4000, **extra):
    override = {"environment": {"scenario": scenario, "seed": seed}, "simulation": {"max_steps": max_steps},
                "dashboard": {"realtime_factor": 0.0}}
    override.update(extra)
    return run(mode_override="DIRECT", cfg_override=override)


def test_auv_flies_a_real_mission_through_every_scenario_and_maps_it_from_sonar():
    for scenario in chk.PROCEDURAL:
        result = mission(scenario)
        m = result["mapping"]["mapping"]
        assert result["final_status"] in ("COMPLETED", "ENDED_BEFORE_GOAL"), (scenario, result["final_status"])
        assert m["update_count"] > 20, (scenario, m)                          # sonar really drove map updates
        # A real mission maps a real, bounded part of the grid. The sonar is forward-facing (90 deg x 30 m, the
        # documented mapper contract), so 40 s down the mission corridor always leaves a large unobserved region
        # behind/beside the vehicle: evidence exists, and the map never claims to be fully explored. The stronger
        # "most of the grid is still unknown" expectation only holds in the confined scenarios (corridors/obstacle
        # fields, where hits cap the rays early) - in open water the free-space fan legitimately covers more.
        total = m["known_cells"] + m["unknown_cells"]
        assert m["free_cells"] > 0, scenario                                 # sonar really wrote free space
        assert 0 < m["unknown_cells"] < total, (scenario, m)                  # partial: nothing is claimed behind it
        assert m["coverage_percent"] < 75.0, (scenario, m["coverage_percent"])  # ... and it never maps the whole map
        if scenario in ("narrow_corridor", "mapping_validation"):
            assert m["unknown_cells"] > 0.5 * total, (scenario, m)            # confined: most of it stays unexplored
        assert result["environment"]["scenario"] == scenario and result["environment"]["procedural"] is True
        assert result["collisions"] == 0, (scenario, result["collisions"])    # obstacles are kept off the mission path
        if scenario in ("mapping_validation", "sparse_obstacles", "dense_obstacles", "mixed_terrain_obstacles", "narrow_corridor"):
            assert m["occupied_cells"] > 0, (scenario, m)                     # obstacles in view were mapped


def test_mission_with_mapping_disabled_still_runs_and_reports_no_map():
    result = mission("sparse_obstacles", max_steps=800, mapping={"enabled": False})
    assert result["final_status"] in ("COMPLETED", "ENDED_BEFORE_GOAL")
    assert result["mapping"] is None or result["mapping"]["mapping"]["update_count"] == 0


def test_mission_runs_in_both_dynamics_modes_on_the_procedural_world():
    for model in ("legacy", "6dof"):
        result = mission("mapping_validation", max_steps=1500, simulation={"max_steps": 1500, "dynamics": {"model": model}})
        assert result["mapping"]["mapping"]["update_count"] > 10, model
        assert result["collisions"] == 0, model


def test_legacy_week4_world_still_builds_unchanged():
    result = run(mode_override="DIRECT", cfg_override={"simulation": {"max_steps": 600}, "dashboard": {"realtime_factor": 0.0}})
    assert result["environment"]["scenario"] == "legacy_week4" and result["environment"]["procedural"] is False


"""Shared REAL-PyBullet environment checks (used by tests/test_pybullet_world.py and
tools/verify_pybullet_environment.py). Requires PyBullet; the generator's analytic twin
(tests/helpers/analytic_world.py) acts as the independent oracle for what PyBullet's rays should hit."""
from __future__ import annotations

import copy

import numpy as np
import yaml

from tests.helpers.analytic_world import world_from_spec

PROCEDURAL = ["open_terrain", "sparse_obstacles", "dense_obstacles", "narrow_corridor",
              "terrain_variation", "mixed_terrain_obstacles", "mapping_validation"]
SEED_SENSITIVE = ["sparse_obstacles", "dense_obstacles", "terrain_variation", "mixed_terrain_obstacles"]


def load_cfg(scenario: str, seed: int = 42, **env_over) -> dict:
    with open("configs/config.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["environment"].update(scenario=scenario, seed=seed, **env_over)
    return cfg


def build(pb, scenario: str, seed: int = 42, **env_over):
    """Connect a fresh DIRECT client and build the Phase 4 world exactly as the simulation does."""
    from src.simulation.environment import Environment
    client = pb.connect(pb.DIRECT)
    cfg = load_cfg(scenario, seed, **env_over)
    env = Environment(cfg, physics_client=client)
    return client, cfg, env


def geometry_report(pb, client, env) -> dict:
    """Does PyBullet really hold collision + visual geometry for the seabed and every generated obstacle?"""
    ids = list(env.obstacle_ids)
    missing_col = [b for b in ids + [env.seabed_id] if not pb.getCollisionShapeData(b, -1, physicsClientId=client)]
    missing_vis = [b for b in ids + [env.seabed_id] if not pb.getVisualShapeData(b, physicsClientId=client)]
    seabed_geom = pb.getCollisionShapeData(env.seabed_id, -1, physicsClientId=client)[0][2]
    positions = [tuple(round(v, 6) for v in pb.getBasePositionAndOrientation(b, physicsClientId=client)[0]) for b in ids]
    spec = env.world_spec
    pos_err = max((max(abs(p[0] - o.x), abs(p[1] - o.y)) for p, o in zip(positions, spec.obstacles)), default=0.0)
    return {"bodies": pb.getNumBodies(physicsClientId=client), "expected_bodies": 1 + len(spec.obstacles),
            "obstacles": len(ids), "missing_collision": missing_col, "missing_visual": missing_vis,
            "seabed_is_mesh": seabed_geom == pb.GEOM_MESH, "max_position_error_m": float(pos_err),
            "positions": positions}


def vantage_points(cfg: dict) -> list:
    sim = cfg["simulation"]
    pts = [list(map(float, sim["start_position"]))] + [list(map(float, w)) for w in sim["waypoints"]]
    return pts


def ray_agreement(pb, client, env, cfg, max_range=30.0, n_az=72, elevations_deg=(-20.0, -8.0, 0.0, 8.0)) -> dict:
    """Cast the same rays through REAL PyBullet and through the generator's analytic geometry and compare."""
    world = world_from_spec(env.world_spec)
    starts, ends = [], []
    for p in vantage_points(cfg):
        for el in np.deg2rad(elevations_deg):
            for az in np.linspace(0, 2 * np.pi, n_az, endpoint=False):
                d = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
                starts.append(p)
                ends.append(np.asarray(p) + max_range * d)
    starts, ends = np.asarray(starts), np.asarray(ends)
    res = pb.rayTestBatch(starts.tolist(), ends.tolist(), physicsClientId=client)
    pb_range = np.array([r[2] * max_range if r[0] != -1 else np.inf for r in res])
    an = np.asarray(world.ray_caster(starts, ends), float)
    an_range = np.where(an < max_range, an, np.inf)
    pb_hit, an_hit = np.isfinite(pb_range), np.isfinite(an_range)
    both = pb_hit & an_hit
    err = np.abs(pb_range[both] - an_range[both])
    return {"rays": len(res), "pybullet_hits": int(pb_hit.sum()), "analytic_hits": int(an_hit.sum()),
            "hit_agreement": float((pb_hit == an_hit).mean()), "both_hit": int(both.sum()),
            "range_within_0.5m": float((err < 0.5).mean()) if both.any() else 1.0,
            "median_abs_error_m": float(np.median(err)) if both.any() else 0.0,
            "max_abs_error_m": float(err.max()) if both.any() else 0.0}


def downward_ray_vs_seabed(pb, client, env, points=((10.0, 10.0), (30.0, 40.0), (70.0, 20.0), (55.0, 80.0))) -> dict:
    """PyBullet's seabed (mesh) vs the generator's height function."""
    errs = []
    for x, y in points:
        hit = pb.rayTest([x, y, -3.0], [x, y, -60.0], physicsClientId=client)[0]
        assert hit[0] == env.seabed_id or hit[0] in env.obstacle_ids, f"no seabed under ({x}, {y})"
        z = -3.0 + hit[2] * (-57.0)
        if hit[0] == env.seabed_id:
            errs.append(abs(z - env.world_spec.seabed_height(x, y)))
    return {"samples": len(errs), "max_abs_error_m": float(max(errs)) if errs else 0.0}

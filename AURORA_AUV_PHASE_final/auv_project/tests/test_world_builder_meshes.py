"""PyBullet-independent guards for the mesh data handed to PyBullet (the 'createVisualShape failed' bug class).

Real PyBullet creation of the same meshes is exercised by tests/test_pybullet_world.py."""
import numpy as np
import yaml

from src.simulation import world_builder as wb
from src.simulation.world_generator import SCENARIOS, generate_world

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))


def spec_for(scenario, seed=7):
    env = dict(CFG["environment"], scenario=scenario, seed=seed)
    return generate_world(env, CFG["simulation"], (100, 100, 1.0))


def test_seabed_mesh_is_valid_bounded_and_covers_the_whole_extent_for_every_scenario():
    for sc in SCENARIOS:
        spec = spec_for(sc)
        verts, idx, normals = wb._heightfield_mesh(spec)
        v, ix = np.asarray(verts), np.asarray(idx)
        assert v.shape[1] == 3 and np.isfinite(v).all(), sc
        assert len(ix) % 3 == 0 and ix.min() == 0 and ix.max() == len(v) - 1, sc
        assert len(ix) // 3 <= wb.MAX_SEABED_TRIANGLES and len(normals) == len(verts), sc
        ex, pad = spec.extent, wb.SEABED_PAD_M
        # the mesh reaches PAD metres beyond the raster on every side: the generator's seabed is edge-clamped, not
        # bounded, so rays leaving the arena must still find a seabed (this was the open_terrain 0.854 failure)
        assert np.isclose(v[:, 0].min(), ex[0] - pad) and np.isclose(v[:, 0].max(), ex[1] + pad), sc
        assert np.isclose(v[:, 1].min(), ex[2] - pad) and np.isclose(v[:, 1].max(), ex[3] + pad), sc
        assert np.all(np.asarray(normals)[:, 2] > 0), sc                  # a seabed faces up


def test_seabed_mesh_follows_the_height_function():
    spec = spec_for("terrain_variation", 5)
    v = np.asarray(wb._heightfield_mesh(spec)[0])
    err = [abs(z - spec.seabed_height(x, y)) for x, y, z in v[::7]]
    assert max(err) < 1e-9                                                  # nodes are exact samples


def test_boulder_hull_has_indices_and_outward_winding():
    spec = spec_for("dense_obstacles", 7)
    boulders = [o for o in spec.obstacles if o.kind == "boulder"]
    assert boulders
    for o in boulders:
        verts, idx, normals = wb._convex_hull_mesh(o.mesh_vertices)       # indices are what createVisualShape needs
        v, t = np.asarray(verts), np.asarray(idx).reshape(-1, 3)
        assert len(t) >= 4 and t.max() < len(v) and len(normals) == len(v)
        centre = v.mean(axis=0)
        n = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
        assert np.all(np.einsum("ij,ij->i", n, v[t[:, 0]] - centre) > 0)


def test_shared_read_returns_the_current_file_content(tmp_path):
    from src.simulation.live_io import atomic_write_json, read_bytes_shared, read_json_retry
    f = tmp_path / "state.json"
    atomic_write_json(f, {"n": 1})
    assert read_json_retry(f) == {"n": 1} and b'"n"' in read_bytes_shared(f)
    assert read_json_retry(tmp_path / "missing.json") is None


def test_every_budget_is_valid_and_the_finer_budget_is_more_faithful():
    spec = spec_for("terrain_variation", 7)
    errs = []
    for budget in wb.SEABED_TRIANGLE_BUDGETS:
        v, i, n = wb._heightfield_mesh(spec, budget)
        assert len(i) // 3 <= budget and len(i) % 3 == 0 and np.isfinite(np.asarray(v)).all()
        errs.append(len(v))
    assert errs == sorted(errs, reverse=True)                              # finest budget first -> most vertices


def test_padded_mesh_matches_the_edge_clamped_height_outside_the_arena():
    spec = spec_for("terrain_variation", 7)
    v = np.asarray(wb._heightfield_mesh(spec)[0])
    outside = v[(v[:, 0] < 0) | (v[:, 0] > 100) | (v[:, 1] < 0) | (v[:, 1] > 100)]
    assert len(outside) > 0
    assert max(abs(z - spec.seabed_height(x, y)) for x, y, z in outside) < 1e-9

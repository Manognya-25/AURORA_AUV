"""WorldSpec -> real PyBullet geometry (collision + visual) for the procedural scenarios.

Every obstacle and the seabed get BOTH a collision shape and a visual shape, so the sonar's
rayTestBatch, DVL-relative terrain, and collision checks all act on the same geometry the GUI shows.
  seabed  : always one static concave triangle mesh built from the height field (flat worlds included)
  pillar / reef_tower : cylinder        wall / wreck parts : box (yawed)
  boulder : convex hull of the generated vertices
"""
from __future__ import annotations

import numpy as np

from src.simulation.world_generator import ObstacleSpec, WorldSpec

SEABED_RGBA = [0.34, 0.41, 0.38, 1.0]       # muted blue-green silt: reads as seabed under the underwater light (rendering only)
OBSTACLE_RGBA = {"pillar": [0.66, 0.46, 0.30, 1], "reef_tower": [0.88, 0.42, 0.36, 1], "wall": [0.52, 0.55, 0.64, 1],
                 "boulder": [0.50, 0.47, 0.42, 1], "wreck_hull": [0.52, 0.40, 0.30, 1],
                 "wreck_tower": [0.56, 0.44, 0.34, 1]}


# PyBullet passes user meshes through fixed-size command buffers; large meshes make createVisualShape /
# createCollisionShape fail with a bare "createVisualShape failed". The seabed is therefore tessellated onto a
# coarser lattice that always fits (<= MAX_SEABED_TRIANGLES), and every mesh is validated before it is sent.
MAX_SEABED_TRIANGLES = 3000                 # proven-safe size (the coarsest fallback below is never larger)
# Finest first. Ray fidelity on ridged terrain improves with resolution (3000 tris: ~91 % of rays within 0.5 m
# of the analytic surface on terrain_variation/seed 7; 6000 tris: ~97 %), but PyBullet's mesh buffers are
# limited, so build_seabed() tries each budget in turn and uses the finest one this PyBullet accepts.
SEABED_TRIANGLE_BUDGETS = (6000, 3000, 1200)
MAX_MESH_VERTICES = 4000


def _vertex_normals(verts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Area-weighted per-vertex normals (one per vertex, as createVisualShape expects)."""
    v = np.asarray(verts, float)
    t = np.asarray(tris, int).reshape(-1, 3)
    fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, t[:, k], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ln[ln < 1e-12] = 1.0
    return n / ln


def _check_mesh(name: str, verts, indices) -> None:
    v = np.asarray(verts, float)
    ix = np.asarray(indices, int)
    if v.ndim != 2 or v.shape[1] != 3 or len(v) < 4 or not np.all(np.isfinite(v)):
        raise ValueError(f"{name}: invalid vertices (shape {v.shape}, finite={bool(np.all(np.isfinite(v)))})")
    if len(ix) == 0 or len(ix) % 3 != 0 or ix.min() < 0 or ix.max() >= len(v):
        raise ValueError(f"{name}: invalid triangle indices (count {len(ix)}, vertices {len(v)})")
    if len(v) > MAX_MESH_VERTICES * 3:
        raise ValueError(f"{name}: mesh too large for PyBullet ({len(v)} vertices)")


# The generator defines the seabed as z = seabed_height(x, y), CLAMPED to the raster edge outside the raster
# (the seabed does not stop at the edge of the arena). The PyBullet mesh must represent the same surface,
# otherwise rays leaving the arena (sonar max range 30 m, AUV waypoints near the border) hit PyBullet's
# nothing while the generator's geometry says "seabed". The mesh is therefore extended by SEABED_PAD_M on
# every side with the edge-clamped height (extra nodes only on the outer ring: no extra resolution cost).
SEABED_PAD_M = 80.0


def _lattice(lo: float, hi: float, n_cells: int, pad: float) -> np.ndarray:
    inner = np.linspace(lo, hi, n_cells + 1)
    return np.concatenate([[lo - pad], inner, [hi + pad]])


def _heightfield_mesh(spec: WorldSpec, max_triangles: int | None = None):
    """Triangle mesh of the seabed surface: the raster extent sampled on a lattice coarse enough to stay within
    PyBullet's limits, plus one padding ring out to SEABED_PAD_M with the edge-clamped height.
    Nodes are exact samples of spec.seabed_height. Returns (vertices, flat int indices, per-vertex normals)."""
    rows, cols = spec.heights.shape
    budget = int(max_triangles or MAX_SEABED_TRIANGLES)
    k = 1
    while 2 * (int(np.ceil(rows / k)) + 2) * (int(np.ceil(cols / k)) + 2) > budget:
        k += 1
    ny, nx = int(np.ceil(rows / k)), int(np.ceil(cols / k))
    ex = spec.extent
    xs, ys = _lattice(ex[0], ex[1], nx, SEABED_PAD_M), _lattice(ex[2], ex[3], ny, SEABED_PAD_M)
    z = np.array([[spec.seabed_height(float(x), float(y)) for x in xs] for y in ys])    # clamps outside the raster
    X, Y = np.meshgrid(xs, ys)
    verts = np.stack([X, Y, z], axis=-1).reshape(-1, 3)
    idx = np.arange(len(ys) * len(xs)).reshape(len(ys), len(xs))
    a, b, c, d = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    tris = np.concatenate([np.stack([a, b, d], -1).reshape(-1, 3), np.stack([a, d, c], -1).reshape(-1, 3)])
    flat = tris.reshape(-1)
    _check_mesh("seabed", verts, flat)
    return verts.tolist(), flat.tolist(), _vertex_normals(verts, tris).tolist()


def _convex_hull_mesh(points):
    """Outward-wound triangles of the convex hull of `points` -> (hull vertices, flat indices, normals)."""
    from scipy.spatial import ConvexHull
    pts = np.asarray(points, float)
    hull = ConvexHull(pts)
    used = np.unique(hull.simplices)
    remap = {int(o): i for i, o in enumerate(used)}
    verts = pts[used]
    centre = verts.mean(axis=0)
    tris = []
    for tri in hull.simplices:
        a, b, c = (pts[tri[0]], pts[tri[1]], pts[tri[2]])
        if np.dot(np.cross(b - a, c - a), (a + b + c) / 3.0 - centre) < 0:      # make the winding outward
            tri = (tri[0], tri[2], tri[1])
        tris.append([remap[int(t)] for t in tri])
    tris = np.asarray(tris, int)
    flat = tris.reshape(-1)
    _check_mesh("boulder", verts, flat)
    return verts.tolist(), flat.tolist(), _vertex_normals(verts, tris).tolist()


def build_seabed(spec: WorldSpec, physics_client: int = 0) -> int:
    """ALWAYS a static triangle mesh (collision + visual), flat or not, so every scenario has the same kind of
    seabed body and sonar/DVL rays hit one consistent surface. Uses the finest mesh PyBullet accepts
    (SEABED_TRIANGLE_BUDGETS); a rejected size is reported on stderr and the next coarser mesh is tried, a
    failure of the coarsest raises. The chosen size is available as build_seabed.last_info."""
    import sys

    import pybullet as p
    errors = []
    for budget in SEABED_TRIANGLE_BUDGETS:
        verts, indices, normals = _heightfield_mesh(spec, budget)
        try:
            col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=indices, meshScale=[1, 1, 1],
                                         flags=p.GEOM_FORCE_CONCAVE_TRIMESH, physicsClientId=physics_client)
            vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=indices, normals=normals,
                                      meshScale=[1, 1, 1], rgbaColor=SEABED_RGBA, physicsClientId=physics_client)
        except p.error as exc:
            errors.append(f"{len(indices) // 3} triangles: {exc}")
            print(f"[world_builder] PyBullet rejected a {len(indices) // 3}-triangle seabed mesh ({exc}); "
                  f"trying a coarser one", file=sys.stderr)
            continue
        if col < 0 or vis < 0:
            errors.append(f"{len(indices) // 3} triangles: shape ids collision={col} visual={vis}")
            continue
        build_seabed.last_info = {"triangles": len(indices) // 3, "vertices": len(verts), "budget": budget}
        return p.createMultiBody(0.0, col, vis, basePosition=[0, 0, 0], physicsClientId=physics_client)
    raise RuntimeError("PyBullet rejected every seabed mesh size: " + "; ".join(errors))


build_seabed.last_info = {}


def build_obstacle(o: ObstacleSpec, physics_client: int = 0) -> int:
    import pybullet as p
    rgba = OBSTACLE_RGBA.get(o.kind, [0.5, 0.5, 0.5, 1])
    pos = [o.x, o.y, o.z_centre]
    orn = p.getQuaternionFromEuler([0, 0, o.yaw])
    if o.kind in ("pillar", "reef_tower"):
        col = p.createCollisionShape(p.GEOM_CYLINDER, radius=o.radius, height=o.height, physicsClientId=physics_client)
        vis = p.createVisualShape(p.GEOM_CYLINDER, radius=o.radius, length=o.height, rgbaColor=rgba,
                                  physicsClientId=physics_client)
    elif o.kind in ("wall", "wreck_hull", "wreck_tower"):
        half = [o.half_extents[0], o.half_extents[1], o.height / 2.0]
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=physics_client)
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=rgba, physicsClientId=physics_client)
    elif o.kind == "boulder":
        # createVisualShape(GEOM_MESH, vertices=...) REQUIRES triangle indices (a bare point cloud is rejected);
        # build an explicit convex-hull triangle mesh and give the identical mesh to collision and visual.
        verts, indices, normals = _convex_hull_mesh(o.mesh_vertices)
        col = p.createCollisionShape(p.GEOM_MESH, vertices=verts, indices=indices, meshScale=[1, 1, 1],
                                     physicsClientId=physics_client)
        vis = p.createVisualShape(p.GEOM_MESH, vertices=verts, indices=indices, normals=normals,
                                  meshScale=[1, 1, 1], rgbaColor=rgba, physicsClientId=physics_client)
        zmin = float(np.asarray(o.mesh_vertices, float)[:, 2].min())
        pos = [o.x, o.y, o.z_bottom - zmin]               # vertices are about their own centre; z_bottom = lowest point
        orn = [0, 0, 0, 1]
    else:
        raise ValueError(f"cannot build obstacle kind {o.kind!r}")
    if col < 0 or vis < 0:
        raise RuntimeError(f"{o.kind} at ({o.x:.1f},{o.y:.1f}): shape creation failed (collision={col}, visual={vis})")
    return p.createMultiBody(0.0, col, vis, basePosition=pos, baseOrientation=orn, physicsClientId=physics_client)


def build_world(spec: WorldSpec, physics_client: int = 0) -> tuple[int, list]:
    """Create all geometry. Returns (seabed_body_id, [obstacle_body_ids])."""
    return build_seabed(spec, physics_client), [build_obstacle(o, physics_client) for o in spec.obstacles]

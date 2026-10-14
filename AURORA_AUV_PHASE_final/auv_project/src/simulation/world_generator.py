"""Deterministic procedural underwater world generation (pure NumPy; no PyBullet import).

A WorldSpec is the *description* of the simulated world (seabed height field + obstacle primitives).
src/simulation/world_builder.py turns it into real PyBullet collision+visual geometry; the sonar,
DVL and collision checks then act on that geometry. Nothing here is ever read by the mapper: it is the
simulation's ground truth (labelled as such in telemetry).

DETERMINISM: every random draw comes from an explicit np.random.default_rng built from
SeedSequence([seed, stream]); the seabed and obstacle streams are independent. Same scenario + seed +
config  ->  identical WorldSpec (tested); a different seed gives a different world (tested).

FRAMES: world +Z up (z negative underwater, surface z = 0). Seabed height field z_s(x, y) is sampled on
a regular grid [row<->y, col<->x] with cell CENTRES at origin + (idx + 0.5)*res, bilinearly
interpolated by seabed_height(). Obstacles stand on the seabed and extend up to z_top.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

OBSTACLE_KINDS = ("pillar", "boulder", "wall", "reef_tower", "wreck_hull", "wreck_tower")

# Used when an operator asks for obstacles.count > 0 on a scenario whose own mix is empty (open_terrain,
# narrow_corridor, mapping_validation): the scenario still gets EXACTLY the requested number of bodies,
# drawn from a representative mix. A scenario that does not ask for obstacles (count 0) is unchanged.
DEFAULT_OBSTACLE_MIX = {"pillar": 3, "wall": 2, "boulder": 3, "reef_tower": 1}

# Scenario defaults. Everything here can be overridden from config.yaml (environment.*).
SCENARIOS: dict[str, dict] = {
    "open_terrain": {
        "seabed": {"type": "flat", "variation_m": 0.0, "roughness_m": 0.0},
        "obstacles": {"count": 0, "mix": {}},
        "description": "Flat seabed, no obstacles.",
    },
    "sparse_obstacles": {
        "seabed": {"type": "rolling", "variation_m": 1.5, "roughness_m": 0.05},
        "obstacles": {"count": 8, "mix": {"pillar": 3, "wall": 2, "boulder": 3}},
        "description": "Gentle seabed with a few tall pillars/walls and low boulders.",
    },
    "dense_obstacles": {
        "seabed": {"type": "rolling", "variation_m": 1.5, "roughness_m": 0.08},
        "obstacles": {"count": 30, "mix": {"pillar": 4, "wall": 2, "boulder": 3, "reef_tower": 1}},
        "description": "Many obstacles; the mission corridor is kept clear.",
    },
    "narrow_corridor": {
        "seabed": {"type": "flat", "variation_m": 0.0, "roughness_m": 0.0},
        "obstacles": {"count": 0, "mix": {}},
        "corridor": {"gap_m": 8.0, "length_m": 36.0, "thickness_m": 1.0},
        "description": "Two parallel walls forming a corridor along the mission line.",
    },
    "terrain_variation": {
        "seabed": {"type": "ridged", "variation_m": 4.0, "roughness_m": 0.15},
        "obstacles": {"count": 2, "mix": {"pillar": 1, "boulder": 1}},
        "description": "Strong seabed relief (ridges, depressions); almost no obstacles.",
    },
    "mixed_terrain_obstacles": {
        "seabed": {"type": "ridged", "variation_m": 3.0, "roughness_m": 0.12},
        "obstacles": {"count": 14, "mix": {"pillar": 3, "wall": 2, "boulder": 4, "reef_tower": 2, "wreck": 1}},
        "description": "Relief plus pillars, walls, reef towers and a wreck-like structure.",
    },
    "mapping_validation": {
        "seabed": {"type": "flat", "variation_m": 0.0, "roughness_m": 0.0},
        "obstacles": {"count": 0, "mix": {}},
        "explicit": [
            {"kind": "pillar", "x": 16.0, "y": 9.0, "radius_m": 0.8, "z_top_m": -2.0},
            {"kind": "wall", "x": 26.0, "y": 12.0, "half_x_m": 0.5, "half_y_m": 12.0, "yaw_deg": 0.0, "z_top_m": -2.0},
        ],
        "description": "Fixed obstacles within sonar range of the default start; for mapping validation.",
    },
}
LEGACY_SCENARIO = "legacy_week4"
ALL_SCENARIOS = (LEGACY_SCENARIO,) + tuple(SCENARIOS)


@dataclass
class ObstacleSpec:
    kind: str
    x: float
    y: float
    z_bottom: float
    z_top: float
    radius: float = 0.0                 # pillar / reef_tower / boulder (mean radius)
    half_extents: tuple = (0.0, 0.0)    # wall / wreck: (half_x, half_y) in the obstacle's own frame
    yaw: float = 0.0                    # rad, about +Z
    mesh_vertices: list | None = None   # boulder: convex-hull vertices relative to (x, y, z_centre)
    group: int = -1                     # reef towers / wreck parts of one structure share a group id
    footprint: list = field(default_factory=list)   # [(x, y), ...] sample points (placement + debug raster)
    footprint_radius: float = 0.0

    @property
    def z_centre(self) -> float:
        return 0.5 * (self.z_bottom + self.z_top)

    @property
    def height(self) -> float:
        return self.z_top - self.z_bottom

    def to_dict(self) -> dict:
        return {"kind": self.kind, "x": self.x, "y": self.y, "z_bottom": self.z_bottom, "z_top": self.z_top,
                "radius": self.radius, "half_extents": list(self.half_extents), "yaw": self.yaw,
                "group": self.group}


@dataclass
class WorldSpec:
    scenario: str
    seed: int
    origin: tuple                       # world (x, y) of the lower-left corner of the seabed raster
    resolution: float                   # metres per height-field cell
    heights: np.ndarray                 # (rows, cols) seabed z, cell-centre samples
    seabed_type: str
    obstacles: list
    mission_points: list                # [(x, y, z)], start first
    meta: dict = field(default_factory=dict)

    @property
    def extent(self) -> tuple:
        r, c = self.heights.shape
        return (self.origin[0], self.origin[0] + c * self.resolution,
                self.origin[1], self.origin[1] + r * self.resolution)

    @property
    def is_flat(self) -> bool:
        return bool(np.ptp(self.heights) < 1e-9)

    def scene_payload(self, max_cells: int = 64) -> dict:
        """The world geometry for the dashboard "Underwater view": a down-sampled seabed height field and the obstacle list.

        GROUND-TRUTH world geometry (what the generator built, the same bodies PyBullet simulates). It is only DRAWN; the
        mapper, planner and controller never receive it."""
        r, c = self.heights.shape
        sr, sc = max(1, int(np.ceil(r / max_cells))), max(1, int(np.ceil(c / max_cells)))
        h = self.heights[::sr, ::sc]
        return {"label": "GROUND TRUTH WORLD GEOMETRY (display only)", "scenario": self.scenario, "seed": self.seed,
                "seabed_type": self.seabed_type, "extent": [float(v) for v in self.extent],
                "x0": float(self.origin[0] + 0.5 * self.resolution), "y0": float(self.origin[1] + 0.5 * self.resolution),
                "dx": float(self.resolution * sc), "dy": float(self.resolution * sr),
                "heights": np.round(h, 2).tolist(),
                "obstacles": [{"kind": o.kind, "x": round(float(o.x), 2), "y": round(float(o.y), 2),
                               "z_bottom": round(float(o.z_bottom), 2), "z_top": round(float(o.z_top), 2),
                               "radius": round(float(o.radius), 2), "half_extents": [round(float(v), 2) for v in o.half_extents],
                               "yaw": round(float(o.yaw), 3), "group": int(o.group)} for o in self.obstacles]}

    def seabed_height(self, x: float, y: float) -> float:
        """Bilinear seabed z at world (x, y); clamped to the raster edge outside it."""
        r, c = self.heights.shape
        fx = (x - self.origin[0]) / self.resolution - 0.5
        fy = (y - self.origin[1]) / self.resolution - 0.5
        fx, fy = float(np.clip(fx, 0, c - 1)), float(np.clip(fy, 0, r - 1))
        x0, y0 = int(np.floor(fx)), int(np.floor(fy))
        x1, y1 = min(x0 + 1, c - 1), min(y0 + 1, r - 1)
        tx, ty = fx - x0, fy - y0
        h = self.heights
        return float((1 - ty) * ((1 - tx) * h[y0, x0] + tx * h[y0, x1]) + ty * ((1 - tx) * h[y1, x0] + tx * h[y1, x1]))

    def obstacle_footprint_mask(self, geom) -> np.ndarray:
        """DEBUG/EVALUATION ONLY: rasterise obstacle footprints onto a map grid (GridGeometry).
        Never an input to mapping."""
        mask = np.zeros(geom.shape, bool)
        rows, cols = np.mgrid[0:geom.rows, 0:geom.cols]
        X, Y = geom.grid_to_world(rows, cols)
        for o in self.obstacles:
            pts = np.asarray(o.footprint if o.footprint else [(o.x, o.y)], float)
            rad = max(o.footprint_radius, 1e-6)
            for px, py in pts:
                mask |= (X - px) ** 2 + (Y - py) ** 2 <= rad ** 2
        return mask

    def summary(self) -> dict:
        kinds: dict[str, int] = {}
        for o in self.obstacles:
            kinds[o.kind] = kinds.get(o.kind, 0) + 1
        return {"scenario": self.scenario, "seed": self.seed, "seabed_type": self.seabed_type,
                "seabed_z_min": float(self.heights.min()), "seabed_z_max": float(self.heights.max()),
                "obstacle_count": len(self.obstacles), "structure_count": len({o.group for o in self.obstacles}),
                "obstacle_kinds": kinds,
                "extent": list(self.extent), "description": self.meta.get("description", ""),
                "placement_attempts": self.meta.get("placement_attempts"),
                "obstacles_requested": self.meta.get("obstacles_requested"),
                "obstacles_fixed": self.meta.get("obstacles_fixed", 0),
                "obstacles_placed": self.meta.get("obstacles_placed")}

    def signature(self) -> dict:
        """Everything that defines the world, for equality tests / provenance."""
        return {"scenario": self.scenario, "seed": self.seed, "heights": self.heights.copy(),
                "obstacles": [o.to_dict() for o in self.obstacles], "origin": self.origin,
                "resolution": self.resolution}


# ----------------------------------------------------------------------------- helpers
def _rng(seed: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(stream)]))


def _merge(base: dict, over: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if v is None:
            continue
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _seg_dist(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    L2 = float(ab @ ab)
    if L2 < 1e-12:
        return np.linalg.norm(points - a, axis=1)
    t = np.clip(((points - a) @ ab) / L2, 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def _dist_to_polyline(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    if len(poly) == 1:
        return np.linalg.norm(points - poly[0], axis=1)
    return np.min([_seg_dist(points, poly[i], poly[i + 1]) for i in range(len(poly) - 1)], axis=0)


def _axis_points(x, y, hx, hy, yaw, step=0.5) -> list:
    """Sample points along the long axis of an (hx, hy) box rotated by yaw."""
    long_is_x = hx >= hy
    half_long = hx if long_is_x else hy
    n = max(2, int(np.ceil(2 * half_long / step)) + 1)
    s = np.linspace(-half_long, half_long, n)
    local = np.stack([s, np.zeros(n)], 1) if long_is_x else np.stack([np.zeros(n), s], 1)
    c, sn = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -sn], [sn, c]])
    return (local @ R.T + np.array([x, y])).tolist()


# ----------------------------------------------------------------------------- seabed
def generate_seabed(rows: int, cols: int, resolution: float, base_z: float, seabed_cfg: dict,
                    seed: int, max_z: float) -> tuple[np.ndarray, dict]:
    """Height field (rows, cols). Returns (heights, info). Heights are clipped to <= max_z."""
    stype = str(seabed_cfg.get("type", "flat"))
    var = float(seabed_cfg.get("variation_m", 0.0))
    rough = float(seabed_cfg.get("roughness_m", 0.0))
    if stype not in ("flat", "rolling", "ridged"):
        raise ValueError(f"seabed.type must be flat|rolling|ridged, got {stype!r}")
    if stype == "flat" or (var == 0.0 and rough == 0.0):
        return np.full((rows, cols), base_z, np.float64), {"clipped_cells": 0}
    rng = _rng(seed, 1)
    sigma = max(2.0, 14.0 / resolution)
    f = gaussian_filter(rng.standard_normal((rows, cols)), sigma=sigma, mode="reflect")
    f = (f - f.mean()) / (f.std() + 1e-12)                         # zero-mean, unit-variance relief
    if stype == "ridged":
        g = gaussian_filter(rng.standard_normal((rows, cols)), sigma=sigma * 0.7, mode="reflect")
        g = (g - g.mean()) / (g.std() + 1e-12)
        f = 0.5 * f + 0.9 * (1.0 - 2.0 * np.abs(np.tanh(g)))      # sharp crests along zero-crossings
        f = (f - f.mean()) / (f.std() + 1e-12)
    heights = base_z + var * 0.4 * f                               # ~ +/-1 sigma == +/-0.4*variation
    if rough > 0:
        n = gaussian_filter(rng.standard_normal((rows, cols)), sigma=1.0)
        heights = heights + rough * n / (n.std() + 1e-12)
    clipped = int(np.sum(heights > max_z))
    heights = np.minimum(heights, max_z)
    return heights, {"clipped_cells": clipped}


# ----------------------------------------------------------------------------- obstacles
def _make_boulder(rng, x, y, z_ground, r, seed_id) -> ObstacleSpec:
    n = 28
    v = rng.standard_normal((n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    radial = 0.7 + 0.5 * rng.random(n)
    scale = np.array([1.0, 0.8 + 0.4 * rng.random(), 0.7 + 0.3 * rng.random()])
    verts = v * radial[:, None] * r * scale
    zc = z_ground + float(np.abs(verts[:, 2].min())) - 0.3         # partly embedded in the seabed
    return ObstacleSpec("boulder", x, y, zc - float(np.abs(verts[:, 2].min())), zc + float(verts[:, 2].max()),
                        radius=float(r), mesh_vertices=verts.tolist(), group=seed_id,
                        footprint=[(x, y)], footprint_radius=float(np.abs(verts[:, :2]).max()))


def _place(world_kind, rng, cfg, ctx):
    """Draw one candidate obstacle (or structure = list of ObstacleSpec) at a random location."""
    ex = ctx["extent"]
    m = ctx["edge_margin"]
    x = rng.uniform(ex[0] + m, ex[1] - m)
    y = rng.uniform(ex[2] + m, ex[3] + 0 - m)
    zt_lo, zt_hi = ctx["top_range"]
    ground = ctx["ground"]
    smin, smax = float(cfg["min_size_m"]), float(cfg["max_size_m"])
    gid = ctx["next_group"]
    if world_kind == "pillar":
        r = rng.uniform(smin, smax) / 2.0
        zt = rng.uniform(zt_lo, zt_hi)
        zb = ground(x, y, r) - 0.5
        return [ObstacleSpec("pillar", x, y, zb, zt, radius=r, footprint=[(x, y)], footprint_radius=r, group=gid)]
    if world_kind == "boulder":
        return [_make_boulder(rng, x, y, ground(x, y, smax / 2), rng.uniform(smin, smax) / 2.0, gid)]
    if world_kind == "wall":
        hx, hy = rng.uniform(2.0, 6.0), rng.uniform(0.25, 0.6)
        yaw = float(rng.uniform(0, np.pi))
        zt = rng.uniform(zt_lo, zt_hi)
        zb = ground(x, y, hx) - 0.5
        fp = _axis_points(x, y, hx, hy, yaw)
        return [ObstacleSpec("wall", x, y, zb, zt, half_extents=(hx, hy), yaw=yaw, footprint=fp,
                             footprint_radius=hy, group=gid)]
    if world_kind == "reef_tower":
        parts = []
        for _ in range(int(rng.integers(3, 6))):
            dx, dy = rng.normal(0, 1.4, 2)
            r = rng.uniform(smin, smax) / 3.0
            parts.append(ObstacleSpec("reef_tower", x + dx, y + dy, ground(x + dx, y + dy, r) - 0.5,
                                      rng.uniform(zt_lo, zt_hi), radius=r, footprint=[(x + dx, y + dy)],
                                      footprint_radius=r, group=gid))
        return parts
    if world_kind == "wreck":
        yaw = float(rng.uniform(0, np.pi))
        hx, hy = 4.0, 1.0
        zb = ground(x, y, hx) - 0.5
        hull = ObstacleSpec("wreck_hull", x, y, zb, zb + 2.0, half_extents=(hx, hy), yaw=yaw,
                            footprint=_axis_points(x, y, hx, hy, yaw), footprint_radius=hy, group=gid)
        c, s = np.cos(yaw), np.sin(yaw)
        tx, ty = x + 1.5 * c, y + 1.5 * s
        tower = ObstacleSpec("wreck_tower", tx, ty, zb, rng.uniform(zt_lo, zt_hi), half_extents=(0.8, 0.8),
                             yaw=yaw, footprint=[(tx, ty)], footprint_radius=1.1, group=gid)
        return [hull, tower]
    raise ValueError(f"unknown obstacle kind {world_kind!r}")


def _violates(parts, placed, poly, clearance, spacing) -> bool:
    for o in parts:
        pts = np.asarray(o.footprint, float)
        if np.any(_dist_to_polyline(pts, poly) < o.footprint_radius + clearance):
            return True
        for q in placed:
            if q.group == o.group:
                continue
            qp = np.asarray(q.footprint, float)
            d = np.linalg.norm(pts[:, None, :] - qp[None, :, :], axis=2).min()
            if d < o.footprint_radius + q.footprint_radius + spacing:
                return True
    return False


def generate_world(env_cfg: dict | None, sim_cfg: dict, grid_extent: tuple | None = None) -> WorldSpec:
    """Build a WorldSpec for the scenario in env_cfg (config `environment:` section).

    sim_cfg: the `simulation` config section (start/waypoints/goal, seabed_z, auv). grid_extent:
    (rows, cols, resolution) of the world raster; defaults from environment.grid_size/resolution.
    """
    env_cfg = env_cfg or {}
    scenario = str(env_cfg.get("scenario", "sparse_obstacles"))
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown procedural scenario {scenario!r}; choose from {sorted(SCENARIOS)}")
    seed = int(env_cfg.get("seed", 42))
    base = SCENARIOS[scenario]
    seabed_cfg = _merge(base["seabed"], env_cfg.get("seabed"))
    obs_over = dict(env_cfg.get("obstacles") or {})
    obs_cfg = _merge({"enabled": True, "count": base["obstacles"]["count"], "mix": base["obstacles"]["mix"],
                      "min_size_m": 1.0, "max_size_m": 3.0, "clearance_m": 3.0, "path_clearance_m": 3.0,
                      "min_spacing_m": 1.5, "edge_margin_m": 2.0, "max_attempts_per_obstacle": 200},
                     obs_over)
    if "mix" in obs_over:                                # an explicit mix REPLACES (not merges with) the default
        obs_cfg["mix"] = dict(obs_over["mix"])
    if obs_cfg["max_size_m"] < obs_cfg["min_size_m"] or obs_cfg["min_size_m"] <= 0:
        raise ValueError("need 0 < obstacles.min_size_m <= obstacles.max_size_m")
    if int(obs_cfg["count"]) < 0:
        raise ValueError("obstacles.count must be >= 0")

    rows, cols, res = grid_extent or (int(env_cfg.get("grid_size", {}).get("rows", 100)),
                                      int(env_cfg.get("grid_size", {}).get("cols", 100)),
                                      float(env_cfg.get("resolution", 1.0)))
    origin = (0.0, 0.0)
    base_z = float(seabed_cfg.get("base_z_m", sim_cfg.get("seabed_z", -15.0)))
    start = [float(v) for v in sim_cfg["start_position"]]
    wps = [[float(v) for v in w] for w in (sim_cfg.get("waypoints") or [sim_cfg["goal_position"]])]
    goal = [float(v) for v in sim_cfg["goal_position"]]
    if wps[-1] != goal:
        wps.append(goal)
    mission = [start] + wps
    mission_z_min = min(p[2] for p in mission)
    vehicle_clear = float(env_cfg.get("min_seabed_clearance_m", 2.0))
    max_z = mission_z_min - vehicle_clear                      # the seabed must stay below the mission depth
    if base_z > max_z:
        raise ValueError(f"seabed base z {base_z} leaves < {vehicle_clear} m below the shallowest mission z {mission_z_min}")

    heights, sinfo = generate_seabed(rows, cols, res, base_z, seabed_cfg, seed, max_z)
    spec = WorldSpec(scenario, seed, origin, res, heights, str(seabed_cfg["type"]), [], mission,
                     {"description": base.get("description", ""), "seabed_clipped_cells": sinfo["clipped_cells"],
                      "obstacles_requested": 0, "placement_attempts": 0})
    ex = spec.extent
    poly = np.asarray([p[:2] for p in mission], float)

    def ground(x, y, r):
        xs = x + np.array([-r, 0, r, 0, 0]); ys = y + np.array([0, -r, 0, r, 0])
        return float(min(spec.seabed_height(a, b) for a, b in zip(xs, ys)))

    top_lo = max(mission_z_min + 1.0, -8.0) if mission_z_min + 1.0 < -0.5 else -3.0
    top_range = (min(top_lo, -1.5), -0.5)                      # tops reach (or pass) the mission depth

    obstacles: list[ObstacleSpec] = []
    gid = 0
    if obs_cfg["enabled"]:
        # --- scenario-fixed structures -------------------------------------------------
        # config `environment.explicit_obstacles` (Phase 5): extra fixed obstacles appended to the scenario's own.
        for e in list(base.get("explicit", [])) + list(env_cfg.get("explicit_obstacles") or []):
            obstacles.append(_explicit(e, spec, gid, ground)); gid += 1
        if "corridor" in base:
            for o in _corridor(base["corridor"], mission, ground, top_range, gid):
                obstacles.append(o)
            gid += 2
        # --- random placement ---------------------------------------------------------
        # `obstacles.count` is the TOTAL number of obstacle BODIES this world must contain (that is how many
        # PyBullet collision+visual bodies build_world() creates, and how many the dashboard reports). The
        # scenario's own fixed structures (explicit + corridor, placed above) count towards it, so the random
        # placement fills the remainder and the requested count == the generated count.
        fixed_count = len(obstacles)
        count = int(obs_cfg["count"])
        random_target = max(0, count - fixed_count)
        mix = {k: float(v) for k, v in obs_cfg["mix"].items() if float(v) > 0}
        for k in mix:
            if k not in ("pillar", "boulder", "wall", "reef_tower", "wreck"):
                raise ValueError(f"unknown obstacle kind in mix: {k!r}")
        if random_target > 0 and not mix:
            mix = dict(DEFAULT_OBSTACLE_MIX)                  # honour an explicit count on an obstacle-less scenario
        spec.meta["obstacles_requested"] = count
        spec.meta["obstacles_fixed"] = fixed_count
        rng = _rng(seed, 2)
        kinds = sorted(mix)
        probs = np.array([mix[k] for k in kinds]) / sum(mix.values()) if kinds else None
        placed_count, attempts = 0, 0
        while placed_count < random_target:
            remaining = random_target - placed_count
            kind = kinds[int(rng.choice(len(kinds), p=probs))]
            for _ in range(int(obs_cfg["max_attempts_per_obstacle"])):
                attempts += 1
                parts = _place(kind, rng, obs_cfg,
                               {"extent": ex, "edge_margin": float(obs_cfg["edge_margin_m"]), "ground": ground,
                                "top_range": top_range, "next_group": gid})
                if len(parts) > remaining:
                    parts = parts[:remaining]                 # a multi-part structure only fills the slots left
                if not _violates(parts, obstacles, poly, max(float(obs_cfg["clearance_m"]), float(obs_cfg["path_clearance_m"])),
                                 float(obs_cfg["min_spacing_m"])):
                    obstacles.extend(parts); gid += 1; placed_count += len(parts)
                    break
            else:
                break                                          # area is full; report fewer than requested
        spec.meta["placement_attempts"] = attempts
        spec.meta["obstacles_placed"] = placed_count
    spec.obstacles = obstacles
    validate_world(spec, obs_cfg)
    return spec


def _explicit(e: dict, spec: WorldSpec, gid: int, ground) -> ObstacleSpec:
    kind, x, y = e["kind"], float(e["x"]), float(e["y"])
    zt = float(e.get("z_top_m", -2.0))
    if kind == "pillar":
        r = float(e["radius_m"])
        return ObstacleSpec("pillar", x, y, ground(x, y, r) - 0.5, zt, radius=r, footprint=[(x, y)],
                            footprint_radius=r, group=gid)
    if kind == "wall":
        hx, hy, yaw = float(e["half_x_m"]), float(e["half_y_m"]), np.deg2rad(float(e.get("yaw_deg", 0.0)))
        return ObstacleSpec("wall", x, y, ground(x, y, max(hx, hy)) - 0.5, zt, half_extents=(hx, hy), yaw=float(yaw),
                            footprint=_axis_points(x, y, hx, hy, yaw), footprint_radius=min(hx, hy), group=gid)
    raise ValueError(f"unsupported explicit obstacle kind {kind!r}")


def _corridor(cc: dict, mission: list, ground, top_range, gid: int) -> list:
    """Two walls parallel to the start->goal line, offset +/- gap/2, centred on the mission."""
    a, b = np.asarray(mission[0][:2]), np.asarray(mission[-1][:2])
    d = (b - a) / np.linalg.norm(b - a)
    n = np.array([-d[1], d[0]])
    mid = 0.5 * (a + b)
    half_len, half_t, gap = cc["length_m"] / 2.0, cc["thickness_m"] / 2.0, cc["gap_m"] / 2.0
    yaw = float(np.arctan2(d[1], d[0]))
    out = []
    for i, sgn in enumerate((+1.0, -1.0)):
        c = mid + sgn * (gap + half_t) * n
        zt = top_range[1] - 0.5
        out.append(ObstacleSpec("wall", float(c[0]), float(c[1]), ground(c[0], c[1], half_len) - 0.5, zt,
                                half_extents=(half_len, half_t), yaw=yaw,
                                footprint=_axis_points(c[0], c[1], half_len, half_t, yaw), footprint_radius=half_t,
                                group=gid + i))
    return out


def validate_world(spec: WorldSpec, obs_cfg: dict) -> None:
    """Hard guarantees: nothing generated blocks the mission start or intersects the mission path."""
    start = np.asarray(spec.mission_points[0][:2], float)
    poly = np.asarray([p[:2] for p in spec.mission_points], float)
    if spec.heights.max() > min(p[2] for p in spec.mission_points) - 1.0 + 1e-9:
        raise ValueError("seabed reaches the mission depth")
    minclear = min(float(obs_cfg.get("clearance_m", 3.0)), float(obs_cfg.get("path_clearance_m", 3.0)))
    for o in spec.obstacles:
        pts = np.asarray(o.footprint, float)
        if np.any(np.linalg.norm(pts - start, axis=1) < o.footprint_radius + 0.5):
            raise ValueError(f"{o.kind} at ({o.x:.1f},{o.y:.1f}) overlaps the AUV start position")
        if np.any(_dist_to_polyline(pts, poly) < o.footprint_radius + 0.5 * minclear):
            raise ValueError(f"{o.kind} at ({o.x:.1f},{o.y:.1f}) blocks the mission path")

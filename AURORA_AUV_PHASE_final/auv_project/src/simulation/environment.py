"""
environment.py
----------------
Builds the PyBullet world: seabed + static obstacles derived from Week 4's
occupancy grid, plus the grid<->world coordinate conversion shared by every
other Week 5 module.

Pipeline implemented here:

    data/processed/occupancy_grid_inflated.npy   (Week 4 output -- NOT recomputed)
    data/processed/navigation_cost_map.npy       (Week 4 output -- loaded, unused
                                                    until Week 6, see class docstring)
                       |
                       v
      load_mapping_metadata() / load_occupancy_grid() / load_cost_map()
                       |
                       v
      grid_to_world() / world_to_grid()   -> shared coordinate mapping (STEP 8)
                       |
                       v
      build_seabed()          -> one static box spanning the mapped area
      build_obstacles()       -> merged static boxes, one per horizontal
                                  run of obstacle cells (STEP 7)
                       |
                       v
             Environment  -> holds body ids, GridInfo, Bounds, collision checks

COORDINATE CONVENTION (documented, not left implicit): grid column -> world
X, grid row -> world Y, both scaled by `resolution` (metres/cell). World Z
is a separate, independent vertical (depth) axis -- the Week 4 occupancy
grid is a 2D top-down map with no per-cell height information, so every
obstacle column is extruded from the seabed up through most of the water
column (see build_obstacles docstring). This is a standard simplification
for 2D-grid-based marine obstacle avoidance and is what the Week 6 planner
will also assume when it reads the same occupancy grid.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pybullet as p


@dataclass
class GridInfo:
    """Grid geometry read from Week 4 outputs -- never hard-coded (STEP 1)."""
    rows: int
    cols: int
    resolution: float
    free_value: int
    obstacle_value: int
    unknown_value: int

    @classmethod
    def from_metadata(cls, metadata: dict) -> "GridInfo":
        occ = metadata["occupancy_values"]
        return cls(
            rows=int(metadata["rows"]),
            cols=int(metadata["columns"]),
            resolution=float(metadata["resolution"]),
            free_value=int(occ["free"]),
            obstacle_value=int(occ["obstacle"]),
            unknown_value=int(occ["unknown"]),
        )


def load_mapping_metadata(path: str) -> dict:
    """Load data/processed/mapping_metadata.json (Week 4 output)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_occupancy_grid(path: str) -> np.ndarray:
    """Load a Week 4 occupancy grid .npy (e.g. occupancy_grid_inflated.npy). Not recomputed here."""
    return np.load(path)


def load_cost_map(path: str) -> np.ndarray:
    """
    Load data/processed/navigation_cost_map.npy (Week 4 output).

    Loaded and stored on the Environment for Week 6 to consume later.
    Week 5 does NOT use this for navigation -- see module docstring.
    """
    return np.load(path)


# --------------------------------------------------------------------------
# Coordinate conversion (STEP 8 / STEP 9)
# --------------------------------------------------------------------------

def grid_to_world(row: int, col: int, resolution: float) -> tuple:
    """
    Input: row, col: grid indices. resolution: metres/cell.
    Processing: world_x = col * resolution, world_y = row * resolution
        (see module docstring for the convention).
    Output: (world_x, world_y) tuple of floats.
    """
    return (float(col) * resolution, float(row) * resolution)


def world_to_grid(x: float, y: float, resolution: float) -> tuple:
    """
    Input: x, y: world coordinates (metres). resolution: metres/cell.
    Processing: inverse of grid_to_world -- col = round(x / resolution),
        row = round(y / resolution).
    Output: (row, col) tuple of ints.
    """
    col = int(round(x / resolution))
    row = int(round(y / resolution))
    return (row, col)


# --------------------------------------------------------------------------
# World construction
# --------------------------------------------------------------------------

def build_seabed(grid_info: GridInfo, seabed_z: float, physics_client: int = 0) -> int:
    """
    Create a static box representing the seabed floor.

    Input:
        grid_info: GridInfo (used to size the slab to the mapped area).
        seabed_z: world Z of the TOP surface of the seabed (config:
                  simulation.seabed_z). The water column exists above this.
        physics_client: PyBullet client id.
    Processing:
        A single static (baseMass=0) box spanning the full grid footprint
        (rows*cols*resolution), thin in Z, positioned so its top face sits
        at `seabed_z`. A bounded box is used instead of PyBullet's infinite
        p.GEOM_PLANE so the seabed matches the actual surveyed extent.
    Output:
        PyBullet body id of the seabed.
    """
    half_x = (grid_info.cols * grid_info.resolution) / 2.0
    half_y = (grid_info.rows * grid_info.resolution) / 2.0
    half_thickness = 0.5

    collision_shape = p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[half_x, half_y, half_thickness], physicsClientId=physics_client
    )
    visual_shape = p.createVisualShape(
        p.GEOM_BOX, halfExtents=[half_x, half_y, half_thickness],
        rgbaColor=[0.76, 0.70, 0.50, 1.0], physicsClientId=physics_client,
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=collision_shape,
        baseVisualShapeIndex=visual_shape,
        basePosition=[half_x, half_y, seabed_z - half_thickness],
        physicsClientId=physics_client,
    )
    return body_id


def _merge_obstacle_runs(occupancy_grid: np.ndarray, obstacle_value: int) -> list:
    """
    Row-wise run-length encoding of obstacle cells.

    Input:
        occupancy_grid: 2D int array.
        obstacle_value: code meaning "obstacle".
    Processing:
        For each row, finds maximal horizontal runs of obstacle cells and
        records (row, col_start, col_end_inclusive). This is a deliberate
        simplification (STEP 7): spawning one PyBullet static body PER
        obstacle cell does not scale (thousands of bodies on a dense
        100x100 grid). Merging horizontally-contiguous cells into single
        boxes cuts the body count roughly by the average run length while
        staying simple to implement and reason about. A full 2D
        (quadtree/rectangle) merge is noted as future work in
        docs/simulation.md.
    Output:
        list of (row, col_start, col_end_inclusive) tuples.
    """
    runs = []
    rows, cols = occupancy_grid.shape
    for r in range(rows):
        c = 0
        while c < cols:
            if occupancy_grid[r, c] == obstacle_value:
                start = c
                while c < cols and occupancy_grid[r, c] == obstacle_value:
                    c += 1
                runs.append((r, start, c - 1))
            else:
                c += 1
    return runs


def build_obstacles(
    occupancy_grid: np.ndarray,
    grid_info: GridInfo,
    obstacle_base_z: float,
    obstacle_height: float,
    physics_client: int = 0,
) -> list:
    """
    Create static obstacle bodies in PyBullet from a Week 4 occupancy grid.

    Input:
        occupancy_grid: 2D int array (Week 4 output; the caller should pass
                        the *_inflated grid so the AUV's safety margin is
                        already baked into the geometry).
        grid_info: GridInfo (resolution, obstacle_value).
        obstacle_base_z: world Z of the BOTTOM of every obstacle (normally
                         == seabed_z, so obstacles rest on the seabed).
        obstacle_height: vertical extent of each obstacle column (metres).
                         Since the 2D occupancy grid carries no per-cell
                         height, obstacles are extruded as full water-column
                         pillars (see module docstring) -- obstacle_height
                         should span from the seabed up past the AUV's
                         cruise depth.
        physics_client: PyBullet client id.
    Processing:
        Runs _merge_obstacle_runs() to collapse horizontal obstacle runs
        into rectangles, then creates one static box body per run,
        positioned via grid_to_world() at the run's centre.
    Output:
        list of PyBullet body ids, one per merged obstacle run.
    """
    runs = _merge_obstacle_runs(occupancy_grid, grid_info.obstacle_value)

    body_ids = []
    for row, col_start, col_end in runs:
        run_len_cells = col_end - col_start + 1
        half_x = (run_len_cells * grid_info.resolution) / 2.0
        half_y = grid_info.resolution / 2.0
        half_z = obstacle_height / 2.0

        x_start, y = grid_to_world(row, col_start, grid_info.resolution)
        center_x = x_start + half_x
        center_y = y
        center_z = obstacle_base_z + half_z

        collision_shape = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[half_x, half_y, half_z], physicsClientId=physics_client
        )
        visual_shape = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[half_x, half_y, half_z],
            rgbaColor=[0.55, 0.27, 0.07, 1.0], physicsClientId=physics_client,
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=collision_shape,
            baseVisualShapeIndex=visual_shape,
            basePosition=[center_x, center_y, center_z],
            physicsClientId=physics_client,
        )
        body_ids.append(body_id)
    return body_ids


@dataclass
class Bounds:
    """World-space safety boundaries (STEP 25), derived from the grid extents."""
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    zmin: float
    zmax: float

    def contains(self, position: tuple) -> bool:
        x, y, z = position
        return (
            self.xmin <= x <= self.xmax
            and self.ymin <= y <= self.ymax
            and self.zmin <= z <= self.zmax
        )


class Environment:
    """Owns the seabed/obstacle bodies, coordinate conversion, and collision checks."""

    def __init__(self, cfg: dict, physics_client: int = 0):
        """
        Input:
            cfg: full project config dict (needs cfg["output"] and cfg["simulation"]).
            physics_client: PyBullet client id to build the world in.
        Processing:
            Loads Week 4's mapping_metadata.json + occupancy_grid_inflated.npy
            (STEP 7: the *inflated* grid, so the safety margin is already
            physical geometry) + navigation_cost_map.npy (STEP 30: loaded,
            not used for navigation), builds GridInfo, then constructs the
            seabed and obstacle bodies and the world safety Bounds (STEP 25).
        Output:
            None. Access via .seabed_id / .obstacle_ids / .grid_info / .bounds.
        """
        self._client = physics_client
        processed_dir = cfg["output"]["processed_dir"]
        sim_cfg = cfg["simulation"]
        env_cfg = cfg.get("environment", {}) or {}
        self.scenario = str(env_cfg.get("scenario", "legacy_week4"))
        self.seed = int(env_cfg.get("seed", 42))
        self.world_spec = None          # WorldSpec for procedural scenarios (ground-truth description)

        seabed_z = float(sim_cfg["seabed_z"])
        obstacle_height = float(sim_cfg["obstacle_height"])
        margin = float(sim_cfg["bounds_margin"])

        if self.scenario == "legacy_week4":
            # Phase 1-3 world: Week 4 maps -> seabed box + extruded obstacle pillars (unchanged).
            self.metadata = load_mapping_metadata(f"{processed_dir}/mapping_metadata.json")
            self.grid_info = GridInfo.from_metadata(self.metadata)
            self.occupancy_grid = load_occupancy_grid(f"{processed_dir}/occupancy_grid_inflated.npy")
            self.cost_map = load_cost_map(f"{processed_dir}/navigation_cost_map.npy")
            self.seabed_id = build_seabed(self.grid_info, seabed_z, self._client)
            self.obstacle_ids = build_obstacles(
                self.occupancy_grid, self.grid_info,
                obstacle_base_z=seabed_z, obstacle_height=obstacle_height,
                physics_client=self._client,
            )
            zmin = seabed_z
        else:
            # Phase 4: deterministic procedural world (seabed height field + obstacles), real collision geometry.
            from src.simulation.world_builder import build_world
            from src.simulation.world_generator import generate_world
            rows = int((env_cfg.get("grid_size") or {}).get("rows", 100))
            cols = int((env_cfg.get("grid_size") or {}).get("cols", 100))
            res = float(env_cfg.get("resolution", 1.0))
            self.world_spec = generate_world(env_cfg, sim_cfg, grid_extent=(rows, cols, res))
            self.seabed_id, self.obstacle_ids = build_world(self.world_spec, self._client)
            self.metadata = {"scenario": self.scenario}
            self.grid_info = GridInfo(rows=rows, cols=cols, resolution=res, free_value=0, obstacle_value=1,
                                      unknown_value=2)
            # DEBUG/EVALUATION raster of obstacle footprints (legacy field name, used only for plots)
            from src.mapping.coordinate_utils import GridGeometry
            geom = GridGeometry.from_extent(res, cols * res, rows * res, 0.0, 0.0)
            self.occupancy_grid = self.world_spec.obstacle_footprint_mask(geom).astype(np.int8)
            self.cost_map = None
            zmin = float(self.world_spec.heights.min())

        self.bounds = Bounds(
            xmin=-margin,
            xmax=self.grid_info.cols * self.grid_info.resolution + margin,
            ymin=-margin,
            ymax=self.grid_info.rows * self.grid_info.resolution + margin,
            zmin=zmin,
            zmax=0.0,
        )

    def describe(self) -> dict:
        """Generation parameters of the simulated world. This is GROUND-TRUTH environment information
        (what the generator built), reported for the dashboard under an explicit label; the mapper
        never receives it."""
        if self.world_spec is None:
            return {"label": "GROUND TRUTH ENVIRONMENT (debug)", "scenario": self.scenario, "seed": self.seed,
                    "terrain_type": "legacy flat seabed + extruded obstacle pillars (Week 4 maps)",
                    "seabed_z_min": float(self.bounds.zmin), "seabed_z_max": float(self.bounds.zmin),
                    "obstacle_count": len(self.obstacle_ids), "structure_count": len(self.obstacle_ids),
                    "procedural": False}
        sm = self.world_spec.summary()
        return {"label": "GROUND TRUTH ENVIRONMENT (debug)", "scenario": sm["scenario"], "seed": sm["seed"],
                "terrain_type": sm["seabed_type"], "seabed_z_min": sm["seabed_z_min"], "seabed_z_max": sm["seabed_z_max"],
                "obstacle_count": sm["obstacle_count"], "structure_count": sm["structure_count"],
                "obstacle_kinds": sm["obstacle_kinds"],
                "obstacles_requested": sm["obstacles_requested"], "obstacles_fixed": sm.get("obstacles_fixed", 0),
                "obstacles_placed": sm.get("obstacles_placed"), "description": sm["description"],
                "procedural": True}

    def scene_payload(self):
        """Ground-truth world geometry for the dashboard's Underwater view (None for the legacy flat Week-4 world)."""
        return None if self.world_spec is None else self.world_spec.scene_payload()

    def grid_to_world(self, row: int, col: int) -> tuple:
        return grid_to_world(row, col, self.grid_info.resolution)

    def world_to_grid(self, x: float, y: float) -> tuple:
        return world_to_grid(x, y, self.grid_info.resolution)

    def check_collision(self, body_id: int) -> bool:
        """
        Input: body_id: the body to check (normally the AUV).
        Processing: p.getContactPoints(bodyA=body_id, bodyB=other) against
            the seabed and every obstacle body (STEP 24); any non-empty
            contact list counts as a collision.
        Output: True if `body_id` is currently in contact with the seabed
            or any obstacle, else False.
        """
        # One contact query for the body, filtered to the seabed + obstacle bodies. Same result as one query per
        # body (bodyA=body_id, bodyB=other) but O(1) PyBullet calls per step instead of O(number of obstacles).
        others = {self.seabed_id, *self.obstacle_ids}
        for contact in p.getContactPoints(bodyA=body_id, physicsClientId=self._client):
            if contact[2] in others:                     # contact tuple: (flag, bodyA, bodyB, ...)
                return True
        return False

    def in_bounds(self, position: tuple) -> bool:
        """True if `position` (x, y, z) is within the world safety Bounds (STEP 25)."""
        return self.bounds.contains(position)

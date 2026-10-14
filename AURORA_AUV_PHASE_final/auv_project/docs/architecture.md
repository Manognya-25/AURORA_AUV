# System Architecture — AUV Project

This document describes the end-to-end architecture of the AI-Powered AUV
simulation pipeline, mapping each conceptual stage to the module(s) that will
implement it and the data that flows between stages.

## High-Level Pipeline

```text
Bathymetry / Sensor Data
          |
          v
Data Preprocessing
          |
          v
Terrain Feature Extraction
          |
          v
Terrain Classification
          |
          v
Environmental Map
          |
          v
Occupancy + Cost Map
          |
          v
A* Path Planner
          |
          v
AUV Controller
          |
          v
PyBullet Simulation
```

## Stage-by-Stage Breakdown

### 1. Bathymetry / Sensor Data
- **Purpose:** Provide raw depth information for a region of seafloor.
- **Input:** Real dataset file (CSV / NPY / NPZ / GeoTIFF) *or* synthetic
  generation parameters from `configs/config.yaml`.
- **Output:** Raw 2D depth grid (NumPy array).
- **Module(s):** `src/data_processing/load_bathymetry.py`,
  `src/data_processing/generate_environment.py`
- **Implemented in:** Week 2

### 2. Data Preprocessing
- **Purpose:** Clean and normalize the raw depth grid (handle missing values,
  scale depth, remove artifacts).
- **Input:** Raw depth grid.
- **Output:** Cleaned, normalized depth grid.
- **Module(s):** `src/data_processing/preprocess.py`
- **Implemented in:** Week 2

### 3. Terrain Feature Extraction
- **Purpose:** Compute per-cell numerical descriptors of terrain shape.
- **Input:** Cleaned depth grid.
- **Output:** Feature grid/array: `[depth, slope, roughness, curvature]` per cell.
- **Module(s):** `src/terrain/feature_extraction.py`
- **Implemented in:** Week 3

### 4. Terrain Classification
- **Purpose:** Assign each cell a navigability class using a trained ML model.
- **Input:** Feature vectors.
- **Output:** Terrain class grid (`0=Flat/Sand, 1=Rocky, 2=Steep/Hazard, 3=Obstacle`).
- **Module(s):** `src/terrain/classifier.py`, `src/terrain/evaluate.py`
- **Implemented in:** Week 3
- **Note:** Initial training labels are synthetically generated from physically
  meaningful thresholds (documented in `docs/dataset.md`), not from a real
  labeled survey. This is explicitly disclosed and never misrepresented.

### 5. Environmental Map
- **Purpose:** Combine depth + terrain classification into a unified
  environmental representation used by the mapping stage.
- **Input:** Depth grid + terrain class grid.
- **Output:** Combined environment map object.
- **Module(s):** `src/data_processing/generate_environment.py`
- **Implemented in:** Week 2 – 3

### 6. Occupancy + Cost Map
- **Purpose:** Translate terrain understanding into a navigation representation.
  - **Occupancy grid** answers: "Can the robot occupy this cell?" (free / obstacle / unknown)
  - **Cost map** answers: "How expensive/risky is it to traverse this cell?"
- **Input:** Terrain class grid, feature grid.
- **Output:** Occupancy grid (0/1/2), cost map (float grid).
- **Module(s):** `src/mapping/occupancy_grid.py`, `src/mapping/cost_map.py`
- **Implemented in:** Week 4

### 7. A* Path Planner
- **Purpose:** Compute a safe, cost-aware path from a start cell to a goal cell.
- **Input:** Occupancy grid, cost map, start coordinate, goal coordinate.
- **Output:** Ordered list of waypoints `[(x1,y1), (x2,y2), ..., (xn,yn)]`.
- **Module(s):** `src/planning/astar.py`
- **Implemented in:** Week 6

### 8. AUV Controller
- **Purpose:** Convert waypoints into low-level thruster / velocity commands for
  the simulated AUV, using a simple proportional/PID-style controller.
- **Input:** Current AUV state, next waypoint.
- **Output:** Control command (thrust / velocity setpoints).
- **Module(s):** `src/simulation/auv.py`
- **Implemented in:** Week 5 (basic controller), used with A* in Week 6

### 9. PyBullet Simulation
- **Purpose:** Physically simulate the AUV moving through the 3D environment,
  following the planned path, subject to simplified underwater drag.
- **Input:** AUV URDF model, environment (seabed + obstacles), control commands.
- **Output:** AUV trajectory, collision events, simulation logs.
- **Module(s):** `src/simulation/environment.py`, `src/simulation/run_simulation.py`
- **Implemented in:** Week 5 – 6

## Supporting Modules

| Module | Responsibility |
|--------|-----------------|
| `src/utils/visualization.py` | Shared plotting helpers (2D/3D terrain, maps, paths, trajectories) |
| `src/utils/metrics.py` | Path length, planning time, terrain risk, energy proxy, collision count |
| `tests/` | Unit tests for every module listed above |

## Design Principles

1. **Modularity** — every stage above is a separate, independently testable module.
2. **Configurability** — all thresholds, weights, and paths live in `configs/config.yaml`.
3. **Explainability** — classical, interpretable methods (Random Forest, manual A*)
   are used before any more advanced technique is considered.
4. **Reproducibility** — a fixed random seed and clear documentation of any
   synthetic data / labels used.
5. **Honesty about limitations** — synthetic labels, simplified physics, and the
   energy model are always explicitly labeled as such, never presented as
   real-world validated results.

## Explicitly Out of Scope (Weeks 1–6)

- Reinforcement Learning
- Model Predictive Control (MPC)
- LLM-based planning or natural-language interfaces
- Streamlit / web dashboard
- Advanced adaptive control

These may be added in later phases of the project, after the Week 6 milestone.

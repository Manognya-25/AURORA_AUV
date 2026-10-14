# AI-Powered Autonomous Underwater Vehicle (AUV) Simulation

**B.Tech Final Year Project**
**Current Stage:** Week 5 — PyBullet AUV Simulation

---

## 1. Project Title

**AI-Powered Autonomous Underwater Vehicle (AUV) for Intelligent Underwater Exploration**

## 2. Problem Statement

Underwater exploration is difficult, expensive, and hazardous for human divers due to
pressure, visibility, and unpredictable terrain. Autonomous Underwater Vehicles (AUVs)
can reduce this risk, but they need to **understand the seafloor terrain** and
**plan safe paths** around hazards (rocks, obstacles, steep slopes) rather than simply
moving in a straight line toward a goal.

This project builds a simulation pipeline that takes raw bathymetric (underwater
depth) data, understands the terrain using machine learning, builds a safe navigation
map, and drives a simulated AUV along a collision-free, cost-aware path toward a goal.

## 3. Motivation

* Real AUV field trials are expensive and slow — a simulation-first pipeline lets
  algorithms be developed and validated safely before any real deployment.
* Combining classical robotics (A* path planning, occupancy grids) with machine
  learning (terrain classification) mirrors how real-world marine robotics research
  approaches the problem.
* This project intentionally starts with explainable, classical methods (Random
  Forest, A*) before introducing more advanced techniques later, so every stage of
  the pipeline can be understood and verified.

## 4. Objectives

1. Build a pipeline that ingests or synthetically generates bathymetric terrain data.
2. Extract meaningful terrain features (depth, slope, roughness, curvature).
3. Classify terrain into navigability categories using a traditional ML classifier.
4. Convert terrain understanding into an occupancy grid and a navigation cost map.
5. Simulate a simple AUV in PyBullet with basic movement and waypoint following.
6. Plan safe, cost-aware paths using the A* algorithm.
7. Integrate the full pipeline end-to-end and evaluate it with real, measured metrics.

## 5. System Architecture

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

See [`docs/architecture.md`](docs/architecture.md) for a more detailed breakdown of
each stage, its inputs/outputs, and the module(s) responsible for it.

## 6. Technology Stack

| Category            | Tools / Libraries                     |
|---------------------|----------------------------------------|
| Language             | Python 3.10+                          |
| Numerical computing  | NumPy, SciPy                          |
| Data handling        | Pandas                                |
| Visualization        | Matplotlib                            |
| Machine learning     | Scikit-learn, Joblib                  |
| Computer vision      | OpenCV                                |
| Graph utilities      | NetworkX                              |
| Configuration        | PyYAML                                |
| Robotics simulation  | PyBullet, URDF                        |
| Testing              | Pytest                                |
| Dev environment      | VS Code (Windows or Ubuntu, platform-independent code) |
| Version control      | Git / GitHub                          |

## 7. Dataset Strategy

Real bathymetric datasets (e.g., multibeam sonar surveys, public bathymetry
archives) may not always be readily available in a usable form for a first
prototype. To keep development unblocked, the project uses a **dual-source
dataset interface**:

* **Real bathymetry mode** — loads CSV / NPY / NPZ / GeoTIFF-style raster data
  through a common loader interface (implemented in Week 2).
* **Synthetic bathymetry mode** — procedurally generates a depth grid using
  gradients, Gaussian hills, depressions, ridges, and noise, so the rest of the
  pipeline (features → classification → mapping → planning → simulation) can be
  developed and tested without waiting on a real dataset.

Both modes feed into the exact same downstream pipeline. When a real dataset is
integrated, its source, format, resolution, units, and licensing will be documented
in [`docs/dataset.md`](docs/dataset.md).

**No datasets are downloaded automatically.** The user explicitly chooses the data
source via `configs/config.yaml` (`environment.data_source: synthetic | real`).

## 8. Development Roadmap

| Week | Milestone |
|------|-----------|
| 1 | Project setup, repository structure, architecture design ✅ |
| 2 | Bathymetric data loading, cleaning, synthetic terrain generation, visualization ✅ *(current stage)* |
| 3 | Terrain feature extraction + Random Forest terrain classification |
| 4 | Occupancy grid + cost map generation |
| 5 | PyBullet AUV model, environment, basic movement/waypoint following ✅ |
| 6 | A* path planning (cost-aware), full pipeline integration, experiments & metrics |

## Week 2 — Bathymetry Processing

**Status:** Completed

**Implemented:**
- Bathymetry loading (`src/data_processing/load_bathymetry.py`) — supports CSV, NPY, NPZ, and GeoTIFF (optional `rasterio`), with automatic format dispatch based on file extension.
- Data validation (`validate_bathymetry()`) — reports shape, dtype, min/max depth, missing values, infinite values, and conservative z-score outliers.
- Cleaning (`clean_bathymetry()`) — converts infinities to missing values, then fills missing cells via a configurable strategy (`nearest` / `linear` / `median`).
- Depth convention handling (`apply_depth_convention()`) — explicitly converts `positive_down` or `negative_down` input into the project's internal `positive_down` standard. See [`docs/dataset.md`](docs/dataset.md).
- Optional normalization (`normalize_depth()`) — `minmax` or `standardize`, always producing a separate array (raw/cleaned data is never overwritten).
- Synthetic bathymetry generation (`generate_synthetic_bathymetry()`) — spatially continuous terrain built from a smooth gradient, Gaussian hills, depressions, elongated ridges, and low-amplitude noise. **Explicitly and honestly labeled as synthetic** — never presented as a real surveyed location.
- Unified environment interface (`create_environment()`) — produces the same structured `Environment` object regardless of whether the source is `"synthetic"` or `"real"`, ready for Week 3 to consume.
- Visualization (`src/utils/visualization.py`) — 2D depth map, 3D terrain surface, depth histogram, and a raw/corrupted/cleaned preprocessing comparison figure.
- Full pipeline orchestration (`src/data_processing/run_pipeline.py`) — loads config, builds the environment, validates, cleans, optionally normalizes, saves processed data + metadata, and generates all visualizations.
- 21 unit tests in `tests/test_bathymetry.py`, covering generation, cleaning, normalization, depth convention conversion, save/load round-trips, and mock real-dataset loading — all passing.

**How to run:**
```bash
python -m src.data_processing.run_pipeline
```

**Outputs:**
- `data/processed/processed_bathymetry.npy` — the primary artifact consumed by Week 3.
- `data/processed/processed_bathymetry_metadata.json` — dataset provenance and statistics.
- `results/figures/bathymetry_2d.png`, `bathymetry_3d.png`, `depth_histogram.png`, `preprocessing_comparison.png`.

See [`docs/dataset.md`](docs/dataset.md) for full documentation of the depth convention, preprocessing pipeline, synthetic generation method, and how to plug in a real dataset later.

## Week 3 — Terrain Feature Extraction and Classification

**Status:** Completed

**Implemented:**
- Feature extraction (`src/terrain/feature_extraction.py`) — per-cell `depth`, `slope` (gradient magnitude), `roughness` (local depth std, configurable window), and `curvature` (discrete Laplacian), all resolution-aware.
- Synthetic label generation (`src/terrain/label_generation.py`) — rule-based terrain classes (`Flat/Sand`, `Rocky`, `Steep/Terrain Hazard`, `Obstacle`) from `slope`/`roughness` thresholds in `configs/config.yaml`.
- **Threshold recalibration (documented):** the thresholds originally in `config.yaml` were placeholders that classified 96.7% of the real grid as `Steep/Obstacle`. They were recalibrated against the real feature distribution (values only, same keys/semantics) to give a non-degenerate split — see [`docs/terrain_classification.md`](docs/terrain_classification.md) for the full before/after numbers and derivation.
- Classifier (`src/terrain/classifier.py`) — stratified 70/15/15 train/val/test split, `RandomForestClassifier` (`n_estimators=200`, `class_weight="balanced"`), joblib save/load.
- Evaluation (`src/terrain/evaluate.py`) — accuracy, macro/weighted F1, confusion matrix, per-class report, feature-importance report. Test accuracy ≈0.999 — documented in `docs/terrain_classification.md` as an expected result of labels being a deterministic function of the same features the model trains on, not evidence of real-world navigability judgment.
- Visualization helpers added to `src/utils/visualization.py` — feature grids, categorical terrain-class maps, confusion-matrix heatmap.
- Full pipeline orchestration (`src/terrain/run_pipeline.py`).
- 37 unit tests in `tests/test_terrain.py` — all passing.

**How to run:**
```bash
python -m src.terrain.run_pipeline
```

**Outputs:**
- `models/terrain_classifier.pkl`
- `results/figures/terrain_features.png`, `terrain_labels_synthetic.png`, `terrain_classification_predicted.png`, `terrain_confusion_matrix.png`

See [`docs/terrain_classification.md`](docs/terrain_classification.md) for full documentation.

## Week 5 — PyBullet AUV Simulation

**Status:** Completed

**Implemented:**
- AUV model (`assets/auv/auv.urdf`) — a single cylindrical rigid body (mass 20 kg, length 1.5 m, radius 0.25 m) with matching visual/collision geometry and closed-form solid-cylinder inertia.
- Simplified underwater physics (`src/simulation/physics.py`) — buoyancy (calibrated for approximate neutral buoyancy — see `docs/simulation.md` section 3 for why the naive `buoyancy_ratio=1.0` was wrong by 14.7x and how it was fixed) and simplified linear drag, applied every step alongside PyBullet's native rigid-body dynamics and gravity.
- AUV state/control (`src/simulation/auv.py`) — `get_state()`/`get_position()`/`get_orientation()`/`get_linear_velocity()`/`get_angular_velocity()`, normalized `[-1, 1]` thruster control (`apply_control`, plus `move_forward`/`move_backward`/`move_left`/`move_right`/`move_up`/`move_down`/`stop`), and a movement-effort proxy (explicitly *not* claimed as real energy — see `docs/simulation.md` section 9).
- Environment (`src/simulation/environment.py`) — loads Week 4's `occupancy_grid_inflated.npy`/`navigation_cost_map.npy`/`mapping_metadata.json` as-is (not recomputed); builds a bounded seabed slab and obstacle geometry via row-wise run-length merging of obstacle cells (2,599 obstacle cells → 286 merged bodies on the current data); documented, tested `grid_to_world`/`world_to_grid` coordinate conversion; world safety `Bounds`; PyBullet-contact-based `check_collision()`.
- Waypoint controller (`src/simulation/controller.py`) — proportional (P) position + yaw control, `reached_waypoint()` tolerance check; drives the AUV through a manually-defined, occupancy-grid-verified waypoint sequence (`simulation.waypoints` in config) — **not** A*, which is reserved for Week 6.
- Proximity sensors (`src/simulation/sensors.py`) — single-ray front/left/right distance readings via `p.rayTest()`, logged every step.
- Metrics (`src/simulation/metrics.py`) — only measured/clearly-labeled values: steps, simulated + wall-clock duration, real trajectory length, final position/distance-to-goal, waypoints reached, collision count, and an explicitly-labeled movement-effort proxy.
- Full orchestration (`src/simulation/run_simulation.py`) — connects PyBullet (GUI or headless DIRECT, `simulation.mode`/`simulation.gui`), builds the environment, runs the waypoint sequence, and saves results.
- 17 unit tests in `tests/test_simulation.py` (URDF existence, DIRECT-mode connection, AUV load/state, forward/vertical motion, stop, energy tracking, coordinate round-trip, controller direction/tolerance, collision detection against a real Week 4 obstacle cell, proximity sensors, metrics computation, and a full headless run). Tests are skipped (not failed) if `pybullet` isn't installed.

**How to run:**
```bash
python -m src.simulation.run_simulation
```

**Outputs:**
- `results/logs/week5_simulation_log.csv` — full per-step trajectory/state/sensor log.
- `results/logs/week5_trajectory.npy` — just the x/y/z columns.
- `results/logs/week5_simulation_metrics.json` — summary metrics.
- `results/figures/auv_trajectory.png` — trajectory plotted over the Week 4 occupancy grid.

See [`docs/simulation.md`](docs/simulation.md) for the full documentation, including the coordinate-system convention, the buoyancy-calibration bug and fix, and all documented limitations.

## Phase 4 — Online Mapping, Cost Map, Procedural Environment, Live Dashboard
Real sonar + EKF estimated pose -> log-odds occupancy grid (UNKNOWN/FREE/OCCUPIED) -> navigation cost map
(obstacle inflation from vehicle radius + safety margin, clearance cost, configurable unknown policy).
Seven deterministic seeded underwater scenarios (`environment.scenario`, `environment.seed`). A functional
dashboard (`python dashboard.py`, http://127.0.0.1:5000; `--port` to use another, and it refuses to start on a port
another dashboard already serves) runs the simulation in its own process and shows only real telemetry. Ground truth never enters mapping; it appears only in labelled debug sections.
See `docs/mapping.md`, `docs/cost_map.md`, `docs/environment.md`, `docs/dashboard.md`.
No planner exists yet (Phase 5).
3D view: `python run_visual_demo.py` opens the real PyBullet window (see `docs/visual_mode.md`); verify the environment
in real PyBullet with `python tools/verify_pybullet_environment.py`.

## 9. Future Modules (Not Implemented Yet)

The following are **explicitly out of scope** for Weeks 1–6 and will only be
considered afterward:

* Reinforcement Learning-based control
* Model Predictive Control (MPC)
* LLM-based mission planning / natural-language interfaces
* Streamlit dashboard / web UI
* Advanced adaptive control

| Concept                      | Implementation Status |
| ----------------------------- | ---------------------- |
| Bathymetric environment       | Yes (Week 2)           |
| Environmental representation  | Yes (Week 2)           |
| Terrain understanding         | Yes (Week 3)           |
| Occupancy grid & cost map     | Yes (Week 4)           |
| AUV simulation (PyBullet)     | Yes (Week 5)           |
| Cost-aware A* planning        | Yes (Week 6)           |
| Reinforcement Learning        | Later (post Week 6)    |
| MPC                           | Later (post Week 6)    |
| LLM integration               | Optional / later        |
| Streamlit dashboard           | Later (post Week 6)    |

This project is **research-inspired** but is a simplified, explainable B.Tech
implementation. It does **not** claim to reproduce every component of any specific
research paper. All reported metrics (accuracy, path length, energy estimates, etc.)
come only from actual code execution — never fabricated or assumed.

## 10. Project Structure

```text
auv_project/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── raw/            # Real bathymetry data (user-provided), if any
│   ├── processed/       # Cleaned / normalized terrain data
│   └── synthetic/       # Generated synthetic bathymetry
│
├── models/              # Saved trained ML models (e.g., terrain_classifier.pkl)
│
├── configs/
│   └── config.yaml       # Central configuration for all modules
│
├── src/
│   ├── data_processing/  # Loading, cleaning, synthetic generation (Week 2)
│   ├── terrain/           # Feature extraction & classification (Week 3)
│   ├── mapping/            # Occupancy grid & cost map (Week 4)
│   ├── simulation/          # PyBullet AUV, environment, sensors (Week 5)
│   ├── planning/            # A* path planning (Week 6)
│   └── utils/                # Visualization & metrics helpers
│
├── tests/                # Unit tests for each module
├── notebooks/            # Exploratory notebooks (optional)
├── results/
│   ├── figures/           # Saved plots
│   ├── maps/               # Saved occupancy/cost/terrain maps
│   └── logs/                # Run logs
└── docs/                 # Detailed per-module documentation
```

## 11. How to Run (Week 2 Status)

```bash
# 1. Clone / open the project folder in VS Code
cd auv_project

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the Week 2 bathymetry processing pipeline
python -m src.data_processing.run_pipeline

# 5. Run the test suite
pytest tests/test_bathymetry.py -v
```

Expected output: the pipeline prints a progress report (grid size, resolution,
depth range, missing-value counts) and saves the processed depth grid plus
four figures under `results/figures/`. All 21 tests should pass.

## 12. Limitations (Week 2–3)

* No real bathymetry dataset has been integrated yet — all current results
  come from the synthetic generator (clearly labeled as such everywhere).
* Terrain class labels are synthetic and rule-based (derived from the same
  slope/roughness features the classifier trains on), not a real labeled
  seafloor survey — the ≈0.999 test accuracy reflects the model recovering
  that rule, not validated real-world terrain judgment. See
  [`docs/terrain_classification.md`](docs/terrain_classification.md).
* No simulation exists yet (Week 5).
* The outlier detector (Week 2) uses a single global z-score threshold,
  which may not suit terrain with legitimately large depth variation.

## 13. Next Steps

Proceed to **Week 4 — Occupancy Grid & Cost Map**, which will consume the
Week 3 terrain classification (`models/terrain_classifier.pkl`) to build a
navigation-ready occupancy grid and traversal cost map.


## Phase 5: autonomous navigation (A* + path following + avoidance + replanning)
Opt-in (`navigation.enabled: true`; default off keeps the Phase 4 mission unchanged). Estimate-only: the planner plans from the
EKF position over the online cost map, follows the path with the existing dynamics/thrusters, detours around newly mapped
obstacles, replans when the map invalidates the path, and reports `IDLE/PLAN/FOLLOW_PATH/AVOID_OBSTACLE/REPLAN/GOAL_REACHED/FAILED`
on the dashboard. See `docs/navigation.md`. Real PyBullet mission: `python tools/run_phase5_mission.py`.

UI + visual demo: every dashboard control is a real API call into the running simulation (goal / start / replan / cancel,
map, speed, sensor faults, next-start config), the goal inputs mirror the authoritative `navigator.goal`, and the
Environment panel reports the obstacle count the run actually built. `environment.obstacles.count` (and
`run_visual_demo.py --obstacles N`) generates EXACTLY N real PyBullet obstacle bodies, deterministically per seed.
`python run_visual_demo.py` opens the same simulation in a realistic underwater scene (deep-water backdrop, light +
shadows, water surface at z=0 -- rendering only, no collision geometry) with follow/top/orbit/zoom/reset camera controls
(keys 1-7); verify it objectively with `python tools/verify_visual_scene.py`.

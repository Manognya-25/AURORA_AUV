"""
run_pipeline.py
----------------
Week 4 entry point: turns Week 3's terrain intelligence into the
environmental representation a path planner (Week 6: A*) can consume.

    1. Load configuration + Week 2 processed bathymetry
    2. Extract terrain features (depth, slope, roughness) -- Week 3 code, reused
    3. Load the trained Week 3 classifier, predict + save the full-grid
       terrain classification map (bridges Week 3 -> Week 4)
    4. Build a binary occupancy grid from the terrain classification
    5. Inflate obstacles by a safety margin (AUV footprint + pose uncertainty)
    6. Build the navigation cost map from terrain features + inflated occupancy
    7. Save all mapping outputs (.npy + metadata) and generate visualizations
    8. Print a summary report

Run with:
    python -m src.mapping.run_pipeline
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import yaml

from src.terrain.feature_extraction import extract_features
from src.terrain.classifier import FEATURE_ORDER, load_model, predict_grid
from src.mapping.occupancy_grid import (
    OccupancyValues,
    build_occupancy_grid,
    format_occupancy_summary,
    occupancy_summary,
)
from src.mapping.obstacle_inflation import inflate_obstacles
from src.mapping.cost_map import CostWeights, build_cost_map
from src.utils.visualization import (
    plot_cost_component,
    plot_navigation_cost_map,
    plot_occupancy_grid,
)

CONFIG_PATH = "configs/config.yaml"
BATHYMETRY_PATH = "data/processed/processed_bathymetry.npy"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and return the project YAML config as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str = CONFIG_PATH, bathymetry_path: str = BATHYMETRY_PATH) -> dict:
    """
    Execute the full Week 4 environmental mapping pipeline.

    Input:
        config_path: path to configs/config.yaml.
        bathymetry_path: path to the Week 2 processed depth grid (.npy).
    Processing:
        See module docstring for the 8-step pipeline.
    Output:
        dict summary with keys: features, class_grid, occupancy_grid,
        inflation_result, cost_map (for programmatic use / testing).
    """
    start = time.time()
    cfg = load_config(config_path)
    terrain_cfg = cfg["terrain"]
    mapping_cfg = cfg["mapping"]
    resolution = cfg["environment"]["resolution"]

    processed_dir = cfg["output"]["processed_dir"]
    figures_dir = cfg["output"]["figures_dir"]

    print(f"[1/8] Loading processed bathymetry from {bathymetry_path} ...")
    depth = np.load(bathymetry_path)
    print(f"      grid shape: {depth.shape}, depth range: {depth.min():.2f}-{depth.max():.2f} m")

    print("[2/8] Extracting terrain features (depth, slope, roughness, curvature) ...")
    features = extract_features(
        depth, resolution=resolution, roughness_window=terrain_cfg["roughness_window"]
    )
    print(features)

    print(f"\n[3/8] Loading trained classifier from {terrain_cfg['model_path']} "
          f"and classifying the full grid ...")
    model = load_model(terrain_cfg["model_path"])
    X = features.as_array(FEATURE_ORDER)
    class_grid = predict_grid(model, X, depth.shape)
    class_map_path = os.path.join(processed_dir, "terrain_classification_map.npy")
    np.save(class_map_path, class_grid)
    print(f"      saved terrain classification map to {class_map_path}")

    print("\n[4/8] Building occupancy grid from terrain classification ...")
    occupancy_values = OccupancyValues.from_config(mapping_cfg)
    occupancy = build_occupancy_grid(class_grid, occupancy_values)
    raw_summary = occupancy_summary(occupancy, occupancy_values)
    print(format_occupancy_summary(raw_summary))

    print(f"\n[5/8] Inflating obstacles by safety radius "
          f"({mapping_cfg['safety_inflation_radius_cells']} cells) ...")
    inflation_result = inflate_obstacles(
        occupancy,
        radius_cells=mapping_cfg["safety_inflation_radius_cells"],
        obstacle_value=occupancy_values.obstacle,
    )
    inflated_occupancy = inflation_result.inflated_grid
    print(inflation_result)
    inflated_summary = occupancy_summary(inflated_occupancy, occupancy_values)
    print(format_occupancy_summary(inflated_summary))

    print("\n[6/8] Building navigation cost map from terrain features + occupancy ...")
    weights = CostWeights.from_config(mapping_cfg)
    cost_map = build_cost_map(
        features,
        inflated_occupancy,
        weights,
        obstacle_value=occupancy_values.obstacle,
        unknown_value=occupancy_values.unknown,
    )
    print(cost_map)

    print(f"\n[7/8] Saving mapping outputs to {processed_dir} and generating visualizations ...")
    os.makedirs(processed_dir, exist_ok=True)
    np.save(os.path.join(processed_dir, "occupancy_grid.npy"), occupancy)
    np.save(os.path.join(processed_dir, "occupancy_grid_inflated.npy"), inflated_occupancy)
    np.save(os.path.join(processed_dir, "navigation_cost_map.npy"), cost_map.total)

    metadata = {
        "rows": int(depth.shape[0]),
        "columns": int(depth.shape[1]),
        "resolution": resolution,
        "occupancy_values": {
            "free": occupancy_values.free,
            "obstacle": occupancy_values.obstacle,
            "unknown": occupancy_values.unknown,
        },
        "obstacle_class_id_source": "terrain classification class 3 (Obstacle)",
        "safety_inflation_radius_cells": mapping_cfg["safety_inflation_radius_cells"],
        "occupancy_before_inflation": raw_summary,
        "occupancy_after_inflation": inflated_summary,
        "newly_blocked_cells_from_inflation": inflation_result.newly_blocked_cells,
        "cost_weights": {
            "w_depth": weights.w_depth,
            "w_slope": weights.w_slope,
            "w_roughness": weights.w_roughness,
            "w_obstacle": weights.w_obstacle,
            "unknown_cell_cost": weights.unknown_cell_cost,
        },
        "cost_stats": {
            "depth_cost": {"min": float(cost_map.depth_cost.min()), "max": float(cost_map.depth_cost.max())},
            "slope_cost": {"min": float(cost_map.slope_cost.min()), "max": float(cost_map.slope_cost.max())},
            "roughness_cost": {"min": float(cost_map.roughness_cost.min()), "max": float(cost_map.roughness_cost.max())},
            "total_cost_traversable_max": float(
                cost_map.total[cost_map.total < weights.w_obstacle].max()
            ) if np.any(cost_map.total < weights.w_obstacle) else None,
        },
    }
    metadata_path = os.path.join(processed_dir, "mapping_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"      saved metadata to {metadata_path}")

    plot_occupancy_grid(
        occupancy, os.path.join(figures_dir, "occupancy_grid.png"),
        occupancy_values, title="Occupancy Grid (from terrain classification)",
    )
    plot_occupancy_grid(
        inflated_occupancy, os.path.join(figures_dir, "occupancy_grid_inflated.png"),
        occupancy_values,
        title=f"Occupancy Grid (inflated by {mapping_cfg['safety_inflation_radius_cells']} cells)",
    )
    plot_cost_component(
        cost_map.depth_cost, os.path.join(figures_dir, "cost_depth.png"),
        title="Depth Cost Component (normalized)",
    )
    plot_cost_component(
        cost_map.slope_cost, os.path.join(figures_dir, "cost_slope.png"),
        title="Slope Cost Component (normalized)", cmap="inferno",
    )
    plot_cost_component(
        cost_map.roughness_cost, os.path.join(figures_dir, "cost_roughness.png"),
        title="Roughness Cost Component (normalized)", cmap="inferno",
    )
    plot_navigation_cost_map(
        cost_map.total, os.path.join(figures_dir, "navigation_cost_map.png"),
        title="Navigation Cost Map (weighted combination + obstacles)",
        obstacle_cost_floor=weights.w_obstacle,
    )

    elapsed = time.time() - start
    print(f"\n[8/8] Done in {elapsed:.2f}s. "
          f"Free: {inflated_summary['free']['fraction']*100:.1f}%  "
          f"Obstacle (incl. inflation): {inflated_summary['obstacle']['fraction']*100:.1f}%  "
          f"Unknown: {inflated_summary['unknown']['fraction']*100:.1f}%")

    return {
        "features": features,
        "class_grid": class_grid,
        "occupancy_grid": occupancy,
        "inflation_result": inflation_result,
        "cost_map": cost_map,
    }


if __name__ == "__main__":
    run()

"""
run_pipeline.py
----------------
Week 2 entry point: executes the full bathymetry processing pipeline.

    1. Load configuration
    2. Generate synthetic environment OR load real bathymetry
    3. Validate
    4. Clean
    5. Normalize if enabled
    6. Save processed data (+ metadata)
    7. Generate visualizations
    8. Print dataset statistics

Run with:
    python -m src.data_processing.run_pipeline
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import yaml

from src.data_processing.generate_environment import create_environment
from src.data_processing.preprocess import validate_bathymetry, normalize_depth
from src.utils.visualization import (
    plot_depth_map_2d,
    plot_depth_map_3d,
    plot_depth_histogram,
    plot_preprocessing_comparison,
)

CONFIG_PATH = "configs/config.yaml"


def load_config(path: str = CONFIG_PATH) -> dict:
    """
    Load the project's YAML configuration file.

    Input:  path to config.yaml.
    Processing: parses YAML into a nested Python dict.
    Output: configuration dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_week2_pipeline(config: dict) -> dict:
    """
    Execute the full Week 2 bathymetry processing pipeline using values from
    the provided configuration dictionary.

    Input:
        config: parsed config.yaml dictionary (must contain "environment",
                 "bathymetry", and "output" sections).
    Processing:
        Runs load/generate -> validate -> clean -> (optional) normalize ->
        save -> visualize, printing progress and statistics throughout.
    Output:
        A summary dictionary describing what was produced (paths, stats),
        useful for tests and for chaining into later weeks.
    """
    print("=" * 40)
    print("AUV ENVIRONMENT PROCESSING — WEEK 2")
    print("=" * 40)
    print()

    env_cfg = config["environment"]
    bathy_cfg = config["bathymetry"]
    output_cfg = config["output"]

    source = bathy_cfg["source"]
    print(f"Source: {source}")
    print()

    rows = env_cfg["grid_size"]["rows"]
    cols = env_cfg["grid_size"]["cols"]
    resolution = env_cfg.get("resolution", 1.0)

    synthetic_params = dict(env_cfg.get("synthetic_generation", {}))
    # "seed" may be specified either under synthetic_generation or fall back
    # to the project-wide random seed for reproducibility.
    synthetic_params.setdefault("seed", config.get("project", {}).get("random_seed", 42))

    print("Generating bathymetric environment..." if source == "synthetic" else "Loading real bathymetry dataset...")
    environment = create_environment(
        source=source,
        rows=rows,
        cols=cols,
        resolution=resolution,
        real_input_path=bathy_cfg.get("raw_input_path"),
        depth_convention=bathy_cfg.get("depth_convention", "positive_down"),
        interpolation_method=bathy_cfg.get("interpolation_method", "linear"),
        synthetic_params=synthetic_params,
    )
    print()
    print(f"Grid:\n{rows} x {cols}")
    print()
    print(f"Resolution:\n{resolution} m/cell")
    print()

    raw_report = environment.metadata["validation_report"]
    print("Raw depth range:")
    print(f"{environment.min_depth:.4f} m  to  {environment.max_depth:.4f} m")
    print()
    print("Missing values:")
    print(raw_report["missing_values_found"])
    print()

    print("Cleaning bathymetry...")
    # create_environment() already cleans internally; re-validate the result
    # so the reported "processed" statistics reflect the final cleaned grid.
    processed_report = validate_bathymetry(environment.depth_grid)
    print()
    print("Processed depth range:")
    print(f"{processed_report.min_depth:.4f} m  to  {processed_report.max_depth:.4f} m")
    print()

    normalized_grid = None
    if bathy_cfg.get("normalize", False):
        method = bathy_cfg.get("normalization_method", "minmax")
        print(f"Normalizing depth grid (method={method})...")
        normalized_grid = normalize_depth(environment.depth_grid, method=method)
        print()

    # ---- Save processed data ----
    print("Saving processed environment...")
    processed_dir = output_cfg["processed_dir"]
    os.makedirs(processed_dir, exist_ok=True)

    processed_npy_path = os.path.join(processed_dir, "processed_bathymetry.npy")
    np.save(processed_npy_path, environment.depth_grid)

    metadata_path = os.path.join(processed_dir, "processed_bathymetry_metadata.json")
    metadata = dict(environment.metadata)
    metadata["min_depth"] = environment.min_depth
    metadata["max_depth"] = environment.max_depth
    metadata["normalized"] = bool(bathy_cfg.get("normalize", False))
    if normalized_grid is not None:
        normalized_path = os.path.join(processed_dir, "processed_bathymetry_normalized.npy")
        np.save(normalized_path, normalized_grid)
        metadata["normalized_path"] = normalized_path
        metadata["normalization_method"] = bathy_cfg.get("normalization_method", "minmax")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print()
    print("Saved:")
    print(processed_npy_path)
    print()

    # ---- Visualizations ----
    print("Generating visualizations...")
    figures_dir = output_cfg["figures_dir"]

    fig_2d = plot_depth_map_2d(
        environment.depth_grid,
        save_path=os.path.join(figures_dir, "bathymetry_2d.png"),
        title=f"Bathymetry Depth Map ({source.capitalize()})",
        resolution=resolution,
    )
    fig_3d = plot_depth_map_3d(
        environment.depth_grid,
        save_path=os.path.join(figures_dir, "bathymetry_3d.png"),
        title=f"Bathymetry 3D Terrain ({source.capitalize()})",
        resolution=resolution,
    )
    fig_hist = plot_depth_histogram(
        environment.depth_grid,
        save_path=os.path.join(figures_dir, "depth_histogram.png"),
        title=f"Depth Distribution ({source.capitalize()} Bathymetry)",
    )

    # Preprocessing before/after comparison figure: demonstrate cleaning on a
    # deliberately corrupted copy of the SAME environment (illustrative only).
    rng = np.random.default_rng(0)
    corrupted = environment.depth_grid.copy()
    corrupt_mask = rng.random(corrupted.shape) < 0.05  # corrupt 5% of cells
    corrupted[corrupt_mask] = np.nan
    # inject a few extreme outliers too
    n_outliers = max(1, corrupted.size // 500)
    flat_idx = rng.choice(corrupted.size, size=n_outliers, replace=False)
    corrupted.flat[flat_idx] = environment.max_depth * 5

    from src.data_processing.preprocess import clean_bathymetry
    cleaned_demo = clean_bathymetry(
        corrupted,
        interpolation_method=bathy_cfg.get("interpolation_method", "linear"),
        filter_outliers=True,
    )

    fig_compare = plot_preprocessing_comparison(
        raw_grid=environment.depth_grid,
        corrupted_grid=corrupted,
        cleaned_grid=cleaned_demo,
        save_path=os.path.join(figures_dir, "preprocessing_comparison.png"),
    )

    print()
    print("Saved:")
    print(fig_2d)
    print(fig_3d)
    print(fig_hist)
    print(fig_compare)
    print()
    print("WEEK 2 PIPELINE COMPLETED")

    return {
        "environment": environment,
        "processed_npy_path": processed_npy_path,
        "metadata_path": metadata_path,
        "figures": [fig_2d, fig_3d, fig_hist, fig_compare],
        "normalized_path": metadata.get("normalized_path"),
    }


def main() -> None:
    """Load config and run the Week 2 pipeline end-to-end."""
    config = load_config()
    run_week2_pipeline(config)


if __name__ == "__main__":
    main()

"""
run_pipeline.py
----------------
Week 3 entry point: executes the full terrain classification pipeline.

    1. Load configuration + Week 2 processed bathymetry
    2. Extract terrain features (depth, slope, roughness, curvature)
    3. Generate synthetic labels from calibrated thresholds
    4. Split into train/val/test
    5. Train RandomForestClassifier
    6. Evaluate on val + test splits
    7. Save model + visualizations
    8. Print a summary report

Run with:
    python -m src.terrain.run_pipeline
"""

from __future__ import annotations

import time

import numpy as np
import yaml

from src.terrain.feature_extraction import extract_features
from src.terrain.label_generation import (
    LabelThresholds,
    class_distribution,
    format_distribution,
    generate_labels,
)
from src.terrain.classifier import (
    FEATURE_ORDER,
    predict_grid,
    save_model,
    split_dataset,
    train_classifier,
)
from src.terrain.evaluate import evaluate_classifier, feature_importance_report
from src.utils.visualization import (
    plot_confusion_matrix,
    plot_feature_grids,
    plot_terrain_classification,
)

CONFIG_PATH = "configs/config.yaml"
BATHYMETRY_PATH = "data/processed/processed_bathymetry.npy"


def load_config(path: str = CONFIG_PATH) -> dict:
    """Load and return the project YAML config as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run(config_path: str = CONFIG_PATH, bathymetry_path: str = BATHYMETRY_PATH) -> dict:
    """
    Execute the full Week 3 terrain classification pipeline.

    Input:
        config_path: path to configs/config.yaml.
        bathymetry_path: path to the Week 2 processed depth grid (.npy).
    Processing:
        See module docstring for the 8-step pipeline.
    Output:
        dict summary with keys: features, labels, splits, model,
        val_result, test_result (for programmatic use / testing).
    """
    start = time.time()
    cfg = load_config(config_path)
    terrain_cfg = cfg["terrain"]
    resolution = cfg["environment"]["resolution"]
    random_seed = cfg["project"]["random_seed"]

    print(f"[1/8] Loading processed bathymetry from {bathymetry_path} ...")
    depth = np.load(bathymetry_path)
    print(f"      grid shape: {depth.shape}, depth range: {depth.min():.2f}-{depth.max():.2f} m")

    print("[2/8] Extracting terrain features (depth, slope, roughness, curvature) ...")
    features = extract_features(
        depth, resolution=resolution, roughness_window=terrain_cfg["roughness_window"]
    )
    print(features)

    print("\n[3/8] Generating synthetic labels from calibrated thresholds ...")
    thresholds = LabelThresholds.from_config(terrain_cfg)
    labels = generate_labels(features, thresholds)
    dist = class_distribution(labels)
    print(format_distribution(dist))

    print("\n[4/8] Splitting into train/val/test ...")
    X = features.as_array(FEATURE_ORDER)
    y = labels.reshape(-1)
    splits = split_dataset(
        X, y,
        train_split=terrain_cfg["train_split"],
        val_split=terrain_cfg["val_split"],
        test_split=terrain_cfg["test_split"],
        random_seed=random_seed,
    )
    print(splits)

    print("\n[5/8] Training RandomForestClassifier ...")
    clf_cfg = terrain_cfg["classifier"]
    model = train_classifier(
        splits.X_train, splits.y_train,
        n_estimators=clf_cfg["n_estimators"],
        max_depth=clf_cfg["max_depth"],
        random_seed=random_seed,
    )
    print(feature_importance_report(model, FEATURE_ORDER))

    print("\n[6/8] Evaluating on validation and test splits ...")
    val_result = evaluate_classifier(model, splits.X_val, splits.y_val)
    test_result = evaluate_classifier(model, splits.X_test, splits.y_test)
    print("\n--- Validation ---")
    print(val_result)
    print("\n--- Test ---")
    print(test_result)

    print(f"\n[7/8] Saving model to {terrain_cfg['model_path']} and generating visualizations ...")
    save_model(model, terrain_cfg["model_path"])
    plot_feature_grids(features, "results/figures/terrain_features.png")
    plot_terrain_classification(labels, "results/figures/terrain_labels_synthetic.png",
                                 title="Synthetic Training Labels (rule-based)")
    predicted_grid = predict_grid(model, X, depth.shape)
    plot_terrain_classification(predicted_grid, "results/figures/terrain_classification_predicted.png",
                                 title="RandomForest Terrain Classification (full grid)")
    plot_confusion_matrix(test_result.confusion, "results/figures/terrain_confusion_matrix.png")

    elapsed = time.time() - start
    print(f"\n[8/8] Done in {elapsed:.2f}s. "
          f"Test accuracy: {test_result.accuracy:.4f}, macro-F1: {test_result.macro_f1:.4f}")

    return {
        "features": features,
        "labels": labels,
        "splits": splits,
        "model": model,
        "val_result": val_result,
        "test_result": test_result,
    }


if __name__ == "__main__":
    run()

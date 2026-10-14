"""
test_terrain.py
----------------
Unit tests for src/terrain/ (Week 3: feature extraction, label generation,
classifier training/evaluation).
"""

from __future__ import annotations

import numpy as np
import pytest
import yaml

from src.terrain.feature_extraction import (
    FeatureSet,
    compute_curvature,
    compute_roughness,
    compute_slope,
    extract_features,
)
from src.terrain.label_generation import (
    CLASS_NAMES,
    LabelThresholds,
    class_distribution,
    generate_labels,
)
from src.terrain.classifier import (
    FEATURE_ORDER,
    load_model,
    predict_grid,
    save_model,
    split_dataset,
    train_classifier,
)
from src.terrain.evaluate import evaluate_classifier, feature_importance_report


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_depth_grid():
    """The actual Week 2 processed bathymetry grid."""
    return np.load("data/processed/processed_bathymetry.npy")


@pytest.fixture(scope="module")
def config():
    with open("configs/config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def flat_grid():
    """A perfectly flat 10x10 grid: slope/roughness/curvature should be ~0."""
    return np.full((10, 10), 50.0)


@pytest.fixture
def ramp_grid():
    """A constant-gradient ramp: slope should be constant, curvature ~0."""
    rows, cols = 20, 20
    x = np.arange(cols, dtype=float)
    return np.tile(x * 2.0, (rows, 1))  # depth increases by 2.0 m per cell in x


@pytest.fixture
def spike_grid():
    """Flat grid with a single spike: roughness should peak at the spike."""
    grid = np.full((11, 11), 10.0)
    grid[5, 5] = 100.0
    return grid


# --------------------------------------------------------------------------
# Phase 1: feature_extraction.py
# --------------------------------------------------------------------------

class TestComputeSlope:
    def test_flat_grid_has_zero_slope(self, flat_grid):
        slope = compute_slope(flat_grid, resolution=1.0)
        assert np.allclose(slope, 0.0)

    def test_ramp_grid_has_constant_slope(self, ramp_grid):
        slope = compute_slope(ramp_grid, resolution=1.0)
        # interior cells should all have slope magnitude ~= 2.0 (the ramp's gradient)
        interior = slope[2:-2, 2:-2]
        assert np.allclose(interior, 2.0, atol=1e-8)

    def test_slope_scales_inversely_with_resolution(self, ramp_grid):
        slope_res1 = compute_slope(ramp_grid, resolution=1.0)
        slope_res2 = compute_slope(ramp_grid, resolution=2.0)
        # doubling resolution (m/cell) halves the computed gradient
        assert np.allclose(slope_res1[5, 5], 2.0 * slope_res2[5, 5], atol=1e-8)

    def test_slope_is_nonnegative(self, real_depth_grid):
        slope = compute_slope(real_depth_grid, resolution=1.0)
        assert np.all(slope >= 0)

    def test_rejects_nonpositive_resolution(self, flat_grid):
        with pytest.raises(ValueError):
            compute_slope(flat_grid, resolution=0.0)


class TestComputeRoughness:
    def test_flat_grid_has_zero_roughness(self, flat_grid):
        roughness = compute_roughness(flat_grid, window=3)
        assert np.allclose(roughness, 0.0, atol=1e-10)

    def test_spike_increases_local_roughness(self, spike_grid):
        roughness = compute_roughness(spike_grid, window=3)
        # roughness at the spike's neighborhood should be far higher than elsewhere
        assert roughness[5, 5] > roughness[0, 0]
        assert roughness[5, 5] > 0

    def test_roughness_is_nonnegative(self, real_depth_grid):
        roughness = compute_roughness(real_depth_grid, window=3)
        assert np.all(roughness >= 0)

    def test_output_shape_matches_input(self, real_depth_grid):
        roughness = compute_roughness(real_depth_grid, window=3)
        assert roughness.shape == real_depth_grid.shape

    def test_rejects_even_window(self, flat_grid):
        with pytest.raises(ValueError):
            compute_roughness(flat_grid, window=4)

    def test_rejects_too_small_window(self, flat_grid):
        with pytest.raises(ValueError):
            compute_roughness(flat_grid, window=1)


class TestComputeCurvature:
    def test_flat_grid_has_zero_curvature(self, flat_grid):
        curvature = compute_curvature(flat_grid, resolution=1.0)
        assert np.allclose(curvature, 0.0, atol=1e-10)

    def test_ramp_grid_has_zero_curvature(self, ramp_grid):
        # a constant-gradient ramp has zero second derivative everywhere (interior)
        curvature = compute_curvature(ramp_grid, resolution=1.0)
        interior = curvature[2:-2, 2:-2]
        assert np.allclose(interior, 0.0, atol=1e-8)

    def test_curvature_output_shape(self, real_depth_grid):
        curvature = compute_curvature(real_depth_grid, resolution=1.0)
        assert curvature.shape == real_depth_grid.shape


class TestExtractFeatures:
    def test_returns_all_four_features(self, real_depth_grid):
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        assert isinstance(fs, FeatureSet)
        for name in ("depth", "slope", "roughness", "curvature"):
            grid = getattr(fs, name)
            assert grid.shape == real_depth_grid.shape

    def test_as_array_shape_and_order(self, real_depth_grid):
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        arr = fs.as_array(("depth", "slope", "roughness", "curvature"))
        rows, cols = real_depth_grid.shape
        assert arr.shape == (rows * cols, 4)
        # column 0 should be the flattened depth grid
        assert np.allclose(arr[:, 0], real_depth_grid.reshape(-1))

    def test_rejects_nan_input(self):
        bad_grid = np.full((5, 5), 1.0)
        bad_grid[2, 2] = np.nan
        with pytest.raises(ValueError):
            extract_features(bad_grid)

    def test_rejects_non_2d_input(self):
        with pytest.raises(ValueError):
            extract_features(np.ones((5, 5, 2)))

    def test_matches_real_data_reported_stats(self, real_depth_grid):
        # sanity-check the stats used to justify the Week 3 threshold recalibration
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        assert 0.0 <= fs.slope.min() < 0.1
        assert 7.0 < fs.slope.max() < 9.0
        assert 0.0 < fs.roughness.min() < 1.0
        assert 3.5 < fs.roughness.max() < 5.5


# --------------------------------------------------------------------------
# Phase 2: label_generation.py
# --------------------------------------------------------------------------

class TestLabelThresholds:
    def test_from_config_reads_calibrated_values(self, config):
        thresholds = LabelThresholds.from_config(config["terrain"])
        assert thresholds.slope_rocky == pytest.approx(1.65)
        assert thresholds.slope_steep == pytest.approx(2.35)
        assert thresholds.roughness_obstacle == pytest.approx(2.8)


class TestGenerateLabels:
    def test_flat_grid_is_all_class_zero(self, flat_grid):
        fs = extract_features(flat_grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds(slope_rocky=0.1, slope_steep=0.5, roughness_obstacle=1.0)
        labels = generate_labels(fs, thresholds)
        assert np.all(labels == 0)

    def test_output_shape_matches_grid(self, real_depth_grid, config):
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds.from_config(config["terrain"])
        labels = generate_labels(fs, thresholds)
        assert labels.shape == real_depth_grid.shape
        assert labels.dtype == np.int64

    def test_labels_only_contain_known_classes(self, real_depth_grid, config):
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds.from_config(config["terrain"])
        labels = generate_labels(fs, thresholds)
        assert set(np.unique(labels)).issubset(set(CLASS_NAMES.keys()))

    def test_high_roughness_always_wins_obstacle(self):
        # even a low-slope cell should be Obstacle if roughness exceeds threshold
        grid = np.full((5, 5), 10.0)
        grid[2, 2] = 200.0  # spike -> huge local roughness, negligible net slope contribution elsewhere
        fs = extract_features(grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds(slope_rocky=100.0, slope_steep=200.0, roughness_obstacle=1.0)
        labels = generate_labels(fs, thresholds)
        assert labels[2, 2] == 3

    def test_calibrated_thresholds_give_nondegenerate_distribution(self, real_depth_grid, config):
        # regression test for the Week 3 recalibration: no class should
        # dominate to the point of being unusable for training
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds.from_config(config["terrain"])
        labels = generate_labels(fs, thresholds)
        dist = class_distribution(labels)
        for class_id, info in dist.items():
            assert 0.01 < info["fraction"] < 0.95, (
                f"class {class_id} ({info['name']}) fraction {info['fraction']:.3f} "
                "is degenerate -- thresholds need recalibration"
            )


class TestClassDistribution:
    def test_fractions_sum_to_one(self, real_depth_grid, config):
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds.from_config(config["terrain"])
        labels = generate_labels(fs, thresholds)
        dist = class_distribution(labels)
        total_fraction = sum(info["fraction"] for info in dist.values())
        assert total_fraction == pytest.approx(1.0)

    def test_counts_sum_to_total_cells(self, real_depth_grid, config):
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=3)
        thresholds = LabelThresholds.from_config(config["terrain"])
        labels = generate_labels(fs, thresholds)
        dist = class_distribution(labels)
        total_count = sum(info["count"] for info in dist.values())
        assert total_count == real_depth_grid.size


# --------------------------------------------------------------------------
# Phase 3: classifier.py + evaluate.py
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def trained_pipeline(real_depth_grid, config):
    """Run feature extraction -> labels -> split -> train once, reused by several tests."""
    terrain_cfg = config["terrain"]
    fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=terrain_cfg["roughness_window"])
    thresholds = LabelThresholds.from_config(terrain_cfg)
    labels = generate_labels(fs, thresholds)

    X = fs.as_array(FEATURE_ORDER)
    y = labels.reshape(-1)
    splits = split_dataset(
        X, y,
        train_split=terrain_cfg["train_split"],
        val_split=terrain_cfg["val_split"],
        test_split=terrain_cfg["test_split"],
        random_seed=config["project"]["random_seed"],
    )
    model = train_classifier(
        splits.X_train, splits.y_train,
        n_estimators=50,  # smaller forest for fast tests
        max_depth=terrain_cfg["classifier"]["max_depth"],
        random_seed=config["project"]["random_seed"],
    )
    return {"features": fs, "labels": labels, "splits": splits, "model": model}


class TestSplitDataset:
    def test_split_sizes_match_config_fractions(self, trained_pipeline):
        splits = trained_pipeline["splits"]
        total = splits.X_train.shape[0] + splits.X_val.shape[0] + splits.X_test.shape[0]
        assert total == 10000
        assert splits.X_train.shape[0] == pytest.approx(0.70 * total, abs=5)
        assert splits.X_val.shape[0] == pytest.approx(0.15 * total, abs=5)
        assert splits.X_test.shape[0] == pytest.approx(0.15 * total, abs=5)

    def test_rejects_bad_split_fractions(self):
        X = np.zeros((100, 4))
        y = np.zeros(100, dtype=int)
        with pytest.raises(ValueError):
            split_dataset(X, y, train_split=0.5, val_split=0.5, test_split=0.5)

    def test_stratification_preserves_minority_class(self, real_depth_grid, config):
        # Obstacle (~4.6%) must appear in all three splits, not just train
        terrain_cfg = config["terrain"]
        fs = extract_features(real_depth_grid, resolution=1.0, roughness_window=terrain_cfg["roughness_window"])
        thresholds = LabelThresholds.from_config(terrain_cfg)
        labels = generate_labels(fs, thresholds)
        X = fs.as_array(FEATURE_ORDER)
        y = labels.reshape(-1)
        splits = split_dataset(X, y, 0.70, 0.15, 0.15, random_seed=42)
        assert 3 in np.unique(splits.y_train)
        assert 3 in np.unique(splits.y_val)
        assert 3 in np.unique(splits.y_test)


class TestTrainClassifier:
    def test_model_predicts_known_classes(self, trained_pipeline):
        model = trained_pipeline["model"]
        splits = trained_pipeline["splits"]
        preds = model.predict(splits.X_test)
        assert set(np.unique(preds)).issubset(set(CLASS_NAMES.keys()))

    def test_reasonable_test_accuracy(self, trained_pipeline):
        # labels are a deterministic function of the input features, so a
        # RandomForest should fit this relationship very well
        model = trained_pipeline["model"]
        splits = trained_pipeline["splits"]
        acc = model.score(splits.X_test, splits.y_test)
        assert acc > 0.85

    def test_predict_grid_reshapes_correctly(self, trained_pipeline, real_depth_grid):
        model = trained_pipeline["model"]
        fs = trained_pipeline["features"]
        X = fs.as_array(FEATURE_ORDER)
        grid = predict_grid(model, X, real_depth_grid.shape)
        assert grid.shape == real_depth_grid.shape


class TestSaveLoadModel:
    def test_save_and_load_round_trip(self, trained_pipeline, tmp_path):
        model = trained_pipeline["model"]
        splits = trained_pipeline["splits"]
        path = tmp_path / "model.pkl"
        save_model(model, str(path))
        loaded = load_model(str(path))
        original_preds = model.predict(splits.X_test)
        loaded_preds = loaded.predict(splits.X_test)
        assert np.array_equal(original_preds, loaded_preds)

    def test_load_missing_model_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(str(tmp_path / "does_not_exist.pkl"))


class TestEvaluateClassifier:
    def test_evaluation_result_fields(self, trained_pipeline):
        model = trained_pipeline["model"]
        splits = trained_pipeline["splits"]
        result = evaluate_classifier(model, splits.X_test, splits.y_test)
        assert 0.0 <= result.accuracy <= 1.0
        assert 0.0 <= result.macro_f1 <= 1.0
        assert result.confusion.shape == (4, 4)

    def test_feature_importance_report_lists_all_features(self, trained_pipeline):
        model = trained_pipeline["model"]
        report = feature_importance_report(model, FEATURE_ORDER)
        for name in FEATURE_ORDER:
            assert name in report

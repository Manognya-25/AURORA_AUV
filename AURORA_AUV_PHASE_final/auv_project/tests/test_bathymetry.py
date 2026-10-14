"""
test_bathymetry.py
-------------------
Unit tests for the Week 2 bathymetry processing pipeline:
    - synthetic terrain generation
    - cleaning (NaN / Inf handling)
    - normalization
    - depth convention conversion
    - saving/loading .npy round-trip
    - real-dataset (mock CSV) loading

Run with:
    pytest tests/test_bathymetry.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Ensure the project root is on sys.path when running pytest from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_processing.generate_environment import generate_synthetic_bathymetry, create_environment
from src.data_processing.preprocess import (
    validate_bathymetry,
    clean_bathymetry,
    normalize_depth,
    apply_depth_convention,
)
from src.data_processing.load_bathymetry import load_csv, load_bathymetry


# --------------------------------------------------------------------------
# Test 1: Synthetic terrain generation -> correct shape
# --------------------------------------------------------------------------

def test_synthetic_generation_shape():
    grid = generate_synthetic_bathymetry(rows=40, cols=60, seed=1)
    assert grid.shape == (40, 60)
    assert grid.dtype == np.float64
    # Depth must be non-negative under the "positive_down" convention.
    assert np.all(grid >= 0)


def test_synthetic_generation_has_spatial_continuity():
    """Neighboring cells should be similar in value (not pure random noise)."""
    grid = generate_synthetic_bathymetry(rows=50, cols=50, seed=2, noise_std=1.0)
    horizontal_diff = np.abs(np.diff(grid, axis=1))
    # Mean adjacent-cell difference should be small relative to the overall
    # depth range, indicating smooth/continuous terrain rather than noise.
    depth_range = grid.max() - grid.min()
    assert np.mean(horizontal_diff) < 0.15 * depth_range


# --------------------------------------------------------------------------
# Test 2: No NaN after preprocessing
# --------------------------------------------------------------------------

def test_clean_bathymetry_removes_nan():
    grid = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [7.0, 8.0, 9.0]])
    cleaned = clean_bathymetry(grid, interpolation_method="linear")
    assert not np.any(np.isnan(cleaned))


# --------------------------------------------------------------------------
# Test 3: No infinite values after preprocessing
# --------------------------------------------------------------------------

def test_clean_bathymetry_removes_infinite():
    grid = np.array([[1.0, np.inf, 3.0], [4.0, -np.inf, 6.0], [7.0, 8.0, 9.0]])
    cleaned = clean_bathymetry(grid, interpolation_method="nearest")
    assert not np.any(np.isinf(cleaned))
    assert not np.any(np.isnan(cleaned))


@pytest.mark.parametrize("method", ["nearest", "linear", "median"])
def test_clean_bathymetry_all_methods(method):
    grid = np.full((10, 10), 5.0)
    grid[3, 3] = np.nan
    grid[7, 7] = np.inf
    cleaned = clean_bathymetry(grid, interpolation_method=method)
    assert not np.any(np.isnan(cleaned))
    assert not np.any(np.isinf(cleaned))
    assert cleaned.shape == grid.shape


# --------------------------------------------------------------------------
# Test 4: Normalization works correctly
# --------------------------------------------------------------------------

def test_normalize_minmax_range():
    grid = np.array([[10.0, 20.0], [30.0, 40.0]])
    normalized = normalize_depth(grid, method="minmax")
    assert np.isclose(normalized.min(), 0.0)
    assert np.isclose(normalized.max(), 1.0)


def test_normalize_standardize_stats():
    rng = np.random.default_rng(0)
    grid = rng.normal(loc=50, scale=10, size=(20, 20))
    normalized = normalize_depth(grid, method="standardize")
    assert np.isclose(np.mean(normalized), 0.0, atol=1e-8)
    assert np.isclose(np.std(normalized), 1.0, atol=1e-8)


def test_normalize_does_not_mutate_original():
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    original_copy = grid.copy()
    _ = normalize_depth(grid, method="minmax")
    assert np.array_equal(grid, original_copy)


# --------------------------------------------------------------------------
# Test 5: Depth convention conversion works
# --------------------------------------------------------------------------

def test_depth_convention_positive_down_noop():
    grid = np.array([[1.0, 2.0], [3.0, 4.0]])
    converted = apply_depth_convention(grid, source_convention="positive_down")
    assert np.array_equal(converted, grid)


def test_depth_convention_negative_down_flips_sign():
    grid = np.array([[-1.0, -2.0], [-3.0, -4.0]])
    converted = apply_depth_convention(grid, source_convention="negative_down")
    assert np.array_equal(converted, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_depth_convention_invalid_raises():
    grid = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError):
        apply_depth_convention(grid, source_convention="sideways")


# --------------------------------------------------------------------------
# Test 6: Saved .npy can be loaded again
# --------------------------------------------------------------------------

def test_npy_save_and_load_roundtrip():
    grid = generate_synthetic_bathymetry(rows=20, cols=20, seed=5)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "test_bathymetry.npy")
        np.save(path, grid)
        loaded = np.load(path)
        assert np.array_equal(grid, loaded)
        assert loaded.shape == (20, 20)


# --------------------------------------------------------------------------
# Test 7: Real dataset loader handles a small mock CSV
# --------------------------------------------------------------------------

def test_load_csv_mock_dataset():
    mock_grid = np.array([[10.0, 11.0, 12.0], [13.0, 14.0, 15.0], [16.0, 17.0, 18.0]])
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "mock_bathymetry.csv")
        pd.DataFrame(mock_grid).to_csv(path, header=False, index=False)

        loaded = load_csv(path)
        assert loaded.shape == (3, 3)
        assert np.allclose(loaded, mock_grid)

        # Also confirm the unified auto-dispatch loader handles it identically.
        auto_loaded = load_bathymetry(path)
        assert np.allclose(auto_loaded, mock_grid)


def test_load_bathymetry_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        load_bathymetry("data/raw/this_file_does_not_exist.csv")


def test_load_bathymetry_unsupported_extension_raises():
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = os.path.join(tmp_dir, "file.unsupported")
        with open(path, "w", encoding="utf-8") as f:
            f.write("not a real bathymetry file")
        with pytest.raises(ValueError):
            load_bathymetry(path)


# --------------------------------------------------------------------------
# Integration test: create_environment() end-to-end (synthetic)
# --------------------------------------------------------------------------

def test_create_environment_synthetic_end_to_end():
    env = create_environment(source="synthetic", rows=30, cols=30, synthetic_params={"seed": 7})
    assert env.depth_grid.shape == (30, 30)
    assert not np.any(np.isnan(env.depth_grid))
    assert not np.any(np.isinf(env.depth_grid))
    assert env.source == "synthetic"
    assert env.min_depth <= env.max_depth
    assert "data_origin_note" in env.metadata
    # Must be explicitly and honestly labeled as synthetic.
    assert "ynthetic" in env.metadata["data_origin_note"]


def test_create_environment_real_requires_path():
    with pytest.raises(ValueError):
        create_environment(source="real", real_input_path=None)


def test_create_environment_invalid_source_raises():
    with pytest.raises(ValueError):
        create_environment(source="imaginary_source")


# --------------------------------------------------------------------------
# Validation report sanity checks
# --------------------------------------------------------------------------

def test_validate_bathymetry_report_counts():
    grid = np.array([[1.0, np.nan, np.inf], [4.0, 5.0, 6.0]])
    report = validate_bathymetry(grid)
    assert report.shape == (2, 3)
    assert report.num_missing == 1
    assert report.num_infinite == 1
    assert report.num_valid_cells == 4  # total 6 cells - 1 nan - 1 inf

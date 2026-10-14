"""
preprocess.py
-------------
Validation, cleaning, depth-convention handling, and normalization for a
bathymetry depth grid.

Pipeline implemented here:

    raw grid -> validate_bathymetry() -> report
                       |
                       v
             clean_bathymetry() -> cleaned grid (NaN/Inf handled)
                       |
                       v
          apply_depth_convention() -> consistent "positive_down" grid
                       |
                       v
             normalize_depth()  -> OPTIONAL separate normalized copy

The raw/cleaned depth grid is NEVER overwritten by normalization -- callers
keep both arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import interpolate as sp_interpolate


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

@dataclass
class ValidationReport:
    """Holds the results of validating a bathymetry grid."""
    shape: tuple
    dtype: str
    min_depth: float
    max_depth: float
    num_missing: int
    num_infinite: int
    num_valid_cells: int
    num_outliers: int = 0

    def __str__(self) -> str:
        lines = [
            "Bathymetry Dataset Report",
            "-------------------------",
            f"Shape: {self.shape[0]} x {self.shape[1]}",
            f"Data type: {self.dtype}",
            "",
            f"Minimum depth: {self.min_depth:.4f}",
            f"Maximum depth: {self.max_depth:.4f}",
            "",
            f"Missing values: {self.num_missing}",
            f"Infinite values: {self.num_infinite}",
            f"Outliers detected: {self.num_outliers}",
            "",
            f"Valid cells: {self.num_valid_cells} / {self.shape[0] * self.shape[1]}",
        ]
        return "\n".join(lines)


def validate_bathymetry(grid: np.ndarray, outlier_zscore_threshold: float = 4.0) -> ValidationReport:
    """
    Inspect a raw bathymetry grid and report data-quality issues.

    Input:
        grid: 2D NumPy array of depth/elevation values (may contain NaN/Inf).
        outlier_zscore_threshold: |z| above this value is flagged as an outlier
                                   (computed only over finite, non-NaN values).
    Processing:
        Computes shape, dtype, missing/infinite counts, valid-cell count,
        min/max over finite values only, and a conservative z-score-based
        outlier count (report-only, does not modify data).
    Output:
        A ValidationReport dataclass instance. Printing it produces a
        human-readable report (see __str__).
    """
    if grid.ndim != 2:
        raise ValueError(f"Expected a 2D grid, got shape {grid.shape}")

    num_missing = int(np.isnan(grid).sum())
    num_infinite = int(np.isinf(grid).sum())

    finite_mask = np.isfinite(grid)
    num_valid_cells = int(finite_mask.sum())

    if num_valid_cells > 0:
        min_depth = float(np.min(grid[finite_mask]))
        max_depth = float(np.max(grid[finite_mask]))
    else:
        # No valid data at all -- report honestly rather than fabricating.
        min_depth = float("nan")
        max_depth = float("nan")

    # Conservative outlier detection via z-score over finite values only.
    num_outliers = 0
    if num_valid_cells > 1:
        finite_values = grid[finite_mask]
        mean = np.mean(finite_values)
        std = np.std(finite_values)
        if std > 0:
            z_scores = np.abs((finite_values - mean) / std)
            num_outliers = int(np.sum(z_scores > outlier_zscore_threshold))

    return ValidationReport(
        shape=grid.shape,
        dtype=str(grid.dtype),
        min_depth=min_depth,
        max_depth=max_depth,
        num_missing=num_missing,
        num_infinite=num_infinite,
        num_valid_cells=num_valid_cells,
        num_outliers=num_outliers,
    )


# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

def _fill_missing_nearest(grid: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Fill missing cells using nearest-neighbor interpolation."""
    rows, cols = grid.shape
    yy, xx = np.mgrid[0:rows, 0:cols]

    known_points = np.column_stack((yy[~missing_mask], xx[~missing_mask]))
    known_values = grid[~missing_mask]
    missing_points = np.column_stack((yy[missing_mask], xx[missing_mask]))

    if len(known_points) == 0:
        # Nothing to interpolate from -- cannot recover any values.
        return grid

    filled_values = sp_interpolate.griddata(
        known_points, known_values, missing_points, method="nearest"
    )
    result = grid.copy()
    result[missing_mask] = filled_values
    return result


def _fill_missing_linear(grid: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """
    Fill missing cells using linear interpolation, with a nearest-neighbor
    fallback for any points linear interpolation cannot reach (e.g. missing
    cells on the grid border, outside the convex hull of known points).
    """
    rows, cols = grid.shape
    yy, xx = np.mgrid[0:rows, 0:cols]

    known_points = np.column_stack((yy[~missing_mask], xx[~missing_mask]))
    known_values = grid[~missing_mask]
    missing_points = np.column_stack((yy[missing_mask], xx[missing_mask]))

    if len(known_points) == 0:
        return grid

    filled_values = sp_interpolate.griddata(
        known_points, known_values, missing_points, method="linear"
    )

    # linear interpolation returns NaN outside the convex hull -- patch those
    # remaining NaNs with nearest-neighbor as a fallback.
    still_missing = np.isnan(filled_values)
    if np.any(still_missing):
        fallback_values = sp_interpolate.griddata(
            known_points, known_values, missing_points[still_missing], method="nearest"
        )
        filled_values[still_missing] = fallback_values

    result = grid.copy()
    result[missing_mask] = filled_values
    return result


def _fill_missing_median(grid: np.ndarray, missing_mask: np.ndarray) -> np.ndarray:
    """Fill missing cells with the global median of known values."""
    known_values = grid[~missing_mask]
    if len(known_values) == 0:
        return grid
    median_value = float(np.median(known_values))
    result = grid.copy()
    result[missing_mask] = median_value
    return result


_MISSING_VALUE_STRATEGIES = {
    "nearest": _fill_missing_nearest,
    "linear": _fill_missing_linear,
    "median": _fill_missing_median,
}


def clean_bathymetry(
    grid: np.ndarray,
    interpolation_method: str = "linear",
    filter_outliers: bool = False,
    outlier_zscore_threshold: float = 4.0,
) -> np.ndarray:
    """
    Clean a raw bathymetry grid: convert infinities to missing values, then
    fill missing values using the chosen interpolation strategy. Optionally
    filter (convert to missing + re-interpolate) statistical outliers.

    Input:
        grid: raw 2D depth/elevation array (may contain NaN and/or Inf).
        interpolation_method: "nearest" | "linear" | "median".
        filter_outliers: if True, z-score outliers are also treated as
                          missing and re-interpolated. Off by default --
                          real bathymetric data should not be aggressively
                          modified without an explicit opt-in.
        outlier_zscore_threshold: threshold used when filter_outliers=True.
    Processing:
        1. Replace +/-Inf with NaN (infinities are never valid depth values).
        2. Optionally flag statistical outliers as NaN too.
        3. Interpolate all NaN cells using the selected strategy.
    Output:
        Cleaned 2D NumPy array with the same shape as the input, containing
        no NaN and no Inf values (as long as at least one valid cell existed).

    Raises:
        ValueError: if interpolation_method is not recognized.
    """
    if interpolation_method not in _MISSING_VALUE_STRATEGIES:
        raise ValueError(
            f"Unknown interpolation_method '{interpolation_method}'. "
            f"Choose from {list(_MISSING_VALUE_STRATEGIES.keys())}"
        )

    cleaned = grid.copy().astype(np.float64)

    # Step 1: infinities are never physically valid depth values -> treat as missing.
    cleaned[np.isinf(cleaned)] = np.nan

    # Step 2 (optional): flag conservative outliers as missing too.
    if filter_outliers:
        finite_mask = np.isfinite(cleaned)
        if finite_mask.sum() > 1:
            finite_values = cleaned[finite_mask]
            mean = np.mean(finite_values)
            std = np.std(finite_values)
            if std > 0:
                z_scores = np.abs((cleaned - mean) / std)
                outlier_mask = finite_mask & (z_scores > outlier_zscore_threshold)
                cleaned[outlier_mask] = np.nan

    # Step 3: interpolate all missing (NaN) cells.
    missing_mask = np.isnan(cleaned)
    if np.any(missing_mask):
        strategy = _MISSING_VALUE_STRATEGIES[interpolation_method]
        cleaned = strategy(cleaned, missing_mask)

    return cleaned


# --------------------------------------------------------------------------
# Depth convention
# --------------------------------------------------------------------------

def apply_depth_convention(grid: np.ndarray, source_convention: str) -> np.ndarray:
    """
    Convert a depth/elevation grid into the project's single internal
    convention: "positive_down" (larger positive value = deeper water).

    Input:
        grid: cleaned 2D depth/elevation array.
        source_convention: the convention the INPUT data already uses.
            "positive_down" -> already correct, no change needed.
            "negative_down" -> elevation-style data where more negative =
                                deeper. Converted via depth = -elevation.
    Processing:
        Applies a sign flip only if the source convention differs from the
        internal standard. Never silently guesses the convention.
    Output:
        2D NumPy array in "positive_down" convention.

    Raises:
        ValueError: if source_convention is not recognized.
    """
    if source_convention == "positive_down":
        return grid.copy()
    elif source_convention == "negative_down":
        return -grid.copy()
    else:
        raise ValueError(
            f"Unknown depth_convention '{source_convention}'. "
            "Expected 'positive_down' or 'negative_down'."
        )


# --------------------------------------------------------------------------
# Normalization (produces a SEPARATE array; never overwrites the original)
# --------------------------------------------------------------------------

def normalize_depth(grid: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Produce a normalized COPY of a depth grid. The original grid passed in
    is never modified.

    Input:
        grid: cleaned 2D depth array (no NaN/Inf expected).
        method: "minmax" -> scales values to [0, 1].
                "standardize" -> zero mean, unit variance (z-score).
    Processing:
        Computes the relevant statistics (min/max or mean/std) and applies
        the transformation to a copy of the array.
    Output:
        2D NumPy array, same shape as input, normalized according to `method`.

    Raises:
        ValueError: if method is not recognized, or if normalization is
                    degenerate (e.g. constant grid for minmax).
    """
    if method == "minmax":
        grid_min = np.min(grid)
        grid_max = np.max(grid)
        denom = grid_max - grid_min
        if denom == 0:
            # Constant grid -- avoid divide-by-zero; return zeros explicitly.
            return np.zeros_like(grid)
        return (grid - grid_min) / denom

    elif method == "standardize":
        mean = np.mean(grid)
        std = np.std(grid)
        if std == 0:
            return np.zeros_like(grid)
        return (grid - mean) / std

    else:
        raise ValueError(
            f"Unknown normalization method '{method}'. Choose 'minmax' or 'standardize'."
        )


if __name__ == "__main__":
    # Manual smoke test.
    sample = np.array(
        [
            [10.0, 11.0, np.nan],
            [13.0, np.inf, 15.0],
            [16.0, 17.0, 18.0],
        ]
    )
    print("Raw:\n", sample)

    report = validate_bathymetry(sample)
    print("\n", report)

    cleaned = clean_bathymetry(sample, interpolation_method="linear")
    print("\nCleaned:\n", cleaned)

    converted = apply_depth_convention(cleaned, source_convention="positive_down")
    print("\nAfter depth convention (positive_down, no change expected):\n", converted)

    normalized = normalize_depth(cleaned, method="minmax")
    print("\nNormalized (minmax):\n", normalized)

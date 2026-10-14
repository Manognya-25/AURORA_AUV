"""
feature_extraction.py
----------------------
Per-cell terrain descriptors computed from a cleaned bathymetry grid.

Pipeline implemented here:

    depth grid -> compute_slope()       -> slope magnitude grid
                  compute_roughness()   -> local roughness grid
                  compute_curvature()   -> curvature (Laplacian) grid
                       |
                       v
             extract_features() -> stacked feature array (rows*cols, 4)
                                    columns: [depth, slope, roughness, curvature]

All spatial derivatives are scaled by `resolution` (metres/cell) so that
slope is a true dimensionless gradient (rise/run) and curvature has units
of metres^-1, independent of grid spacing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import generic_filter, uniform_filter


# --------------------------------------------------------------------------
# Individual feature computations
# --------------------------------------------------------------------------

def compute_slope(depth: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """
    Compute per-cell slope magnitude from a depth grid.

    Input:
        depth: 2D array of depth values (positive_down convention).
        resolution: metres per grid cell (spacing used for the gradient).
    Processing:
        Central-difference gradient (np.gradient) in both grid directions,
        scaled by `resolution`, combined into a gradient magnitude
        sqrt(dz/dx^2 + dz/dy^2). Edge cells use one-sided differences
        (handled automatically by np.gradient).
    Output:
        2D array, same shape as `depth`, of dimensionless slope magnitude
        (rise/run). Always >= 0.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    grad_y, grad_x = np.gradient(depth, resolution)
    slope = np.sqrt(grad_x ** 2 + grad_y ** 2)
    return slope


def compute_roughness(depth: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Compute per-cell local roughness (terrain irregularity).

    Input:
        depth: 2D array of depth values.
        window: side length of the square neighbourhood (must be odd, >= 3).
    Processing:
        Roughness at cell (i, j) is the standard deviation of depth values
        within a `window` x `window` neighbourhood centred on (i, j).
        Border cells are handled with reflected padding so the output has
        the same shape as the input (no edge shrinkage).
    Output:
        2D array, same shape as `depth`, of local depth standard deviation
        (metres). Always >= 0.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError(f"window must be an odd integer >= 3, got {window}")

    mean_sq = uniform_filter(depth ** 2, size=window, mode="reflect")
    sq_mean = uniform_filter(depth, size=window, mode="reflect") ** 2
    # Guard against tiny negative values from floating-point cancellation.
    variance = np.clip(mean_sq - sq_mean, a_min=0.0, a_max=None)
    roughness = np.sqrt(variance)
    return roughness


def compute_curvature(depth: np.ndarray, resolution: float = 1.0) -> np.ndarray:
    """
    Compute per-cell curvature (second-derivative / Laplacian) of the terrain.

    Input:
        depth: 2D array of depth values.
        resolution: metres per grid cell.
    Processing:
        Discrete Laplacian: d^2z/dx^2 + d^2z/dy^2, computed via two
        successive np.gradient passes and scaled by `resolution`.
        Positive curvature => locally concave (basin/depression in
        positive_down convention); negative => locally convex (ridge/peak).
    Output:
        2D array, same shape as `depth`, of curvature (m^-1). Signed.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    grad_y, grad_x = np.gradient(depth, resolution)
    grad_yy, _ = np.gradient(grad_y, resolution)
    _, grad_xx = np.gradient(grad_x, resolution)
    curvature = grad_xx + grad_yy
    return curvature


# --------------------------------------------------------------------------
# Combined extraction
# --------------------------------------------------------------------------

@dataclass
class FeatureSet:
    """Container for the individual and stacked terrain features."""
    depth: np.ndarray
    slope: np.ndarray
    roughness: np.ndarray
    curvature: np.ndarray
    resolution: float
    roughness_window: int

    @property
    def shape(self) -> tuple:
        return self.depth.shape

    def as_array(self, feature_order: tuple = ("depth", "slope", "roughness", "curvature")) -> np.ndarray:
        """
        Flatten and stack the requested features into an (N, F) array,
        where N = rows * cols (row-major / C order) and F = len(feature_order).
        """
        columns = []
        for name in feature_order:
            grid = getattr(self, name)
            columns.append(grid.reshape(-1))
        return np.stack(columns, axis=1)

    def __str__(self) -> str:
        lines = [
            "Terrain Feature Summary",
            "------------------------",
            f"Grid shape: {self.shape[0]} x {self.shape[1]}  (resolution={self.resolution} m/cell)",
            f"Roughness window: {self.roughness_window}",
            "",
            f"{'feature':<12}{'min':>10}{'max':>10}{'mean':>10}",
        ]
        for name in ("depth", "slope", "roughness", "curvature"):
            grid = getattr(self, name)
            lines.append(f"{name:<12}{grid.min():>10.4f}{grid.max():>10.4f}{grid.mean():>10.4f}")
        return "\n".join(lines)


def extract_features(
    depth: np.ndarray,
    resolution: float = 1.0,
    roughness_window: int = 3,
) -> FeatureSet:
    """
    Compute the full terrain feature set for a cleaned bathymetry grid.

    Input:
        depth: 2D array of depth values (positive_down convention), no
               NaN/Inf (validated/cleaned upstream in Week 2).
        resolution: metres per grid cell.
        roughness_window: neighbourhood size for compute_roughness.
    Processing:
        Runs compute_slope, compute_roughness, compute_curvature over the
        same grid and packages the results (plus the raw depth) together.
    Output:
        FeatureSet with .depth, .slope, .roughness, .curvature grids
        (each same shape as input) and an .as_array() helper for ML use.
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be a 2D array, got shape {depth.shape}")
    if not np.all(np.isfinite(depth)):
        raise ValueError("depth grid contains NaN/Inf values; clean it first (Week 2 preprocess.py)")

    slope = compute_slope(depth, resolution=resolution)
    roughness = compute_roughness(depth, window=roughness_window)
    curvature = compute_curvature(depth, resolution=resolution)

    return FeatureSet(
        depth=depth,
        slope=slope,
        roughness=roughness,
        curvature=curvature,
        resolution=resolution,
        roughness_window=roughness_window,
    )

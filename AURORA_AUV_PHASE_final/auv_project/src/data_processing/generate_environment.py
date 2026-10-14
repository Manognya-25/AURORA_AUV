"""
generate_environment.py
------------------------
Synthetic bathymetry generation, and a unified `create_environment()` entry
point that produces a structured environment object regardless of whether
the data source is synthetic or real.

IMPORTANT (scientific honesty):
    Synthetic terrain generated here is clearly labeled as SYNTHETIC. It does
    NOT represent any real surveyed location and must never be described as
    real ocean terrain or claimed to have real-world positional accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.data_processing.load_bathymetry import load_bathymetry
from src.data_processing.preprocess import (
    validate_bathymetry,
    clean_bathymetry,
    apply_depth_convention,
)


# --------------------------------------------------------------------------
# Synthetic bathymetry generation
# --------------------------------------------------------------------------

def _add_gaussian_bump(grid: np.ndarray, cx: float, cy: float, amplitude: float, sigma: float) -> np.ndarray:
    """
    Add a smooth Gaussian "bump" to a grid. A positive amplitude raises the
    surface (hill); a negative amplitude lowers it (depression).

    Input:
        grid: current depth grid to add the bump onto.
        cx, cy: bump center in grid coordinates (column, row).
        amplitude: peak height/depth change at the center.
        sigma: spread of the Gaussian (larger = wider, gentler bump).
    Processing:
        Computes a 2D Gaussian centered at (cx, cy) and adds it elementwise.
    Output:
        New grid (grid + gaussian_bump), same shape as input.
    """
    rows, cols = grid.shape
    yy, xx = np.mgrid[0:rows, 0:cols]
    bump = amplitude * np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2)))
    return grid + bump


def _add_ridge(
    grid: np.ndarray,
    start: tuple,
    end: tuple,
    amplitude: float,
    width: float,
) -> np.ndarray:
    """
    Add an elongated ridge (or trench, if amplitude is negative) running
    from `start` to `end` in grid coordinates.

    Input:
        grid: current depth grid.
        start, end: (col, row) endpoints of the ridge centerline.
        amplitude: height/depth change along the ridge crest.
        width: how wide/gentle the ridge cross-section is (like a sigma).
    Processing:
        For every cell, computes the perpendicular distance to the line
        segment from `start` to `end`, then applies a Gaussian falloff of
        that distance so the ridge is highest along the centerline and
        fades smoothly away from it.
    Output:
        New grid with the ridge added, same shape as input.
    """
    rows, cols = grid.shape
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)

    x1, y1 = start
    x2, y2 = end
    dx, dy = (x2 - x1), (y2 - y1)
    seg_len_sq = dx ** 2 + dy ** 2
    if seg_len_sq == 0:
        seg_len_sq = 1e-9  # avoid divide-by-zero for degenerate zero-length ridge

    # Project each grid cell onto the line segment, clamped to [0, 1].
    t = ((xx - x1) * dx + (yy - y1) * dy) / seg_len_sq
    t_clamped = np.clip(t, 0.0, 1.0)

    closest_x = x1 + t_clamped * dx
    closest_y = y1 + t_clamped * dy

    dist = np.sqrt((xx - closest_x) ** 2 + (yy - closest_y) ** 2)
    ridge = amplitude * np.exp(-(dist ** 2) / (2 * width ** 2))
    return grid + ridge


def generate_synthetic_bathymetry(
    rows: int = 100,
    cols: int = 100,
    base_depth: float = 50.0,
    gradient_strength: float = 0.4,
    num_hills: int = 3,
    num_depressions: int = 2,
    num_ridges: int = 2,
    noise_std: float = 1.5,
    seed: Optional[int] = 42,
) -> np.ndarray:
    """
    Generate a SYNTHETIC bathymetry (depth) grid with spatial continuity:
    a smooth base gradient, several Gaussian hills (shallower mounds),
    depressions (deeper pits), elongated ridges, and low-amplitude noise.

    Depth convention used internally: "positive_down" -- larger value = deeper.

    Input:
        rows, cols: grid dimensions.
        base_depth: average sea-floor depth in meters before features are added.
        gradient_strength: how much a smooth linear gradient varies depth
                             across the grid (small value = gentle slope).
        num_hills: number of shallow Gaussian mounds (raise the seafloor,
                    i.e. reduce depth locally).
        num_depressions: number of Gaussian pits (increase depth locally).
        num_ridges: number of elongated ridge features (raise the seafloor
                     along a line).
        noise_std: standard deviation of low-frequency random noise added
                    at the end (kept small relative to feature amplitudes
                    so the terrain has spatial continuity, not pixel noise).
        seed: random seed for reproducibility. None -> non-deterministic.
    Processing:
        1. Start from a constant `base_depth` plane.
        2. Add a smooth diagonal gradient.
        3. Add `num_hills` Gaussian hills (negative amplitude -> shallower).
        4. Add `num_depressions` Gaussian depressions (positive amplitude -> deeper).
        5. Add `num_ridges` elongated ridges (negative amplitude -> shallower).
        6. Add smooth-ish low-amplitude Gaussian noise (NOT pure random pixel noise --
           noise is generated once per-cell but kept small relative to feature scale
           so terrain remains visually continuous).
        7. Clip depth to be non-negative (depth cannot be negative in this convention).
    Output:
        2D NumPy array of shape (rows, cols), dtype float64, representing
        SYNTHETIC depth values in meters ("positive_down" convention).
    """
    rng = np.random.default_rng(seed)

    grid = np.full((rows, cols), fill_value=base_depth, dtype=np.float64)

    # --- Smooth gradient (diagonal, gentle) ---
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float64)
    gradient = gradient_strength * ((xx / max(cols - 1, 1)) + (yy / max(rows - 1, 1))) * base_depth * 0.2
    grid += gradient

    # --- Hills (shallower mounds -> negative amplitude reduces depth) ---
    for _ in range(num_hills):
        cx = rng.uniform(0.15, 0.85) * cols
        cy = rng.uniform(0.15, 0.85) * rows
        amplitude = -rng.uniform(0.25, 0.5) * base_depth   # shallower = less depth
        sigma = rng.uniform(0.08, 0.15) * min(rows, cols)
        grid = _add_gaussian_bump(grid, cx, cy, amplitude, sigma)

    # --- Depressions (deeper pits -> positive amplitude increases depth) ---
    for _ in range(num_depressions):
        cx = rng.uniform(0.15, 0.85) * cols
        cy = rng.uniform(0.15, 0.85) * rows
        amplitude = rng.uniform(0.3, 0.6) * base_depth
        sigma = rng.uniform(0.06, 0.12) * min(rows, cols)
        grid = _add_gaussian_bump(grid, cx, cy, amplitude, sigma)

    # --- Ridges (elongated shallower features) ---
    for _ in range(num_ridges):
        start = (rng.uniform(0, cols), rng.uniform(0, rows))
        end = (rng.uniform(0, cols), rng.uniform(0, rows))
        amplitude = -rng.uniform(0.2, 0.4) * base_depth
        width = rng.uniform(0.03, 0.06) * min(rows, cols)
        grid = _add_ridge(grid, start, end, amplitude, width)

    # --- Low-amplitude noise for local roughness (kept small vs. feature scale) ---
    noise = rng.normal(loc=0.0, scale=noise_std, size=(rows, cols))
    grid += noise

    # Depth cannot physically be negative under "positive_down" -- clip.
    grid = np.clip(grid, a_min=0.0, a_max=None)

    return grid


# --------------------------------------------------------------------------
# Unified environment object
# --------------------------------------------------------------------------

@dataclass
class Environment:
    """
    Structured container for a processed bathymetric environment.
    This is the object Week 3 (terrain feature extraction) will consume.
    """
    depth_grid: np.ndarray           # cleaned, convention-normalized depth grid
    source: str                       # "synthetic" | "real"
    resolution: float                 # meters per grid cell
    min_depth: float
    max_depth: float
    metadata: dict = field(default_factory=dict)


def create_environment(
    source: str = "synthetic",
    rows: int = 100,
    cols: int = 100,
    resolution: float = 1.0,
    real_input_path: Optional[str] = None,
    depth_convention: str = "positive_down",
    interpolation_method: str = "linear",
    synthetic_params: Optional[dict] = None,
) -> Environment:
    """
    Unified entry point that produces a fully cleaned Environment object,
    regardless of whether the underlying data is synthetic or real.

    Input:
        source: "synthetic" or "real".
        rows, cols: grid size (used only when source == "synthetic").
        resolution: meters represented by one grid cell (simulation assumption).
        real_input_path: path to a real dataset file (required if source == "real").
        depth_convention: the convention of the INPUT data ("positive_down" |
                            "negative_down"); internally always converted to
                            "positive_down".
        interpolation_method: strategy used to fill missing values during cleaning.
        synthetic_params: optional dict of kwargs forwarded to
                            generate_synthetic_bathymetry() (e.g. base_depth,
                            num_hills, seed, etc.).
    Processing:
        1. Obtain a raw depth grid (generate synthetic OR load real file).
        2. Validate it (report only; does not raise on data-quality issues).
        3. Clean it (missing values, infinities).
        4. Apply the depth convention conversion.
        5. Package everything into an Environment object with metadata.
    Output:
        An Environment dataclass instance.

    Raises:
        ValueError: if source == "real" but real_input_path is not provided,
                    or if source is not recognized.
    """
    synthetic_params = synthetic_params or {}

    if source == "synthetic":
        raw_grid = generate_synthetic_bathymetry(rows=rows, cols=cols, **synthetic_params)
        data_origin_note = "Synthetically generated terrain (not a real surveyed location)."
    elif source == "real":
        if not real_input_path:
            raise ValueError("real_input_path must be provided when source='real'.")
        raw_grid = load_bathymetry(real_input_path)
        data_origin_note = f"Loaded from real dataset file: {real_input_path}"
    else:
        raise ValueError(f"Unknown source '{source}'. Expected 'synthetic' or 'real'.")

    report = validate_bathymetry(raw_grid)

    cleaned_grid = clean_bathymetry(raw_grid, interpolation_method=interpolation_method)
    converted_grid = apply_depth_convention(cleaned_grid, source_convention=depth_convention)

    metadata = {
        "source": source,
        "data_origin_note": data_origin_note,
        "rows": converted_grid.shape[0],
        "columns": converted_grid.shape[1],
        "resolution": resolution,
        "depth_convention_input": depth_convention,
        "depth_convention_internal": "positive_down",
        "interpolation_method": interpolation_method,
        "validation_report": {
            "missing_values_found": report.num_missing,
            "infinite_values_found": report.num_infinite,
            "outliers_flagged": report.num_outliers,
            "valid_cells": report.num_valid_cells,
        },
    }

    return Environment(
        depth_grid=converted_grid,
        source=source,
        resolution=resolution,
        min_depth=float(np.min(converted_grid)),
        max_depth=float(np.max(converted_grid)),
        metadata=metadata,
    )


if __name__ == "__main__":
    env = create_environment(source="synthetic", rows=50, cols=50, synthetic_params={"seed": 1})
    print("Synthetic environment created.")
    print("Shape:", env.depth_grid.shape)
    print("Depth range:", env.min_depth, "-", env.max_depth)
    print("Metadata:", env.metadata)

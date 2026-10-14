"""
load_bathymetry.py
-------------------
Loads a bathymetric (underwater depth) dataset from disk into a clean 2D
NumPy array, OR reports that no real dataset is available so the caller can
fall back to synthetic generation.

Supported real-dataset formats:
    - CSV   (.csv)   -> a 2D numeric grid, rows = one line per row of depths
    - NPY   (.npy)   -> a single saved NumPy array
    - NPZ   (.npz)   -> a saved archive; the depth grid is expected under a
                        key such as "depth" / "elevation" / "z" / "arr_0"
    - GeoTIFF (.tif/.tiff) -> raster elevation/depth data (optional; requires
                        the 'rasterio' package. If rasterio isn't installed,
                        a clear error is raised rather than silently failing.)

IMPORTANT: This module does NOT assume any one fixed real-world dataset.
It only extracts a numerical depth/elevation grid. Terrain *labels*
(sand/rock/obstacle/etc.) are NOT handled here -- that begins in Week 3.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Individual format loaders
# --------------------------------------------------------------------------

def load_csv(path: str) -> np.ndarray:
    """
    Load a bathymetry grid stored as a plain numeric CSV file.

    Input:
        path: path to a .csv file where each row of the file is a row of
              depth/elevation values (no header, no index column expected;
              if a header/index exists, pandas will still often parse it
              correctly, but a clean numeric-only CSV is safest).
    Processing:
        Reads the CSV with pandas (robust to whitespace / minor formatting
        quirks) and converts the result to a float NumPy array.
    Output:
        2D NumPy array of shape (rows, columns), dtype float64.
    """
    df = pd.read_csv(path, header=None)
    grid = df.to_numpy(dtype=np.float64)
    return grid


def load_npy(path: str) -> np.ndarray:
    """
    Load a bathymetry grid saved as a raw NumPy .npy file.

    Input:  path to a .npy file containing a 2D array.
    Processing: np.load, then cast to float64 for consistent downstream math.
    Output: 2D NumPy array of shape (rows, columns).
    """
    grid = np.load(path)
    return np.asarray(grid, dtype=np.float64)


def load_npz(path: str, key: Optional[str] = None) -> np.ndarray:
    """
    Load a bathymetry grid from a NumPy .npz archive.

    Input:
        path: path to a .npz file.
        key:  optional explicit key to read. If not provided, the loader
              tries a list of common key names, then falls back to the
              first array stored in the archive.
    Processing:
        Opens the archive, resolves which array represents the depth grid.
    Output:
        2D NumPy array of shape (rows, columns).
    """
    archive = np.load(path)
    if key is not None:
        if key not in archive:
            raise KeyError(
                f"Key '{key}' not found in {path}. Available keys: {list(archive.keys())}"
            )
        return np.asarray(archive[key], dtype=np.float64)

    # Try common conventional key names before giving up.
    candidate_keys = ["depth", "elevation", "z", "bathymetry", "arr_0"]
    for candidate in candidate_keys:
        if candidate in archive:
            return np.asarray(archive[candidate], dtype=np.float64)

    # Fall back to the first array in the archive.
    first_key = list(archive.keys())[0]
    return np.asarray(archive[first_key], dtype=np.float64)


def load_geotiff(path: str) -> np.ndarray:
    """
    Load a bathymetry/elevation grid from a GeoTIFF raster file.

    Input:  path to a .tif / .tiff file.
    Processing:
        Requires the optional 'rasterio' package (not in the default
        requirements.txt since most B.Tech setups won't have a GeoTIFF
        dataset). If rasterio is missing, raises a clear, actionable error
        instead of silently failing.
    Output: 2D NumPy array of shape (rows, columns) taken from band 1.
    """
    try:
        import rasterio  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Reading GeoTIFF files requires the 'rasterio' package, which is "
            "not installed. Install it with: pip install rasterio"
        ) from exc

    with rasterio.open(path) as dataset:
        grid = dataset.read(1)  # first band
    return np.asarray(grid, dtype=np.float64)


# --------------------------------------------------------------------------
# Unified entry point
# --------------------------------------------------------------------------

_EXTENSION_LOADERS = {
    ".csv": load_csv,
    ".npy": load_npy,
    ".npz": load_npz,
    ".tif": load_geotiff,
    ".tiff": load_geotiff,
}


def load_bathymetry(path: str, npz_key: Optional[str] = None) -> np.ndarray:
    """
    Automatically determine the file format from its extension and load a
    bathymetry grid from disk.

    Input:
        path: path to a real bathymetry dataset file (.csv/.npy/.npz/.tif/.tiff)
        npz_key: optional key to use if the file is a .npz archive.
    Processing:
        1. Verifies the file exists.
        2. Dispatches to the correct format-specific loader based on extension.
        3. Ensures the result is a 2D numeric array.
    Output:
        2D NumPy array (rows, columns) representing raw depth/elevation.

    Raises:
        FileNotFoundError: if the path does not exist.
        ValueError: if the file extension is unsupported, or the loaded
                    data is not a 2D array.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bathymetry file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in _EXTENSION_LOADERS:
        raise ValueError(
            f"Unsupported bathymetry file extension '{ext}'. "
            f"Supported extensions: {list(_EXTENSION_LOADERS.keys())}"
        )

    if ext == ".npz":
        grid = load_npz(path, key=npz_key)
    else:
        grid = _EXTENSION_LOADERS[ext](path)

    grid = np.asarray(grid, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError(
            f"Loaded bathymetry data from {path} has shape {grid.shape}, "
            "but a 2D (rows, columns) grid is required."
        )

    return grid


if __name__ == "__main__":
    # Small manual smoke test: create a temp CSV, load it back, print result.
    import tempfile

    sample = np.array([[10.0, 11.0, 12.0], [13.0, np.nan, 15.0]])
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
        pd.DataFrame(sample).to_csv(f.name, header=False, index=False)
        tmp_path = f.name

    loaded = load_bathymetry(tmp_path)
    print("Loaded grid:\n", loaded)
    os.remove(tmp_path)

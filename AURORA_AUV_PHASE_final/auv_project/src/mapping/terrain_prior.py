"""Terrain-risk PRIOR layer (optional, DISABLED by default).

This is a-priori environmental knowledge (a pre-dive bathymetric chart), kept strictly separate from
the sensor-derived occupancy grid. It is built from bathymetry features run through the EXISTING
Random Forest classifier (no new labels are invented) and mapped to a risk in [0, 1] per terrain class
(mapping.terrain_risk.class_risk). The 'Obstacle' class defaults to risk 0 so the prior cannot
smuggle obstacle locations into the cost map: obstacles must come from the sonar.
"""
from __future__ import annotations

import numpy as np

from src.mapping.coordinate_utils import GridGeometry


def risk_from_classes(class_grid: np.ndarray, class_risk: dict) -> np.ndarray:
    risk = np.zeros(class_grid.shape, np.float32)
    for k, v in class_risk.items():
        risk[class_grid == int(k)] = float(v)
    return risk


def resample_to_geometry(src_grid: np.ndarray, src_origin_xy: tuple, src_resolution: float,
                         geom: GridGeometry, fill: float = 0.0) -> np.ndarray:
    """Nearest-neighbour resample of a source raster (row<->y, col<->x, cell CENTRES at
    origin + (idx + 0.5) * res) onto the map grid. Cells outside the source get `fill`."""
    rows, cols = np.mgrid[0:geom.rows, 0:geom.cols]
    x, y = geom.grid_to_world(rows, cols)
    sc = np.floor((x - src_origin_xy[0]) / src_resolution).astype(int)
    sr = np.floor((y - src_origin_xy[1]) / src_resolution).astype(int)
    ok = (sr >= 0) & (sr < src_grid.shape[0]) & (sc >= 0) & (sc < src_grid.shape[1])
    out = np.full(geom.shape, fill, np.float32)
    out[ok] = src_grid[sr[ok], sc[ok]]
    return out


def classes_from_depth(depth_positive_down: np.ndarray, resolution: float, model_path: str,
                       roughness_window: int = 3) -> np.ndarray:
    """Run the existing terrain pipeline (features -> Random Forest) on a bathymetry grid."""
    from src.terrain.classifier import load_model, predict_grid
    from src.terrain.feature_extraction import extract_features
    feats = extract_features(depth_positive_down, resolution=resolution, roughness_window=roughness_window)
    return predict_grid(load_model(model_path), feats.as_array(), depth_positive_down.shape)

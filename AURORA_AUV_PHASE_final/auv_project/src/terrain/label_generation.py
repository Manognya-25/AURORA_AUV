"""
label_generation.py
--------------------
Rule-based synthetic terrain-class labels, used only because no real
surveyed/labeled seafloor classification exists for this project. Labels
are derived from the same slope/roughness features used at inference time,
via thresholds stored in `configs/config.yaml` (`terrain.synthetic_label_thresholds`).

IMPORTANT — threshold recalibration (documented, not silent):
The thresholds originally checked into config.yaml (slope_rocky=0.15,
slope_steep=0.35, roughness_obstacle=2.5) were written before any feature
statistics existed for the real Week 2 grid. Measured on the actual
processed_bathymetry.npy (resolution=1.0 m/cell):

    slope magnitude:  min 0.0095, max 7.98, mean 1.82
    roughness (w=3):  min 0.34,   max 4.47, mean 1.72

At the original thresholds, 96.7% of cells fall in Steep/Obstacle -- an
unusable, degenerate label distribution for training a classifier. The
threshold *values* were recalibrated against this data's actual scale
(see `configs/config.yaml` comments for the before/after numbers and the
percentile search used to choose them). The threshold *keys*/semantics are
unchanged. Resulting distribution on the real grid:
    Flat/Sand ~49%, Rocky ~23%, Steep/Hazard ~23%, Obstacle ~5%.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.terrain.feature_extraction import FeatureSet

CLASS_NAMES = {
    0: "Flat/Sand",
    1: "Rocky",
    2: "Steep/Terrain Hazard",
    3: "Obstacle",
}


@dataclass
class LabelThresholds:
    """Thresholds used to derive synthetic terrain-class labels."""
    slope_rocky: float
    slope_steep: float
    roughness_obstacle: float

    @classmethod
    def from_config(cls, terrain_config: dict) -> "LabelThresholds":
        t = terrain_config["synthetic_label_thresholds"]
        return cls(
            slope_rocky=float(t["slope_rocky"]),
            slope_steep=float(t["slope_steep"]),
            roughness_obstacle=float(t["roughness_obstacle"]),
        )


def generate_labels(features: FeatureSet, thresholds: LabelThresholds) -> np.ndarray:
    """
    Assign a terrain class to every cell using slope + roughness thresholds.

    Input:
        features: FeatureSet from feature_extraction.extract_features().
        thresholds: LabelThresholds (slope_rocky, slope_steep, roughness_obstacle).
    Processing:
        Decision order (roughness checked first -- a locally very irregular
        cell is an Obstacle regardless of slope):
            roughness > roughness_obstacle          -> 3 (Obstacle)
            elif slope > slope_steep                -> 2 (Steep/Terrain Hazard)
            elif slope > slope_rocky                -> 1 (Rocky)
            else                                    -> 0 (Flat/Sand)
    Output:
        2D int array, same shape as the input grid, with values in {0,1,2,3}.
    """
    slope = features.slope
    roughness = features.roughness

    labels = np.zeros(slope.shape, dtype=np.int64)
    labels[slope > thresholds.slope_rocky] = 1
    labels[slope > thresholds.slope_steep] = 2
    labels[roughness > thresholds.roughness_obstacle] = 3  # overrides slope-based class
    return labels


def class_distribution(labels: np.ndarray) -> dict:
    """
    Summarize a label grid as per-class counts and fractions.

    Input:
        labels: 2D (or flattened) int array of class ids {0,1,2,3}.
    Processing:
        Counts occurrences of each of the four known classes (0 for any
        class absent from this particular grid), and their fraction of
        total cells.
    Output:
        dict keyed by class id -> {"name": str, "count": int, "fraction": float}.
    """
    flat = labels.reshape(-1)
    total = flat.size
    result = {}
    for class_id, name in CLASS_NAMES.items():
        count = int(np.sum(flat == class_id))
        result[class_id] = {"name": name, "count": count, "fraction": count / total if total else 0.0}
    return result


def format_distribution(dist: dict) -> str:
    """Human-readable rendering of the dict returned by class_distribution()."""
    lines = ["Class Distribution", "-------------------"]
    for class_id in sorted(dist):
        info = dist[class_id]
        lines.append(f"{class_id} {info['name']:<22} {info['count']:>6} cells  ({info['fraction']*100:5.1f}%)")
    return "\n".join(lines)

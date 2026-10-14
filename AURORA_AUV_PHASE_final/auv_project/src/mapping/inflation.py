"""Distance-based obstacle inflation + clearance field for the ONLINE cost map.

Complements (does not replace) src/mapping/obstacle_inflation.py, which dilates a Week-4 integer-cell
grid by a whole number of cells. Here the radius is metric: inflation_radius = vehicle_radius +
safety_margin, applied through a Euclidean distance transform, so it is exactly circular at any map
resolution. The vehicle footprint is approximated by a circle (heading-dependent footprint of the
1.5 m x 0.5 m hull is NOT modelled; pick vehicle_radius_m conservatively).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def obstacle_distance_m(occupied: np.ndarray, resolution: float) -> np.ndarray:
    """Distance (m) from each cell centre to the nearest OCCUPIED cell centre; +inf if none exist."""
    if not occupied.any():
        return np.full(occupied.shape, np.inf, dtype=np.float32)
    return (distance_transform_edt(~occupied) * resolution).astype(np.float32)


def inflation_masks(dist_m: np.ndarray, occupied: np.ndarray, inflation_radius_m: float):
    """Returns (inflated_mask, in_radius) where inflated = within the radius and not itself occupied."""
    within = dist_m <= inflation_radius_m
    return within & ~occupied, within


def clearance_cost(dist_m: np.ndarray, inflation_radius_m: float, decay_m: float, weight: float) -> np.ndarray:
    """Deterministic additive cost: `weight` at the inflation boundary, linearly decaying to 0 over
    `decay_m` further from the obstacle, 0 beyond. Cells inside the radius get the full weight."""
    frac = 1.0 - np.clip((dist_m - inflation_radius_m) / decay_m, 0.0, 1.0)
    frac = np.where(np.isfinite(dist_m), frac, 0.0)
    return (weight * frac).astype(np.float32)

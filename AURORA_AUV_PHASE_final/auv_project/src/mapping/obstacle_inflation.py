"""
obstacle_inflation.py
----------------------
Grows obstacle regions in the occupancy grid outward by a safety margin, so
the AUV's finite physical size (and pose/localization uncertainty) is
accounted for by the planner instead of treating the vehicle as a
dimensionless point.

Pipeline implemented here:

    occupancy grid {free, obstacle, unknown}
                       |
                       v
             inflate_obstacles()  -> occupancy grid with a safety buffer
                                      of `radius_cells` added around every
                                      original obstacle cell
                       |
                       v
             inflation_summary()  -> how many cells were newly blocked

Only `obstacle` cells are grown; `unknown` cells are left as-is (growing
into unknown space is a planning-time risk decision, not a mapping-time
one -- e.g. the cost map / planner can decide how to price unknown cells).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure


@dataclass
class InflationResult:
    """Occupancy grid after obstacle inflation, plus bookkeeping for reporting."""
    inflated_grid: np.ndarray
    radius_cells: int
    newly_blocked_cells: int

    def __str__(self) -> str:
        return (
            "Obstacle Inflation\n"
            "-------------------\n"
            f"radius: {self.radius_cells} cell(s)\n"
            f"newly blocked cells: {self.newly_blocked_cells}"
        )


def inflate_obstacles(
    occupancy_grid: np.ndarray,
    radius_cells: int,
    obstacle_value: int,
) -> InflationResult:
    """
    Grow every obstacle cell outward by `radius_cells` to add a safety margin.

    Input:
        occupancy_grid: 2D int array of occupancy codes (free/obstacle/unknown).
        radius_cells: safety margin in grid cells (config:
                       mapping.safety_inflation_radius_cells). Should reflect
                       AUV half-width plus a localization-error margin,
                       converted to cells via config.environment.resolution.
        obstacle_value: the integer code that means "obstacle" (config:
                        mapping.occupancy_values.obstacle).
    Processing:
        Builds a boolean mask of obstacle cells, then applies binary
        dilation with a disk-like structuring element (8-connectivity,
        iterated `radius_cells` times, which approximates a circular
        buffer well for small integer radii). `free`/`unknown` cells that
        fall inside the dilated mask are overwritten with `obstacle_value`;
        cells that were already `obstacle` or already outside the buffer
        are left untouched (so `unknown` cells far from any obstacle stay
        `unknown`, and existing obstacle cells are unaffected).
    Output:
        InflationResult with the inflated grid and a count of cells that
        changed from non-obstacle to obstacle.
    """
    if radius_cells < 0:
        raise ValueError(f"radius_cells must be >= 0, got {radius_cells}")

    obstacle_mask = occupancy_grid == obstacle_value

    if radius_cells == 0:
        return InflationResult(
            inflated_grid=occupancy_grid.copy(),
            radius_cells=0,
            newly_blocked_cells=0,
        )

    structure = generate_binary_structure(rank=2, connectivity=2)  # 8-connected
    dilated_mask = binary_dilation(obstacle_mask, structure=structure, iterations=radius_cells)

    inflated_grid = occupancy_grid.copy()
    newly_blocked_mask = dilated_mask & ~obstacle_mask
    inflated_grid[newly_blocked_mask] = obstacle_value

    return InflationResult(
        inflated_grid=inflated_grid,
        radius_cells=radius_cells,
        newly_blocked_cells=int(np.sum(newly_blocked_mask)),
    )

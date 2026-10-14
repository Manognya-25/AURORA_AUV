"""
cost_map.py
------------
Turns continuous terrain features (depth, slope, roughness) plus the
inflated occupancy grid into a single scalar navigation cost per cell,
suitable for a weighted-graph planner (Week 6: A*).

Pipeline implemented here:

    Week 3 features (depth, slope, roughness)
                       |
                       v
      normalize_feature()  -> each feature rescaled independently to [0, 1]
                       |
                       v
      combine_cost_map()   -> weighted sum (config.mapping.cost_weights)
                               + a large fixed penalty on inflated-obstacle
                                 cells + a configurable penalty on unknown
                                 cells
                       |
                       v
             NavigationCostMap  -> per-feature cost grids + final total

Why normalize per-feature: depth (metres), slope (rise/run) and roughness
(metres, local std-dev) live on unrelated numeric scales. Combining them
with a weighted sum only makes sense once each is rescaled to a common
[0, 1] range -- otherwise `w_depth`/`w_slope`/`w_roughness` wouldn't mean
what they say (e.g. depth in metres could dominate the sum purely because
its raw numbers are bigger, regardless of the configured weight).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class CostWeights:
    """Weights combining normalized per-feature costs into one total cost."""
    w_depth: float
    w_slope: float
    w_roughness: float
    w_obstacle: float
    unknown_cell_cost: float

    @classmethod
    def from_config(cls, mapping_config: dict) -> "CostWeights":
        w = mapping_config["cost_weights"]
        return cls(
            w_depth=float(w["w_depth"]),
            w_slope=float(w["w_slope"]),
            w_roughness=float(w["w_roughness"]),
            w_obstacle=float(w["w_obstacle"]),
            unknown_cell_cost=float(mapping_config["unknown_cell_cost"]),
        )


@dataclass
class NavigationCostMap:
    """Container for the per-feature cost grids and the combined total cost."""
    depth_cost: np.ndarray
    slope_cost: np.ndarray
    roughness_cost: np.ndarray
    total: np.ndarray
    weights: CostWeights

    @property
    def shape(self) -> tuple:
        return self.total.shape

    def __str__(self) -> str:
        lines = [
            "Navigation Cost Map Summary",
            "-----------------------------",
            f"Grid shape: {self.shape[0]} x {self.shape[1]}",
            f"Weights: depth={self.weights.w_depth}, slope={self.weights.w_slope}, "
            f"roughness={self.weights.w_roughness}, obstacle={self.weights.w_obstacle}, "
            f"unknown={self.weights.unknown_cell_cost}",
            "",
            f"{'component':<16}{'min':>14}{'max':>14}{'mean':>14}",
        ]
        panels = [
            ("depth_cost", self.depth_cost),
            ("slope_cost", self.slope_cost),
            ("roughness_cost", self.roughness_cost),
            ("total", self.total),
        ]
        for name, grid in panels:
            # `total` can span from ~0-1 (traversable cells) up to w_obstacle
            # (~1e6, an effectively-infinite blocked-cell marker); a general
            # format (`g`) keeps both ranges readable in one fixed-width column,
            # whereas a fixed-decimal format squashes the large obstacle value.
            lines.append(
                f"{name:<16}{grid.min():>14.4g}{grid.max():>14.4g}{grid.mean():>14.4g}"
            )
        return "\n".join(lines)


def normalize_feature(grid: np.ndarray) -> np.ndarray:
    """
    Rescale a feature grid to [0, 1] via min-max normalization.

    Input:
        grid: 2D array of raw feature values (any unit/sign).
    Processing:
        (grid - min) / (max - min). If the grid is constant (max == min --
        e.g. perfectly flat terrain), returns an all-zero grid rather than
        dividing by zero, since a constant feature carries no discriminating
        cost information.
    Output:
        2D array, same shape as `grid`, with values in [0, 1].
    """
    grid_min = float(grid.min())
    grid_max = float(grid.max())
    if np.isclose(grid_max, grid_min):
        return np.zeros_like(grid, dtype=np.float64)
    return (grid - grid_min) / (grid_max - grid_min)


def build_cost_map(
    features,
    inflated_occupancy: np.ndarray,
    weights: CostWeights,
    obstacle_value: int,
    unknown_value: int | None = None,
) -> NavigationCostMap:
    """
    Combine normalized terrain features + occupancy into one navigation cost grid.

    Input:
        features: a FeatureSet (src.terrain.feature_extraction.FeatureSet)
                  exposing .depth, .slope, .roughness 2D grids, same shape
                  as inflated_occupancy.
        inflated_occupancy: 2D int occupancy grid *after* obstacle_inflation
                            (so the safety margin is already priced in).
        weights: CostWeights (config.mapping.cost_weights + unknown_cell_cost).
        obstacle_value: occupancy code meaning "obstacle" (mapping.occupancy_values.obstacle).
        unknown_value: occupancy code meaning "unknown" (mapping.occupancy_values.unknown).
                       If None, no unknown-cell penalty is applied (all cells
                       are treated as known -- correct for Week 4).
    Processing:
        1. Each of depth/slope/roughness is min-max normalized to [0, 1]
           independently (see normalize_feature).
        2. total = w_depth*depth_cost + w_slope*slope_cost + w_roughness*roughness_cost.
        3. Cells that are `obstacle` in inflated_occupancy get total = w_obstacle
           (an effectively-infinite fixed cost, overriding the weighted sum --
           a blocked cell isn't "a bit expensive", it's impassable).
        4. Cells that are `unknown` (if unknown_value is given) get
           total += unknown_cell_cost, added on top of the weighted terrain
           cost so unknown terrain is discouraged but not treated as a hard
           wall like a confirmed obstacle.
    Output:
        NavigationCostMap with the three normalized per-feature cost grids
        and the combined `total` cost grid (same shape as the inputs).
    """
    if features.shape != inflated_occupancy.shape:
        raise ValueError(
            f"features shape {features.shape} does not match "
            f"inflated_occupancy shape {inflated_occupancy.shape}"
        )

    depth_cost = normalize_feature(features.depth)
    slope_cost = normalize_feature(features.slope)
    roughness_cost = normalize_feature(features.roughness)

    total = (
        weights.w_depth * depth_cost
        + weights.w_slope * slope_cost
        + weights.w_roughness * roughness_cost
    )

    if unknown_value is not None:
        total = total + weights.unknown_cell_cost * (inflated_occupancy == unknown_value)

    obstacle_mask = inflated_occupancy == obstacle_value
    total = np.where(obstacle_mask, weights.w_obstacle, total)

    return NavigationCostMap(
        depth_cost=depth_cost,
        slope_cost=slope_cost,
        roughness_cost=roughness_cost,
        total=total,
        weights=weights,
    )

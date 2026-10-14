"""
occupancy_grid.py
------------------
Converts a Week 3 terrain-classification grid into a binary occupancy grid
that a path planner can query directly: "is this cell drivable or not?"

Pipeline implemented here:

    terrain class grid (0=Flat/Sand, 1=Rocky, 2=Steep/Hazard, 3=Obstacle)
                       |
                       v
             build_occupancy_grid()  -> binary {free, obstacle} grid
                       |
                       v
             occupancy_summary()     -> free/obstacle cell counts

Design note: only the Obstacle class (id 3) is treated as impassable.
Rocky and Steep/Terrain Hazard cells are *traversable but expensive* --
that distinction is exactly what the Week 4 cost map (cost_map.py) is for.
Collapsing "hazardous" and "impassable" into the same occupancy value would
throw away information the planner needs; occupancy answers "can I go
there at all", cost answers "how much do I want to avoid it".

An optional `known_mask` lets the grid mark cells as UNKNOWN rather than
FREE/OBSTACLE. Week 4 classifies the entire surveyed grid, so by default
every cell is treated as known; `known_mask` exists so later weeks (e.g.
a sensor with limited range in Week 5+) can reuse this module without a
redesign.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Terrain class id (from src.terrain.label_generation.CLASS_NAMES) that maps
# to an impassable occupancy cell. Kept as a named constant rather than a
# bare "3" so the coupling to the Week 3 class scheme is explicit and greppable.
OBSTACLE_CLASS_ID = 3


@dataclass
class OccupancyValues:
    """Integer codes used in the occupancy grid (config: mapping.occupancy_values)."""
    free: int
    obstacle: int
    unknown: int

    @classmethod
    def from_config(cls, mapping_config: dict) -> "OccupancyValues":
        v = mapping_config["occupancy_values"]
        return cls(free=int(v["free"]), obstacle=int(v["obstacle"]), unknown=int(v["unknown"]))


def build_occupancy_grid(
    class_grid: np.ndarray,
    occupancy_values: OccupancyValues,
    known_mask: np.ndarray | None = None,
    obstacle_class_id: int = OBSTACLE_CLASS_ID,
) -> np.ndarray:
    """
    Convert a terrain-classification grid into a binary occupancy grid.

    Input:
        class_grid: 2D int array of terrain class ids, shape (rows, cols),
                    as produced by src.terrain.classifier.predict_grid().
        occupancy_values: OccupancyValues (config.mapping.occupancy_values)
                          giving the integer codes to write for free/obstacle/unknown.
        known_mask: optional 2D bool array, same shape as class_grid, True
                    where a cell has actually been observed/classified. If
                    None (default), every cell is treated as known -- correct
                    for Week 4, where the full grid was classified in Week 3.
        obstacle_class_id: terrain class id treated as impassable (default:
                            OBSTACLE_CLASS_ID, i.e. class 3 "Obstacle").
    Processing:
        Starts every cell as `free`, sets cells where class_grid ==
        obstacle_class_id to `obstacle`, then overwrites any cell where
        known_mask is False with `unknown` (unknown status takes precedence
        since we cannot claim a cell is free or occupied without having
        observed it).
    Output:
        2D int array, same shape as class_grid, containing only the codes
        in occupancy_values (free / obstacle / unknown).
    """
    if class_grid.ndim != 2:
        raise ValueError(f"class_grid must be a 2D array, got shape {class_grid.shape}")
    if known_mask is not None and known_mask.shape != class_grid.shape:
        raise ValueError(
            f"known_mask shape {known_mask.shape} does not match class_grid shape {class_grid.shape}"
        )

    occupancy = np.full(class_grid.shape, occupancy_values.free, dtype=np.int64)
    occupancy[class_grid == obstacle_class_id] = occupancy_values.obstacle

    if known_mask is not None:
        occupancy[~known_mask] = occupancy_values.unknown

    return occupancy


def occupancy_summary(occupancy_grid: np.ndarray, occupancy_values: OccupancyValues) -> dict:
    """
    Summarize an occupancy grid as per-category cell counts and fractions.

    Input:
        occupancy_grid: 2D int array of occupancy codes.
        occupancy_values: OccupancyValues used to interpret the codes.
    Processing:
        Counts cells equal to each of free/obstacle/unknown and divides by
        the total cell count.
    Output:
        dict keyed by category name ("free"/"obstacle"/"unknown") ->
        {"count": int, "fraction": float}.
    """
    total = occupancy_grid.size
    categories = {
        "free": occupancy_values.free,
        "obstacle": occupancy_values.obstacle,
        "unknown": occupancy_values.unknown,
    }
    result = {}
    for name, code in categories.items():
        count = int(np.sum(occupancy_grid == code))
        result[name] = {"count": count, "fraction": count / total if total else 0.0}
    return result


def format_occupancy_summary(summary: dict) -> str:
    """Human-readable rendering of the dict returned by occupancy_summary()."""
    lines = ["Occupancy Grid Summary", "-----------------------"]
    for name in ("free", "obstacle", "unknown"):
        info = summary[name]
        lines.append(f"{name:<10} {info['count']:>6} cells  ({info['fraction']*100:5.1f}%)")
    return "\n".join(lines)

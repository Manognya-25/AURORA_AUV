"""
visualization.py
-----------------
Shared Matplotlib plotting helpers for the bathymetry / terrain pipeline.
All functions save a PNG to disk and also return the output path.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend: safe for headless/server use
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)


def _ensure_dir(path: str) -> None:
    """Create the parent directory of `path` if it does not already exist."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def plot_depth_map_2d(
    depth_grid: np.ndarray,
    save_path: str,
    title: str = "Bathymetry Depth Map (Synthetic)",
    resolution: float = 1.0,
) -> str:
    """
    Plot a 2D top-down depth map with a colorbar.

    Input:
        depth_grid: 2D array of depth values ("positive_down" convention;
                     deeper = larger value).
        save_path: file path (including filename) to save the PNG to.
        title: plot title.
        resolution: meters per cell, used to label axes in meters.
    Processing:
        Uses imshow with a perceptually-ordered colormap ("viridis_r" so
        deeper areas appear darker/cooler, consistent with intuitive
        underwater-depth reading), adds axis labels in meters and a colorbar
        labeled with depth units.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)
    rows, cols = depth_grid.shape
    extent = [0, cols * resolution, rows * resolution, 0]  # y-axis inverted (row 0 at top)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(depth_grid, cmap="viridis_r", extent=extent, aspect="equal")
    ax.set_title(title)
    ax.set_xlabel(f"X position (meters, {resolution} m/cell)")
    ax.set_ylabel(f"Y position (meters, {resolution} m/cell)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Depth (meters, positive = deeper)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_depth_map_3d(
    depth_grid: np.ndarray,
    save_path: str,
    title: str = "Bathymetry 3D Terrain (Synthetic)",
    resolution: float = 1.0,
) -> str:
    """
    Plot a 3D surface view of the terrain.

    Input:
        depth_grid: 2D array of depth values.
        save_path: file path to save the PNG to.
        title: plot title.
        resolution: meters per cell.
    Processing:
        Builds an (X, Y) meshgrid scaled by resolution, plots a 3D surface
        with the Z-axis inverted so deeper areas appear lower (visually
        intuitive for underwater terrain), using a depth-ordered colormap.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)
    rows, cols = depth_grid.shape
    x = np.arange(cols) * resolution
    y = np.arange(rows) * resolution
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(xx, yy, depth_grid, cmap="viridis_r", linewidth=0, antialiased=True)

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Depth (m)")
    ax.invert_zaxis()  # deeper values point "down" visually
    fig.colorbar(surf, ax=ax, shrink=0.6, label="Depth (m)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_depth_histogram(
    depth_grid: np.ndarray,
    save_path: str,
    title: str = "Depth Distribution (Synthetic Bathymetry)",
    bins: int = 40,
) -> str:
    """
    Plot a histogram of depth values across the grid.

    Input:
        depth_grid: 2D array of depth values.
        save_path: file path to save the PNG to.
        title: plot title.
        bins: number of histogram bins.
    Processing:
        Flattens the grid and plots a standard Matplotlib histogram with
        mean/median reference lines for quick interpretation.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)
    flat = depth_grid.flatten()

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(flat, bins=bins, color="steelblue", edgecolor="black", alpha=0.8)
    ax.axvline(np.mean(flat), color="red", linestyle="--", label=f"Mean = {np.mean(flat):.2f} m")
    ax.axvline(np.median(flat), color="orange", linestyle="--", label=f"Median = {np.median(flat):.2f} m")
    ax.set_title(title)
    ax.set_xlabel("Depth (meters)")
    ax.set_ylabel("Cell count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_preprocessing_comparison(
    raw_grid: np.ndarray,
    corrupted_grid: np.ndarray,
    cleaned_grid: np.ndarray,
    save_path: str,
    title: str = "Preprocessing Comparison: Raw -> Corrupted -> Cleaned",
) -> str:
    """
    Plot a 3-panel side-by-side comparison: raw, corrupted (with injected
    NaN/outliers), and cleaned depth grids. Useful for demonstrating that
    the cleaning pipeline actually works.

    Input:
        raw_grid: original depth grid before corruption.
        corrupted_grid: same grid with missing/outlier values injected.
        cleaned_grid: result after running clean_bathymetry() on the corrupted grid.
        save_path: file path to save the PNG to.
        title: overall figure title.
    Processing:
        Plots all three grids with the same color scale for direct visual
        comparison, using imshow with a shared vmin/vmax range.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)

    all_finite_values = np.concatenate(
        [raw_grid[np.isfinite(raw_grid)], cleaned_grid[np.isfinite(cleaned_grid)]]
    )
    vmin, vmax = float(np.min(all_finite_values)), float(np.max(all_finite_values))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("Raw", raw_grid),
        ("Corrupted (NaN / outliers injected)", corrupted_grid),
        ("Cleaned", cleaned_grid),
    ]
    for ax, (panel_title, panel_grid) in zip(axes, panels):
        im = ax.imshow(panel_grid, cmap="viridis_r", vmin=vmin, vmax=vmax)
        ax.set_title(panel_title)
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Week 3: terrain feature / classification plotting helpers
# --------------------------------------------------------------------------

def plot_feature_grids(
    features,
    save_path: str,
    title: str = "Terrain Feature Grids",
) -> str:
    """
    Plot the four terrain feature grids (depth, slope, roughness, curvature)
    side by side.

    Input:
        features: a FeatureSet (src.terrain.feature_extraction.FeatureSet),
                   or any object exposing .depth/.slope/.roughness/.curvature
                   2D arrays.
        save_path: file path to save the PNG to.
        title: overall figure title.
    Processing:
        One imshow panel per feature, each with its own colorbar/colormap
        (diverging "coolwarm" for signed curvature, sequential "viridis"
        otherwise) since the four features have unrelated units/ranges.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)
    panels = [
        ("Depth (m)", features.depth, "viridis_r"),
        ("Slope (rise/run)", features.slope, "viridis"),
        ("Roughness (m, local std)", features.roughness, "viridis"),
        ("Curvature (1/m)", features.curvature, "coolwarm"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    for ax, (panel_title, grid, cmap) in zip(axes, panels):
        if cmap == "coolwarm":
            vmax = float(np.max(np.abs(grid)))
            im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax)
        else:
            im = ax.imshow(grid, cmap=cmap)
        ax.set_title(panel_title)
        ax.set_xlabel("X (cells)")
        ax.set_ylabel("Y (cells)")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_terrain_classification(
    class_grid: np.ndarray,
    save_path: str,
    class_names: Optional[dict] = None,
    title: str = "Terrain Classification Map",
) -> str:
    """
    Plot a discrete terrain-class grid as a categorical map with a legend.

    Input:
        class_grid: 2D int array of class ids (e.g. {0,1,2,3}).
        save_path: file path to save the PNG to.
        class_names: optional {id: name} dict for the legend
                     (defaults to src.terrain.label_generation.CLASS_NAMES).
        title: plot title.
    Processing:
        Uses a fixed discrete colormap (one flat color per class, ordered
        by increasing hazard: Flat/Sand -> Rocky -> Steep/Hazard -> Obstacle)
        via ListedColormap + BoundaryNorm, with a labeled legend rather than
        a continuous colorbar (classes are categorical, not ordinal in value).
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    if class_names is None:
        from src.terrain.label_generation import CLASS_NAMES as class_names  # noqa: N811

    _ensure_dir(save_path)
    colors = ["#2ca25f", "#fee08b", "#fc8d59", "#b2182b"]  # flat->rocky->steep->obstacle
    n_classes = len(class_names)
    cmap = ListedColormap(colors[:n_classes])
    bounds = np.arange(n_classes + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(class_grid, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")

    legend_elements = [
        Patch(facecolor=colors[i], label=f"{i}: {class_names[i]}") for i in sorted(class_names)
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_confusion_matrix(
    confusion: np.ndarray,
    save_path: str,
    class_names: Optional[dict] = None,
    title: str = "Terrain Classifier Confusion Matrix",
) -> str:
    """
    Plot a confusion matrix heatmap with per-cell counts annotated.

    Input:
        confusion: (n_classes, n_classes) array, rows=true, cols=predicted
                   (as returned by src.terrain.evaluate.evaluate_classifier).
        save_path: file path to save the PNG to.
        class_names: optional {id: name} dict for axis tick labels
                     (defaults to src.terrain.label_generation.CLASS_NAMES).
        title: plot title.
    Processing:
        imshow heatmap of raw counts with each cell's count annotated as
        text, tick labels replaced with human-readable class names.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    if class_names is None:
        from src.terrain.label_generation import CLASS_NAMES as class_names  # noqa: N811

    _ensure_dir(save_path)
    labels = [class_names[i] for i in sorted(class_names)]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    vmax = confusion.max() if confusion.size else 1
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            value = confusion[i, j]
            color = "white" if value > vmax * 0.5 else "black"
            ax.text(j, i, str(value), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Week 4: occupancy grid / cost map plotting helpers
# --------------------------------------------------------------------------

def plot_occupancy_grid(
    occupancy_grid: np.ndarray,
    save_path: str,
    occupancy_values,
    title: str = "Occupancy Grid",
) -> str:
    """
    Plot a categorical occupancy grid (free / obstacle / unknown).

    Input:
        occupancy_grid: 2D int array of occupancy codes.
        save_path: file path to save the PNG to.
        occupancy_values: src.mapping.occupancy_grid.OccupancyValues giving
                           the integer codes for free/obstacle/unknown.
        title: plot title.
    Processing:
        Fixed 3-color discrete colormap (green=free, red=obstacle,
        grey=unknown) via ListedColormap + BoundaryNorm, with a legend
        rather than a continuous colorbar since the values are categorical.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    _ensure_dir(save_path)
    code_to_color = {
        occupancy_values.free: "#2ca25f",
        occupancy_values.obstacle: "#b2182b",
        occupancy_values.unknown: "#969696",
    }
    code_to_label = {
        occupancy_values.free: "Free",
        occupancy_values.obstacle: "Obstacle",
        occupancy_values.unknown: "Unknown",
    }
    codes_sorted = sorted(code_to_color)
    colors = [code_to_color[c] for c in codes_sorted]
    cmap = ListedColormap(colors)
    # Map raw codes -> contiguous 0..N-1 bins so BoundaryNorm works regardless
    # of the actual integer values used in config.mapping.occupancy_values.
    remapped = np.zeros_like(occupancy_grid, dtype=np.int64)
    for i, code in enumerate(codes_sorted):
        remapped[occupancy_grid == code] = i
    bounds = np.arange(len(codes_sorted) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(remapped, cmap=cmap, norm=norm)
    ax.set_title(title)
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")

    legend_elements = [
        Patch(facecolor=code_to_color[c], label=code_to_label[c]) for c in codes_sorted
    ]
    ax.legend(handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_cost_component(
    cost_grid: np.ndarray,
    save_path: str,
    title: str,
    cmap: str = "inferno",
) -> str:
    """
    Plot a single normalized cost component (e.g. depth_cost, slope_cost,
    roughness_cost) as a heatmap.

    Input:
        cost_grid: 2D array of normalized cost values, expected in [0, 1].
        save_path: file path to save the PNG to.
        title: plot title.
        cmap: Matplotlib colormap name (sequential, since cost is unsigned).
    Processing:
        imshow with a fixed 0-1 color scale so depth/slope/roughness cost
        panels are directly visually comparable to each other.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cost_grid, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized cost (0=cheap, 1=expensive)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


def plot_navigation_cost_map(
    total_cost: np.ndarray,
    save_path: str,
    title: str = "Navigation Cost Map",
    obstacle_cost_floor: Optional[float] = None,
) -> str:
    """
    Plot the final combined navigation cost map used by the path planner.

    Input:
        total_cost: 2D array of combined per-cell cost (output of
                    src.mapping.cost_map.build_cost_map(...).total).
        save_path: file path to save the PNG to.
        title: plot title.
        obstacle_cost_floor: if given, cells with cost >= this value (i.e.
                              hard obstacles, w_obstacle in config) are
                              clipped to the 99th percentile of the
                              *non-obstacle* costs before plotting. Without
                              this, the ~1e6 obstacle penalty washes out all
                              color contrast among the merely-expensive
                              (rocky/steep) cells.
        title: plot title.
    Processing:
        Optionally clips the display range (not the returned/saved data --
        this function only writes a PNG) so obstacle cells still read as
        clearly "off the scale" (dark border in the colorbar) while the
        traversable cost gradient remains visible.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)

    display_grid = total_cost
    vmax = float(total_cost.max())
    if obstacle_cost_floor is not None:
        non_obstacle = total_cost[total_cost < obstacle_cost_floor]
        if non_obstacle.size:
            vmax = float(np.percentile(non_obstacle, 99))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display_grid, cmap="magma", vmin=0.0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("X (cells)")
    ax.set_ylabel("Y (cells)")
    cbar = fig.colorbar(im, ax=ax, extend="max")
    cbar.set_label("Navigation cost (higher = more expensive/blocked)")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path


# --------------------------------------------------------------------------
# Week 5: AUV trajectory plotting helper
# --------------------------------------------------------------------------

def plot_auv_trajectory(
    log_df,
    save_path: str,
    start: tuple,
    goal: tuple,
    occupancy_grid: Optional[np.ndarray] = None,
    resolution: float = 1.0,
    obstacle_value: int = 1,
    title: str = "AUV Trajectory (Week 5 waypoint run)",
) -> str:
    """
    Plot a top-down (X-Y) view of the AUV's logged trajectory, with the
    start/goal points and (optionally) the Week 4 occupancy grid as a
    backdrop, for a quick visual sanity check of the waypoint run.

    Input:
        log_df: pandas DataFrame with at least "x" and "y" columns, as
                written to results/simulation/waypoint_run.csv.
        save_path: file path to save the PNG to.
        start: (x, y, z) start position.
        goal: (x, y, z) goal/waypoint position.
        occupancy_grid: optional 2D int array (Week 4 output) drawn as a
                        light backdrop so the trajectory can be checked
                        against obstacle locations.
        resolution: metres/cell, used to scale the occupancy_grid extent
                    to world coordinates (must match environment.py's
                    grid_to_world convention: column->X, row->Y).
        obstacle_value: occupancy code meaning "obstacle".
        title: plot title.
    Processing:
        If occupancy_grid is given, draws it as a background image
        (obstacle cells shaded) using the same X=column, Y=row convention
        as src.simulation.environment.grid_to_world, then overlays the
        logged (x, y) trajectory as a line, with start/goal marked.
    Output:
        Saves a PNG to `save_path` and returns that path.
    """
    _ensure_dir(save_path)

    fig, ax = plt.subplots(figsize=(7, 7))

    if occupancy_grid is not None:
        rows, cols = occupancy_grid.shape
        extent = [0, cols * resolution, 0, rows * resolution]
        backdrop = (occupancy_grid == obstacle_value).astype(float)
        ax.imshow(backdrop, cmap="Greys", extent=extent, origin="lower", alpha=0.5, vmin=0, vmax=1)

    ax.plot(log_df["x"], log_df["y"], color="tab:blue", linewidth=1.5, label="Trajectory")
    ax.scatter([start[0]], [start[1]], color="tab:green", s=80, zorder=5, label="Start")
    ax.scatter([goal[0]], [goal[1]], color="tab:red", marker="*", s=150, zorder=5, label="Goal")

    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=3)
    ax.set_aspect("equal")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path

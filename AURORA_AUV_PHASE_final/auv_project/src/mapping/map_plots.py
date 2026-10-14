"""Static PNG renderings of the online occupancy grid and cost map (saved at the end of a run)."""
from __future__ import annotations

import os

import numpy as np


def save_map_figures(mapping, out_dir: str, est_xy=None, truth_xy=None) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    os.makedirs(out_dir, exist_ok=True)
    g = mapping.grid.geom
    ext = g.extent
    state = mapping.grid.state()
    code = np.zeros(state.shape, np.uint8)
    code[state == 0] = 1
    code[state == 100] = 2
    paths = {}
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(code, origin="lower", extent=(ext[0], ext[1], ext[2], ext[3]), interpolation="nearest",
              cmap=ListedColormap(["#3a3a3a", "#d9f0ff", "#c0392b"]), vmin=0, vmax=2)
    if est_xy is not None and len(est_xy):
        ax.plot(est_xy[:, 0], est_xy[:, 1], "c-", lw=1.5, label="EKF estimated path")
    if truth_xy is not None and len(truth_xy):
        ax.plot(truth_xy[:, 0], truth_xy[:, 1], "y--", lw=1, label="ground truth path (DEBUG ONLY)")
    ax.set_title("Online occupancy grid (sonar + EKF pose): unknown=grey free=light occupied=red")
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)"); ax.legend(loc="upper right", fontsize=7)
    paths["occupancy"] = os.path.join(out_dir, "online_occupancy_map.png")
    fig.savefig(paths["occupancy"], dpi=120, bbox_inches="tight"); plt.close(fig)

    cm = mapping.cost_map
    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(np.minimum(cm.cost, mapping.cost_cfg.inflated_cost), origin="lower",
                   extent=(ext[0], ext[1], ext[2], ext[3]), interpolation="nearest", cmap="magma")
    fig.colorbar(im, ax=ax, label="cost (clamped at inflated_cost)")
    ax.set_title("Online navigation cost map (derived from the occupancy grid)")
    ax.set_xlabel("world X (m)"); ax.set_ylabel("world Y (m)")
    paths["costmap"] = os.path.join(out_dir, "online_cost_map.png")
    fig.savefig(paths["costmap"], dpi=120, bbox_inches="tight"); plt.close(fig)
    return paths

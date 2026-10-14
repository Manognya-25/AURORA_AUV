"""PathData: an arc-length parameterised 3-D path (x, y from the planner, z = depth profile)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.planning.grid_utils import cumulative_length, point_at_arc, project_onto_polyline


@dataclass
class PathData:
    xy: np.ndarray                      # (N, 2)
    z: np.ndarray                       # (N,)
    kind: str = "global"                # global | detour
    version: int = 0
    created_time: float = 0.0
    map_version: int = -1
    cum: np.ndarray = field(init=False)

    def __post_init__(self):
        self.xy = np.asarray(self.xy, float).reshape(-1, 2)
        self.z = np.asarray(self.z, float).reshape(-1)
        if len(self.xy) != len(self.z):
            raise ValueError("xy and z must have the same length")
        self.cum = cumulative_length(self.xy)

    @classmethod
    def from_xy(cls, xy, z_start: float, z_goal: float, **kw) -> "PathData":
        """Depth profile: linear in arc length from z_start to z_goal (constant if they are equal)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        cum = cumulative_length(xy)
        total = float(cum[-1]) if len(cum) else 0.0
        t = np.zeros(len(xy)) if total < 1e-9 else cum / total
        return cls(xy, z_start + t * (z_goal - z_start), **kw)

    @property
    def length(self) -> float:
        return float(self.cum[-1]) if len(self.cum) else 0.0

    @property
    def n(self) -> int:
        return len(self.xy)

    def point(self, s: float) -> np.ndarray:
        """(x, y, z) at arc length s."""
        p = point_at_arc(self.xy, self.cum, s)
        z = float(np.interp(np.clip(s, 0.0, self.length), self.cum, self.z)) if self.n > 1 else float(self.z[0])
        return np.array([p[0], p[1], z])

    def project(self, p_xy, s_min: float = 0.0, s_max: float | None = None):
        return project_onto_polyline(self.xy, self.cum, p_xy, s_min, s_max)

    def next_vertex_index(self, s: float) -> int:
        """Index of the first vertex strictly ahead of arc length s (the 'current waypoint')."""
        i = int(np.searchsorted(self.cum, s + 1e-6, side="left"))
        return min(i, self.n - 1)

    def remaining_xy(self, s: float, start_xy=None) -> np.ndarray:
        """Polyline from the projection point at arc s (or start_xy) through the remaining vertices."""
        s = float(np.clip(s, 0.0, self.length))
        head = point_at_arc(self.xy, self.cum, s) if start_xy is None else np.asarray(start_xy, float)
        tail = self.xy[self.cum > s + 1e-9]
        return np.vstack([head[None, :], tail]) if len(tail) else head[None, :]

    def segment_xy(self, s_a: float, s_b: float) -> np.ndarray:
        """Polyline of the path between arc lengths s_a <= s_b (end points interpolated, vertices in between kept)."""
        s_a, s_b = float(np.clip(s_a, 0.0, self.length)), float(np.clip(s_b, 0.0, self.length))
        mid = self.xy[(self.cum > s_a + 1e-9) & (self.cum < s_b - 1e-9)]
        a, b = point_at_arc(self.xy, self.cum, s_a), point_at_arc(self.xy, self.cum, s_b)
        return np.vstack([a[None, :], mid, b[None, :]])

    def as_points(self, decimals: int = 3) -> list:
        return np.round(self.xy, decimals).tolist()

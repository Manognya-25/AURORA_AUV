"""Grid geometry + exact ray traversal for the online occupancy grid.

FRAMES (all right-handed, identical to Phase 2/3):
  world / map : +X east-ish, +Y left of +X (CCW about +Z), +Z UP, depth = -z. The map frame IS the
                world frame: the 2-D grid is a horizontal slice indexed by world (x, y).
  body        : X forward, Y left, Z up; quaternion (x, y, z, w) body -> world.
  sonar       : same axes as body, rotated by the mounting rotation and offset by the mount position.
  yaw         : rotation about +Z, 0 = body X along world +X, positive CCW (toward +Y).

GRID CONVENTION:
  array[row, col];  col <-> world X,  row <-> world Y  (row grows with +Y: the array is NOT flipped,
  so plot with origin='lower').  `origin` is the world coordinate of the lower-left CORNER of cell
  (0, 0).  Cell (row, col) covers x in [ox + col*res, ox + (col+1)*res), y likewise; its centre is at
  (+0.5 cell).  world->grid uses floor(), so a point exactly on a cell boundary belongs to the cell
  with the larger index.  (This differs from the legacy Week-5 environment helper, which put cell
  centres at integer*res; the two are unrelated.)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridGeometry:
    resolution: float
    rows: int
    cols: int
    origin_x: float
    origin_y: float

    @classmethod
    def from_extent(cls, resolution: float, width_m: float, height_m: float,
                    origin_x: float, origin_y: float) -> "GridGeometry":
        return cls(float(resolution), int(round(height_m / resolution)), int(round(width_m / resolution)),
                   float(origin_x), float(origin_y))

    @property
    def shape(self) -> tuple:
        return (self.rows, self.cols)

    @property
    def extent(self) -> tuple:
        """(xmin, xmax, ymin, ymax) in metres."""
        return (self.origin_x, self.origin_x + self.cols * self.resolution,
                self.origin_y, self.origin_y + self.rows * self.resolution)

    def world_to_grid(self, x, y):
        """(row, col) ints for world (x, y). May be out of bounds; test with in_bounds(). Vectorised."""
        col = np.floor((np.asarray(x, float) - self.origin_x) / self.resolution).astype(np.int64)
        row = np.floor((np.asarray(y, float) - self.origin_y) / self.resolution).astype(np.int64)
        return row, col

    def grid_to_world(self, row, col):
        """World (x, y) of the CENTRE of cell (row, col). Vectorised."""
        x = self.origin_x + (np.asarray(col, float) + 0.5) * self.resolution
        y = self.origin_y + (np.asarray(row, float) + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        row, col = np.asarray(row), np.asarray(col)
        return (row >= 0) & (row < self.rows) & (col >= 0) & (col < self.cols)

    def flat_index(self, row, col):
        return np.asarray(row) * self.cols + np.asarray(col)


def trace_segment(geom: GridGeometry, x0: float, y0: float, x1: float, y1: float):
    """Exact grid traversal (supercover DDA) of the segment (x0,y0)->(x1,y1).

    Returns (rows, cols) int arrays of EVERY in-bounds cell the segment passes through, ordered from
    start to end; the first cell contains the start point and the last cell contains the end point
    (when in bounds). Cells are found by cutting the segment at every x- and y-grid-line crossing and
    classifying each piece by its midpoint, so no cell is skipped or duplicated.
    """
    res = geom.resolution
    dx, dy = x1 - x0, y1 - y0
    ts = [0.0, 1.0]
    if dx != 0.0:
        lo, hi = sorted((x0, x1))
        k0 = int(np.floor((lo - geom.origin_x) / res)) + 1
        k1 = int(np.ceil((hi - geom.origin_x) / res)) - 1
        if k1 >= k0:
            ts.append((geom.origin_x + np.arange(k0, k1 + 1) * res - x0) / dx)
    if dy != 0.0:
        lo, hi = sorted((y0, y1))
        k0 = int(np.floor((lo - geom.origin_y) / res)) + 1
        k1 = int(np.ceil((hi - geom.origin_y) / res)) - 1
        if k1 >= k0:
            ts.append((geom.origin_y + np.arange(k0, k1 + 1) * res - y0) / dy)
    t = np.unique(np.clip(np.concatenate([np.atleast_1d(a) for a in ts]), 0.0, 1.0))
    if len(t) < 2:                                   # zero-length segment: just the one cell
        rows, cols = geom.world_to_grid(np.array([x0]), np.array([y0]))
    else:
        tm = 0.5 * (t[:-1] + t[1:])
        rows, cols = geom.world_to_grid(x0 + tm * dx, y0 + tm * dy)
    keep = geom.in_bounds(rows, cols)
    return rows[keep], cols[keep]

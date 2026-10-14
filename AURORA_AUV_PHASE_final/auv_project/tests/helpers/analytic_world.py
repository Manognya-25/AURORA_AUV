"""TEST-ONLY analytic geometry + ray caster.

Stands in for PyBullet's rayTestBatch in tests that must run where PyBullet is unavailable. It is NOT
used by any product code path and tests built on it are reported separately from the real-PyBullet
tests (tests/test_pybullet_world.py). Geometry: axis-aligned boxes, vertical cylinders, spheres, and a
seabed given as z = seabed_fn(x, y).
"""
from __future__ import annotations

import numpy as np


class AnalyticWorld:
    def __init__(self, boxes=(), cylinders=(), spheres=(), seabed_fn=None, seabed_z=-15.0):
        # boxes: (centre, half_extents) or (centre, half_extents, yaw_rad)
        self.boxes = [(np.asarray(b[0], float), np.asarray(b[1], float), float(b[2]) if len(b) > 2 else 0.0)
                      for b in boxes]
        self.cylinders = [(np.asarray(c, float), float(r), float(hh)) for c, r, hh in cylinders]  # centre, radius, half-height
        self.spheres = [(np.asarray(c, float), float(r)) for c, r in spheres]
        self.seabed_fn = seabed_fn
        self.seabed_z = seabed_z

    # -- per-primitive nearest-hit distances along unit direction d from point o ------------------
    @staticmethod
    def _box(o, d, c, h):
        lo, hi = c - h, c + h
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        tmin = np.nanmax(np.where(np.isnan(t1), -np.inf, np.minimum(t1, t2)))
        tmax = np.nanmin(np.where(np.isnan(t1), np.inf, np.maximum(t1, t2)))
        for k in range(3):                                   # parallel rays outside the slab never hit
            if d[k] == 0 and not (lo[k] <= o[k] <= hi[k]):
                return np.inf
        return tmin if (tmax >= max(tmin, 0.0) and tmin >= 0.0) else np.inf

    @staticmethod
    def _cyl(o, d, c, r, hh):
        best = np.inf
        a = d[0] ** 2 + d[1] ** 2
        if a > 1e-12:
            b = 2 * ((o[0] - c[0]) * d[0] + (o[1] - c[1]) * d[1])
            cc = (o[0] - c[0]) ** 2 + (o[1] - c[1]) ** 2 - r * r
            disc = b * b - 4 * a * cc
            if disc >= 0:
                t = (-b - np.sqrt(disc)) / (2 * a)
                if t >= 0 and abs(o[2] + t * d[2] - c[2]) <= hh:
                    best = t
        return best

    @staticmethod
    def _sphere(o, d, c, r):
        oc = o - c
        b = 2 * np.dot(oc, d)
        disc = b * b - 4 * (np.dot(oc, oc) - r * r)
        if disc < 0:
            return np.inf
        t = (-b - np.sqrt(disc)) / 2
        return t if t >= 0 else np.inf

    def _seabed(self, o, d, tmax):
        if d[2] >= 0:
            return np.inf
        if self.seabed_fn is None:
            t = (self.seabed_z - o[2]) / d[2]
            return t if 0 <= t <= tmax else np.inf
        ts = np.arange(0.0, tmax, 0.1)
        pts = o[None, :] + ts[:, None] * d[None, :]
        below = pts[:, 2] < np.array([self.seabed_fn(x, y) for x, y in pts[:, :2]])
        if not below.any():
            return np.inf
        i = int(np.argmax(below))
        return float(ts[i])

    def ray_caster(self, starts: np.ndarray, ends: np.ndarray) -> np.ndarray:
        out = np.full(len(starts), np.inf)
        for i, (s, e) in enumerate(zip(starts, ends)):
            v = e - s
            L = float(np.linalg.norm(v))
            d = v / L
            best = np.inf
            for c, h, yaw in self.boxes:
                if yaw != 0.0:                       # rotate the ray into the box frame
                    cy, sy = np.cos(-yaw), np.sin(-yaw)
                    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
                    o_l, d_l = c + Rz @ (s - c), Rz @ d
                    best = min(best, self._box(o_l, d_l, c, h))
                else:
                    best = min(best, self._box(s, d, c, h))
            for c, r, hh in self.cylinders:
                best = min(best, self._cyl(s, d, c, r, hh))
            for c, r in self.spheres:
                best = min(best, self._sphere(s, d, c, r))
            best = min(best, self._seabed(s, d, L))
            out[i] = best if best <= L else np.inf
        return out


def world_from_spec(spec) -> "AnalyticWorld":
    """Analytic twin of a generated WorldSpec (boulders approximated by their mean-radius sphere)."""
    boxes, cyls, spheres = [], [], []
    for o in spec.obstacles:
        zc, hh = o.z_centre, o.height / 2.0
        if o.kind in ("pillar", "reef_tower"):
            cyls.append(((o.x, o.y, zc), o.radius, hh))
        elif o.kind in ("wall", "wreck_hull", "wreck_tower"):
            boxes.append(((o.x, o.y, zc), (o.half_extents[0], o.half_extents[1], hh), o.yaw))
        elif o.kind == "boulder":
            spheres.append(((o.x, o.y, 0.5 * (o.z_bottom + o.z_top)), 0.5 * o.height))
    return AnalyticWorld(boxes=boxes, cylinders=cyls, spheres=spheres, seabed_fn=None if spec.is_flat else spec.seabed_height,
                         seabed_z=float(spec.heights.flat[0]))

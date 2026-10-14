"""Sonar -> occupancy-grid integration (inverse sensor model + exact ray traversal).

INPUTS (and nothing else): a SonarMeasurement (post noise/dropout model), the ESTIMATED vehicle pose
(position + rotation from the EKF), and the map configuration. This module does not import PyBullet,
the simulation, or any truth object; there is no code path through which ground truth can enter.

FRAME CHAIN per ray i (see coordinate_utils for conventions):
    d_sensor = [cos(el)cos(az), cos(el)sin(az), sin(el)]          (az +left, el +up, from the measurement)
    d_world  = R_est @ R_mount @ d_sensor                          R_mount: sensor->body calibration
    origin   = p_est + R_est @ mount_position                      sensor origin in the world/map frame
    point(t) = origin + t * d_world                                t = range along the ray (m)

PER-RAY SEMANTICS (documented contract, tested):
  invalid ray (dropout / blanking / failed)  -> NO update at all (no free, no occupied)
  scan-level invalid or sensor FAILED        -> whole scan ignored, counted, map unchanged
  valid ray, detected hit inside the slice   -> cells before the hit FREE, hit cell OCCUPIED
  valid ray, detected hit OUTSIDE the slice  -> FREE only while the ray is inside the slice; no obstacle
                                                (e.g. the seabed seen by a downward-pointing ray)
  valid ray, nothing detected (no-hit /
  max-range / missed detection)              -> cells along the ray up to min(sensor max range,
                                                max_update_range) FREE with weight no_return_free_weight;
                                                nothing beyond; no obstacle invented
  hit beyond max_update_range                -> treated like a no-return up to max_update_range
Cells beyond a hit are never touched (the sonar says nothing about what is behind an obstacle).

2-D SLICE: the grid is a horizontal map, the sonar is 3-D. A ray only informs cells while its altitude
is within +/- slice_half_height_m of the ESTIMATED vehicle z, so obstacles are mapped at the depth
the vehicle operates in and sloping seabed hits do not become phantom walls.

Per scan each cell receives at most one free and one occupied update (several elevation rays through
the same cell are one observation); a cell hit by any ray is occupied for that scan (occupied wins).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.mapping.coordinate_utils import trace_segment
from src.mapping.log_odds_grid import LogOddsGrid
from src.mapping.map_types import MapConfig
from src.sensors.health import SensorHealth
from src.utils.transforms import euler_to_quat, quat_to_rot


_DIR_DECIMALS = 12       # unit-vector component resolution (rad-equivalent ~1e-12)
_POS_DECIMALS = 9        # metres (1 nm)


@dataclass
class ScanStats:
    processed: bool = False
    reason: str = ""                 # why a scan was not processed ("" if processed)
    rays_total: int = 0
    rays_valid: int = 0
    rays_invalid: int = 0
    rays_hit: int = 0                # valid rays with a detection
    rays_no_return: int = 0          # valid rays without a detection
    hits_mapped: int = 0             # hits that produced an OCCUPIED update (inside slice/range/bounds)
    cells_updated: int = 0
    min_valid_range: float | None = None
    max_valid_range: float | None = None


class SonarMapper:
    def __init__(self, config: MapConfig):
        self.cfg = config
        self._fallback_R = quat_to_rot(euler_to_quat(*np.deg2rad(config.sonar_mount_rpy_deg)))
        self._fallback_p = np.asarray(config.sonar_mount_position_m, float)

    def integrate(self, grid: LogOddsGrid, meas, position, rotation) -> ScanStats:
        """Integrate one SonarMeasurement using the ESTIMATED pose (position (3,), rotation (3,3))."""
        c, geom = self.cfg, grid.geom
        n = len(meas.azimuths)
        st = ScanStats(rays_total=n)
        position = np.asarray(position, float)
        rotation = np.asarray(rotation, float)
        if not (np.all(np.isfinite(position)) and np.all(np.isfinite(rotation))):
            st.reason = "pose_not_finite"
            return st
        if (not meas.valid) or meas.health == SensorHealth.FAILED:
            st.reason = "sonar_invalid_or_failed"
            st.rays_invalid = n
            return st

        status = meas.ray_status()
        st.rays_invalid = int(np.sum(status < 0))
        st.rays_hit = int(np.sum(status == 1))
        st.rays_no_return = int(np.sum(status == 0))
        st.rays_valid = st.rays_hit + st.rays_no_return
        ranges = np.asarray(meas.value, float)
        good = (status >= 0) & np.isfinite(ranges)
        if st.rays_valid:
            vr = ranges[status == 1]
            if len(vr):
                st.min_valid_range, st.max_valid_range = float(vr.min()), float(vr.max())

        R_m = self._fallback_R if meas.mount_rotation is None else np.asarray(meas.mount_rotation, float)
        p_m = self._fallback_p if meas.mount_position is None else np.asarray(meas.mount_position, float)
        az, el = np.asarray(meas.azimuths, float), np.asarray(meas.elevations, float)
        d_s = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        # Frame chain: sensor -> (R_mount, p_mount) -> body -> (EKF R_est, p_est) -> world/map.
        # Rotation matrices built from quaternions carry ~1e-16 residue (cos(90deg) is not exactly 0, and its
        # sign differs between platform math libraries). A beam lying on a grid line (x=0, y=1.0, ...) would
        # then fall into the left or right neighbour cell depending on the OS. Snapping to 1e-12 (directions)
        # and 1e-9 m (origin) removes that ulp-level ambiguity without moving anything physically.
        d_w = np.round((rotation @ R_m @ d_s.T).T, _DIR_DECIMALS) + 0.0      # +0.0 turns -0.0 into 0.0
        origin = np.round(position + rotation @ p_m, _POS_DECIMALS) + 0.0
        z0 = float(np.round(position[2], _POS_DECIMALS))
        h = c.slice_half_height_m
        L_max = min(float(meas.max_range), c.max_update_range_m)

        free_idx, free_w, occ_idx = [], [], []
        for i in np.flatnonzero(good):
            dz = d_w[i, 2]
            # parameter interval [t_a, t_b] of the ray inside the vertical slice
            if abs(dz) < 1e-9:
                if abs(origin[2] - z0) > h:
                    continue
                t_a, t_b = 0.0, np.inf
            else:
                t1, t2 = (z0 - h - origin[2]) / dz, (z0 + h - origin[2]) / dz
                t_a, t_b = max(0.0, min(t1, t2)), max(t1, t2)
                if t_b <= 0.0:
                    continue
            hit = status[i] == 1 and ranges[i] <= L_max
            t_end = float(ranges[i]) if hit else L_max
            mapped_hit = hit and (t_a <= t_end <= t_b)
            t_free_end = min(t_end, t_b)
            if t_a >= t_free_end and not mapped_hit:
                continue
            p0 = origin + t_a * d_w[i]
            p1 = origin + min(max(t_free_end, t_a), t_end) * d_w[i]
            rows, cols = trace_segment(geom, p0[0], p0[1], p1[0], p1[1])
            if len(rows) == 0:
                continue
            flat = geom.flat_index(rows, cols)
            if mapped_hit:
                end_row, end_col = geom.world_to_grid(p1[0], p1[1])
                if geom.in_bounds(end_row, end_col):
                    end_flat = int(geom.flat_index(end_row, end_col))
                    occ_idx.append(end_flat)
                    st.hits_mapped += 1
                    flat = flat[flat != end_flat]            # the hit cell is not free evidence
            if len(flat):
                free_idx.append(flat)
                free_w.append(np.full(len(flat), 1.0 if status[i] == 1 else c.no_return_free_weight))

        occ = np.unique(np.asarray(occ_idx, dtype=np.int64)) if occ_idx else np.zeros(0, np.int64)
        if free_idx:
            fi, fw = np.concatenate(free_idx), np.concatenate(free_w)
            order = np.argsort(fi, kind="stable")
            fi, fw = fi[order], fw[order]
            uniq, start = np.unique(fi, return_index=True)
            fw = np.maximum.reduceat(fw, start)               # one free update per cell, strongest evidence
            keep = ~np.isin(uniq, occ)                        # occupied wins within a scan
            fi, fw = uniq[keep], fw[keep]
        else:
            fi, fw = np.zeros(0, np.int64), np.zeros(0)
        st.cells_updated = grid.apply(fi, fw, occ)
        st.processed = True
        return st

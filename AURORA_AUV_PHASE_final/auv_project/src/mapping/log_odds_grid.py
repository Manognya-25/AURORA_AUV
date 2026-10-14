"""Online probabilistic occupancy grid (log-odds, Bayesian binary filter per cell).

Each cell stores l = log(p / (1 - p)), p = P(cell occupied). l = 0 (p = 0.5) is the prior for every
cell, so a fresh map is entirely UNKNOWN. Independent evidence adds in log space:
    l <- clip(l + l_meas, l_min, l_max)
The clamp keeps a cell revisable (it can never saturate to p = 0 or 1) and bounds numerical range.
A cell's STATE is derived from p, not stored:  p >= occupied_threshold -> OCCUPIED,
p <= free_threshold -> FREE, otherwise UNKNOWN. Consequently one free observation does not make a
cell FREE; "unknown until sufficient evidence" is a property of the update parameters.
"""
from __future__ import annotations

import numpy as np

from src.mapping.coordinate_utils import GridGeometry
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, MapConfig


class LogOddsGrid:
    def __init__(self, config: MapConfig):
        self.cfg = config
        self.geom = GridGeometry.from_extent(config.resolution_m, config.width_m, config.height_m,
                                             config.origin_x_m, config.origin_y_m)
        self._l_occ_thr = float(np.log(config.occupied_threshold / (1 - config.occupied_threshold)))
        self._l_free_thr = float(np.log(config.free_threshold / (1 - config.free_threshold)))
        self.reset()

    def reset(self) -> None:
        self.log_odds = np.zeros(self.geom.shape, dtype=np.float32)
        self.observed = np.zeros(self.geom.shape, dtype=bool)      # ever touched by a measurement
        self.update_count = 0

    # ---------------------------------------------------------------- updates
    def apply(self, free_idx: np.ndarray, free_w: np.ndarray, occ_idx: np.ndarray) -> int:
        """Apply one scan's worth of evidence. Indices are FLAT cell indices, each cell at most once
        per list. free cells get log_odds_free * weight, occupied cells get log_odds_occupied.
        Returns the number of cells touched."""
        c = self.cfg
        flat = self.log_odds.reshape(-1)
        seen = self.observed.reshape(-1)
        if len(free_idx):
            flat[free_idx] = np.clip(flat[free_idx] + c.log_odds_free * free_w, c.log_odds_min, c.log_odds_max)
            seen[free_idx] = True
        if len(occ_idx):
            flat[occ_idx] = np.clip(flat[occ_idx] + c.log_odds_occupied, c.log_odds_min, c.log_odds_max)
            seen[occ_idx] = True
        self.update_count += 1
        return int(len(free_idx) + len(occ_idx))

    # ---------------------------------------------------------------- views
    def probability(self) -> np.ndarray:
        """P(occupied) per cell, float32 in (0, 1)."""
        return (1.0 / (1.0 + np.exp(-self.log_odds.astype(np.float64)))).astype(np.float32)

    def state(self) -> np.ndarray:
        """int8 grid of UNKNOWN / FREE / OCCUPIED derived from the thresholds."""
        s = np.full(self.geom.shape, UNKNOWN, dtype=np.int8)
        s[self.log_odds <= self._l_free_thr] = FREE
        s[self.log_odds >= self._l_occ_thr] = OCCUPIED
        return s

    def counts(self, state: np.ndarray | None = None) -> dict:
        s = self.state() if state is None else state
        n = int(s.size)
        free, occ = int(np.sum(s == FREE)), int(np.sum(s == OCCUPIED))
        known = free + occ
        return {"total_cells": n, "free_cells": free, "occupied_cells": occ, "known_cells": known,
                "unknown_cells": n - known, "observed_cells": int(self.observed.sum()),
                "coverage_percent": 100.0 * known / n, "occupied_percent": 100.0 * occ / n}

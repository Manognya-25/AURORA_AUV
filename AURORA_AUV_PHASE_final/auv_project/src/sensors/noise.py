"""Noise primitives. Every sensor owns one np.random.Generator seeded from config
and ALWAYS draws the same number of variates per sample, so streams stay
reproducible regardless of dropouts."""
from __future__ import annotations

import numpy as np


def make_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


class RandomWalkBias:
    """b(t+dt) = b(t) + N(0, walk_std^2 * dt); walk_std in [unit / sqrt(s)]."""

    def __init__(self, initial, walk_std: float, rng: np.random.Generator):
        self.value = np.atleast_1d(np.asarray(initial, float)).copy()
        self.walk_std = float(walk_std)
        self._rng = rng

    def step(self, dt: float) -> np.ndarray:
        draw = self._rng.standard_normal(self.value.shape)   # always drawn
        if dt > 0:
            self.value = self.value + self.walk_std * np.sqrt(dt) * draw
        return self.value

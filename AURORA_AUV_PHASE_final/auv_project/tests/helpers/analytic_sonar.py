"""TEST-ONLY: the real Sonar sensor model with the PyBullet ray cast replaced by an analytic one.

Everything else (ray geometry, noise, dropout, detection probability, min-range blanking, health,
extrinsics) is the product Sonar code. Used where PyBullet is unavailable; reported separately from
the real-PyBullet tests.
"""
import numpy as np

from src.sensors.sonar import Sonar


class AnalyticSonar(Sonar):
    def __init__(self, world, config=None):
        super().__init__(body_id=0, physics_client=0, config=config)
        self.world = world

    def _ray_ranges(self, start, dirs_w):
        n = self.n_rays
        starts = np.repeat(start[None, :], n, axis=0)
        r = self.world.ray_caster(starts, starts + dirs_w * self.cfg.max_range_m)
        return np.where(r < self.cfg.max_range_m, r, np.inf)

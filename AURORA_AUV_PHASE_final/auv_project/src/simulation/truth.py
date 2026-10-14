"""PyBullet -> TruthSnapshot adapter (the one place simulated ground truth is read for the sensors)."""
from __future__ import annotations

import numpy as np
import pybullet as p

from src.sensors.models import TruthSnapshot


def snapshot_from_pybullet(body_id: int, time: float, physics_client: int = 0) -> TruthSnapshot:
    pos, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
    lin, ang = p.getBaseVelocity(body_id, physicsClientId=physics_client)
    return TruthSnapshot(time=time, position=np.array(pos), quat=np.array(quat),
                         velocity_world=np.array(lin), angular_velocity_world=np.array(ang))

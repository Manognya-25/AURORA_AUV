"""
sensors.py
-----------
Minimal simulated proximity sensors for the AUV: front/left/right distance
readings using PyBullet ray casting. Not realistic sonar physics -- the
purpose is only to demonstrate that the simulated AUV can obtain basic
environmental information about nearby obstacles (STEP 8), which later
weeks (e.g. reactive/local-avoidance behaviors) may build on.

Pipeline implemented here:

    AUV position + orientation
                 |
                 v
      ray_distance()       -> single-direction distance-to-obstacle reading
                 |
                 v
      read_proximity()     -> {"front": ..., "left": ..., "right": ...}
"""

from __future__ import annotations

import numpy as np
import pybullet as p


def ray_distance(
    body_id: int,
    direction_body_frame: tuple,
    max_range: float,
    physics_client: int = 0,
) -> float:
    """
    Cast a single ray from the AUV outward in a body-frame direction and
    return the distance to the nearest hit (or max_range if nothing is hit).

    Input:
        body_id: the AUV's PyBullet body id.
        direction_body_frame: (dx, dy, dz) unit-ish vector in the AUV's own
                              body frame (e.g. (1,0,0) = straight ahead,
                              (0,1,0) = to the left, given the URDF's body
                              X = forward convention -- see auv.urdf).
        max_range: maximum sensing distance in metres.
        physics_client: PyBullet client id.
    Processing:
        Rotates `direction_body_frame` into the world frame using the
        AUV's current orientation, casts a ray from the AUV's position to
        `position + direction * max_range` via p.rayTest(), and converts
        the returned hit fraction into a distance.
    Output:
        Distance to the nearest obstacle along that ray, in metres. Equal
        to max_range if nothing was hit within range.
    """
    position, quat = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
    rot_matrix = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
    world_direction = rot_matrix @ np.array(direction_body_frame, dtype=float)
    norm = np.linalg.norm(world_direction)
    if norm > 0:
        world_direction = world_direction / norm

    ray_start = position
    ray_end = tuple(np.array(position) + world_direction * max_range)

    result = p.rayTest(ray_start, ray_end, physicsClientId=physics_client)[0]
    hit_object_id, _, hit_fraction, _, _ = result[0], result[1], result[2], result[3], result[4]

    if hit_object_id == -1 or hit_object_id == body_id:
        return float(max_range)
    return float(hit_fraction * max_range)


def read_proximity(body_id: int, max_range: float, physics_client: int = 0) -> dict:
    """
    Read simplified front/left/right proximity distances for the AUV.

    Input:
        body_id: the AUV's PyBullet body id.
        max_range: maximum sensing distance in metres (config:
                   simulation.sensors.max_range).
        physics_client: PyBullet client id.
    Processing:
        Casts three rays in the AUV's body frame: front (+X), left (+Y),
        right (-Y), using ray_distance() for each. These are NOT modeled
        as sonar cones/beams -- each is a single ray, which is a
        deliberate simplification for a B.Tech-level demonstration rather
        than a physically accurate sensor model.
    Output:
        dict {"front": float, "left": float, "right": float}, each a
        distance in metres (== max_range if no obstacle is within range).
    """
    return {
        "front": ray_distance(body_id, (1.0, 0.0, 0.0), max_range, physics_client),
        "left": ray_distance(body_id, (0.0, 1.0, 0.0), max_range, physics_client),
        "right": ray_distance(body_id, (0.0, -1.0, 0.0), max_range, physics_client),
    }

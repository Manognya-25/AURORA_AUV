"""
metrics.py
-----------
Computes Week 5 run metrics from the recorded trajectory log. Every value
here comes directly from something measured during the run -- nothing is
estimated or assumed. The one exception is clearly labeled: since Week 5
has no physically modeled thrust/energy budget, `movement_effort_proxy` is
a simple |control| integral, explicitly named and documented as a proxy,
not a real energy figure (see docstring below).

Pipeline implemented here:

    trajectory records (from run_simulation.py's step loop)
                       |
                       v
      compute_metrics()  -> dict of only-measured summary statistics
                       |
                       v
      save_metrics()     -> results/logs/week5_simulation_metrics.json
"""

from __future__ import annotations

import json
import os

import numpy as np


def compute_metrics(
    records: list,
    start_position: tuple,
    goal_position: tuple,
    reached_goal: bool,
    waypoints_reached: int,
    total_waypoints: int,
    collisions: int,
    simulation_wall_time: float,
    dt: float,
) -> dict:
    """
    Compute summary metrics for one Week 5 waypoint-following run.

    Input:
        records: list of per-step dicts (as built in run_simulation.py's
                 loop), each with at least "x", "y", "z".
        start_position, goal_position: (x, y, z) tuples.
        reached_goal: whether the final goal was reached (STEP 10).
        waypoints_reached: count of intermediate waypoints reached en route.
        total_waypoints: total number of waypoints in the planned sequence
                         (excluding the start, including the final goal).
        collisions: number of steps at which a collision was detected.
        simulation_wall_time: wall-clock seconds the run took.
        dt: simulation timestep (seconds/step), used to get simulated time.
    Processing:
        - total_steps = len(records).
        - simulation_duration_seconds = total_steps * dt (SIMULATED time,
          not wall-clock -- both are reported, clearly labeled).
        - trajectory_length_m = sum of consecutive-step Euclidean distances
          (a real measured path length, not a straight-line estimate).
        - final_position / final_distance_to_goal read directly from the
          last record.
        - movement_effort_proxy: sum over steps of dt * (|forward| +
          |lateral| + |vertical| + |yaw|), i.e. the same quantity
          AUV.apply_control() integrates as "energy" -- reported here
          under a name that does NOT imply real physical energy, since no
          real thrust/energy model exists (per Week 5 spec: "do not claim
          real energy consumption"). Only included if the records contain
          an "energy" field.
    Output:
        dict of the metrics above -- every field is either directly
        measured or explicitly labeled as a proxy/estimate.
    """
    total_steps = len(records)
    simulation_duration_seconds = total_steps * dt

    positions = np.array([[r["x"], r["y"], r["z"]] for r in records], dtype=float)
    if len(positions) > 1:
        deltas = np.diff(positions, axis=0)
        trajectory_length_m = float(np.sum(np.linalg.norm(deltas, axis=1)))
    else:
        trajectory_length_m = 0.0

    final_position = tuple(positions[-1]) if len(positions) else tuple(start_position)
    final_distance_to_goal = float(np.linalg.norm(np.array(final_position) - np.array(goal_position)))

    metrics = {
        "total_simulation_steps": total_steps,
        "simulation_duration_seconds_simulated": simulation_duration_seconds,
        "simulation_wall_time_seconds": simulation_wall_time,
        "trajectory_length_m": trajectory_length_m,
        "start_position": list(start_position),
        "goal_position": list(goal_position),
        "final_position": list(final_position),
        "final_distance_to_goal_m": final_distance_to_goal,
        "waypoints_reached": waypoints_reached,
        "total_waypoints": total_waypoints,
        "goal_reached": bool(reached_goal),
        "collision_count": collisions,
    }

    if records and "energy" in records[-1]:
        metrics["movement_effort_proxy"] = {
            "value": records[-1]["energy"],
            "note": (
                "NOT a physical energy measurement -- Week 5 has no calibrated "
                "thrust/energy model. This is dt * thrust_coefficient * "
                "sum(|control components|), integrated over the run, reported "
                "only as a relative movement-effort proxy for comparing runs."
            ),
        }

    return metrics


def save_metrics(metrics: dict, path: str) -> str:
    """
    Input: metrics: dict from compute_metrics(). path: destination .json path.
    Processing: creates the parent directory if needed, writes indented JSON.
    Output: the path written to.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path

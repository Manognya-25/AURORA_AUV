"""Estimation pipeline: TRUE state -> sensors -> measurements -> EKF -> ESTIMATED state.

Keeps the three layers separate. `update()` is the only method that touches truth, and only to
feed the sensors. The EKF is initialised from the mission start pose (a known initial condition,
as for any dead-reckoning navigation system), never from per-step ground truth.
"""
from __future__ import annotations

import numpy as np

from src.localization.ekf import EKF, EKFConfig
from src.sensors.suite import SensorSuite


class EstimationPipeline:
    def __init__(self, cfg: dict, body_id: int, start_position, physics_client: int = 0,
                 start_quat=(0.0, 0.0, 0.0, 1.0)):
        self.suite = SensorSuite(cfg, body_id=body_id, physics_client=physics_client)
        self.ekf_enabled = bool((cfg.get("ekf", {}) or {}).get("enabled", True)) and self.suite.enabled
        self.body_id, self.client = body_id, physics_client
        self.ekf = EKF(EKFConfig.from_config(cfg), position=start_position, quat=start_quat) \
            if self.ekf_enabled else None
        self.last_measurements: dict = {}   # sensors sampled on the most recent step only
        self.latest: dict = {}              # most recent Measurement per sensor (sonar is 5 Hz, DVL/depth 10 Hz)
        self.truth = None

    def update(self, time: float):
        """Sample sensors at `time` and run one EKF cycle. Returns (measurements, EstimatedState|None)."""
        from src.simulation.truth import snapshot_from_pybullet   # lazy: needs PyBullet
        return self.update_from_truth(snapshot_from_pybullet(self.body_id, time, self.client))

    def update_from_truth(self, truth):
        """Same cycle as update() but with the TruthSnapshot supplied by the caller (the only way
        truth reaches the sensors). Lets the pipeline run without PyBullet (tests, replays)."""
        time = truth.time
        self.truth = truth
        meas = self.suite.sample(self.truth)
        self.last_measurements = meas
        self.latest.update(meas)
        if self.ekf is None:
            return meas, None
        imu = meas.get("imu")
        if imu is not None:
            self.ekf.predict(imu)
        else:
            self.ekf.coast(time)
        self.ekf.update_dvl(meas.get("dvl"))
        self.ekf.update_depth(meas.get("depth"))
        self.ekf.update_compass(meas.get("compass"))        # absolute heading (only when sensors.compass is configured)
        return meas, self.ekf.state

    def log_fields(self, est, meas: dict) -> dict:
        """Flat dict for CSV/telemetry: estimate, error vs truth (EVALUATION ONLY), sensor flags."""
        out = {}
        if est is not None and self.truth is not None:
            e = est.euler
            err = est.position - self.truth.position
            out.update({
                "est_x": est.position[0], "est_y": est.position[1], "est_z": est.position[2],
                "est_roll": e[0], "est_pitch": e[1], "est_yaw": e[2],
                "est_vx": est.velocity[0], "est_vy": est.velocity[1], "est_vz": est.velocity[2],
                "est_err_x": err[0], "est_err_y": err[1], "est_err_z": err[2],
                "est_pos_err": float(np.linalg.norm(err)),
                "est_pos_std": float(np.linalg.norm(est.position_std)),
            })
        for k in ("dvl", "depth", "compass", "sonar"):
            m = meas.get(k)
            if m is not None:
                out[f"{k}_valid"] = bool(m.valid)
        s = meas.get("sonar")
        if s is not None:
            # None (not NaN): telemetry is serialised to JSON and browsers reject the NaN token
            out["sonar_min_range"] = float(np.min(s.value)) if s.valid else None
            out["sonar_detections"] = int(np.sum(s.detected))
        return out

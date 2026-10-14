"""Error-state (multiplicative) EKF for AUV navigation: IMU propagation, DVL + depth updates.

Nominal state: position p (world), velocity v (world), unit quaternion q (x,y,z,w; body->world),
accelerometer bias b_a, gyro bias b_g.  Error state (15): [dp, dv, dtheta, dba, dbg] with
R_true = R(q) (I + [dtheta]x)  (dtheta expressed in the BODY frame).

Propagation (IMU specific force f, rate w; g_w = [0,0,-g]):
    a_b = f - b_a,  w_b = w - b_g,  a_w = R a_b + g_w
    p += v dt + 1/2 a_w dt^2;  v += a_w dt;  q <- q (x) exp(w_b dt)
Error dynamics:  d(dp)=dv; d(dv)=-R[a_b]x dtheta - R dba - R n_a; d(dtheta)=-[w_b]x dtheta - dbg - n_g;
                 d(dba)=n_ba; d(dbg)=n_bg.   F = I + A dt + (A dt)^2/2.
Process noise per step: dv (sigma_a dt)^2, dp (sigma_a dt^2/2)^2, dtheta (sigma_g dt)^2, biases sigma^2 dt.
Updates (Joseph form, optional chi-square gate, covariance symmetrised):
    DVL   z = R^T v (body frame, 3):  H_v = R^T,  H_theta = [R^T v]x
    Depth z = -p_z (1):               H_p = [0, 0, -1]
    Compass (optional) z = yaw(R):    H_theta = e_z^T R  (third row of R: yaw change for a small body-frame rotation)
Not estimated: sensor scale factors, depth-sensor bias, compass bias, current. Horizontal position has no
absolute reference (no GPS/USBL): it is dead-reckoned from DVL velocity and drifts with distance travelled. Yaw
is bounded by the compass when one is configured; without it yaw is dead-reckoned too and drifts (see
docs/localization.md, "Why the physical goal error was ~1.4 m").
Without IMU data the filter coasts (constant velocity, held attitude) with inflated covariance.
No ground truth, no PyBullet: the only inputs are Measurement objects and an initial pose.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.localization.state import BA, BG, P, TH, V, EstimatedState
from src.sensors.health import SensorHealth
from src.utils.transforms import quat_integrate, quat_multiply, quat_normalize, quat_to_rot, skew

# chi-square 99.9 % points
_CHI2_999 = {1: 10.83, 3: 16.27}


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass
class EKFConfig:
    enabled: bool = True
    gravity: float = 9.8
    # process noise (match the sensor per-sample stds unless deliberately mistuned)
    accel_noise_std: float = 0.05
    gyro_noise_std: float = 0.005
    accel_bias_walk_std: float = 1e-4
    gyro_bias_walk_std: float = 1e-5
    # measurement noise
    dvl_noise_std: float = 0.02
    depth_noise_std: float = 0.02
    compass_noise_std: float = 0.02        # rad (matches sensors.compass.noise_std)
    degraded_r_scale: float = 16.0         # R multiplier when the sensor reports DEGRADED
    # initial covariance (1-sigma)
    init_position_std: float = 0.5
    init_velocity_std: float = 0.1
    init_attitude_std: float = 0.05
    init_accel_bias_std: float = 0.05
    init_gyro_bias_std: float = 0.005
    # behaviour
    gate_enabled: bool = True
    gate_max_consecutive: int = 5          # consecutive gate rejections from one sensor => filter is overconfident
    gate_recovery_inflation_std: float = 0.5   # added 1-sigma (m/s for DVL velocity, m for depth) on recovery
    max_dt: float = 0.5                    # larger gaps are coasted, not integrated blindly
    imu_loss_accel_inflation: float = 1.0  # extra accel sigma (m/s^2) injected per second when IMU is missing
    max_time_skew: float = 0.05            # reject measurements older than this relative to filter time

    @classmethod
    def from_config(cls, cfg: dict) -> "EKFConfig":
        """cfg: FULL project config; reads `ekf` plus simulation.gravity."""
        e = dict(cfg.get("ekf", {}) or {})
        e.setdefault("gravity", abs(float(cfg.get("simulation", {}).get("gravity", -9.8))))
        return cls(**e)


class EKF:
    def __init__(self, config: EKFConfig | None = None, position=(0.0, 0.0, 0.0),
                 quat=(0.0, 0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0), time: float = 0.0):
        self.cfg = config or EKFConfig()
        c = self.cfg
        self.p = np.asarray(position, float).copy()
        self.v = np.asarray(velocity, float).copy()
        self.q = quat_normalize(np.asarray(quat, float))
        self.ba = np.zeros(3)
        self.bg = np.zeros(3)
        self.w_b = np.zeros(3)
        self.time = float(time)
        self.P = np.diag(np.r_[[c.init_position_std] * 3, [c.init_velocity_std] * 3, [c.init_attitude_std] * 3,
                               [c.init_accel_bias_std] * 3, [c.init_gyro_bias_std] * 3]) ** 2
        self.g_w = np.array([0.0, 0.0, -c.gravity])
        self.stats = {"predictions": 0, "imu_missing": 0, "dvl_updates": 0, "depth_updates": 0, "compass_updates": 0,
                      "rejected_invalid": 0, "rejected_gate": 0, "rejected_stale": 0, "gate_recoveries": 0}
        self.last_update_time = {"dvl": None, "depth": None, "compass": None}
        self._consecutive_gate_rejects = {"dvl": 0, "depth": 0, "compass": 0}

    # ------------------------------------------------------------------ prediction
    def predict(self, imu) -> bool:
        """Propagate to imu.timestamp using an IMU Measurement. Returns True if integrated.

        An invalid/failed/non-finite IMU sample, or a gap > max_dt, falls back to coast()."""
        t = float(imu.timestamp)
        dt = t - self.time
        if dt <= 0:
            return False
        if (not imu.valid) or imu.health == SensorHealth.FAILED or not np.all(np.isfinite(imu.value)) or dt > self.cfg.max_dt:
            self.coast(t)
            return False
        c = self.cfg
        f, w = imu.value[:3], imu.value[3:]
        a_b, self.w_b = f - self.ba, w - self.bg
        R = quat_to_rot(self.q)
        a_w = R @ a_b + self.g_w

        # error-state transition (uses pre-update R, a_b, w_b)
        A = np.zeros((15, 15))
        A[P, V] = np.eye(3)
        A[V, TH] = -R @ skew(a_b)
        A[V, BA] = -R
        A[TH, TH] = -skew(self.w_b)
        A[TH, BG] = -np.eye(3)
        Adt = A * dt
        F = np.eye(15) + Adt + 0.5 * Adt @ Adt
        sa, sg = c.accel_noise_std, c.gyro_noise_std
        if imu.health == SensorHealth.DEGRADED and imu.covariance is not None:
            sa, sg = float(np.sqrt(imu.covariance[0, 0])), float(np.sqrt(imu.covariance[3, 3]))
        Q = np.zeros((15, 15))
        Q[P, P] = np.eye(3) * (0.5 * sa * dt ** 2) ** 2
        Q[V, V] = np.eye(3) * (sa * dt) ** 2
        Q[TH, TH] = np.eye(3) * (sg * dt) ** 2
        Q[BA, BA] = np.eye(3) * c.accel_bias_walk_std ** 2 * dt
        Q[BG, BG] = np.eye(3) * c.gyro_bias_walk_std ** 2 * dt
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)

        # nominal-state propagation
        self.p = self.p + self.v * dt + 0.5 * a_w * dt ** 2
        self.v = self.v + a_w * dt
        self.q = quat_integrate(self.q, self.w_b, dt)
        self.time = t
        self.stats["predictions"] += 1
        return True

    def coast(self, t: float) -> None:
        """Prediction without IMU: constant velocity, held attitude, covariance inflated."""
        dt = float(t) - self.time
        if dt <= 0:
            return
        c = self.cfg
        F = np.eye(15)
        F[P, V] = np.eye(3) * dt
        Q = np.zeros((15, 15))
        s = c.imu_loss_accel_inflation
        Q[P, P] = np.eye(3) * (0.5 * s * dt ** 2) ** 2
        Q[V, V] = np.eye(3) * (s * dt) ** 2
        Q[TH, TH] = np.eye(3) * (c.gyro_noise_std * dt + 0.05 * dt) ** 2
        Q[BA, BA] = np.eye(3) * c.accel_bias_walk_std ** 2 * dt
        Q[BG, BG] = np.eye(3) * c.gyro_bias_walk_std ** 2 * dt
        self.P = F @ self.P @ F.T + Q
        self.P = 0.5 * (self.P + self.P.T)
        self.p = self.p + self.v * dt
        self.time = float(t)
        self.stats["imu_missing"] += 1

    # ------------------------------------------------------------------ updates
    def _usable(self, m, dim: int) -> bool:
        if m is None:
            return False
        if (not m.valid) or m.health == SensorHealth.FAILED or not np.all(np.isfinite(m.value)):
            self.stats["rejected_invalid"] += 1
            return False
        if m.timestamp < self.time - self.cfg.max_time_skew:
            self.stats["rejected_stale"] += 1
            return False
        return True

    def _gate_reject(self, key: str) -> None:
        """Count a gate rejection. A run of them means the filter, not the sensor, is wrong (bad
        initialisation, long outage, overconfident covariance). Without a way out the gate would
        reject every future measurement forever, so inflate the affected covariance; the next samples
        then pass the gate and are absorbed through the normal update (no blind acceptance)."""
        self.stats["rejected_gate"] += 1
        self._consecutive_gate_rejects[key] += 1
        if self._consecutive_gate_rejects[key] >= self.cfg.gate_max_consecutive:
            infl = self.cfg.gate_recovery_inflation_std ** 2
            if key == "dvl":
                self.P[V, V] += np.eye(3) * infl
            elif key == "compass":
                self.P[TH, TH] += np.eye(3) * (0.2 ** 2)          # 0.2 rad: re-admit the heading reference
            else:
                self.P[2, 2] += infl
            self._consecutive_gate_rejects[key] = 0
            self.stats["gate_recoveries"] += 1

    def _update(self, y: np.ndarray, H: np.ndarray, Rm: np.ndarray, gate_key: int, name: str = "") -> bool:
        S = H @ self.P @ H.T + Rm
        if self.cfg.gate_enabled:
            d2 = float(y @ np.linalg.solve(S, y))
            if d2 > _CHI2_999[gate_key]:
                self._gate_reject(name)
                return False
        self._consecutive_gate_rejects[name] = 0
        K = np.linalg.solve(S, H @ self.P).T          # P H^T S^-1 (S, P symmetric)
        dx = K @ y
        I_KH = np.eye(15) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ Rm @ K.T      # Joseph form
        self.P = 0.5 * (self.P + self.P.T)
        # inject error into the nominal state
        self.p = self.p + dx[P]
        self.v = self.v + dx[V]
        dth = dx[TH]
        dq = np.r_[0.5 * dth, 1.0]
        self.q = quat_normalize(quat_multiply(self.q, quat_normalize(dq)))
        self.ba = self.ba + dx[BA]
        self.bg = self.bg + dx[BG]
        return True

    def update_dvl(self, m) -> bool:
        """Body-frame, seabed-relative velocity update."""
        if not self._usable(m, 3):
            return False
        R = quat_to_rot(self.q)
        v_b = R.T @ self.v
        H = np.zeros((3, 15))
        H[:, V] = R.T
        H[:, TH] = skew(v_b)
        scale = self.cfg.degraded_r_scale if m.health == SensorHealth.DEGRADED else 1.0
        ok = self._update(m.value - v_b, H, np.eye(3) * (self.cfg.dvl_noise_std ** 2) * scale, 3, "dvl")
        if ok:
            self.stats["dvl_updates"] += 1
            self.last_update_time["dvl"] = self.time
        return ok

    def update_compass(self, m) -> bool:
        """Absolute heading update (rad). The innovation is wrapped to [-pi, pi]."""
        if not self._usable(m, 1):
            return False
        R = quat_to_rot(self.q)
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        H = np.zeros((1, 15))
        H[0, TH] = R[2, :]
        scale = self.cfg.degraded_r_scale if m.health == SensorHealth.DEGRADED else 1.0
        y = np.array([_wrap(float(m.value[0]) - yaw)])
        ok = self._update(y, H, np.array([[self.cfg.compass_noise_std ** 2 * scale]]), 1, "compass")
        if ok:
            self.stats["compass_updates"] += 1
            self.last_update_time["compass"] = self.time
        return ok

    def update_depth(self, m) -> bool:
        """Depth (positive down) update: z_meas = -p_z."""
        if not self._usable(m, 1):
            return False
        H = np.zeros((1, 15))
        H[0, 2] = -1.0
        scale = self.cfg.degraded_r_scale if m.health == SensorHealth.DEGRADED else 1.0
        y = np.array([float(m.value[0]) - (-self.p[2])])
        ok = self._update(y, H, np.array([[self.cfg.depth_noise_std ** 2 * scale]]), 1, "depth")
        if ok:
            self.stats["depth_updates"] += 1
            self.last_update_time["depth"] = self.time
        return ok

    # ------------------------------------------------------------------ convenience
    def step(self, imu=None, dvl=None, depth=None, t: float | None = None) -> EstimatedState:
        """One filter cycle: predict (IMU, or coast to t) then apply any DVL/depth measurements."""
        if imu is not None:
            self.predict(imu)
        elif t is not None:
            self.coast(t)
        if dvl is not None:
            self.update_dvl(dvl)
        if depth is not None:
            self.update_depth(depth)
        return self.state

    @property
    def state(self) -> EstimatedState:
        return EstimatedState(
            time=self.time, position=self.p.copy(), velocity=self.v.copy(), quat=self.q.copy(),
            angular_velocity=self.w_b.copy(), accel_bias=self.ba.copy(), gyro_bias=self.bg.copy(),
            covariance=self.P.copy(), status=dict(self.stats))

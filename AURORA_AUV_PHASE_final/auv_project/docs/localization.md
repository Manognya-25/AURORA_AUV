# Localization: error-state EKF (Phase 3)

Package `src/localization/`. The EKF consumes `Measurement` objects only. It imports nothing from PyBullet or
`src.simulation` (enforced by a test). It is initialised from the known mission start pose, as any dead-reckoning
navigation system is, never from per-step ground truth.

## Layers (kept separate)
TRUE state (PyBullet) → sensors → measurements → EKF → ESTIMATED state (`EstimatedState`).
In Phase 3 the controller still uses the true state. The estimate is **only logged** (`est_*` columns in the log CSV,
plus `est_*_err` vs truth for evaluation).

## State
Nominal: position `p` (world), velocity `v` (world), unit quaternion `q` (x,y,z,w), accel bias `b_a`, gyro bias `b_g`.
Error state (15): `[δp, δv, δθ (body-frame small rotation), δb_a, δb_g]`, with `R_true = R(q)(I + [δθ]×)`.
Angular velocity is reported as the last gyro reading minus the estimated bias.

## Prediction (IMU)
`a_b = f − b_a`, `ω_b = ω − b_g`, `a_w = R a_b + g_w`;
`p += v dt + ½ a_w dt²`, `v += a_w dt`, `q ← q ⊗ exp(ω_b dt)`.
Transition `F = I + A dt + (A dt)²/2` with `A_pv = I`, `A_vθ = −R[a_b]×`, `A_vba = −R`, `A_θθ = −[ω_b]×`, `A_θbg = −I`.
Process noise from accel/gyro per-sample noise and bias random-walk. If the IMU sample is invalid or failed, or the gap
exceeds `max_dt`, the filter **coasts**: constant velocity, held attitude, covariance inflated.

## Updates (Joseph form, symmetrised)
- **DVL:** `z = Rᵀ v` (body frame). `H_v = Rᵀ`, `H_θ = [Rᵀ v]×`. Checked against a numerical derivative in a test.
- **Depth:** `z = −p_z`, `H = [0 0 −1]`.
- Invalid, NaN, failed or stale measurements are skipped and counted. A DEGRADED sensor has its R scaled by
  `degraded_r_scale`.
- **Chi-square gate** (99.9 %) rejects outliers. A run of `gate_max_consecutive` rejections means the *filter* is
  overconfident (bad init or long outage), so the relevant covariance is inflated and later samples pass the gate
  normally. Without this a naive gate rejects everything forever; this was reproduced and a regression test covers it.

## What is and is not observable
- Velocity, roll and pitch, and depth are observable and converge with DVL, depth and gravity.
- **Yaw and horizontal position have no absolute reference** (no compass, GPS or USBL). They are dead-reckoned and
  drift, and the covariance stays open to say so. After a DVL blackout, body-frame velocity recovers but world-frame
  velocity error stays about `speed × yaw error`. Tests assert this honestly instead of hiding it.

## Configuration
`ekf:` in `configs/config.yaml` (process noise, measurement noise, initial 1-sigma covariance, gating, `enabled`).
Initial covariance should match the real initial error: the filter is only as honest as this number.

## Results seen (not guarantees)
Full mission, noisy sensors, DVL ~96–98 % valid: RMS position error 0.23 m (legacy) and 0.38 m (6dof), with error
growing toward the end of the run as yaw drift accumulates.

## Limitations
No sonar integration (by design); depth-sensor bias and DVL scale not estimated; ocean current not estimated;
no lever arm; Euler singularity not an issue (quaternion) but pitch is clipped at ±90° in the Euler output;
tuning uses the same noise values as the simulated sensors, so a mismatched-noise study has not been done.

## Why the physical goal error was ~1.4 m (and the compass)

The Phase 5 PyBullet mission ended with an *estimated* goal distance of 0.32 m but a *true* (evaluation-only) distance of
~1.42 m; the EKF position error was 1.66 m. The log (`results/logs/week5_simulation_log.csv`) shows the cause: the EKF yaw
error grew from ~0 to 3-9 degrees during the avoidance turns. Without an absolute heading reference yaw is gyro dead
reckoning, so every DVL velocity is rotated by the wrong heading and position error grows with distance travelled
(1 degree of heading error ~ 1.7 % of the distance). The navigator was *correct* to stop where the EKF said the goal was;
the EKF was wrong about where the vehicle was. Horizontal position still has no absolute fix (no GPS/USBL), so it stays
dead-reckoned.

Fix (legitimate sensing, no ground truth in control): `sensors.compass` (`src/sensors/compass.py`) is a magnetic
heading sensor (5 Hz, 1.1 deg noise, 0.3 deg constant bias the EKF does not know about, 2 % dropouts, health/fault
injection) and `EKF.update_compass` fuses it (wrapped innovation, chi-square gated). On a scripted weaving trajectory
(real sensors + EKF, `tests/test_compass.py`) the final position error drops from 0.8-3.6 m to 0.06-0.22 m and the yaw
error from 5-17 deg to <1 deg. Disable with `sensors.compass.enabled: false` to get the old behaviour. Whether the real
PyBullet mission now ends within 0.6 m must be confirmed on a PyBullet run (UNVALIDATED here).

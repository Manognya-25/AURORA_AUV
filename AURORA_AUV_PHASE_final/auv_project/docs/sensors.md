# Sensors (Phase 3)

Package `src/sensors/`. All parameters in `configs/config.yaml` under top-level `sensors:` are **illustrative simulation
parameters, not calibrated specifications of any real device**. Noise standard deviations are **per sample**.

## Architecture
```
PyBullet body --(src/simulation/truth.py)--> TruthSnapshot --> IMU / DVL / Depth / Sonar --> Measurement
```
`TruthSnapshot` is the only way ground truth enters the sensor path. `SensorSuite` samples each sensor at its own
`rate_hz` (IMU every physics step). A `Measurement` holds `timestamp, value, valid, health, covariance`.
Invalid samples carry **NaN**, never ground truth. Sonar returns `SonarMeasurement` (ranges plus per-ray detail).
Every sensor owns a seeded `np.random.Generator` and draws a **fixed number of variates per sample**, so a stream is
reproducible and a valid sample's noise does not depend on the dropout setting.

## Frames and conventions (inherited from Phase 2, unchanged)
| Item | Convention |
|---|---|
| World | +Z up, depth = −z, seabed plane z = `simulation.seabed_z` (−15) |
| Body / sensor | X forward, Y left, Z up; IMU and DVL at the CoG; sonar at its mount |
| Quaternion | (x, y, z, w), body→world (PyBullet order). Euler: intrinsic Z-Y-X |
| Gravity | `g_w = [0, 0, −9.8]` |
| Depth | positive down, metres |

No frame inconsistency was found in Phase 2 that needed fixing. The legacy controller commanding world-frame forces is a
documented Phase 1/2 simplification and was left alone.

## IMU (`imu.py`)
- Accelerometer (specific force): `f_b = Rᵀ(a_w − g_w) + b_a + n_a`. At rest and level it reads `[0, 0, +9.8]`.
- Gyroscope: `ω_b = Rᵀ ω_w + b_g + n_g`.
- `a_w` is the finite difference of the simulated world velocity between samples (the first sample uses 0). It is
  derived from motion, not generated.
- Bias is an initial constant plus a random walk: `b += N(0, walk_std²·dt)`.
- Degraded health multiplies noise by `degraded_noise_scale`; failed gives invalid NaN.
- Not modelled: lever arm, centripetal terms, scale factor, axis misalignment, temperature.

## DVL (`dvl.py`)
Seabed-relative velocity in the **body frame**: `v = (1+s)·Rᵀ v_w + n`. Ocean current does not appear (bottom lock).
Invalid when random dropout (`dropout_probability`), altitude outside `[min_altitude_m, max_altitude_m]` (seabed is the
plane `z = seabed_z`), or failure. Not modelled: beam geometry, water-track mode, speed-of-sound error, terrain slope.

## Depth (`depth.py`)
`depth = −z + bias + n`, positive down. Dropout and failure give NaN. Not modelled: pressure physics, density, drift.

## Sonar (`sonar.py`)
Real PyBullet ray casting (`p.rayTestBatch`) against the simulated seabed and obstacles. **Not acoustic physics**: each
beam is one geometric ray (no beam pattern, multipath, attenuation, or sound-speed effects).
- **Mount:** `mount_position_m` in the body frame (default `[0.8, 0, 0]`, just ahead of the 0.75 m nose so rays do not
  start inside the hull). **Orientation:** `mount_rpy_deg` relative to the body (default 0 = looks along +X).
- **Ray frame:** azimuth positive toward +Y (left), elevation positive toward +Z (up);
  `d = [cos el cos az, cos el sin az, sin el]`. Angles are evenly spread over ±FOV/2; one ray sits at 0.
  Index = `i_elev · n_az + i_az`.
- **Per ray, fixed draw order:** dropout → detection probability → Gaussian range noise. Hits closer than `min_range_m`,
  at or beyond `max_range_m`, or on the vehicle's own body give no return.
- **Output:** `value` = ranges (`max_range` where nothing detected), `detected` mask, `azimuths`, `elevations`,
  `points_sensor` (body-frame points, NaN where not detected). `valid` is False if the whole sensor failed or over half
  the rays dropped out.
- Requires PyBullet; `src.sensors` imports without it (lazy import).

## Health (`health.py`)
Canonical states: `HEALTHY / DEGRADED / FAILED / DISCARDED / LINK_CUT`. Consecutive invalid samples escalate status
(degraded after 5, failed after 50; sonar 3/30). This is a **status report only** and does not latch the sensor off,
so a sensor recovers when samples return. Only an **injected** fault (`sensor.inject_fault(...)`) changes behaviour.

### Operator-facing state names (dashboard / API)
The operator selects four names; each is applied live to the RUNNING sensor and really changes its output
(`canonical_state()` maps the name to the canonical state, so the rest of the pipeline never sees the alias):

| Operator name | Canonical state | What the sensor model really does |
|---|---|---|
| ENABLED | HEALTHY | normal noise/bias; readings carry information |
| UNHEALTHY | DEGRADED | samples keep coming, noise multiplied by `degraded_noise_scale`; the EKF keeps using them |
| FAILED | FAILED | a real sample every cycle but its output is **stuck at 0.0** (`valid=False`, the EKF rejects it) |
| DISABLED | DISCARDED | no sample at all (the suite skips it), so its readings stop changing; also `LINK_CUT` |

### Safety stop: every sensor failed/unhealthy
The run loop (`run_simulation.py`) checks `SensorSuite.all_unusable()` at the top of every step: when **every**
configured sensor is FAILED or UNHEALTHY (DEGRADED) the vehicle has no usable navigation input at all, so the robot
is commanded to stop and the run ends with status `STOPPED` (recorded in the `safety` block of the final telemetry).
DISABLED (DISCARDED / LINK_CUT) is deliberately **not** part of this condition: there the sensor system delivers
nothing and the EKF simply coasts, which is the documented behaviour the dashboard already reports
(`sensors_all_stopped`).

## Configuration
See `sensors:` in `configs/config.yaml`. `sensors.enabled: false` removes sensors and EKF entirely; Phase 2 behaviour is
bit-identical (verified for legacy and 6dof). The older `simulation.sensors.max_range` 3-ray proximity sensor is untouched.

## Limitations
Simplified error models only; no sensor latency or timestamp jitter; independent noise per sensor; sonar is geometric,
not acoustic; the seabed for the DVL is a flat plane; sensor rates are idealised.

"""SensorSuite: builds the configured sensors and samples them at their own rates.

Reads the top-level `sensors:` config section. Truth enters only through the
TruthSnapshot passed to sample(); nothing here talks to the estimator.
"""
from __future__ import annotations

import numpy as np

from src.sensors.compass import Compass, CompassConfig
from src.sensors.depth import DepthConfig, DepthSensor
from src.sensors.dvl import DVL, DVLConfig
from src.sensors.imu import IMU, IMUConfig
from src.sensors.models import TruthSnapshot


class SensorSuite:
    def __init__(self, cfg: dict, body_id: int | None = None, physics_client: int = 0):
        """cfg: the FULL project config (needs `sensors`; uses simulation.gravity / seabed_z)."""
        s = cfg.get("sensors", {}) or {}
        sim = cfg.get("simulation", {})
        gravity = abs(float(sim.get("gravity", -9.8)))
        seabed_z = float(sim.get("seabed_z", -15.0))
        self.enabled = bool(s.get("enabled", True))
        self.sensors, self.period = {}, {}
        dt = float(sim.get("time_step", 0.01))

        def add(key, obj, default_rate):
            if (s.get(key, {}) or {}).get("enabled", True):
                self.sensors[key] = obj
                self.period[key] = 1.0 / float((s.get(key, {}) or {}).get("rate_hz", default_rate))

        add("imu", IMU(IMUConfig.from_config(s.get("imu"), gravity)), 1.0 / dt)
        add("dvl", DVL(DVLConfig.from_config(s.get("dvl"), seabed_z)), 10.0)
        add("depth", DepthSensor(DepthConfig.from_config(s.get("depth"))), 10.0)
        if "compass" in s and (s.get("compass") or {}).get("enabled", True):     # opt-in: only when configured
            add("compass", Compass(CompassConfig.from_config(s.get("compass"))), 5.0)
        sonar_cfg = s.get("sonar", {}) or {}
        if sonar_cfg.get("enabled", True) and body_id is not None:
            from src.sensors.sonar import Sonar, SonarConfig
            self.sensors["sonar"] = Sonar(body_id, physics_client, SonarConfig.from_config(sonar_cfg))
            self.period["sonar"] = 1.0 / float(sonar_cfg.get("rate_hz", 5.0))
        self._next = {k: -np.inf for k in self.sensors}

    def sample(self, truth: TruthSnapshot) -> dict:
        """Return {name: Measurement} for every sensor whose sample time has come.

        A sensor the operator DISCARDED or whose LINK IS CUT produces no sample at all: it is skipped here, so
        nothing reaches the EKF, the mapper or the telemetry and the sensor's last reading stops changing. The
        sample clock is not advanced either, so clearing the fault resumes sampling immediately.
        """
        out = {}
        if not self.enabled:
            return out
        for k, sensor in self.sensors.items():
            monitor = getattr(sensor, "health_monitor", None)
            if monitor is not None and monitor.stopped:
                continue
            if truth.time + 1e-9 >= self._next[k]:
                out[k] = sensor.measure(truth)
                self._next[k] = (truth.time if self._next[k] == -np.inf else self._next[k]) + self.period[k]
        return out

    def all_unusable(self) -> bool:
        """True when EVERY configured sensor is FAILED or UNHEALTHY (DEGRADED).

        That is the operator's "all sensors failed/unhealthy" condition: no configured sensor is delivering
        usable data, so the run loop stops the vehicle (see src/simulation/run_simulation.py). DISCARDED /
        LINK_CUT do not count here - with only those the sensor system delivers nothing and the EKF simply coasts,
        which is the documented behaviour the dashboard already reports (sensors_all_stopped). A suite with no
        sensors (or sensors disabled) is never "all unusable" - there is nothing to be unhealthy.
        """
        if not self.enabled or not self.sensors:
            return False
        return all(getattr(sensor.health_monitor, "unusable", False) for sensor in self.sensors.values())

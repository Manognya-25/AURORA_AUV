"""Simulated sensors (Phase 3): IMU, DVL, depth, forward multi-ray sonar.

Frame conventions (inherited from Phase 2, see docs/sensors.md):
  world: +Z up, depth = -z.   body/sensor: X forward, Y left, Z up.
  quaternions: (x, y, z, w), body -> world.
"""
from src.sensors.models import Measurement, SonarMeasurement, TruthSnapshot
from src.sensors.health import HealthMonitor, SensorHealth
from src.sensors.imu import IMU, IMUConfig
from src.sensors.dvl import DVL, DVLConfig
from src.sensors.depth import DepthSensor, DepthConfig
from src.sensors.sonar import Sonar, SonarConfig
from src.sensors.suite import SensorSuite

__all__ = [
    "Measurement", "SonarMeasurement", "TruthSnapshot", "HealthMonitor", "SensorHealth",
    "IMU", "IMUConfig", "DVL", "DVLConfig", "DepthSensor", "DepthConfig",
    "Sonar", "SonarConfig", "SensorSuite",
]

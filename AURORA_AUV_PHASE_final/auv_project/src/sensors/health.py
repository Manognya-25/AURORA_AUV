"""Basic sensor health model: HEALTHY / DEGRADED / FAILED / DISCARDED / LINK_CUT.

Health comes from (a) a forced state injected by the operator while the simulation runs and (b) the
number of CONSECUTIVE invalid samples. An isolated dropout does not change health. This is a
sensor-level model only; vehicle-level failure management is a later phase.

Operator-injected states and what the sensor model really does with each (the dashboard shows the
live reading of every state, so the difference is observable, not a label):

  HEALTHY    normal noise/bias, samples carry information.
  DEGRADED   samples keep coming but the noise is multiplied (degraded_noise_scale); the EKF keeps using them.
  FAILED     the sensor keeps producing a sample at its own rate, but its output is STUCK AT ZERO (0.0) and
             does not change. `valid` is False, so the EKF rejects it (a stuck-zero sensor is a fault, not truth).
  DISCARDED  the sensor's output is discarded: it produces no sample at all, so nothing reaches the
             EKF/telemetry and its last reading stops changing (SensorSuite skips it).
  LINK_CUT   the datalink to the sensor is cut: identical observable behaviour to DISCARDED (no sample
             arrives), kept as a separate state so the operator sees which fault was applied.
"""
from __future__ import annotations

from enum import IntEnum


class SensorHealth(IntEnum):
    HEALTHY = 0
    DEGRADED = 1
    FAILED = 2
    DISCARDED = 3
    LINK_CUT = 4


# Strings accepted from the dashboard/API/tests -> canonical state (extra aliases make the API forgiving).
# The operator-facing names ENABLED / DISABLED / UNHEALTHY map onto the canonical states the sensor models use:
#   ENABLED   -> HEALTHY    (normal output)
#   DISABLED  -> DISCARDED  (no output at all: the sensor produces no sample)
#   UNHEALTHY -> DEGRADED   (samples keep coming, but much noisier; still used by the EKF)
#   FAILED    -> FAILED     (samples keep coming, output stuck at 0.0; the EKF rejects them)
STATE_ALIASES = {"": "", "RESET": "HEALTHY", "OK": "HEALTHY", "ENABLE": "HEALTHY",
                 "ON": "HEALTHY", "DISCARD": "DISCARDED", "DISABLE": "DISCARDED", "OFF": "DISCARDED",
                 "CUT_LINK": "LINK_CUT", "CUTLINK": "LINK_CUT", "CUT": "LINK_CUT",
                 "NO_LINK": "LINK_CUT", "NOLINK": "LINK_CUT",
                 "ENABLED": "HEALTHY", "DISABLED": "DISCARDED", "UNHEALTHY": "DEGRADED"}
FORCED_STATES = ("HEALTHY", "DEGRADED", "FAILED", "DISCARDED", "LINK_CUT")
STOPPED_STATES = ("DISCARDED", "LINK_CUT")
# States that mean the sensor is not delivering usable data at all. When EVERY configured sensor is in one of
# these the vehicle has no navigation input and the run must stop (see SensorSuite.all_unusable / run_simulation).
UNUSABLE_STATES = ("FAILED", "DEGRADED")


def canonical_state(name: str) -> str:
    """Normalise an operator-supplied state string (empty string = 'clear', i.e. HEALTHY)."""
    key = str(name or "").strip().upper()
    return STATE_ALIASES.get(key, key)


class HealthMonitor:
    def __init__(self, degrade_after: int = 5, fail_after: int = 50):
        if not 0 < degrade_after <= fail_after:
            raise ValueError("need 0 < degrade_after <= fail_after")
        self.degrade_after = int(degrade_after)
        self.fail_after = int(fail_after)
        self.forced: SensorHealth | None = None
        self.consecutive_invalid = 0

    def inject(self, health: SensorHealth | str | None) -> None:
        """Force a state (None clears the fault). Accepts a SensorHealth, a state name or an alias."""
        if isinstance(health, str):
            health = SensorHealth[canonical_state(health)]
        self.forced = health

    @property
    def fault_failed(self) -> bool:
        """True only for an INJECTED failure (stuck-at-zero output). Auto-escalation from dropouts is a status
        report and must not latch the sensor off (otherwise it could never recover)."""
        return self.forced == SensorHealth.FAILED

    @property
    def fault_degraded(self) -> bool:
        return self.forced == SensorHealth.DEGRADED

    @property
    def fault_discarded(self) -> bool:
        """The operator discarded this sensor's output: it produces no sample at all."""
        return self.forced == SensorHealth.DISCARDED

    @property
    def fault_link_cut(self) -> bool:
        """The datalink to this sensor is cut: no sample arrives (same effect as DISCARDED)."""
        return self.forced == SensorHealth.LINK_CUT

    @property
    def stopped(self) -> bool:
        """True while NO sample is produced at all (discarded / link cut): the readings stop changing."""
        return self.forced in (SensorHealth.DISCARDED, SensorHealth.LINK_CUT)

    @property
    def unusable(self) -> bool:
        """True while the sensor is FAILED or UNHEALTHY (DEGRADED): its output is not usable navigation data.
        DISCARDED / LINK_CUT (DISABLED) are deliberately NOT counted here - a run with only those keeps going by
        the existing contract (no sample arrives, the EKF coasts)."""
        return self.health in (SensorHealth.FAILED, SensorHealth.DEGRADED)

    @property
    def zeroed(self) -> bool:
        """True while the output is a stuck-at-zero sample (FAILED): the reading is 0.0 and stays 0.0."""
        return self.forced == SensorHealth.FAILED

    def record(self, valid: bool) -> SensorHealth:
        self.consecutive_invalid = 0 if valid else self.consecutive_invalid + 1
        return self.health

    @property
    def health(self) -> SensorHealth:
        auto = SensorHealth.HEALTHY
        if self.consecutive_invalid >= self.fail_after:
            auto = SensorHealth.FAILED
        elif self.consecutive_invalid >= self.degrade_after:
            auto = SensorHealth.DEGRADED
        if self.forced is None:
            return auto
        return self.forced if self.stopped else max(auto, self.forced)

    def reset(self) -> None:
        self.consecutive_invalid = 0
        self.forced = None

"""Pybullet-free command + config-merge helpers used by the live simulation loop."""
import yaml

from src.mapping.map_manager import MappingSystem
from src.simulation.commands import apply_command, deep_merge
from src.simulation.estimation import EstimationPipeline


def test_deep_merge_is_recursive_non_mutating_and_overrides_leaves():
    base = {"a": {"b": 1, "c": {"d": 2, "e": 3}}, "x": 1}
    over = {"a": {"c": {"d": 9}}, "y": 2}
    out = deep_merge(base, over)
    assert out == {"a": {"b": 1, "c": {"d": 9, "e": 3}}, "x": 1, "y": 2}      # regression: nested merge recursed into a missing name
    assert base["a"]["c"]["d"] == 2                                             # input untouched
    assert deep_merge(base, None) == base


def make():
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    est = EstimationPipeline(cfg, 0, (5.0, 5.0, -5.0))
    return cfg, est, MappingSystem(cfg)


def test_set_mapping_enabled_and_reset_mapping_act_on_the_mapper():
    cfg, est, ms = make()
    ok, detail = apply_command({"cmd": "set_mapping_enabled", "enabled": False}, ms, est, {})
    assert ok and not ms.enabled and "disabled" in detail
    ok, _ = apply_command({"cmd": "set_mapping_enabled", "enabled": True}, ms, est, {})
    assert ok and ms.enabled
    ms.stats["map_updates"] = 7
    ok, detail = apply_command({"cmd": "reset_mapping"}, ms, est, {})
    assert ok and ms.stats["map_updates"] == 0


def test_set_sensor_health_injects_a_real_fault_and_clears_it():
    cfg, est, ms = make()
    from src.sensors.health import SensorHealth
    ok, _ = apply_command({"cmd": "set_sensor_health", "sensor": "dvl", "health": "failed"}, ms, est, {})
    assert ok and est.suite.sensors["dvl"].health_monitor.health == SensorHealth.FAILED
    ok, _ = apply_command({"cmd": "set_sensor_health", "sensor": "dvl", "health": "HEALTHY"}, ms, est, {})
    assert ok and est.suite.sensors["dvl"].health_monitor.health == SensorHealth.HEALTHY


def test_set_sensor_health_supports_compass_reset_alias_and_reports_available_sensors():
    cfg, est, ms = make()
    from src.sensors.health import SensorHealth
    # compass parity: the dashboard offers a compass selector, so the command layer must act on the REAL compass
    assert "compass" in est.suite.sensors
    ok, detail = apply_command({"cmd": "set_sensor_health", "sensor": "compass", "health": "DEGRADED"}, ms, est, {})
    assert ok and est.suite.sensors["compass"].health_monitor.forced == SensorHealth.DEGRADED
    ok, _ = apply_command({"cmd": "set_sensor_health", "sensor": "compass", "health": "RESET"}, ms, est, {})
    assert ok and est.suite.sensors["compass"].health_monitor.forced is None       # RESET clears the injected fault
    ok, detail = apply_command({"cmd": "set_sensor_health", "sensor": "compass", "health": "HEALTHY"}, ms, est, {})
    assert ok and "cleared" in detail
    # a sensor that is not configured in this run fails with a message naming what IS available
    del est.suite.sensors["compass"]
    ok, detail = apply_command({"cmd": "set_sensor_health", "sensor": "compass", "health": "FAILED"}, ms, est, {})
    assert not ok and "compass" in detail and "dvl" in detail


def test_invalid_commands_are_rejected_with_a_reason_and_change_nothing():
    cfg, est, ms = make()
    state = {"realtime_factor": 1.0}
    for cmd in ({"cmd": "set_sensor_health", "sensor": "lidar", "health": "FAILED"},
                {"cmd": "set_sensor_health", "sensor": "dvl", "health": "EXPLODED"},
                {"cmd": "set_realtime_factor", "value": "fast"},
                {"cmd": "set_realtime_factor", "value": -1},
                {"cmd": "self_destruct"}):
        ok, detail = apply_command(cmd, ms, est, state)
        assert not ok and detail, cmd
    assert state["realtime_factor"] == 1.0
    ok, detail = apply_command({"cmd": "reset_mapping"}, None, est, state)
    assert not ok and "unavailable" in detail
    ok, detail = apply_command({"cmd": "set_sensor_health", "sensor": "dvl", "health": "FAILED"}, ms, None, state)
    assert not ok


def test_realtime_factor_command_updates_the_loop_state():
    cfg, est, ms = make()
    state = {}
    ok, _ = apply_command({"cmd": "set_realtime_factor", "value": 4}, ms, est, state)
    assert ok and state["realtime_factor"] == 4.0

"""Week 9 adaptive P controller: the plain P law is kept far from the target and only the PROPORTIONAL GAIN is
re-shaped -- scheduled on the error magnitude and bounded-self-tuned on the observed closing speed.

Pure numpy (src/simulation/controller.py and adaptive_controller.py import no PyBullet), so these tests run in the
stand-in environment. The controller is fed a PERFECT state; the closed-loop test at the end uses a minimal
first-order body with a constant opposing force as a stand-in for the residual drag/buoyancy that leaves a plain
P controller short of its set-point -- it is NOT a PyBullet result.
"""
import os

import numpy as np
import pytest
import yaml

from src.simulation.adaptive_controller import AdaptivePController, gain_schedule, yaw_gain_multiplier
from src.simulation.controller import AdaptiveParams, ControllerParams, WaypointController

CONFIG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml"))
DT = 0.05
TGT = (10.0, 0.0, -5.0)
POS = (0.0, 0.0, -5.0)


def load_cfg():
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def base_params(**adaptive):
    cfg = load_cfg()["simulation"]["controller"]
    cfg = {**cfg, "adaptive": {**cfg["adaptive"], **adaptive}}
    return ControllerParams.from_config(cfg)


# ------------------------------------------------------------------------------ gain schedule
def test_gain_schedule_is_nominal_far_and_boosted_near():
    assert gain_schedule(50.0, 1.0, 6.0, 3.0) == pytest.approx(1.0)
    assert gain_schedule(6.0, 1.0, 6.0, 3.0) == pytest.approx(1.0)
    assert gain_schedule(1.0, 1.0, 6.0, 3.0) == pytest.approx(3.0)
    assert gain_schedule(0.05, 1.0, 6.0, 3.0) == pytest.approx(3.0)
    assert gain_schedule(3.5, 1.0, 6.0, 3.0) == pytest.approx(2.0)          # linear in between
    assert gain_schedule(-3.5, 1.0, 6.0, 3.0) == pytest.approx(2.0)         # magnitude only


def test_yaw_gain_is_full_when_aligned_and_floored_for_large_errors():
    assert yaw_gain_multiplier(0.0, 0.25, 1.2, 0.5) == pytest.approx(1.0)
    assert yaw_gain_multiplier(0.25, 0.25, 1.2, 0.5) == pytest.approx(1.0)
    assert yaw_gain_multiplier(1.2, 0.25, 1.2, 0.5) == pytest.approx(0.5)
    assert yaw_gain_multiplier(3.0, 0.25, 1.2, 0.5) == pytest.approx(0.5)
    assert yaw_gain_multiplier(-0.725, 0.25, 1.2, 0.5) == pytest.approx(0.75)   # midway between the two


# ------------------------------------------------------------------------------ plain-P equivalence
def test_far_from_the_target_it_is_the_plain_p_controller():
    p = base_params()
    assert p.adaptive.enabled
    ad, plain = AdaptivePController(p), WaypointController(p)
    a = ad.compute_control(POS, 0.0, TGT, dt=DT)          # 10 m away -> gain multiplier 1.0
    b = plain.compute_control(POS, 0.0, TGT)
    assert a == pytest.approx(b)
    assert ad.last_info["kp_position_multiplier"] == pytest.approx(1.0)
    assert ad.last_info["kp_position_effective"] == pytest.approx(p.kp_position)


def test_disabled_adaptation_is_bit_for_bit_the_plain_p_controller():
    p = base_params(enabled=False)
    ad = AdaptivePController(p)
    for e in (10.0, 0.3, 0.05):                            # far, near and inside the deadband
        pos = (TGT[0] - e, TGT[1], TGT[2])
        assert ad.compute_control(pos, 0.0, TGT, dt=DT) == WaypointController(p).compute_control(pos, 0.0, TGT)
    assert ad.adaptive_gain == pytest.approx(1.0)


def test_near_the_target_the_gain_is_boosted_and_the_command_is_larger():
    p = base_params()
    near = (TGT[0] - 0.3, TGT[1], TGT[2])
    boosted = AdaptivePController(p).compute_control(near, 0.0, TGT)
    plain = WaypointController(p).compute_control(near, 0.0, TGT)
    assert p.adaptive.near_m == 1.0 and 0.3 <= p.adaptive.near_m
    assert boosted["forward"] > plain["forward"]
    assert boosted["forward"] == pytest.approx(plain["forward"] * p.adaptive.gain_boost, rel=1e-6)


# ------------------------------------------------------------------------------ self-tuning
def test_self_tuning_ramps_the_gain_up_while_the_vehicle_is_stalled():
    p = base_params()
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])                  # between near_m and far_m: the schedule alone is < boost
    c = AdaptivePController(p)
    first = c.compute_control(pos, 0.0, TGT, dt=DT)        # the vehicle never moves -> closing speed 0 -> stalled
    for _ in range(200):
        last = c.compute_control(pos, 0.0, TGT, dt=DT)
    assert last["forward"] > first["forward"]
    assert p.adaptive.gain_boost == pytest.approx(3.0)
    assert c.adaptive_gain == pytest.approx(p.adaptive.gain_boost)          # ramped up to the cap ...
    assert c.last_info["self_tuning_gain"] == pytest.approx(p.adaptive.gain_boost)
    assert c.adaptive_gain <= p.adaptive.gain_boost                         # ... and never beyond it


def test_self_tuning_relaxes_back_while_the_vehicle_tracks_well():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(200):
        c.compute_control(pos, 0.0, TGT, dt=DT)
    assert c.adaptive_gain > 1.5
    x = TGT[0] - 3.0
    for _ in range(60):                                   # closing at 0.25 m/s -> tracking well -> relax
        x += 0.25 * DT
        c.compute_control((x, TGT[1], TGT[2]), 0.0, TGT, dt=DT)
    assert x < TGT[0] and c.last_info["position_error_m"] > p.adaptive.near_m   # still approaching, never overshot
    assert c.adaptive_gain == pytest.approx(1.0)
    assert c.last_info["self_tuning_gain"] == pytest.approx(1.0)


def test_without_dt_it_schedules_but_never_self_tunes():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(50):
        c.compute_control(pos, 0.0, TGT)                  # no dt -> no adaptation
    assert c.adaptive_gain == pytest.approx(1.0)
    assert c.last_info["self_tuning_gain"] == pytest.approx(1.0)
    assert c.last_info["schedule_multiplier"] == pytest.approx(gain_schedule(3.0, 1.0, 6.0, 3.0))


def test_a_new_target_reseeds_the_adaptation_instead_of_adapting_on_the_jump():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(200):
        c.compute_control(pos, 0.0, TGT, dt=DT)
    gain_before = c.adaptive_gain
    assert gain_before > 1.5
    far = (100.0, 0.0, -5.0)                              # a jump >> target_change_jump_m: a NEW target
    c.compute_control(pos, 0.0, far, dt=DT)
    assert c.adaptive_gain == pytest.approx(gain_before)  # re-seeded, not adapted on the phantom error change


def test_reset_adaptation_returns_the_gain_to_nominal():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(100):
        c.compute_control(pos, 0.0, TGT, dt=DT)
    assert c.adaptive_gain > 1.0
    c.reset_adaptation()
    assert c.adaptive_gain == pytest.approx(1.0)


# ------------------------------------------------------------------------------ yaw + interface
def test_the_yaw_gain_is_floored_while_the_heading_error_is_large():
    p = base_params()
    large = AdaptivePController(p).compute_control(POS, 0.0, (0.0, 10.0, -5.0), dt=DT)   # want +90 deg, facing 0
    plain = WaypointController(p).compute_control(POS, 0.0, (0.0, 10.0, -5.0))
    assert abs(large["yaw"]) < abs(plain["yaw"])          # do not spin while the horizontal error is reduced
    assert large["yaw"] == pytest.approx(plain["yaw"] * p.adaptive.yaw_gain_min, rel=1e-6)


def test_output_is_still_normalised_and_the_interface_is_unchanged():
    p = base_params()
    c = AdaptivePController(p)
    out = c.compute_control(POS, 0.0, TGT, current_velocity=(0.0, 0.0, 0.0), vertical_trim=0.1, dt=DT)
    assert set(out) == {"forward", "lateral", "vertical", "yaw"}
    assert all(-1.0 <= v <= 1.0 for v in out.values())
    assert isinstance(c, WaypointController)              # drop-in: distance_to / reached_waypoint are inherited
    assert c.reached_waypoint(POS, TGT) is False and c.reached_waypoint((0.0, 0.0, -5.0), (0.1, 0.0, -5.0)) is True
    assert c.distance_to(POS, TGT) == pytest.approx(float(np.linalg.norm(np.array(TGT) - np.array(POS))))


# ------------------------------------------------------------------------------ config
def test_adaptive_params_load_from_the_config_and_validate():
    cfg = load_cfg()["simulation"]["controller"]
    assert cfg["type"] == "adaptive P"
    p = ControllerParams.from_config(cfg)
    assert p.adaptive.enabled is True
    assert p.adaptive.near_m == 1.0 and p.adaptive.far_m == 6.0 and p.adaptive.gain_boost == 3.0
    for bad in ({"far_m": 0.5}, {"gain_boost": 0.5}, {"near_m": 0.0},
                {"stall_progress_mps": 0.5, "healthy_progress_mps": 0.1}, {"yaw_gain_min": 0.0}):
        with pytest.raises(ValueError):
            AdaptiveParams(**bad)
    assert p.adaptive.enabled and ControllerParams.from_config({**cfg, "adaptive": {"enabled": False}}).adaptive.enabled is False


# ------------------------------------------------------------------------------ closed loop (stand-in body)
def _settle(controller, bias_force=0.6, drag=2.0, mass=9.0, max_thrust=10.0, seconds=40.0, dt=0.01):
    """Minimal first-order body with a CONSTANT opposing force (a stand-in for residual drag/buoyancy, NOT PyBullet).

    Steady state of a proportional law: kp_effective * error = bias_force, so the vehicle stops SHORT of the target by
    bias_force / kp_effective. That is exactly the failure the adaptive controller addresses.
    """
    x = v = 0.0
    adaptive = isinstance(controller, AdaptivePController)
    for _ in range(int(seconds / dt)):
        kw = {"dt": dt} if adaptive else {}
        control = controller.compute_control((x, 0.0, -5.0), 0.0, (TGT[0], 0.0, -5.0), **kw)
        f = control["forward"] * max_thrust - drag * v - bias_force
        v += f / mass * dt
        x += v * dt
    return float(TGT[0] - x)


def test_adaptive_gain_closes_more_of_the_residual_gap_than_the_plain_p_controller():
    p = base_params()
    plain_gap = _settle(WaypointController(p))
    ad_gap = _settle(AdaptivePController(p))
    assert plain_gap > 0.5                                 # the plain P law really does stop short
    assert ad_gap < 0.5 * plain_gap                        # the adaptive law closes most of it
    assert ad_gap == pytest.approx(0.6 / p.adaptive.gain_boost, abs=0.05)   # bounded by the gain cap, as documented


# ------------------------------------------------------------------------------ gain tracking
def test_the_tracked_direction_agrees_with_what_the_gain_really_did():
    """The tracked action and reason come from the branch that moved the gain, so they can never disagree with it:
    ramping = the gain rose, relaxing = it fell, holding = it did not move."""
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    c.compute_control(pos, 0.0, TGT, dt=DT)
    assert c.last_info["tracking"]["action"] == "holding"          # the very first sample has nothing to compare to
    assert "first sample" in c.last_info["tracking"]["reason"]
    for _ in range(200):                                          # stalled: the vehicle never moves
        c.compute_control(pos, 0.0, TGT, dt=DT)
        assert c.last_info["tracking"]["action"] == "ramping"     # ... so the gain only ever ramps up
    assert "stalled" in c.last_info["tracking"]["reason"]
    assert c.adaptive_gain == pytest.approx(p.adaptive.gain_boost)
    gained = c.adaptive_gain
    x = TGT[0] - 3.0
    for _ in range(10):                                           # now closing at 0.25 m/s = tracking well
        x += 0.25 * DT
        c.compute_control((x, TGT[1], TGT[2]), 0.0, TGT, dt=DT)
        assert c.last_info["tracking"]["action"] == "relaxing"
    assert c.adaptive_gain < gained
    assert "tracking well" in c.last_info["tracking"]["reason"]
    assert c.last_info["tracking"]["closing_speed_mps"] == pytest.approx(0.25, rel=1e-6)
    assert c.last_info["closing_speed_mps"] == pytest.approx(c.last_info["tracking"]["closing_speed_mps"])


def test_the_tracked_gain_history_counts_the_adaptation_that_really_happened():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(200):                                          # 200 stalled steps of dt = 0.05 s
        c.compute_control(pos, 0.0, TGT, dt=DT)
    tr = c.tracking
    assert tr["steps"] == 200 and tr["elapsed_s"] == pytest.approx(200 * DT)
    assert tr["ramp_s"] == pytest.approx(tr["elapsed_s"] - DT)   # every step after the first ramped ...
    assert tr["relax_s"] == 0.0                                   # ... none relaxed ...
    assert tr["hold_s"] == pytest.approx(DT)                      # ... and only the first sample held (nothing to compare)
    assert tr["episodes"] == 1                                    # ONE episode: the gain ramped up once and stayed up
    assert tr["peak_self_tuning"] == pytest.approx(p.adaptive.gain_boost)
    assert tr["peak_active"] == pytest.approx(p.adaptive.gain_boost)
    assert 0.0 < tr["at_cap_s"] <= tr["boosted_s"] <= tr["elapsed_s"]
    assert 1.0 < tr["mean_active"] <= p.adaptive.gain_boost
    assert tr["peak_time_s"] is not None and DT <= tr["peak_time_s"] <= tr["elapsed_s"]
    assert tr["reseeds"] == 0


def test_each_time_the_gain_starts_ramping_again_a_new_episode_is_counted():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(200):
        c.compute_control(pos, 0.0, TGT, dt=DT)
    assert c.tracking["episodes"] == 1
    x = TGT[0] - 3.0
    for _ in range(40):                                           # close fast: the gain relaxes all the way back
        x += 0.25 * DT
        c.compute_control((x, TGT[1], TGT[2]), 0.0, TGT, dt=DT)
    assert c.adaptive_gain == pytest.approx(1.0)
    stopped = (x, TGT[1], TGT[2])
    for _ in range(5):                                            # ... and stalls again at the new position
        c.compute_control(stopped, 0.0, TGT, dt=DT)
    assert c.tracking["episodes"] == 2                            # a SECOND episode, counted when the ramp restarted
    assert c.tracking["ramp_s"] > 0.0 and c.tracking["relax_s"] > 0.0
    assert c.tracking["ramp_s"] + c.tracking["relax_s"] + c.tracking["hold_s"] == pytest.approx(c.tracking["elapsed_s"])


def test_a_new_target_is_counted_as_a_re_seed_and_never_as_adaptation():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(10):
        c.compute_control(pos, 0.0, TGT, dt=DT)
    tr = c.tracking
    steps, elapsed, hold, gain = tr["steps"], tr["elapsed_s"], tr["hold_s"], tr["peak_active"]
    c.compute_control(pos, 0.0, (100.0, 0.0, -5.0), dt=DT)        # a jump >> target_change_jump_m: a NEW target
    tr = c.tracking
    assert tr["reseeds"] == 1 and "re-seeded" in tr["reason"]
    assert tr["steps"] == steps + 1 and tr["elapsed_s"] == pytest.approx(elapsed + DT)   # tracked as a step ...
    assert tr["hold_s"] == pytest.approx(hold + DT) and tr["peak_active"] >= gain         # ... but not as adaptation
    assert tr["episodes"] == 1


def test_without_dt_or_with_adaptation_off_nothing_is_tracked_and_reset_clears_the_history():
    p = base_params()
    c = AdaptivePController(p)
    pos = (TGT[0] - 3.0, TGT[1], TGT[2])
    for _ in range(20):                                           # no dt -> the scheduler runs, the tracking does not
        c.compute_control(pos, 0.0, TGT)
    assert c.tracking["steps"] == 0 and c.tracking["elapsed_s"] == 0.0
    assert "no adaptation step" in c.last_info["tracking"]["reason"]
    off = AdaptivePController(base_params(enabled=False))
    for _ in range(20):
        off.compute_control(pos, 0.0, TGT, dt=DT)
    assert off.tracking["steps"] == 0 and off.tracking["peak_active"] == 1.0
    assert off.last_info["tracking"]["action"] == "off"
    assert "disabled" in off.last_info["tracking"]["reason"]
    for _ in range(50):                                           # now really adapt ...
        c.compute_control(pos, 0.0, TGT, dt=DT)
    assert c.tracking["steps"] == 50 and c.tracking["episodes"] == 1
    c.reset_adaptation()                                          # ... and a new mission must not inherit any of it
    assert c.tracking["steps"] == 0 and c.tracking["episodes"] == 0
    assert c.tracking["peak_active"] == 1.0 and c.tracking["elapsed_s"] == 0.0


def test_the_tracked_history_in_telemetry_is_a_copy_so_a_reader_cannot_corrupt_it():
    p = base_params()
    c = AdaptivePController(p)
    c.compute_control((TGT[0] - 3.0, TGT[1], TGT[2]), 0.0, TGT, dt=DT)
    before = c.tracking
    published = c.last_info["tracking"]
    published["steps"] = 999
    published["peak_active"] = 99.0
    assert c.tracking == before

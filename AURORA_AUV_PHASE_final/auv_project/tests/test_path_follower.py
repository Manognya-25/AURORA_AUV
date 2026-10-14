"""Phase 5 path follower: converges, holds depth, handles yaw and the command frame. Uses the legacy-dynamics rigid body
(tests/helpers/closed_loop.NumpyVehicle). The follower is fed an EstimatedState built from that body, i.e. a PERFECT
estimate: these tests isolate the controller; estimator error is covered by the closed-loop and PyBullet tests."""
import inspect

import numpy as np
import pytest
import yaml

from src.localization.state import EstimatedState
from src.planning.config import FollowerConfig
from src.planning.path_data import PathData
from src.planning.path_follower import PathFollower
from src.utils.transforms import euler_to_quat
from tests.helpers.closed_loop import NumpyVehicle, load_cfg

DT = 0.01


def make_est(veh, t):
    return EstimatedState(time=t, position=veh.p.copy(), velocity=veh.v.copy(), quat=euler_to_quat(0.0, 0.0, veh.yaw),
                          angular_velocity=np.array([0.0, 0.0, veh.r]), accel_bias=np.zeros(3), gyro_bias=np.zeros(3),
                          covariance=np.eye(15) * 1e-3)


def drive(xy, z0=-5.0, z1=-5.0, start=None, yaw0=0.0, t_max=80.0, speed_scale=1.0, cfg=None):
    cfg = cfg or load_cfg()
    fol = PathFollower(FollowerConfig.from_config(cfg, "world"))
    path = PathData.from_xy(np.asarray(xy, float), z0, z1)
    veh = NumpyVehicle(cfg, start if start is not None else [xy[0][0], xy[0][1], z0], yaw=yaw0)
    fol.set_path(path)
    log = []
    for k in range(int(t_max / DT)):
        t = k * DT
        cmd = fol.compute(make_est(veh, t), speed_scale)
        veh.step(cmd.as_control(), DT)
        log.append((t, veh.p.copy(), veh.v.copy(), fol.cross_track, cmd))
        if np.linalg.norm(veh.p[:2] - path.xy[-1]) < 0.15 and np.linalg.norm(veh.v) < 0.05:
            break
    return fol, veh, log, path


def test_straight_path_converges_and_stops_at_the_end():
    fol, veh, log, path = drive([[0, 0], [30, 0]])
    assert np.linalg.norm(veh.p[:2] - [30, 0]) < 0.25
    assert np.linalg.norm(veh.v) < 0.1
    assert max(abs(l[1][1]) for l in log) < 0.3                               # cross-track
    assert max(np.linalg.norm(l[2][:2]) for l in log) < 1.15 * fol.cfg.cruise_speed_mps   # no speed overshoot
    assert fol.remaining_m < 0.5


def test_corner_is_followed_with_bounded_cross_track():
    fol, veh, log, path = drive([[0, 0], [15, 0], [15, 15]])
    assert np.linalg.norm(veh.p[:2] - [15, 15]) < 0.3
    assert max(l[3] for l in log) < 1.6                                       # lookahead rounds the corner, but stays near the path


def test_depth_is_held_toward_the_path_depth_with_limited_vertical_speed():
    fol, veh, log, path = drive([[0, 0], [25, 0]], z0=-3.0, z1=-6.0, start=[0, 0, -3.0], t_max=60.0)
    assert abs(veh.p[2] - (-6.0)) < 0.2
    assert max(abs(l[2][2]) for l in log) <= fol.cfg.max_vertical_speed_mps * 1.6
    assert all(abs(l[4].vertical) <= 1.0 and abs(l[4].forward) <= 1.0 and abs(l[4].lateral) <= 1.0 and abs(l[4].yaw) <= 1.0 for l in log)


def test_yaw_aligns_with_the_direction_of_travel():
    fol, veh, log, path = drive([[0, 0], [30, 20]], yaw0=-2.5, t_max=25.0)    # starts pointing the wrong way
    want = np.arctan2(20, 30)
    yaw = (veh.yaw + np.pi) % (2 * np.pi) - np.pi
    assert abs(((yaw - want) + np.pi) % (2 * np.pi) - np.pi) < 0.25
    # while the heading error was large the vehicle was slowed (so the forward sonar looks where it goes)
    early = [np.linalg.norm(l[2][:2]) for l in log[: int(2.0 / DT)]]
    assert max(early) < 0.6


def test_speed_scale_zero_brakes_the_vehicle():
    fol, veh, log, path = drive([[0, 0], [30, 0]], t_max=10.0)
    cfg = load_cfg()
    veh2 = NumpyVehicle(cfg, [0, 0, -5.0])
    fol2 = PathFollower(FollowerConfig.from_config(cfg, "world"))
    fol2.set_path(PathData.from_xy(np.array([[0.0, 0.0], [30.0, 0.0]]), -5.0, -5.0))
    for k in range(int(8.0 / DT)):
        veh2.step(fol2.compute(make_est(veh2, k * DT), 1.0).as_control(), DT)
    assert np.linalg.norm(veh2.v) > 0.5
    for k in range(int(12.0 / DT)):
        veh2.step(fol2.compute(make_est(veh2, 8.0 + k * DT), 0.0).as_control(), DT)
    assert np.linalg.norm(veh2.v) < 0.08


def test_progress_never_goes_backwards_and_is_projected_from_the_estimate():
    cfg = load_cfg()
    fol = PathFollower(FollowerConfig.from_config(cfg, "world"))
    fol.set_path(PathData.from_xy(np.array([[0.0, 0.0], [40.0, 0.0]]), -5.0, -5.0), start_s=19.0)   # progress window is bounded on purpose
    veh = NumpyVehicle(cfg, [20.0, 3.0, -5.0])
    fol.compute(make_est(veh, 0.0))
    assert fol.s == pytest.approx(20.0, abs=0.3) and fol.cross_track == pytest.approx(3.0, abs=1e-6)
    veh.p[:] = [10.0, 0.0, -5.0]                                              # estimate jumps back
    fol.compute(make_est(veh, 0.1))
    assert fol.s >= 19.7


def test_body_frame_command_rotates_the_world_force_by_the_ESTIMATED_yaw():
    cfg = load_cfg()
    world = FollowerConfig.from_config(cfg, "world")
    body = FollowerConfig.from_config(cfg, "body")
    path = PathData.from_xy(np.array([[0.0, 0.0], [30.0, 0.0]]), -5.0, -5.0)
    out = {}
    for name, c in (("world", world), ("body", body)):
        fol = PathFollower(c)
        fol.set_path(path)
        veh = NumpyVehicle(cfg, [0.0, 0.0, -5.0], yaw=np.pi / 2)              # heading +y, path along +x
        out[name] = fol.compute(make_est(veh, 0.0))
    w, b = out["world"], out["body"]
    assert w.forward > 0.1 and abs(w.lateral) < 0.05                          # world frame: push along +x
    assert abs(b.forward) < 0.05 and b.lateral < -0.1                         # body frame: +x is to the right of a +y heading
    assert np.hypot(w.forward, w.lateral) == pytest.approx(np.hypot(b.forward, b.lateral), rel=1e-6)


def test_hold_command_damps_motion_and_holds_depth():
    cfg = load_cfg()
    fol = PathFollower(FollowerConfig.from_config(cfg, "world"))
    veh = NumpyVehicle(cfg, [0.0, 0.0, -4.0])
    veh.v[:] = [0.5, -0.3, 0.0]
    cmd = fol.hold(make_est(veh, 0.0), target_z=-5.0)
    assert cmd.forward < 0 and cmd.lateral > 0 and cmd.vertical < 0           # brake x, brake y, descend toward -5


def test_follower_only_accepts_an_estimate_no_truth_parameter_exists():
    params = list(inspect.signature(PathFollower.compute).parameters)
    assert params == ["self", "estimate", "speed_scale"]
    assert list(inspect.signature(PathFollower.hold).parameters) == ["self", "estimate", "target_z"]


def test_follower_config_comes_from_the_simulation_config_and_validates():
    cfg = load_cfg()
    f = FollowerConfig.from_config(cfg)
    assert f.max_thrust_n == cfg["simulation"]["controller"]["max_thrust"]
    assert f.drag_feedforward == cfg["simulation"]["physics"]["linear_drag"]
    with pytest.raises(ValueError):
        FollowerConfig(command_frame="sideways")
    with pytest.raises(ValueError):
        FollowerConfig(cruise_speed_mps=0.0)


def test_the_gain_report_is_the_gains_and_schedule_that_were_really_applied():
    """The telemetry block the dashboard's gain panel reads: the loop gains in the units they are applied in, the
    live speed schedule as a fraction of cruise, and the real factors that produced it - all read from the command
    that was just computed, not copied from the config."""
    fol, veh, log, path = drive([[0, 0], [30, 0]])
    rep = fol.gain_report()
    c = fol.cfg
    assert rep["available"] is True and rep["source"] == "path follower (Phase 5)"
    assert rep["mode"] == "follow" and rep["command_frame"] == "world"
    assert rep["kv_n_per_mps"] == pytest.approx(c.kv)
    assert rep["kp_yaw_nm_per_rad"] == pytest.approx(c.kp_yaw)
    assert rep["kd_yaw_nm_per_rad_s"] == pytest.approx(c.kd_yaw)
    assert rep["kp_depth_mps_per_m"] == pytest.approx(c.kp_depth)
    assert rep["ki_depth_n_per_m_s"] == pytest.approx(c.ki_depth)
    assert rep["cruise_speed_mps"] == pytest.approx(c.cruise_speed_mps)
    # the schedule is live and bounded, and the speed really commanded matches the factor it reports
    for key in ("speed_factor", "goal_factor", "heading_factor", "sonar_factor"):
        assert 0.0 <= rep[key] <= 1.0 + 1e-9, (key, rep[key])
    last = log[-1][4]
    assert rep["desired_speed_mps"] == pytest.approx(np.linalg.norm(last.info["desired_velocity"][:2]), rel=1e-9)
    assert rep["speed_factor"] == pytest.approx(min(1.0, rep["desired_speed_mps"] / c.cruise_speed_mps), rel=1e-9)
    assert rep["goal_factor"] == pytest.approx(last.info["goal_factor"])
    assert rep["cross_track_m"] == pytest.approx(fol.cross_track)
    assert abs(rep["depth_integral_n"]) <= c.depth_i_limit_n               # the clamped depth-hold integral
    assert np.isfinite(rep["depth_integral_n"]) and np.isfinite(rep["lookahead_m"])
    assert set(rep["command"]) == {"forward", "lateral", "vertical", "yaw"}


def test_the_speed_schedule_really_moves_with_the_situation():
    """The schedule the panel charts is dynamic, not a constant: it drops to the goal-slowdown inside the last
    metres of the path and is scaled by the sonar/avoidance factor it is given."""
    fol, veh, log, path = drive([[0, 0], [30, 0]])
    near_goal = [l[4].info["goal_factor"] for l in log if np.linalg.norm(l[1][:2] - path.xy[-1]) < 1.0]
    early = [l[4].info["goal_factor"] for l in log if np.linalg.norm(l[1][:2]) < 5.0]
    assert near_goal and early
    assert max(near_goal) < min(early)                     # the goal slowdown really engaged near the end
    assert fol.gain_report()["goal_factor"] <= 1.0
    # a halved speed scale halves the commanded speed through the sonar/avoidance factor
    fol2 = PathFollower(fol.cfg)
    fol2.set_path(PathData.from_xy(np.asarray([[0, 0], [30, 0]], float), -5.0, -5.0))
    est = make_est(NumpyVehicle(load_cfg(), [2.0, 0.0, -5.0]), 0.0)
    full = fol2.compute(est, 1.0)
    half = fol2.compute(est, 0.5)
    assert half.info["sonar_factor"] == pytest.approx(0.5)
    assert np.linalg.norm(half.info["desired_velocity"][:2]) < np.linalg.norm(full.info["desired_velocity"][:2])
    assert fol2.gain_report()["speed_factor"] == pytest.approx(min(1.0, np.linalg.norm(half.info["desired_velocity"][:2]) / fol2.cfg.cruise_speed_mps))


def test_the_tracked_schedule_history_adds_up_to_the_commanded_time():
    fol, veh, log, path = drive([[0, 0], [30, 0]])
    tr = fol.tracking
    assert tr["steps"] > 100 and tr["elapsed_s"] > 0.0
    assert tr["follow_s"] + tr["hold_s"] + tr["station_s"] == pytest.approx(tr["elapsed_s"], rel=1e-9)
    assert tr["full_speed_s"] + tr["slowed_s"] + tr["braked_s"] == pytest.approx(tr["elapsed_s"], rel=1e-9)
    assert tr["mode"] == "follow"
    assert 0.0 <= tr["speed_factor_min"] <= tr["speed_factor_mean"] <= 1.0 + 1e-9
    assert 0.0 <= tr["speed_factor_now"] <= 1.0 + 1e-9
    assert tr["slowed_s"] > 0.0 and tr["slows"] >= 1                # it really had to slow down for the goal
    assert tr["braked_s"] >= 0.0 and tr["brakes"] >= 0
    assert tr["heading_factor_min"] <= 1.0 and tr["sonar_factor_min"] <= 1.0
    assert fol.gain_report()["tracking"]["steps"] == tr["steps"]


def test_a_slow_down_episode_is_counted_once_and_a_new_mission_clears_the_tracking():
    fol = PathFollower(FollowerConfig.from_config(load_cfg(), "world"))
    fol.set_path(PathData.from_xy(np.asarray([[0, 0], [30, 0]], float), -5.0, -5.0))
    veh = NumpyVehicle(load_cfg(), [2.0, 0.0, -5.0])
    for k in range(20):                                              # hold the scale at 0.5: one episode, not twenty
        fol.compute(make_est(veh, k * DT), 0.5)
    assert fol.tracking["slows"] == 1 and fol.tracking["slowed_s"] > 0.0
    assert fol.tracking["brakes"] == 0
    for k in range(20, 40):                                          # scale 0: the vehicle is held, counted once
        fol.compute(make_est(veh, k * DT), 0.0)
    assert fol.tracking["brakes"] == 1 and fol.tracking["braked_s"] > 0.0
    assert fol.tracking["steps"] in (39, 40)                        # the first command has no dt, so it cannot be timed
    fol.reset_tracking()                                             # a new navigation mission must not inherit it
    assert fol.tracking["steps"] == 0 and fol.tracking["slows"] == 0 and fol.tracking["brakes"] == 0
    assert fol.tracking["elapsed_s"] == 0.0 and fol.gain_report()["tracking"]["steps"] == 0


def test_path_data_geometry():
    p = PathData.from_xy(np.array([[0.0, 0.0], [3.0, 4.0], [3.0, 10.0]]), -2.0, -8.0)
    assert p.length == pytest.approx(11.0) and p.n == 3
    assert p.point(5.0)[:2] == pytest.approx([3.0, 4.0]) and p.point(11.0)[2] == pytest.approx(-8.0)
    assert p.point(2.5)[2] == pytest.approx(-2.0 - 6.0 * 2.5 / 11.0)
    s, cross, q = p.project([5.0, 7.0])
    assert s == pytest.approx(8.0) and cross == pytest.approx(2.0) and q == pytest.approx([3.0, 7.0])
    assert p.next_vertex_index(1.0) == 1 and p.next_vertex_index(5.0) == 2
    seg = p.segment_xy(2.5, 8.0)
    assert seg[0] == pytest.approx([1.5, 2.0]) and seg[-1] == pytest.approx([3.0, 7.0]) and len(seg) == 3
    with pytest.raises(ValueError):
        PathData(np.zeros((3, 2)), np.zeros(2))

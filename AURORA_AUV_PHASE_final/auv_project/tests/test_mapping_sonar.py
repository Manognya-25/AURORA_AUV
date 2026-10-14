"""Sonar -> occupancy semantics, extrinsics, EKF-pose use, and the end-to-end chain.

These tests use the REAL SonarMeasurement type, the REAL SonarMapper/LogOddsGrid/MappingSystem and (for the
end-to-end chain) the REAL Sonar sensor model, IMU/DVL/depth sensors and 15-state EKF. Only the ray cast is
analytic (tests/helpers/analytic_world.py) because PyBullet is not installable in every environment; the
same chain against real PyBullet geometry is tests/test_pybullet_world.py.
"""
import numpy as np
import pytest
import yaml

from src.localization.ekf import EKF, EKFConfig
from src.mapping.coordinate_utils import GridGeometry
from src.mapping.log_odds_grid import LogOddsGrid
from src.mapping.map_manager import MappingSystem
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, MapConfig
from src.mapping.sonar_mapper import SonarMapper
from src.sensors.health import SensorHealth
from src.sensors.models import SonarMeasurement, TruthSnapshot
from src.sensors.sonar import SonarConfig
from src.simulation.estimation import EstimationPipeline
from src.utils.transforms import euler_to_quat, quat_to_rot
from tests.helpers.analytic_sonar import AnalyticSonar
from tests.helpers.analytic_world import AnalyticWorld

I3 = np.eye(3)


def cfg_map(**kw):
    base = dict(resolution_m=0.25, width_m=40.0, height_m=40.0, origin_x_m=-5.0, origin_y_m=-20.0,
                slice_half_height_m=1.0, max_update_range_m=30.0)
    base.update(kw)
    return MapConfig(**base)


def scan(ranges, az_deg, detected, ray_valid=None, el_deg=None, valid=True, health=SensorHealth.HEALTHY,
         mount_pos=(0, 0, 0), mount_rot=I3, max_range=30.0):
    az = np.deg2rad(np.asarray(az_deg, float))
    n = len(az)
    el = np.deg2rad(np.zeros(n) if el_deg is None else np.asarray(el_deg, float))
    return SonarMeasurement(
        sensor="sonar", timestamp=0.0, value=np.asarray(ranges, float), valid=valid, health=health,
        azimuths=az, elevations=el, detected=np.asarray(detected, bool), points_sensor=np.full((n, 3), np.nan),
        max_range=max_range, min_range=0.2, ray_valid=None if ray_valid is None else np.asarray(ray_valid, bool),
        mount_position=np.asarray(mount_pos, float), mount_rotation=np.asarray(mount_rot, float))


def integrate(meas, pos=(0.0, 0.0, -5.0), R=I3, cfg=None, times=1):
    cfg = cfg or cfg_map()
    grid, mapper = LogOddsGrid(cfg), SonarMapper(cfg)
    st = None
    for _ in range(times):
        st = mapper.integrate(grid, meas, np.asarray(pos, float), R)
    return grid, st


def cell(grid, x, y):
    r, c = grid.geom.world_to_grid(x, y)
    return int(r), int(c)


def state_at(grid, x, y):
    r, c = cell(grid, x, y)
    return int(grid.state()[r, c])


# ------------------------------------------------------------------ hit / free / unknown-behind
def test_hit_marks_free_before_and_occupied_at_endpoint_and_leaves_beyond_unknown():
    g, st = integrate(scan([10.0], [0.0], [True]), times=6)
    assert state_at(g, 10.0, 0.0) == OCCUPIED
    for x in (1.0, 4.0, 8.0, 9.5):
        assert state_at(g, x, 0.0) == FREE
    for x in (10.6, 12.0, 20.0):                       # behind the obstacle: the sonar says nothing
        assert state_at(g, x, 0.0) == UNKNOWN
    assert state_at(g, 5.0, 8.0) == UNKNOWN            # untouched space stays unknown
    assert st.rays_hit == 1 and st.hits_mapped == 1


def test_no_hit_ray_marks_free_up_to_max_range_and_invents_no_obstacle():
    g, st = integrate(scan([30.0], [0.0], [False]), times=8)
    assert not (g.state() == OCCUPIED).any()
    assert state_at(g, 10.0, 0.0) == FREE and state_at(g, 28.0, 0.0) == FREE
    assert state_at(g, 31.0, 0.0) == UNKNOWN           # nothing beyond the valid measurement range
    assert st.rays_no_return == 1 and st.rays_hit == 0


def test_no_return_free_evidence_is_weaker_than_a_hit_ray_free_evidence():
    g_hit, _ = integrate(scan([20.0], [0.0], [True]))
    g_nr, _ = integrate(scan([30.0], [0.0], [False]))
    r, c = cell(g_hit, 5.0, 0.0)
    assert g_nr.log_odds[r, c] > g_hit.log_odds[r, c]      # both negative; no-return is closer to 0


def test_max_update_range_limits_integration():
    cfg = cfg_map(max_update_range_m=12.0)
    g, _ = integrate(scan([30.0], [0.0], [False]), cfg=cfg, times=8)
    assert state_at(g, 10.0, 0.0) == FREE
    assert state_at(g, 14.0, 0.0) == UNKNOWN


def test_hit_beyond_max_update_range_is_not_mapped_as_obstacle():
    cfg = cfg_map(max_update_range_m=12.0)
    g, st = integrate(scan([20.0], [0.0], [True]), cfg=cfg, times=8)
    assert not (g.state() == OCCUPIED).any() and st.hits_mapped == 0


# ------------------------------------------------------------------ dropout / invalid
def test_dropped_ray_updates_nothing():
    m = scan([10.0, 10.0], [-20.0, 20.0], [True, True], ray_valid=[False, True])
    g, st = integrate(m, times=6)
    assert st.rays_invalid == 1 and st.rays_valid == 1
    assert state_at(g, 9.0, 9.0 * np.tan(np.deg2rad(-20))) == UNKNOWN       # dropped ray: no free, no hit
    assert state_at(g, 9.0, 9.0 * np.tan(np.deg2rad(20))) == FREE           # valid ray did update


def test_scan_level_invalid_or_failed_sensor_leaves_map_unchanged():
    for m in (scan([10.0], [0.0], [True], valid=False), scan([10.0], [0.0], [True], health=SensorHealth.FAILED)):
        g, st = integrate(m)
        assert not st.processed and st.reason == "sonar_invalid_or_failed"
        assert g.update_count == 0 and (g.log_odds == 0).all() and (g.state() == UNKNOWN).all()


def test_nonfinite_pose_is_rejected_without_touching_the_map():
    g, st = integrate(scan([10.0], [0.0], [True]), pos=(np.nan, 0, -5))
    assert not st.processed and st.reason == "pose_not_finite" and (g.log_odds == 0).all()


def test_min_range_blanked_hit_is_invalid_not_an_obstacle():
    # a Sonar hit closer than min_range is ray_valid=False in the real sensor model (see test below)
    m = scan([0.1], [0.0], [False], ray_valid=[False])
    g, st = integrate(m, times=4)
    assert st.rays_invalid == 1 and (g.state() == UNKNOWN).all()


def test_real_sonar_model_reports_dropouts_and_blanking_as_invalid_rays():
    world = AnalyticWorld(boxes=[((0.95, 0, -5), (0.05, 5, 5))])             # wall 0.15 m in front of the sensor
    cfg = SonarConfig(horizontal_rays=5, vertical_rays=1, range_noise_std_m=0.0, dropout_probability=0.0,
                      detection_probability=1.0, max_range_m=30.0, min_range_m=0.5, mount_position_m=[0.8, 0, 0])
    t = TruthSnapshot(0.0, np.array([0, 0, -5.0]), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3))
    m = AnalyticSonar(world, cfg).measure(t)
    assert m.ray_status().tolist() == [-1, -1, -1, -1, -1]                    # all blanked: no information
    cfg2 = SonarConfig(horizontal_rays=5, vertical_rays=1, range_noise_std_m=0.0, dropout_probability=1.0,
                       detection_probability=1.0)
    m2 = AnalyticSonar(AnalyticWorld(), cfg2).measure(t)
    assert (m2.ray_status() == -1).all() and not m2.valid


# ------------------------------------------------------------------ frames / extrinsics / pose
def test_world_ray_direction_uses_estimated_yaw():
    R90 = quat_to_rot(euler_to_quat(0, 0, np.pi / 2))                        # vehicle faces +Y
    g, _ = integrate(scan([8.0], [0.0], [True]), pos=(0, 0, -5), R=R90, times=6)
    assert state_at(g, 0.0, 8.0) == OCCUPIED and state_at(g, 8.0, 0.0) == UNKNOWN


def test_azimuth_is_positive_toward_left_plus_y():
    g, _ = integrate(scan([8.0], [30.0], [True]), times=6)
    assert state_at(g, 8.0 * np.cos(np.deg2rad(30)), 8.0 * np.sin(np.deg2rad(30))) == OCCUPIED
    assert state_at(g, 8.0 * np.cos(np.deg2rad(30)), -8.0 * np.sin(np.deg2rad(30))) == UNKNOWN


def test_nonzero_mount_position_offsets_the_ray_origin():
    g, _ = integrate(scan([5.0], [0.0], [True], mount_pos=(0.8, 0.5, 0.0)), pos=(2.0, 0.0, -5), times=6)
    # origin = (2.8, 0.5) -> hit at (7.8, 0.5)
    assert state_at(g, 7.8, 0.5) == OCCUPIED and state_at(g, 7.8, 0.0) != OCCUPIED


def test_nonzero_mount_rotation_turns_the_beam_in_the_body_frame():
    Rm = quat_to_rot(euler_to_quat(0, 0, np.deg2rad(90)))                    # sonar looks to the vehicle's left
    g, _ = integrate(scan([6.0], [0.0], [True], mount_rot=Rm), times=6)
    assert state_at(g, 0.0, 6.0) == OCCUPIED and state_at(g, 6.0, 0.0) == UNKNOWN


def test_mount_and_pose_compose_in_the_right_order():
    Rm = quat_to_rot(euler_to_quat(0, 0, np.deg2rad(90)))
    Rv = quat_to_rot(euler_to_quat(0, 0, np.deg2rad(90)))                    # vehicle yaw 90 + mount yaw 90 = 180
    g, _ = integrate(scan([5.0], [0.0], [True], mount_pos=(1.0, 0, 0), mount_rot=Rm), pos=(10.0, 0, -5), R=Rv, times=6)
    # origin = (10,0) + Rv*(1,0,0) = (10, 1);  direction = Rv*Rm*x = -x  ->  hit at (5, 1)
    assert state_at(g, 5.0, 1.0) == OCCUPIED


def test_slice_ignores_rays_far_above_or_below_the_vehicle_depth():
    # steep downward ray: hits the seabed 10 m below. Map slice is +/-1 m, so no obstacle is created.
    m = scan([20.0], [0.0], [True], el_deg=[-30.0])
    g, st = integrate(m, times=6)
    assert not (g.state() == OCCUPIED).any() and st.hits_mapped == 0
    # slice is +/-1 m: a -30 deg ray leaves it after 2 m of range (x = 1.73 m)
    assert state_at(g, 1.0, 0.0) == FREE                                      # in-slice part still counts as free
    assert state_at(g, 3.0, 0.0) == UNKNOWN                                   # out-of-slice part says nothing


def test_mapper_uses_the_pose_it_is_given_not_any_other():
    m = scan([8.0], [0.0], [True])
    ga, _ = integrate(m, pos=(0, 0, -5), times=6)
    gb, _ = integrate(m, pos=(0, 3, -5), times=6)
    assert state_at(ga, 8.0, 0.0) == OCCUPIED and state_at(gb, 8.0, 0.0) != OCCUPIED
    assert state_at(gb, 8.0, 3.0) == OCCUPIED


# ------------------------------------------------------------------ MappingSystem behaviour
def project_cfg(**map_over):
    c = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    c["mapping"].update(map_over)
    return c


class Est:                                  # minimal stand-in for EstimatedState (only what the mapper reads)
    def __init__(self, t, pos, yaw=0.0):
        self.time, self.position, self.rotation = t, np.asarray(pos, float), quat_to_rot(euler_to_quat(0, 0, yaw))


def test_mapping_system_rate_limits_dedupes_and_rejects_stale_pose():
    ms = MappingSystem(project_cfg(update_rate_hz=5.0))
    s = scan([10.0], [0.0], [True]); s.timestamp = 1.0
    assert ms.process(1.0, s, Est(1.0, (0, 0, -5)))
    assert not ms.process(1.0, s, Est(1.0, (0, 0, -5)))                       # same scan again: ignored
    s2 = scan([10.0], [0.0], [True]); s2.timestamp = 1.05
    assert not ms.process(1.05, s2, Est(1.05, (0, 0, -5)))                    # faster than update_rate_hz
    assert ms.stats["scans_skipped_rate"] == 1
    s3 = scan([10.0], [0.0], [True]); s3.timestamp = 2.0
    assert not ms.process(2.0, s3, Est(1.0, (0, 0, -5)))                      # estimate is 1 s old
    assert ms.last_reject_reason == "stale_pose" and ms.stats["scans_rejected_stale_pose"] == 1


def test_disabled_mapper_does_nothing_and_enable_resumes():
    ms = MappingSystem(project_cfg())
    ms.set_enabled(False)
    s = scan([10.0], [0.0], [True]); s.timestamp = 1.0
    assert not ms.process(1.0, s, Est(1.0, (0, 0, -5))) and ms.stats["map_updates"] == 0
    ms.set_enabled(True)
    s.timestamp = 2.0
    assert ms.process(2.0, s, Est(2.0, (0, 0, -5))) and ms.stats["map_updates"] == 1


def test_reset_returns_to_unknown_and_zeroes_counters_and_cost_map():
    ms = MappingSystem(project_cfg())
    for k in range(8):
        s = scan([10.0], [0.0], [True]); s.timestamp = 1.0 + k
        ms.process(1.0 + k, s, Est(1.0 + k, (0, 0, -5)))
    assert ms.telemetry()["mapping"]["occupied_cells"] > 0 and ms.cost_map.stats["inflated_count"] > 0
    v = ms.version
    ms.reset()
    t = ms.telemetry()
    assert t["mapping"]["known_cells"] == 0 and t["mapping"]["update_count"] == 0
    assert t["mapping"]["occupied_cells"] == 0 and t["cost_map"]["occupied_count"] == 0 and t["cost_map"]["inflated_count"] == 0
    assert t["sonar_mapping"]["valid_returns"] == 0 and t["sonar_mapping"]["rays_processed"] == 0
    assert ms.version > v                                                     # cache key never repeats


def test_telemetry_is_computed_from_the_grid_and_is_strict_json():
    import json
    from src.utils.jsonsafe import jsonsafe
    ms = MappingSystem(project_cfg())
    for k in range(6):
        s = scan([10.0, 30.0], [0.0, 25.0], [True, False]); s.timestamp = 1.0 + k
        ms.process(1.0 + k, s, Est(1.0 + k, (0, 0, -5)))
    t = ms.telemetry()
    st = ms.grid.state()
    assert t["mapping"]["free_cells"] == int((st == FREE).sum()) and t["mapping"]["occupied_cells"] == int((st == OCCUPIED).sum())
    assert t["mapping"]["known_cells"] + t["mapping"]["unknown_cells"] == st.size
    json.dumps(jsonsafe(t), allow_nan=False)
    json.dumps(jsonsafe(ms.occupancy_payload()), allow_nan=False)
    json.dumps(jsonsafe(ms.costmap_payload()), allow_nan=False)


def test_rle_payload_roundtrips_to_the_actual_grid():
    from src.mapping.map_manager import rle_decode
    ms = MappingSystem(project_cfg())
    for k in range(6):
        s = scan([10.0], [0.0], [True]); s.timestamp = 1.0 + k
        ms.process(1.0 + k, s, Est(1.0 + k, (0, 0, -5)))
    p = ms.occupancy_payload()
    dec = rle_decode(p["data"], tuple(p["shape"]))
    st = ms.grid.state()
    assert ((dec == 2) == (st == OCCUPIED)).all() and ((dec == 1) == (st == FREE)).all() and ((dec == 0) == (st == UNKNOWN)).all()


# ------------------------------------------------------------------ end-to-end: sensors -> EKF -> mapper
def e2e_cfg():
    c = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    c["sensors"]["sonar"].update(dropout_probability=0.0, range_noise_std_m=0.05, detection_probability=1.0)
    c["mapping"].update(update_rate_hz=5.0)
    return c


def run_e2e(cfg, world, seconds=8.0, speed=0.5, start=(5.0, 5.0, -5.0)):
    """Drive a kinematic truth (test-only) through the REAL sensors + EKF + mapper."""
    dt = cfg["simulation"]["time_step"]
    pipe = EstimationPipeline(cfg, 0, start)
    pipe.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
    ms = MappingSystem(cfg)
    pos = np.array(start, float)
    trace = []
    for k in range(int(seconds / dt)):
        pos = pos + np.array([speed * dt, 0.0, 0.0])
        t = TruthSnapshot(k * dt, pos.copy(), np.array([0, 0, 0, 1.0]), np.array([speed, 0, 0.0]), np.zeros(3))
        meas, est = pipe.update_from_truth(t)
        ms.process(k * dt, meas.get("sonar"), est)
        trace.append((pos.copy(), est.position.copy()))
    return ms, pipe, trace


def test_end_to_end_unknown_to_free_to_occupied_to_inflated_cost():
    world = AnalyticWorld(cylinders=[((15.0, 5.0, -5.0), 0.8, 8.0)])            # pillar 10 m ahead of the start
    ms, pipe, trace = run_e2e(e2e_cfg(), world)
    st = ms.grid.state()
    g = ms.grid.geom
    r, c = g.world_to_grid(14.2, 5.0)                                           # near face of the pillar
    assert st[int(r), int(c)] == OCCUPIED or (st[int(r) - 1:int(r) + 2, int(c) - 1:int(c) + 2] == OCCUPIED).any()
    r, c = g.world_to_grid(8.0, 5.0)
    assert st[int(r), int(c)] == FREE                                           # observed free along the ray
    r, c = g.world_to_grid(5.0, 40.0)
    assert st[int(r), int(c)] == UNKNOWN                                        # never observed
    r, c = g.world_to_grid(18.0, 5.0)
    assert st[int(r), int(c)] == UNKNOWN                                        # behind the pillar
    cm = ms.cost_map
    assert cm.stats["occupied_count"] > 0 and cm.stats["inflated_count"] > 0
    occ_rc = np.argwhere(st == OCCUPIED)
    inflated = cm.cls == 3
    near = [(rr + dr, cc + dc) for rr, cc in occ_rc[:5] for dr in (-1, 0, 1) for dc in (-1, 0, 1)]
    assert any(inflated[a, b] for a, b in near if 0 <= a < inflated.shape[0] and 0 <= b < inflated.shape[1])
    # the mapped obstacle is where the pillar really is, to within EKF drift + one pillar radius (assertion only)
    mapped = g.grid_to_world(occ_rc[:, 0], occ_rc[:, 1])
    cx, cy = float(np.mean(mapped[0])), float(np.mean(mapped[1]))
    assert abs(cx - 14.2) < 1.2 and abs(cy - 5.0) < 1.2


def test_mapper_is_driven_by_the_ekf_estimate_not_truth():
    """Same truth, same sensors; the only difference is a deliberately wrong estimated pose.
    The map must follow the ESTIMATE."""
    world = AnalyticWorld(cylinders=[((15.0, 5.0, -5.0), 0.8, 8.0)])
    cfg = e2e_cfg()
    pipe = EstimationPipeline(cfg, 0, (5.0, 5.0, -5.0))
    pipe.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
    t = TruthSnapshot(0.0, np.array([5.0, 5.0, -5.0]), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3))
    meas, est = pipe.update_from_truth(t)
    sonar = meas["sonar"]
    maps = {}
    for label, dy in (("honest", 0.0), ("shifted", 6.0)):
        ms = MappingSystem(cfg)
        for k in range(8):
            s = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"])).measure(
                TruthSnapshot(1.0 + k, t.position, t.quat, t.velocity_world, t.angular_velocity_world))
            e = Est(1.0 + k, est.position + np.array([0, dy, 0]))
            ms.process(1.0 + k, s, e)
        maps[label] = ms.grid.state()
    a, b = np.argwhere(maps["honest"] == OCCUPIED), np.argwhere(maps["shifted"] == OCCUPIED)
    assert len(a) and len(b)
    g = MappingSystem(cfg).grid.geom
    ya = g.grid_to_world(a[:, 0], a[:, 1])[1].mean(); yb = g.grid_to_world(b[:, 0], b[:, 1])[1].mean()
    assert yb - ya == pytest.approx(6.0, abs=0.3)


def test_sonar_noise_changes_the_map_but_not_its_structure():
    world = AnalyticWorld(cylinders=[((15.0, 5.0, -5.0), 0.8, 8.0)])
    outs = []
    for std in (0.0, 0.5):
        c = e2e_cfg(); c["sensors"]["sonar"]["range_noise_std_m"] = std
        ms, _, _ = run_e2e(c, world, seconds=6.0)
        outs.append(ms.grid.log_odds.copy())
    assert not np.array_equal(outs[0], outs[1])                                  # noise really reaches the map
    assert ((outs[0] > 0).sum() > 0) and ((outs[1] > 0).sum() > 0)


def test_sonar_dropout_never_creates_obstacles_and_reduces_updates():
    world = AnalyticWorld()                                                      # empty world
    c = e2e_cfg(); c["sensors"]["sonar"]["dropout_probability"] = 0.5
    ms, _, _ = run_e2e(c, world, seconds=6.0)
    assert not (ms.grid.state() == OCCUPIED).any()
    assert ms.stats["dropouts"] > 0
    c0 = e2e_cfg()
    ms0, _, _ = run_e2e(c0, world, seconds=6.0)
    assert ms.stats["cells_updated"] < ms0.stats["cells_updated"]


def test_total_sonar_failure_leaves_map_unknown():
    world = AnalyticWorld(cylinders=[((15.0, 5.0, -5.0), 0.8, 8.0)])
    cfg = e2e_cfg()
    pipe = EstimationPipeline(cfg, 0, (5.0, 5.0, -5.0))
    pipe.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
    pipe.suite.sensors["sonar"].inject_fault("FAILED")
    ms = MappingSystem(cfg)
    for k in range(300):
        t = TruthSnapshot(k * 0.01, np.array([5.0, 5.0, -5.0]), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3))
        meas, est = pipe.update_from_truth(t)
        ms.process(k * 0.01, meas.get("sonar"), est)
    assert (ms.grid.state() == UNKNOWN).all() and ms.stats["map_updates"] == 0 and ms.stats["scans_rejected"] > 0


# ------------------------------------------------------------------ platform-independence of exact-grid-line beams
@pytest.mark.parametrize("eps", [-6e-16, 0.0, 6e-16])
def test_beam_on_a_grid_line_is_not_flipped_by_ulp_level_rotation_residue(eps):
    """cos(90deg) is ~6e-17 with either sign depending on the platform libm. A beam along x=0 / y=1.0 must land
    in the same cell either way (this was the Windows-only 'expected 100, got -1' failure)."""
    R = np.array([[eps, -1.0, 0.0], [1.0, eps, 0.0], [0.0, 0.0, 1.0]])           # yaw +90deg with residue
    g, _ = integrate(scan([6.0], [0.0], [True]), R=R, times=6)
    assert state_at(g, 0.0, 6.0) == OCCUPIED and state_at(g, 6.0, 0.0) == UNKNOWN
    Rm = np.array([[-1.0, eps, 0.0], [-eps, -1.0, 0.0], [0.0, 0.0, 1.0]])       # yaw 180deg with residue
    g2, _ = integrate(scan([5.0], [0.0], [True], mount_pos=(1.0, 0, 0), mount_rot=Rm), pos=(10.0, 0, -5), R=R, times=6)
    # (R @ Rm) x-axis = R[:,0]*(-1) + ... : beam points along -y after the 90deg vehicle yaw; origin = (10, 1)
    assert (g2.state() == OCCUPIED).sum() >= 1

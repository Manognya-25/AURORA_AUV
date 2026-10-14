"""Sonar -> EKF pose -> occupancy grid -> cost map.

Uses the PRODUCT Sonar model, SensorSuite, EKF, SonarMapper, MappingSystem. The only substitution is
the ray caster (tests/helpers/analytic_*) because PyBullet may be unavailable; the same scenarios against
real PyBullet geometry live in tests/test_pybullet_world.py.
"""
import numpy as np
import pytest
import yaml

from src.localization.state import EstimatedState
from src.mapping.coordinate_utils import GridGeometry
from src.mapping.inflation import clearance_cost, obstacle_distance_m
from src.mapping.map_manager import MappingSystem, rle_decode, rle_encode
from src.mapping.map_types import FREE, OCCUPIED, UNKNOWN, CostMapConfig
from src.mapping.online_cost_map import CLASS_INFLATED, build_cost_map
from src.sensors.health import SensorHealth
from src.sensors.models import TruthSnapshot
from src.sensors.sonar import SonarConfig
from src.utils.transforms import euler_to_quat, quat_to_rot
from tests.helpers.analytic_sonar import AnalyticSonar
from tests.helpers.analytic_world import AnalyticWorld

CFG = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))


def sonar_cfg(**kw):
    base = dict(dropout_probability=0.0, detection_probability=1.0, range_noise_std_m=0.0,
                horizontal_rays=21, vertical_rays=1, vertical_fov_deg=0.0, max_range_m=30.0, seed=1)
    base.update(kw)
    return SonarConfig(**base)


def truth(t, pos, yaw=0.0):
    return TruthSnapshot(t, pos, euler_to_quat(0, 0, yaw), [0, 0, 0], [0, 0, 0])


def est_at(t, pos, yaw=0.0):
    return EstimatedState(t, np.asarray(pos, float), np.zeros(3), euler_to_quat(0, 0, yaw), np.zeros(3),
                          np.zeros(3), np.zeros(3), np.eye(15) * 0.01)


PILLAR_X, PILLAR_R = 10.0, 0.8
FACE_X = PILLAR_X - PILLAR_R          # x of the pillar's near face (9.2 m) -- the cell that CONTAINS it is the hit cell


def pillar_world(x=10.0, y=0.0, r=0.8):
    return AnalyticWorld(cylinders=[((x, y, -5.0), r, 10.0)], seabed_z=-15.0)


def run_scans(ms, sonar, true_pos, est_pos, n=8, yaw=0.0, est_yaw=None):
    for k in range(n):
        t = 0.2 * k
        meas = sonar.measure(truth(t, true_pos, yaw))
        ms.process(t, meas, est_at(t, est_pos, yaw if est_yaw is None else est_yaw))
    return ms


def cell_state(ms, x, y):
    r, c = ms.grid.geom.world_to_grid(x, y)
    return int(ms.grid.state()[int(r), int(c)])


# ------------------------------------------------------------------ the core behaviour
def test_unknown_to_free_to_occupied_with_unknown_behind_the_obstacle():
    ms = MappingSystem(CFG)
    assert ms.telemetry()["mapping"]["known_cells"] == 0                      # starts entirely unknown
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5])
    assert cell_state(ms, 5.0, 0.0) == FREE                                   # observed ray -> free
    assert cell_state(ms, FACE_X, 0.0) == OCCUPIED                            # detected obstacle face -> occupied
    assert cell_state(ms, 20.0, 0.0) == UNKNOWN                               # behind the obstacle: never claimed free
    assert cell_state(ms, 0.0, 20.0) == UNKNOWN                               # outside the field of view
    t = ms.telemetry()
    assert t["mapping"]["occupied_cells"] >= 1 and t["mapping"]["free_cells"] > 100
    assert t["mapping"]["unknown_cells"] > 40000                              # the overwhelming majority is still unknown


def test_cost_map_inflates_the_mapped_obstacle_and_keeps_belief_separate():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5])
    cm = ms.cost_map
    state = ms.grid.state()
    occ = state == OCCUPIED
    assert cm.stats["occupied_count"] == int(occ.sum()) > 0
    assert cm.stats["inflated_count"] > 0
    dist = obstacle_distance_m(occ, ms.grid.geom.resolution)
    rad = ms.cost_cfg.inflation_radius_m
    assert rad == pytest.approx(0.35 + 0.25)                                  # vehicle radius + safety margin
    assert np.all(cm.cls[(dist <= rad) & ~occ] == CLASS_INFLATED)             # every cell within the radius
    assert not np.any(cm.cls[(dist > rad)] == CLASS_INFLATED)                 # and nothing beyond it
    assert not cm.traversable[occ].any() and not cm.traversable[cm.cls == CLASS_INFLATED].any()
    assert (state == OCCUPIED).sum() == int(occ.sum())                        # the belief grid was not modified by inflation
    free_far = (state == FREE) & (dist > rad + 2.0)
    assert free_far.any() and np.all(cm.cost[free_far] == ms.cost_cfg.free_cost)


def test_mapper_uses_the_estimated_pose_not_the_true_pose():
    """True vehicle at x=0; the EKF estimate claims x=3. The obstacle (truly at x=10) must land at ~x=13."""
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [3, 0, -5])
    assert cell_state(ms, FACE_X + 3.0, 0.0) == OCCUPIED                      # shifted by the estimate error (+3 m)
    assert cell_state(ms, FACE_X, 0.0) != OCCUPIED                            # where the TRUE pose would have put it


def test_estimated_yaw_rotates_the_map_projection():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5], est_yaw=np.pi / 2, yaw=0.0)
    # sonar saw an obstacle dead ahead (true yaw 0) but the estimate says "facing +Y" -> mapped along +Y
    assert cell_state(ms, 0.0, FACE_X) == OCCUPIED
    assert cell_state(ms, FACE_X, 0.0) != OCCUPIED


def test_nonzero_sonar_mount_extrinsics_are_applied():
    cfg = sonar_cfg(mount_position_m=[0.0, 2.0, 0.0], mount_rpy_deg=[0.0, 0.0, 30.0])
    w = AnalyticWorld(cylinders=[((0.0, 12.0, -5.0), 0.6, 10.0)], seabed_z=-15.0)        # straight ahead of the mount, +30 deg
    s = AnalyticSonar(pillar_world(), cfg)
    m = s.measure(truth(0.0, [0, 0, -5]))
    assert np.allclose(m.mount_position, [0, 2, 0]) and np.allclose(m.mount_rotation, quat_to_rot(euler_to_quat(0, 0, np.deg2rad(30))))
    # a pillar placed along the mounted sonar's boresight: origin (0,2) + t*(cos30, sin30)
    px, py = 0.0 + 12 * np.cos(np.deg2rad(30)), 2.0 + 12 * np.sin(np.deg2rad(30))
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(AnalyticWorld(cylinders=[((px, py, -5.0), 0.6, 10.0)], seabed_z=-15.0), cfg),
              [0, 0, -5], [0, 0, -5])
    # near face of the pillar along the boresight
    fx, fy = px - 0.6 * np.cos(np.deg2rad(30)), py - 0.6 * np.sin(np.deg2rad(30))
    assert cell_state(ms, fx, fy) == OCCUPIED
    # ignoring the extrinsics (mount at origin, no rotation) would have put it elsewhere
    assert cell_state(ms, fx - 0.0, fy - 2.0) != OCCUPIED


# ------------------------------------------------------------------ dropout / no-hit / invalid
def test_dropout_rays_change_nothing():
    ms = MappingSystem(CFG)
    s = AnalyticSonar(pillar_world(), sonar_cfg(dropout_probability=1.0))
    run_scans(ms, s, [0, 0, -5], [0, 0, -5])
    t = ms.telemetry()
    assert t["mapping"]["known_cells"] == 0 and t["mapping"]["occupied_cells"] == 0
    assert (ms.grid.log_odds == 0).all()                                      # not even a sub-threshold nudge
    assert t["sonar_mapping"]["valid_returns"] == 0


def test_failed_sonar_is_ignored_and_counted():
    ms = MappingSystem(CFG)
    s = AnalyticSonar(pillar_world(), sonar_cfg())
    s.inject_fault(SensorHealth.FAILED)
    run_scans(ms, s, [0, 0, -5], [0, 0, -5], n=3)
    assert (ms.grid.log_odds == 0).all()
    assert ms.stats["scans_rejected"] == 3 and ms.last_reject_reason == "sonar_invalid_or_failed"


def test_partial_dropout_degrades_but_does_not_invent_obstacles():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg(dropout_probability=0.3, seed=3)), [0, 0, -5], [0, 0, -5], n=25)
    assert cell_state(ms, FACE_X, 0.0) == OCCUPIED
    state = ms.grid.state()
    X, Y = ms.grid.geom.grid_to_world(*np.mgrid[0:ms.grid.geom.rows, 0:ms.grid.geom.cols])
    far_from_pillar = (np.hypot(X - 10.0, Y) > 2.0)
    assert not np.any(state[far_from_pillar] == OCCUPIED)                      # every occupied cell is near the real pillar


def test_no_hit_rays_mark_free_space_but_never_obstacles_or_beyond_range():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(AnalyticWorld(seabed_z=-15.0), sonar_cfg()), [0, 0, -5], [0, 0, -5], n=12)
    t = ms.telemetry()
    assert t["mapping"]["occupied_cells"] == 0 and t["mapping"]["free_cells"] > 100
    assert cell_state(ms, 12.0, 0.0) == FREE
    assert cell_state(ms, 33.0, 0.0) == UNKNOWN                                # beyond max range (30 m from x=0.8 mount)
    assert t["sonar_mapping"]["no_returns"] > 0 and t["sonar_mapping"]["valid_returns"] == 0


def test_missed_detection_is_not_an_obstacle_and_not_full_free_evidence():
    ms = MappingSystem(CFG)
    s = AnalyticSonar(pillar_world(), sonar_cfg(detection_probability=0.0))
    run_scans(ms, s, [0, 0, -5], [0, 0, -5], n=1)
    assert ms.telemetry()["mapping"]["occupied_cells"] == 0
    assert cell_state(ms, 5.0, 0.0) == UNKNOWN                                 # one weak no-return scan is not yet 'free'
    run_scans(ms, s, [0, 0, -5], [0, 0, -5], n=12)
    assert cell_state(ms, 5.0, 0.0) == FREE


def test_stale_pose_scan_is_rejected():
    ms = MappingSystem(CFG)
    s = AnalyticSonar(pillar_world(), sonar_cfg())
    m = s.measure(truth(1.0, [0, 0, -5]))
    assert ms.process(1.0, m, est_at(0.5, [0, 0, -5])) is False
    assert ms.stats["scans_rejected_stale_pose"] == 1 and (ms.grid.log_odds == 0).all()


def test_non_finite_pose_never_corrupts_the_map():
    ms = MappingSystem(CFG)
    m = AnalyticSonar(pillar_world(), sonar_cfg()).measure(truth(0.0, [0, 0, -5]))
    ms.process(0.0, m, est_at(0.0, [np.nan, 0, -5]))
    assert np.isfinite(ms.grid.log_odds).all() and (ms.grid.log_odds == 0).all()


def test_slice_height_filter_ignores_seabed_hits():
    """A downward-looking ray hits the seabed: it must not create a phantom wall in the 2-D map."""
    ms = MappingSystem(CFG)
    s = AnalyticSonar(AnalyticWorld(seabed_z=-15.0),
                      sonar_cfg(mount_rpy_deg=[0.0, 40.0, 0.0], max_range_m=30.0))   # pitched down 40 deg
    run_scans(ms, s, [0, 0, -5], [0, 0, -5], n=8)
    assert ms.telemetry()["mapping"]["occupied_cells"] == 0


def test_repeated_scans_do_not_saturate_and_a_moved_obstacle_is_revised():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5], n=60)
    assert np.isfinite(ms.grid.log_odds).all()
    assert ms.grid.log_odds.max() <= ms.cfg.log_odds_max and ms.grid.log_odds.min() >= ms.cfg.log_odds_min
    run_scans(ms, AnalyticSonar(AnalyticWorld(seabed_z=-15.0), sonar_cfg()), [0, 0, -5], [0, 0, -5], n=40)
    assert cell_state(ms, 9.25, 0.0) != OCCUPIED                               # evidence was revisable


def test_rate_limit_skips_scans_and_reports_it():
    ms = MappingSystem(CFG)
    s = AnalyticSonar(pillar_world(), sonar_cfg())
    for k in range(4):
        t = 0.05 * k                                                           # 20 Hz offered, 5 Hz allowed
        ms.process(t, s.measure(truth(t, [0, 0, -5])), est_at(t, [0, 0, -5]))
    assert ms.stats["map_updates"] == 1 and ms.stats["scans_skipped_rate"] == 3


def test_disabled_mapping_does_nothing_and_enable_resumes():
    ms = MappingSystem(CFG)
    ms.set_enabled(False)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5], n=3)
    assert ms.stats["map_updates"] == 0 and (ms.grid.log_odds == 0).all() and ms.telemetry()["mapping"]["enabled"] is False
    ms.set_enabled(True)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5], n=3)
    assert ms.stats["map_updates"] == 3


def test_reset_returns_everything_to_unknown_and_zero():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5])
    v = ms.version
    assert ms.telemetry()["mapping"]["occupied_cells"] > 0
    ms.reset()
    t = ms.telemetry()
    assert t["mapping"]["known_cells"] == 0 and t["mapping"]["occupied_cells"] == 0 and t["mapping"]["update_count"] == 0
    assert t["cost_map"]["inflated_count"] == 0 and t["sonar_mapping"]["rays_processed"] == 0
    assert t["mapping"]["last_update_time"] is None and ms.version > v          # version keeps increasing (dashboard cache key)


# ------------------------------------------------------------------ cost-map policy
def test_unknown_cost_policy_cost_vs_blocked():
    st = np.full((20, 20), UNKNOWN, np.int8)
    st[10, 10] = OCCUPIED
    st[2:6, 2:6] = FREE
    soft = build_cost_map(st, 0.25, CostMapConfig(unknown_policy="cost"))
    hard = build_cost_map(st, 0.25, CostMapConfig(unknown_policy="blocked"))
    cfg = CostMapConfig()
    d = float(np.hypot(5, 5)) * 0.25                                            # (15,15) is 5 cells diagonal from the obstacle
    unknown_expected = cfg.unknown_cost + float(clearance_cost(np.array([d]), cfg.inflation_radius_m,
                                                               cfg.clearance_decay_m, cfg.clearance_weight)[0])
    assert soft.cost[15, 15] == pytest.approx(unknown_expected) and soft.traversable[15, 15]   # unknown cost + clearance
    assert soft.cost[15, 15] >= cfg.unknown_cost
    assert not hard.traversable[15, 15] and hard.traversable[3, 3]
    assert soft.cost[3, 3] == pytest.approx(cfg.free_cost)                        # free, far from the obstacle: just free cost
    assert soft.cost[3, 3] < soft.cost[15, 15]


def test_clearance_cost_is_monotonic_deterministic_and_bounded():
    d = np.linspace(0, 5, 51)
    c = clearance_cost(d, 0.6, 1.5, 10.0)
    assert c[0] == 10.0 and c[-1] == 0.0 and np.all(np.diff(c) <= 1e-6)
    assert np.array_equal(c, clearance_cost(d, 0.6, 1.5, 10.0))
    assert np.all(clearance_cost(np.full(3, np.inf), 0.6, 1.5, 10.0) == 0)


def test_inflation_radius_follows_configured_vehicle_radius_and_margin():
    st = np.full((60, 60), FREE, np.int8)
    st[30, 30] = OCCUPIED
    for rv, mg in ((0.35, 0.25), (0.5, 0.5), (0.0, 0.0)):
        cm = build_cost_map(st, 0.25, CostMapConfig(vehicle_radius_m=rv, safety_margin_m=mg))
        infl = np.argwhere(cm.cls == CLASS_INFLATED)
        far = 0.0 if len(infl) == 0 else float(np.max(np.hypot(infl[:, 0] - 30, infl[:, 1] - 30)) * 0.25)
        assert far <= rv + mg + 1e-6
        assert (len(infl) == 0) == (rv + mg < 0.25)


def test_terrain_risk_adds_cost_only_when_enabled_and_never_marks_obstacles():
    st = np.full((10, 10), FREE, np.int8)
    risk = np.zeros((10, 10), np.float32); risk[5, 5] = 1.0
    off = build_cost_map(st, 0.25, CostMapConfig(terrain_enabled=False), risk)
    on = build_cost_map(st, 0.25, CostMapConfig(terrain_enabled=True, terrain_weight=5.0), risk)
    assert off.cost[5, 5] == 1.0 and on.cost[5, 5] == 6.0
    assert on.stats["occupied_count"] == 0 and on.traversable.all()


def test_terrain_prior_uses_real_random_forest_and_bathymetry():
    from src.mapping.terrain_prior import classes_from_depth, resample_to_geometry, risk_from_classes
    depth = np.load("data/processed/processed_bathymetry.npy")
    classes = classes_from_depth(depth, 1.0, "models/terrain_classifier.pkl")
    assert classes.shape == depth.shape and set(np.unique(classes)) <= {0, 1, 2, 3} and len(np.unique(classes)) > 1
    risk = risk_from_classes(classes, {0: 0.0, 1: 0.4, 2: 0.8, 3: 0.0})
    r = resample_to_geometry(risk, (0.0, 0.0), 1.0, GridGeometry.from_extent(0.25, 60, 60, -5, -5))
    assert r.shape == (240, 240) and r.max() == pytest.approx(0.8) and (r[:20, :] == 0).all()   # cells at y<0 are outside the chart
    ms = MappingSystem({**CFG, "mapping": {**CFG["mapping"], "terrain_risk": {**CFG["mapping"]["terrain_risk"], "enabled": True}}}, terrain_risk=r)
    assert ms.cost_map.stats["terrain_risk_used"] and ms.cost_map.cost.max() > 1.0 + 5.0 * 0.3
    assert ms.telemetry()["mapping"]["occupied_cells"] == 0                    # the prior never creates obstacles


# ------------------------------------------------------------------ payloads / JSON
def test_rle_roundtrip_and_payloads_are_strict_json_and_decode_to_the_grid():
    import json
    a = np.random.default_rng(0).integers(0, 3, (7, 9)).astype(np.uint8)
    assert np.array_equal(rle_decode(rle_encode(a), a.shape), a)
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5])
    for payload in (ms.occupancy_payload(), ms.costmap_payload(), ms.telemetry()):
        json.dumps(payload, allow_nan=False)
    p = ms.occupancy_payload()
    code = rle_decode(p["data"], tuple(p["shape"]))
    s = ms.grid.state()
    assert np.array_equal(code == 2, s == OCCUPIED) and np.array_equal(code == 1, s == FREE) and np.array_equal(code == 0, s == UNKNOWN)


def test_telemetry_values_are_computed_from_the_grid():
    ms = MappingSystem(CFG)
    run_scans(ms, AnalyticSonar(pillar_world(), sonar_cfg()), [0, 0, -5], [0, 0, -5])
    s, m = ms.grid.state(), ms.telemetry()["mapping"]
    assert m["free_cells"] == int((s == FREE).sum()) and m["occupied_cells"] == int((s == OCCUPIED).sum())
    assert m["known_cells"] + m["unknown_cells"] == s.size
    assert m["coverage_percent"] == pytest.approx(100.0 * m["known_cells"] / s.size)


# ------------------------------------------------------------------ the full chain incl. EKF
def test_end_to_end_sensor_suite_to_ekf_to_map_with_noise_and_dropout():
    """IMU/DVL/depth -> EKF estimate (real) ; sonar (noisy, 5% dropout) ; mapper gets ONLY estimate + scan."""
    from src.simulation.estimation import EstimationPipeline
    cfg = yaml.safe_load(open("configs/config.yaml", encoding="utf-8"))
    est = EstimationPipeline(cfg, 0, [0.0, 0.0, -5.0])
    est.suite.sensors["sonar"] = AnalyticSonar(pillar_world(14.0, 0.0, 0.8),
                                               SonarConfig.from_config(cfg["sensors"]["sonar"]))
    ms = MappingSystem(cfg)
    dt, pos, vel = 0.01, np.array([0.0, 0.0, -5.0]), np.array([1.0, 0.0, 0.0])
    for k in range(int(8.0 / dt)):
        pos = pos + vel * dt
        tr = TruthSnapshot(k * dt, pos.copy(), [0, 0, 0, 1], vel.copy(), [0, 0, 0], acceleration_world=np.zeros(3))
        meas, estimate = est.update_from_truth(tr)
        ms.process(k * dt, meas.get("sonar"), estimate)
    assert est.ekf.stats["dvl_updates"] > 50 and ms.stats["map_updates"] > 20
    err = np.linalg.norm(est.ekf.state.position - pos)
    assert err < 1.0
    # the pillar face (x = 14 - 0.8) is mapped within a couple of cells of the truth, and via the ESTIMATE
    xs = np.argwhere(ms.grid.state() == OCCUPIED)
    wx, wy = ms.grid.geom.grid_to_world(xs[:, 0], xs[:, 1])
    assert len(xs) > 0 and abs(float(np.min(wx)) - 13.2) < 1.0
    assert abs(float(np.min(wx)) - 13.2) <= err + 0.6
    assert ms.telemetry()["cost_map"]["inflated_count"] > 0

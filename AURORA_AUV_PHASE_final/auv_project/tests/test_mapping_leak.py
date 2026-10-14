"""Ground-truth leak prevention for the operational mapping path.

Operational = src/mapping online modules + the operational sections of the dashboard snapshot.
Ground truth may exist only in: tests, the simulation (feeding sensors), the clearly labelled
`debug_ground_truth` snapshot section and the `environment` description (labelled GROUND TRUTH).
"""
import ast
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

from src.mapping.map_manager import MappingSystem
from src.sensors.models import SonarMeasurement
from src.simulation.telemetry_snapshot import SensorStatsTracker, build_snapshot
from tests.test_mapping_sonar import Est, e2e_cfg, scan
from tests.helpers.analytic_sonar import AnalyticSonar
from tests.helpers.analytic_world import AnalyticWorld

ONLINE_MODULES = ["coordinate_utils", "log_odds_grid", "sonar_mapper", "map_manager", "map_types",
                  "online_cost_map", "inflation", "terrain_prior"]
# Debug/visualisation only: draws a labelled ground-truth path next to the map. It never touches a grid,
# and test_operational_modules_do_not_import_the_debug_plotter proves nothing operational imports it.
DEBUG_MODULES = ["map_plots"]
MAP_DIR = Path("src/mapping")
FORBIDDEN_IMPORTS = ("pybullet", "pybullet_data", "src.simulation", "src.sensors.sonar", "src.sensors.imu",
                     "src.sensors.dvl", "src.sensors.depth", "src.localization.ekf", "tests")
FORBIDDEN_WORDS = ("ground_truth", "groundtruth", "truth", "true_pos", "true_orient", "true_state", "true_pose",
                   "true_vel", "perfect", "oracle", "gt_pos", "gt_pose", "world_spec", "worldspec", "obstacle_footprint_mask",
                   "getBasePositionAndOrientation", "rayTest", "obstacle_ids", "seabed_id")


def imports_of(path: Path) -> list:
    tree = ast.parse(path.read_text("utf-8"))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out += [a.name for a in n.names]
        elif isinstance(n, ast.ImportFrom):
            out.append(n.module or "")
    return out


def test_online_mapping_modules_import_no_simulation_pybullet_or_truth_code():
    for mod in ONLINE_MODULES:
        for name in imports_of(MAP_DIR / f"{mod}.py"):
            assert not any(name == f or name.startswith(f + ".") for f in FORBIDDEN_IMPORTS), (mod, name)


def test_online_mapping_modules_never_reference_truth_or_environment_geometry():
    for mod in ONLINE_MODULES:
        code = (MAP_DIR / f"{mod}.py").read_text("utf-8")
        tree = ast.parse(code)
        # inspect identifiers / attribute names / string constants of EXECUTABLE code (docstrings excluded)
        names = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Attribute):
                names.add(n.attr)
            elif isinstance(n, ast.arg):
                names.add(n.arg)
            elif isinstance(n, ast.FunctionDef):
                names.add(n.name)
        low = {x.lower() for x in names}
        for w in FORBIDDEN_WORDS:
            assert not any(w.lower() in x for x in low), (mod, w)


def test_operational_modules_do_not_import_the_debug_plotter():
    for mod in ONLINE_MODULES:
        assert not any(n.endswith("map_plots") for n in imports_of(MAP_DIR / f"{mod}.py")), mod
    for p in Path("src").rglob("*.py"):
        if p.name == "run_simulation.py" or p.name == "map_plots.py":
            continue
        assert not any(n.endswith("map_plots") for n in imports_of(p)), p
    code = (MAP_DIR / "map_plots.py").read_text("utf-8")
    assert "grid.apply" not in code and ".log_odds[" not in code and "reset(" not in code   # read-only consumer
    assert "DEBUG ONLY" in code


def test_importing_the_mapper_does_not_load_pybullet_or_the_simulation():
    code = ("import sys; import src.mapping.map_manager, src.mapping.sonar_mapper, src.mapping.online_cost_map;"
            "bad=[m for m in sys.modules if m.split('.')[0]=='pybullet' or m.startswith('src.simulation')];"
            "print(bad); sys.exit(1 if bad else 0)")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={"PYTHONPATH": ".", "PATH": ""})
    assert r.returncode == 0, r.stdout + r.stderr


def test_process_signature_accepts_only_time_sonar_and_estimate():
    import inspect
    assert list(inspect.signature(MappingSystem.process).parameters) == ["self", "sim_time", "sonar", "estimate"]
    assert list(inspect.signature(MappingSystem.__init__).parameters) == ["self", "cfg", "terrain_risk"]


class StrictEstimate:
    """Raises on ANY attribute other than the three the mapper is entitled to read."""
    def __init__(self, est):
        object.__setattr__(self, "_e", est)
        object.__setattr__(self, "reads", set())

    def __getattr__(self, name):
        if name not in ("time", "position", "rotation"):
            raise AssertionError(f"mapper read estimate.{name}")
        self.reads.add(name)
        return getattr(self._e, name)


def test_mapper_reads_only_time_position_and_rotation_from_the_estimate():
    ms = MappingSystem(e2e_cfg())
    s = scan([10.0], [0.0], [True]); s.timestamp = 1.0
    est = StrictEstimate(Est(1.0, (0, 0, -5)))
    assert ms.process(1.0, s, est)
    assert est.reads == {"time", "position", "rotation"}


def record_inputs(truth_shift):
    """Run the real sensors+EKF once and record (t, sonar, estimate); `truth_shift` perturbs ONLY the truth
    object that the debug section would carry, after the measurements were produced."""
    from src.sensors.models import TruthSnapshot
    from src.sensors.sonar import SonarConfig
    from src.simulation.estimation import EstimationPipeline
    cfg = e2e_cfg()
    world = AnalyticWorld(cylinders=[((15.0, 5.0, -5.0), 0.8, 8.0)])
    pipe = EstimationPipeline(cfg, 0, (5.0, 5.0, -5.0))
    pipe.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
    rec, dbg = [], []
    pos = np.array([5.0, 5.0, -5.0])
    for k in range(400):
        pos = pos + np.array([0.005, 0, 0])
        t = TruthSnapshot(k * 0.01, pos.copy(), np.array([0, 0, 0, 1.0]), np.array([0.5, 0, 0]), np.zeros(3))
        meas, est = pipe.update_from_truth(t)
        rec.append((k * 0.01, meas.get("sonar"), est))
        dbg.append(pos + truth_shift)               # what a debug consumer would see: differs between the runs
    return rec, dbg


def test_identical_estimate_and_measurements_give_identical_map_whatever_the_ground_truth():
    rec, dbg_a = record_inputs(np.zeros(3))
    _, dbg_b = record_inputs(np.array([40.0, -25.0, 3.0]))
    assert not np.allclose(dbg_a, dbg_b)                                      # ground truth really differs
    maps = []
    for _truth in (dbg_a, dbg_b):
        ms = MappingSystem(e2e_cfg())
        for t, sonar, est in rec:
            ms.process(t, sonar, est)
        maps.append((ms.grid.log_odds.copy(), ms.cost_map.cost.copy(), json.dumps(ms.telemetry(), default=str)))
    assert np.array_equal(maps[0][0], maps[1][0]) and np.array_equal(maps[0][1], maps[1][1])
    ta, tb = json.loads(maps[0][2]), json.loads(maps[1][2])
    for key in ("mapping", "cost_map", "sonar_mapping"):
        assert ta[key] == tb[key], key
    assert (maps[0][0] != 0).any()                                            # and the map is not trivially empty


def test_wrong_estimate_changes_the_map_even_when_truth_is_perfect():
    rec, _ = record_inputs(np.zeros(3))
    good, bad = MappingSystem(e2e_cfg()), MappingSystem(e2e_cfg())
    for t, sonar, est in rec:
        good.process(t, sonar, est)

        e2 = Est(est.time, est.position + np.array([0.0, 8.0, 0.0]))
        e2.rotation = est.rotation
        bad.process(t, sonar, e2)
    assert not np.array_equal(good.grid.log_odds, bad.grid.log_odds)          # the map follows the estimate


def snapshot_with_truth(truth_pos):
    from src.sensors.models import TruthSnapshot
    from src.sensors.sonar import SonarConfig
    from src.simulation.estimation import EstimationPipeline
    cfg = e2e_cfg()
    world = AnalyticWorld(cylinders=[((15.0, 5.0, -5.0), 0.8, 8.0)])
    pipe = EstimationPipeline(cfg, 0, (5.0, 5.0, -5.0))
    pipe.suite.sensors["sonar"] = AnalyticSonar(world, SonarConfig.from_config(cfg["sensors"]["sonar"]))
    ms, tr = MappingSystem(cfg), SensorStatsTracker()
    for k in range(300):
        t = TruthSnapshot(k * 0.01, np.array([5.0, 5.0, -5.0]), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3))
        meas, est = pipe.update_from_truth(t)
        tr.observe(meas)
        ms.process(k * 0.01, meas.get("sonar"), est)
    return build_snapshot(step=299, sim_time=2.99, dt=0.01, status="RUNNING", dynamics_mode="x", estimation=pipe,
                          estimate=est, measurements=meas, sensor_tracker=tr, mapping=ms,
                          env_desc={"scenario": "s"}, truth_position=truth_pos, truth_euler=(0, 0, 0), auv=None,
                          control=None, controller_info={}, perf={}, last_command_id=0,
                          mapping_unavailable_reason=None)


def test_operational_dashboard_sections_do_not_depend_on_ground_truth():
    a = snapshot_with_truth(np.array([5.0, 5.0, -5.0]))
    b = snapshot_with_truth(np.array([90.0, 70.0, -9.0]))
    for key in ("estimate", "sensors", "sonar", "mapping", "cost_map", "sonar_mapping"):
        assert json.dumps(a[key], default=str, sort_keys=True) == json.dumps(b[key], default=str, sort_keys=True), key
    assert a["debug_ground_truth"] != b["debug_ground_truth"]
    assert a["debug_ground_truth"]["LABEL"].startswith("GROUND TRUTH")
    flat = json.dumps({k: v for k, v in a.items() if k not in ("debug_ground_truth", "wall_time")}, default=str)
    assert "estimate_error" not in flat                                        # error vs truth only in the debug section


def test_sensor_measurements_never_carry_truth_when_invalid():
    s = scan([10.0], [0.0], [True], valid=False)
    assert isinstance(s, SonarMeasurement)
    from src.sensors.models import TruthSnapshot
    from src.sensors.sonar import SonarConfig
    cfg = SonarConfig(horizontal_rays=5, vertical_rays=1, dropout_probability=1.0, range_noise_std_m=0.0)
    t = TruthSnapshot(0.0, np.array([0, 0, -5.0]), np.array([0, 0, 0, 1.0]), np.zeros(3), np.zeros(3))
    m = AnalyticSonar(AnalyticWorld(cylinders=[((6.0, 0.0, -5.0), 0.5, 5.0)]), cfg).measure(t)
    assert not m.detected.any() and np.isnan(m.points_sensor).all() and (m.value == cfg.max_range_m).all()


def test_truth_snapshot_is_consumed_only_by_sensors_in_the_simulation_code():
    """Repository audit: which non-test source files mention the truth snapshot type or PyBullet pose queries."""
    allowed = {"src/simulation/truth.py", "src/simulation/estimation.py", "src/sensors/models.py",
               "src/sensors/imu.py", "src/sensors/dvl.py", "src/sensors/depth.py", "src/sensors/compass.py", "src/sensors/sonar.py",
               "src/sensors/suite.py", "src/simulation/run_simulation.py", "src/simulation/telemetry_snapshot.py",
               "src/simulation/auv.py", "src/simulation/physics.py", "src/simulation/environment.py",
               "src/simulation/sensors.py", "src/simulation/controller.py", "src/simulation/metrics.py",
               "src/simulation/live_io.py", "src/simulation/commands.py", "src/sensors/__init__.py",
               # DEBUG-ONLY (3-D window overlays): reads the true pose to PLACE drawings; guarded by
               # test_gui_visuals_is_a_read_only_debug_consumer_imported_only_by_the_simulation_loop below.
               "src/simulation/gui_visuals.py"}
    offenders = []
    for p in Path("src").rglob("*.py"):
        text = p.read_text("utf-8")
        if ("TruthSnapshot" in text or "getBasePositionAndOrientation" in text) and p.as_posix() not in allowed:
            offenders.append(p.as_posix())
    assert offenders == []


def test_gui_visuals_is_a_read_only_debug_consumer_imported_only_by_the_visual_entry_points():
    """gui_visuals draws what the run already published. Only the two 3-D window entry points may import it:
    the run loop's own window, and the live viewer that renders an already-running headless run. No mapper, EKF,
    sensor, controller or dashboard module may pull it in (it is a debug/visual consumer, not a pipeline stage)."""
    visual_entry_points = ["src/simulation/live_viewer.py",           # live window for a running headless run
                           "src/simulation/run_simulation.py"]         # the run loop's own PyBullet GUI path
    importers = []
    for base in ("src", "dashboard"):
        for p in Path(base).rglob("*.py"):
            if any(n.endswith("gui_visuals") for n in imports_of(p)) or "gui_visuals" in p.read_text("utf-8").replace("src/simulation/gui_visuals.py", ""):
                importers.append(p.as_posix())
    importers = [i for i in importers if i != "src/simulation/gui_visuals.py"]
    assert sorted(importers) == sorted(visual_entry_points)
    code = Path("src/simulation/gui_visuals.py").read_text("utf-8")
    for forbidden in (".apply(", ".reset(", ".process(", "log_odds[", "set_enabled", "update_cost_map", "inject_fault"):
        assert forbidden not in code, forbidden                                # it never writes to the map / EKF / sensors
    # A window is a viewer: it renders telemetry and must never steer the vehicle, map a scan or fault a sensor.
    viewer = Path("src/simulation/live_viewer.py").read_text("utf-8")
    for forbidden in ("apply_command", "inject_fault", "set_sensor_health", "update_cost_map", ".process("):
        assert forbidden not in viewer, forbidden

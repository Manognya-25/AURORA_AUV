"""3-D overlay logic (src/simulation/gui_visuals.py) against a RECORDING stand-in for pybullet.

This verifies WHAT would be drawn and where (ray endpoints == measured ranges, dropped rays absent, estimate marker
at the EKF estimate, mapped points == occupied grid cells, camera follows). It cannot verify how PyBullet renders it;
that needs a human with the GUI window (python run_visual_demo.py)."""
import numpy as np
import pytest
import yaml

from src.mapping.map_manager import MappingSystem
from src.mapping.map_types import OCCUPIED
from src.sensors.models import TruthSnapshot
from src.simulation.gui_visuals import GuiVisuals
from src.utils.transforms import euler_to_quat, quat_to_rot
from tests.test_mapping_sonar import Est, e2e_cfg, scan


class FakePB:
    GEOM_SPHERE = 2

    def __init__(self):
        self.lines, self.points, self.texts, self.bodies, self.removed = [], [], [], [], []
        self.camera = (0, 0, None, None, None, None, None, None, 40.0, -30.0, 12.0, [0, 0, 0])
        self.camera_sets, self.marker_moves, self._n = [], [], 100

    def _id(self):
        self._n += 1
        return self._n

    def createVisualShape(self, *a, **k): return 1
    def createMultiBody(self, **k): self.bodies.append(k); return self._id()
    def addUserDebugText(self, text, pos, **k): self.texts.append((text, pos))
    def addUserDebugLine(self, a, b, **k): self.lines.append((np.array(a), np.array(b), k)); return self._id()
    def addUserDebugPoints(self, pts, cols, **k): self.points.append((np.array(pts), k)); return k.get("replaceItemUniqueId") or self._id()
    def removeUserDebugItem(self, i, **k): self.removed.append(i)
    def getDebugVisualizerCamera(self, **k): return self.camera
    def resetDebugVisualizerCamera(self, **k): self.camera_sets.append(k)
    def resetBasePositionAndOrientation(self, body, pos, orn, **k): self.marker_moves.append((body, list(pos)))
    # camera rig extras: no buttons pressed, no keys, an empty collision world
    def addUserDebugParameter(self, name, *a, **k): return self._id()
    def readUserDebugParameter(self, pid, **k): return 0.0
    def getKeyboardEvents(self, **k): return {}
    def rayTestBatch(self, a, b, **k): return [(-1, -1, 1.0, [0, 0, 0], [0, 0, 1])] * len(a)
    def rayTest(self, a, b, **k): return [(-1, -1, 1.0, [0, 0, 0], [0, 0, 1])]


class Env:
    class bounds:
        xmin, xmax, ymin, ymax = -2.0, 102.0, -2.0, 102.0

    def describe(self):
        return {"scenario": "mixed_terrain_obstacles", "seed": 3, "obstacle_count": 24, "seabed_z_min": -12.0, "seabed_z_max": -8.0}


def truth_at(pos, yaw=0.0):
    return TruthSnapshot(0.0, np.array(pos, float), euler_to_quat(0, 0, yaw), np.zeros(3), np.zeros(3))


def make(**cfg):
    pb = FakePB()
    return pb, GuiVisuals(0, cfg, Env(), pb=pb)


def test_scenario_label_and_legend_are_drawn_from_the_real_environment_description():
    pb, g = make()
    assert any("mixed_terrain_obstacles" in t and "seed 3" in t and "24 collision bodies" in t for t, _ in pb.texts)
    assert any("EKF estimate" in t for t, _ in pb.texts)


def test_hit_rays_end_exactly_at_the_measured_range_in_the_world_frame():
    pb, g = make(update_every_n_steps=1, free_ray_stride=1)
    yaw, pos = np.deg2rad(30), [10.0, 4.0, -5.0]
    m = scan([8.0, 30.0], [0.0, 20.0], [True, False], mount_pos=(0.8, 0.0, 0.0))
    g.update(0, truth_at(pos, yaw), m, None, None, 0.0)
    assert len(pb.lines) == 2                                                       # one hit + one no-return (no trail yet: first point)
    hit = [l for l in pb.lines if l[2]["lineColorRGB"] == [1.0, 0.25, 0.1]][0]
    R = quat_to_rot(euler_to_quat(0, 0, yaw))
    origin = np.array(pos) + R @ np.array([0.8, 0, 0])
    assert np.allclose(hit[0], origin)
    assert np.linalg.norm(hit[1] - hit[0]) == pytest.approx(8.0)                    # length == MEASURED range
    assert np.allclose((hit[1] - hit[0]) / 8.0, R @ np.array([1.0, 0, 0]))          # direction through true yaw
    free = [l for l in pb.lines if l[2]["lineColorRGB"] == [0.25, 0.55, 1.0]][0]
    assert np.linalg.norm(free[1] - free[0]) == pytest.approx(30.0)                  # no-return drawn to max range only


def test_dropped_rays_are_not_drawn_and_a_repeated_scan_is_not_redrawn():
    pb, g = make(update_every_n_steps=1, free_ray_stride=1)
    m = scan([10.0, 10.0, 10.0], [-10.0, 0.0, 10.0], [True, True, True], ray_valid=[True, False, True])
    g.update(0, truth_at([0, 0, -5]), m, None, None, 0.0)
    ray_lines = lambda: [l for l in pb.lines if l[2]["lineColorRGB"] != [0.2, 0.9, 1.0]]   # exclude the cyan trail
    assert len(ray_lines()) == 2                                                    # the dropped middle ray has no line
    g.update(1, truth_at([0, 0, -5]), m, None, None, 0.01)                         # same scan timestamp
    assert len(ray_lines()) == 2                                                    # not redrawn


def test_failed_sonar_draws_no_rays_at_all():
    from src.sensors.health import SensorHealth
    pb, g = make(update_every_n_steps=1)
    m = scan([10.0, 10.0], [0.0, 10.0], [True, True], valid=False, health=SensorHealth.FAILED, ray_valid=[False, False])
    g.update(0, truth_at([0, 0, -5]), m, None, None, 0.0)
    assert pb.lines == []


def test_estimate_marker_follows_the_ekf_estimate_not_the_truth():
    pb, g = make(update_every_n_steps=1)
    est = Est(0.0, (3.0, 4.0, -5.0))
    g.update(0, truth_at([50.0, 60.0, -5.0]), None, est, None, 0.0)
    assert pb.marker_moves[-1][1] == [3.0, 4.0, -5.0]


def test_camera_is_a_third_person_follow_view_centred_on_the_auv():
    pb, g = make(update_every_n_steps=1)
    g.update(0, truth_at([7.0, 8.0, -5.0], yaw=0.0), None, None, None, 0.0)
    c = pb.camera_sets[-1]
    tx, ty, tz = c["cameraTargetPosition"]
    assert (tx, ty, tz) == pytest.approx((7.8, 8.0, -5.0))                         # aim: AUV + 0.8 m look-ahead along its heading
    assert c["cameraDistance"] == pytest.approx(5.0) and c["cameraPitch"] == pytest.approx(-20.0)
    assert c["cameraYaw"] == pytest.approx(-90.0)                                   # heading +X  ->  PyBullet yaw -90
    pb2, g2 = make(update_every_n_steps=1, camera_follow=False)
    g2.update(0, truth_at([7.0, 8.0, -5.0]), None, None, None, 0.0)
    assert pb2.camera_sets == []                                                    # --no-follow: the camera is left alone


def test_mapped_points_are_exactly_the_occupied_cells_of_the_sensor_map_at_the_estimated_depth():
    pb, g = make(update_every_n_steps=1)
    ms = MappingSystem(e2e_cfg())
    for k in range(6):
        s = scan([10.0], [0.0], [True]); s.timestamp = 1.0 + k
        ms.process(1.0 + k, s, Est(1.0 + k, (0, 0, -5)))
    est = Est(7.0, (0, 0, -5.0))
    g.update(0, truth_at([0, 0, -5]), None, est, ms, 7.0)
    pts, kw = pb.points[-1]
    rows, cols = np.nonzero(ms.grid.state() == OCCUPIED)
    x, y = ms.grid.geom.grid_to_world(rows, cols)
    assert len(pts) == len(rows) > 0
    assert np.allclose(np.sort(pts[:, 0]), np.sort(x)) and np.allclose(np.sort(pts[:, 1]), np.sort(y))
    assert np.allclose(pts[:, 2], -5.0)
    g.update(1, truth_at([0, 0, -5]), None, est, ms, 7.1)                           # within the refresh period: no redraw
    assert len(pb.points) == 1
    ms.reset()
    g.update(2, truth_at([0, 0, -5]), None, est, ms, 9.0)                           # map reset -> points removed
    assert pb.removed, "mapped points were not cleared after the map was reset"


def test_a_drawing_failure_disables_only_that_overlay_and_never_raises():
    pb, g = make(update_every_n_steps=1, free_ray_stride=1)
    pb.addUserDebugLine = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no connection"))
    m = scan([10.0], [0.0], [True])
    import contextlib, io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        g.update(0, truth_at([0, 0, -5]), m, Est(0.0, (0, 0, -5)), None, 0.0)       # must not raise
        g.update(1, truth_at([0, 0, -5]), m, Est(0.0, (1, 0, -5)), None, 0.1)
    out = buf.getvalue()
    assert out.count("overlay 'trail' disabled") == 1 and "overlay 'rays' disabled" in out   # reported once, not silent
    assert pb.marker_moves[-1][1] == [1.0, 0.0, -5.0]                                # other overlays keep working

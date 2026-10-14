"""Third-person follow camera (src/simulation/gui_camera.py): geometry, heading tracking, collision avoidance, view
controls and mouse offsets, against a recording stand-in for pybullet whose collision world is analytic (a wall
plane, a seabed plane). A cross-check against the REAL pybullet view matrix runs when PyBullet is installed.
The camera never touches the simulation: these tests also assert it creates no bodies and only sets the camera."""
import math

import numpy as np
import pytest

from src.sensors.models import TruthSnapshot
from src.simulation import gui_camera as gc
from src.simulation.gui_camera import CameraRig
from src.utils.transforms import euler_to_quat, quat_to_rot


def truth(pos, yaw=0.0, pitch=0.0):
    return TruthSnapshot(0.0, np.array(pos, float), euler_to_quat(0, pitch, yaw), np.zeros(3), np.zeros(3))


class World:
    """Fake pybullet with analytic collision geometry: wall = plane x >= wall_x; seabed = plane z <= floor_z."""
    KEY_WAS_TRIGGERED = 2

    def __init__(self, wall_x=None, floor_z=None, auv_id=7):
        self.wall_x, self.floor_z, self.auv_id = wall_x, floor_z, auv_id
        self.sets, self.cam = [], (0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, [0, 0, 0])
        self.params, self.keys, self.bodies_created, self._n = {}, {}, 0, 0

    def _hit(self, a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        best = 1.0
        if self.wall_x is not None and (a[0] - self.wall_x) * (b[0] - self.wall_x) < 0:
            best = min(best, (self.wall_x - a[0]) / (b[0] - a[0]))
        if self.floor_z is not None and (a[2] - self.floor_z) * (b[2] - self.floor_z) < 0:
            best = min(best, (self.floor_z - a[2]) / (b[2] - a[2]))
        return (5, -1, best, [0, 0, 0], [0, 0, 1]) if best < 1.0 else (-1, -1, 1.0, [0, 0, 0], [0, 0, 1])

    def rayTestBatch(self, A, B, **k): return [self._hit(a, b) for a, b in zip(A, B)]
    def rayTest(self, a, b, **k): return [self._hit(a, b)]
    def addUserDebugParameter(self, name, *a, **k): self._n += 1; self.params[self._n] = 0.0; self.params[name] = self._n; return self._n
    def readUserDebugParameter(self, pid, **k): return self.params[pid]
    def getKeyboardEvents(self, **k): ev, self.keys = dict(self.keys), {}; return ev
    def getDebugVisualizerCamera(self, **k): return self.cam
    def createMultiBody(self, **k): self.bodies_created += 1; return 1
    def resetDebugVisualizerCamera(self, **k):
        self.sets.append(k)
        self.cam = (0, 0, 0, 0, 0, 0, 0, 0, k["cameraYaw"], k["cameraPitch"], k["cameraDistance"], k["cameraTargetPosition"])

    def press(self, name): self.params[self.params[name]] += 1.0


def rig(world=None, **cfg):
    w = world or World()
    cfg.setdefault("poll_every_n_updates", 1)
    return w, CameraRig(0, cfg, w, auv_body_id=w.auv_id)


def eye_of(c): return gc.view_eye(c["cameraTargetPosition"], c["cameraYaw"], c["cameraPitch"], c["cameraDistance"])


# ------------------------------------------------------------------------------------------ geometry / heading
@pytest.mark.parametrize("heading_deg", [0, 45, 90, 135, 180, -90, -135])
def test_camera_sits_behind_and_above_the_auv_for_any_heading(heading_deg):
    w, r = rig()
    pos = np.array([10.0, -4.0, -5.0])
    r.update(0, truth(pos, math.radians(heading_deg)), 0.0)
    c = w.sets[-1]
    eye = eye_of(c)
    fwd = np.array([math.cos(math.radians(heading_deg)), math.sin(math.radians(heading_deg)), 0.0])
    rel = eye - pos
    assert rel @ fwd < -3.5                                   # behind the AUV (opposite its heading), ~5 m * cos(20 deg) - look-ahead
    assert rel[2] > 1.2                                       # and clearly above it
    assert np.linalg.norm(rel[:2] - (rel @ fwd) * fwd[:2]) < 1e-6   # directly behind (no sideways offset)
    assert np.linalg.norm(np.array(c["cameraTargetPosition"]) - pos) < 1.0   # AUV within 1 m of the aim point = near-centred


def test_view_direction_matches_the_heading_and_the_look_down_angle():
    w, r = rig()
    r.update(0, truth([0, 0, -5], math.radians(90)), 0.0)
    c = w.sets[-1]
    look = np.array(c["cameraTargetPosition"]) - eye_of(c)
    look /= np.linalg.norm(look)
    assert look[1] > 0.9 and abs(look[0]) < 1e-9 and look[2] < -0.3          # looking along +Y (the AUV heading), downwards
    assert math.degrees(math.asin(-look[2])) == pytest.approx(20.0)


def test_camera_yaw_follows_a_turn_smoothly_through_the_wrap_around():
    w, r = rig(update_every_n_steps=1)
    yaws, t = [], 0.0
    for k in range(60):                                       # AUV turns from heading 170 deg through 180 to -170 deg
        h = math.radians(170 + k * 0.4)
        r.update(k, truth([0, 0, -5], h), t); t += 0.05
        yaws.append(w.sets[-1]["cameraYaw"])
    steps = np.abs([gc.wrap180(b - a) for a, b in zip(yaws[:-1], yaws[1:])])
    assert steps.max() < 3.0                                  # no 360-degree flip across +-180
    assert abs(gc.wrap180(yaws[-1] - gc.yaw_deg_for_heading(math.radians(170 + 59 * 0.4)))) < 8.0   # converged to the heading


def test_vertical_hull_holds_the_last_heading_instead_of_spinning():
    h0 = gc.heading_rad(quat_to_rot(euler_to_quat(0, 0, 1.0)))
    steep = quat_to_rot(euler_to_quat(0, math.radians(89.5), 2.5))             # nose straight down
    assert gc.heading_rad(steep, previous=h0) == pytest.approx(h0)


def test_aim_point_tracks_the_moving_auv_with_no_visible_lag():
    w, r = rig(update_every_n_steps=1)
    t = 0.0
    for k in range(80):
        pos = [k * 0.05, 0.0, -5.0]                           # 1 m/s along +X
        r.update(k, truth(pos, 0.0), t); t += 0.05
        tgt = np.array(w.sets[-1]["cameraTargetPosition"])
        assert abs(tgt[0] - (pos[0] + 0.8)) < 0.12 and abs(tgt[1]) < 1e-9


# ------------------------------------------------------------------------------------------ collision avoidance
def test_a_wall_behind_the_auv_never_sits_between_the_camera_and_the_auv():
    w = World(wall_x=-1.0)                                    # a wall 1 m behind an AUV at x=0 heading +X
    _, r = rig(w)
    r.update(0, truth([0, 0, -5], 0.0), 0.0)
    c = w.sets[-1]
    eye, tgt = eye_of(c), np.array(c["cameraTargetPosition"])
    assert eye[0] > -1.0 - 1e-9                               # the camera is NOT behind the wall
    assert w._hit([0, 0, -5], eye)[0] < 0                     # the sight line AUV -> camera is clear
    assert r.stats["rerouted"] + r.stats["pulled_in"] > 0


def test_camera_never_dips_below_the_seabed_or_touches_it():
    w = World(floor_z=-5.6)                                   # seabed only 0.6 m under the AUV; camera wants to look down from above
    _, r = rig(w, pitch_deg=5.0)                              # (pitch clamped to look down), preferred view skims the floor
    for k in range(20):
        r.update(k, truth([k * 0.1, 0, -5], 0.0), k * 0.05)
        assert eye_of(w.sets[-1])[2] > -5.6 + 0.0             # above the seabed


def test_sight_line_stays_clear_while_driving_along_a_wall():
    w = World(wall_x=3.0)
    _, r = rig(w, update_every_n_steps=1)
    t = 0.0
    for k in range(100):
        pos = [2.2, k * 0.05, -5.0]                           # AUV close beside a wall, heading +X (towards it)
        r.update(k, truth(pos, 0.0), t); t += 0.05
        eye = eye_of(w.sets[-1])
        assert w._hit(pos, eye)[0] < 0 and eye[0] < 3.0


def test_camera_returns_to_the_preferred_distance_after_the_obstruction_ends():
    w = World(wall_x=-1.0)
    _, r = rig(w, update_every_n_steps=1)
    t = 0.0
    for k in range(10):
        r.update(k, truth([0, 0, -5], 0.0), t); t += 0.05
    squeezed = w.sets[-1]["cameraDistance"]
    w.wall_x = None
    for k in range(10, 200):
        r.update(k, truth([0, 0, -5], 0.0), t); t += 0.05
    assert w.sets[-1]["cameraDistance"] > squeezed and w.sets[-1]["cameraDistance"] == pytest.approx(5.0, abs=0.2)


def test_probe_failure_degrades_to_plain_follow_without_crashing():
    class NoRays(World):
        def rayTestBatch(self, *a, **k): raise RuntimeError("no ray tests")
    w = NoRays()
    _, r = rig(w)
    assert r.update(0, truth([0, 0, -5], 0.0), 0.0) is True and w.sets
    assert any("probing disabled" in n for n in r.notes)


# ------------------------------------------------------------------------------------------ controls
def test_top_view_button_looks_straight_down_with_north_up_and_follow_returns():
    w, r = rig(update_every_n_steps=1)
    r.update(0, truth([5, 6, -5], 1.2), 0.0)
    w.press(gc.BUTTON_TOP)
    r.update(1, truth([5, 6, -5], 1.2), 0.05)
    c = w.sets[-1]
    assert c["cameraPitch"] <= -85.0 and abs(gc.wrap180(c["cameraYaw"])) < 1e-6 and c["cameraDistance"] == pytest.approx(18.0, abs=0.1)
    assert np.allclose(c["cameraTargetPosition"][:2], [5, 6], atol=1e-6)         # centred on the AUV
    w.press(gc.BUTTON_FOLLOW)
    r.update(2, truth([5, 6, -5], 1.2), 0.10)
    assert w.sets[-1]["cameraPitch"] == pytest.approx(-20.0) and r.mode == gc.FOLLOW


def test_keys_1_2_3_switch_views_and_reset():
    w, r = rig(update_every_n_steps=1)
    r.update(0, truth([0, 0, -5], 0.0), 0.0)
    w.keys = {ord("2"): w.KEY_WAS_TRIGGERED}; r.update(1, truth([0, 0, -5], 0.0), 0.05)
    assert r.mode == gc.TOP
    w.keys = {ord("1"): w.KEY_WAS_TRIGGERED}; r.update(2, truth([0, 0, -5], 0.0), 0.10)
    assert r.mode == gc.FOLLOW
    w.keys = {ord("3"): w.KEY_WAS_TRIGGERED}; r.update(3, truth([0, 0, -5], 0.0), 0.15)
    assert r.stats["resets"] == 1


def test_mouse_rotation_and_zoom_are_kept_as_an_offset_and_reset_view_clears_it():
    w, r = rig(update_every_n_steps=1)
    r.update(0, truth([0, 0, -5], 0.0), 0.0)
    y0, d0 = w.cam[8], w.cam[10]
    w.cam = (*w.cam[:8], y0 + 40.0, w.cam[9], d0 * 2.0, w.cam[11])               # the user drags +40 deg and zooms out x2
    r.update(1, truth([0, 0, -5], 0.0), 0.05)
    assert gc.wrap180(w.sets[-1]["cameraYaw"] - y0) == pytest.approx(40.0, abs=2.0) and w.sets[-1]["cameraDistance"] > 1.4 * d0
    w.press(gc.BUTTON_RESET)
    for k in range(2, 150):
        r.update(k, truth([0, 0, -5], 0.0), k * 0.05)
    assert gc.wrap180(w.sets[-1]["cameraYaw"] - y0) == pytest.approx(0.0, abs=1.0) and w.sets[-1]["cameraDistance"] == pytest.approx(5.0, abs=0.2)


def test_keys_4_to_7_orbit_and_zoom_and_reset_view_clears_them():
    w, r = rig(update_every_n_steps=1)
    r.update(0, truth([0, 0, -5], 0.0), 0.0)
    y0, d0 = w.cam[8], w.cam[10]
    w.keys = {ord("4"): w.KEY_WAS_TRIGGERED}; r.update(1, truth([0, 0, -5], 0.0), 0.05)   # orbit left
    assert gc.wrap180(w.sets[-1]["cameraYaw"] - y0) == pytest.approx(-30.0) and r.stats["orbits"] == 1
    w.keys = {ord("5"): w.KEY_WAS_TRIGGERED}; r.update(2, truth([0, 0, -5], 0.0), 0.10)   # orbit right -> back to 0
    assert gc.wrap180(w.sets[-1]["cameraYaw"] - y0) == pytest.approx(0.0)
    w.keys = {ord("6"): w.KEY_WAS_TRIGGERED}; r.update(3, truth([0, 0, -5], 0.0), 0.15)   # zoom in
    assert w.sets[-1]["cameraDistance"] == pytest.approx(d0 * 0.75, abs=1e-6) and r.stats["zooms"] == 1
    w.keys = {ord("7"): w.KEY_WAS_TRIGGERED}; r.update(4, truth([0, 0, -5], 0.0), 0.20)   # zoom out
    assert w.sets[-1]["cameraDistance"] == pytest.approx(d0, abs=1e-6)
    w.press(gc.BUTTON_RESET); r.update(5, truth([0, 0, -5], 0.0), 0.25)
    assert r.stats["resets"] == 1 and w.sets[-1]["cameraDistance"] == pytest.approx(d0, abs=1e-6)
    assert r.stats["orbits"] == 2 and r.stats["zooms"] == 2


def test_orbit_and_zoom_side_panel_buttons_appear_only_when_enabled():
    w, r = rig(update_every_n_steps=1, orbit_zoom_buttons=True)
    assert all(b in r._buttons for b in gc.ORBIT_ZOOM_BUTTONS)
    r.update(0, truth([0, 0, -5], 0.0), 0.0)
    y0, d0 = w.cam[8], w.cam[10]
    w.press(gc.BUTTON_ORBIT_RIGHT); r.update(1, truth([0, 0, -5], 0.0), 0.05)
    assert gc.wrap180(w.sets[-1]["cameraYaw"] - y0) == pytest.approx(30.0)
    w.press(gc.BUTTON_ZOOM_IN); r.update(2, truth([0, 0, -5], 0.0), 0.10)
    assert w.sets[-1]["cameraDistance"] == pytest.approx(d0 * 0.75, abs=1e-6)
    w2, r2 = rig(update_every_n_steps=1)                                  # default: the lean 3-button panel
    assert [b for b in r2._buttons] == list(gc.BASE_BUTTONS)


def test_camera_only_sets_the_camera_and_creates_no_bodies():
    w, r = rig(update_every_n_steps=1)
    for k in range(10):
        r.update(k, truth([k, 0, -5], 0.0), k * 0.05)
    assert w.bodies_created == 0 and len(w.sets) == 10


def test_update_cadence_and_missing_truth():
    w, r = rig(update_every_n_steps=3)
    assert r.update(1, truth([0, 0, -5]), 0.0) is False and r.update(0, None, 0.0) is False and w.sets == []
    assert r.update(3, truth([0, 0, -5]), 0.0) is True


def test_initial_camera_is_behind_the_start_looking_to_the_first_waypoint_and_overview_is_the_legacy_one():
    c = gc.initial_follow_camera((5, 5, -5), (45, 5, -5))
    assert c["cameraYaw"] == pytest.approx(-90.0) and c["cameraDistance"] == 5.0
    eye = gc.view_eye(c["cameraTargetPosition"], c["cameraYaw"], c["cameraPitch"], c["cameraDistance"])
    assert eye[0] < 5.0 and eye[2] > -5.0                                          # behind (-X side) and above
    o = gc.overview_free_camera((5, 5, -5), [(40, 12, -5)], (75, 82, -5))
    assert o["cameraDistance"] >= 22.0 and o["cameraYaw"] == 45.0


# ------------------------------------------------------------------------------------------ real PyBullet cross-check
def test_view_eye_matches_the_real_pybullet_view_matrix():
    pb = pytest.importorskip("pybullet")
    for yaw, pitch, dist in [(-90, -20, 5), (0, -89, 18), (37, -45, 3), (170, -10, 7)]:
        tgt = [3.0, -2.0, -5.0]
        vm = np.array(pb.computeViewMatrixFromYawPitchRoll(tgt, dist, yaw, pitch, 0, 2)).reshape(4, 4).T
        eye_real = np.linalg.inv(vm)[:3, 3]
        assert np.allclose(eye_real, gc.view_eye(tgt, yaw, pitch, dist), atol=1e-4), (yaw, pitch, dist)


def test_steady_state_costs_only_three_pybullet_round_trips_per_update():
    """In p.GUI every PyBullet call is a GUI round trip (~12 ms in software GL), so the follow camera must stay lean:
    read camera + ONE batched ray test + set camera, controls polled only every few updates."""
    w = World()
    calls = []
    for name in ("getDebugVisualizerCamera", "rayTestBatch", "rayTest", "resetDebugVisualizerCamera", "readUserDebugParameter", "getKeyboardEvents"):
        orig = getattr(w, name)
        setattr(w, name, (lambda n, f: (lambda *a, **k: (calls.append(n), f(*a, **k))[1]))(name, orig))
    r = CameraRig(0, {"poll_every_n_updates": 6}, w, auv_body_id=w.auv_id)
    for k in range(1, 61):
        r.update(k, truth([k * 0.02, 0, -5], 0.0), k * 0.017)
    per_update = len(calls) / r.stats["updates"]
    assert calls.count("rayTestBatch") <= r.stats["updates"] and "rayTest" not in calls          # ground ray rides in the batch
    assert per_update <= 4.0, (per_update, {n: calls.count(n) for n in set(calls)})

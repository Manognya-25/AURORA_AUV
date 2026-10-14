"""Underwater look (src/simulation/underwater_look.py): rendering of the REAL world only.

The atmosphere must never touch physics: no collision shape may be created, and a drawing failure must only
disable that part of the atmosphere (never raise into the run). Verified against a stand-in PyBullet; the
wiring into the GUI setup path is verified with real PyBullet (DIRECT) when it is installed.
"""
import pytest

from src.simulation.underwater_look import DEFAULT_BACKGROUND, DEFAULT_LIGHT, apply


class FakePB:
    COV_ENABLE_SHADOWS = 2
    GEOM_BOX = 3
    VISUAL_SHAPE_DOUBLE_SIDED = 4

    def __init__(self, fail_on=()):
        self.debug_calls, self.visual_shapes, self.bodies = [], [], []
        self.fail_on = set(fail_on)

    def configureDebugVisualizer(self, *a, **k):
        if "debug" in self.fail_on:
            raise RuntimeError("no debug visualizer")
        self.debug_calls.append((a, k))

    def createVisualShape(self, *a, **k):
        if "visual" in self.fail_on:
            raise RuntimeError("shape rejected")
        self.visual_shapes.append((a, k))
        return 11

    def createMultiBody(self, **k):
        if "body" in self.fail_on:
            raise RuntimeError("body rejected")
        self.bodies.append(k)
        return 22

    def createCollisionShape(self, *a, **k):                 # the atmosphere must never create one
        raise AssertionError("the underwater look must not create collision geometry")


class Bounds:
    xmin, xmax, ymin, ymax = -1.0, 101.0, -2.0, 102.0


def test_atmosphere_sets_background_light_and_shadows_and_only_adds_visual_geometry():
    pb = FakePB()
    rep = apply(3, {}, pb=pb, world_bounds=Bounds())
    assert rep["applied"] is True and rep["shadows"] is True
    assert rep["background"] == DEFAULT_BACKGROUND and rep["light"] == DEFAULT_LIGHT
    assert len(pb.debug_calls) == 1
    _args, kw = pb.debug_calls[0]
    assert _args[0] == pb.COV_ENABLE_SHADOWS and _args[1] == 1
    assert kw["rgbBackground"] == DEFAULT_BACKGROUND and kw["lightPosition"] == DEFAULT_LIGHT
    assert kw["physicsClientId"] == 3
    assert len(pb.bodies) == 1
    body = pb.bodies[0]
    assert body["baseCollisionShapeIndex"] == -1             # VISUAL ONLY: nothing to collide with / ray-cast
    assert body["baseMass"] == 0.0 and body["basePosition"][2] == 0.0      # the water surface sits at z = 0
    assert rep["surface_body"] == 22
    assert len(pb.visual_shapes) == 1 and pb.visual_shapes[0][1]["rgbaColor"][3] < 1.0   # translucent sheet


def test_theme_disabled_or_failed_debug_visualizer_creates_nothing():
    pb = FakePB()
    rep = apply(0, {"underwater_theme": False}, pb=pb, world_bounds=Bounds())
    assert rep["applied"] is False and rep["surface_body"] is None and pb.debug_calls == [] and pb.bodies == []
    pb2 = FakePB(fail_on={"debug"})
    rep2 = apply(0, {}, pb=pb2, world_bounds=Bounds())          # must report, never raise
    assert rep2["applied"] is False and rep2["surface_body"] is None and pb2.bodies == []
    assert rep2["reason"]


def test_a_drawing_failure_disables_only_the_surface_and_never_raises():
    pb = FakePB(fail_on={"body"})
    rep = apply(0, {}, pb=pb, world_bounds=Bounds())
    assert rep["applied"] is True and rep["shadows"] is True     # background + light still applied
    assert rep["surface_body"] is None and rep["reason"]


def test_surface_frame_covers_the_world_bounds():
    from src.simulation.underwater_look import _surface_frame
    cx, cy, half = _surface_frame(Bounds())
    assert (cx, cy) == (50.0, 50.0)
    assert half >= 2.0 * max(Bounds.xmax - Bounds.xmin, Bounds.ymax - Bounds.ymin)
    cx2, cy2, half2 = _surface_frame(None)                       # no bounds: a generous default, no crash
    assert half2 >= 200.0


def test_gui_setup_really_applies_the_atmosphere():
    """_setup_gui_visuals (the code path run_visual_demo.py and the dashboard's 3-D window share) calls it."""
    pb = pytest.importorskip("pybullet")
    from src.simulation import underwater_look
    from src.simulation.run_simulation import _setup_gui_visuals
    calls, real = [], underwater_look.apply
    def spy(client, vis_cfg, pb=None, world_bounds=None):
        calls.append(client)
        return {"applied": True, "background": [0, 0, 0], "shadows": True, "light": [0, 0, -1], "surface_body": 1}
    underwater_look.apply = spy
    client = pb.connect(pb.DIRECT)
    try:
        class B:
            xmin, xmax, ymin, ymax = -1.0, 101.0, -1.0, 101.0
        class E:
            bounds = B()
        _setup_gui_visuals((5, 5, -5), [[12, 12, -5]], (20, 20, -5), E(), client,
                           {"camera_follow": False, "underwater_theme": True})
    finally:
        underwater_look.apply = real
        pb.disconnect(client)
    assert calls == [client]

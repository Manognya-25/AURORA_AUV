"""Presentation pass (rendering only): thin/dim sonar, highlighted AUV, small EKF dot, window options, demo flags.

Checked against the recording fake pybullet of tests/test_gui_visuals.py: WHAT is drawn and with which parameters. How it
looks on screen needs a human with the GUI window. Nothing here touches physics, sensors, the EKF or navigation."""
import numpy as np

from run_visual_demo import build_override, main
from src.simulation.gui_visuals import GuiVisuals, SONAR_FREE_RGB, SONAR_HIT_RGB
from tests.test_gui_visuals import Env, FakePB, truth_at
from tests.test_mapping_sonar import scan


class PB(FakePB):
    def __init__(self):
        super().__init__()
        self.visual_changes, self.parented_texts = [], []

    def getNumJoints(self, body, **k): return 2
    def changeVisualShape(self, body, link, **k): self.visual_changes.append((body, link, k))

    def addUserDebugText(self, text, pos, **k):
        super().addUserDebugText(text, pos, **k)
        if "parentObjectUniqueId" in k:
            self.parented_texts.append((text, k["parentObjectUniqueId"]))


def make(**cfg):
    pb = PB()
    return pb, GuiVisuals(0, cfg, Env(), pb=pb, auv_body_id=9)


def test_the_auv_is_recoloured_and_tagged_without_touching_anything_physical():
    pb, g = make()
    assert {c[:2] for c in pb.visual_changes} == {(9, -1), (9, 0), (9, 1)}               # base + every link
    assert all(set(c[2]) <= {"rgbaColor", "physicsClientId"} for c in pb.visual_changes)    # colour only: no shape/collision args
    assert pb.parented_texts == [("AURORA AUV", 9)]                                        # the tag follows the body
    pb2, _ = make(highlight_auv=False)
    assert pb2.visual_changes == [] and pb2.parented_texts == []


def test_estimate_marker_is_small_by_default_and_configurable():
    pb, g = make()
    assert g.estimate_radius == 0.15
    assert make(estimate_marker_radius_m=0.3)[1].estimate_radius == 0.3


def _fan():
    az = np.linspace(-30, 30, 40)
    return scan([8.0 if i % 4 == 0 else 30.0 for i in range(40)], az, [i % 4 == 0 for i in range(40)])


def test_sonar_fan_is_thin_and_sparse_by_default_and_every_hit_is_drawn():
    pb, g = make(update_every_n_steps=1)
    g.update(0, truth_at([0, 0, -5]), _fan(), None, None, 0.0)
    hits = [k for _, _, k in pb.lines if list(k["lineColorRGB"]) == list(SONAR_HIT_RGB)]
    free = [k for _, _, k in pb.lines if list(k["lineColorRGB"]) == list(SONAR_FREE_RGB)]
    assert len(hits) == 10 and all(k["lineWidth"] == 1.5 for k in hits)                    # all 10 real hits, thin
    assert 0 < len(free) <= 30 // 6 + 1 and all(k["lineWidth"] == 1.0 for k in free)       # no-return fan: every 6th ray only
    assert g.free_ray_stride == 6 and g.show_free_rays


def test_hits_only_mode_draws_no_free_rays():
    pb, g = make(update_every_n_steps=1, show_free_rays=False)
    g.update(0, truth_at([0, 0, -5]), _fan(), None, None, 0.0)
    assert all(list(k["lineColorRGB"]) != list(SONAR_FREE_RGB) for _, _, k in pb.lines)
    assert sum(list(k["lineColorRGB"]) == list(SONAR_HIT_RGB) for _, _, k in pb.lines) == 10


def test_window_options_and_demo_flags():
    import pathlib, re
    src = (pathlib.Path(__file__).resolve().parents[1] / "src" / "simulation" / "run_simulation.py").read_text(encoding="utf-8")
    ns = {}
    exec(src[src.index("def gui_window_options"):src.index("def _setup_gui_visuals")], ns)
    assert ns["gui_window_options"](None) == "--width=1600 --height=900"
    assert ns["gui_window_options"]({"window": {"width": 1920, "height": 1080}}) == "--width=1920 --height=1080"
    assert main(["--window", "bad", "--list"]) == 0                                       # --list exits before PyBullet / parsing
    ov = build_override("mixed_terrain_obstacles", 3, "long", {"show_free_rays": False, "window": {"width": 800, "height": 600}})
    assert ov["visualization"]["show_free_rays"] is False and ov["visualization"]["window"]["width"] == 800

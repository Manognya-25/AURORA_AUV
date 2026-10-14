"""Phase 5 final approach: the navigator's CURRENT goal is authoritative; GOAL_REACHED station-keeps on the ESTIMATE;
GoalMarker follows the authoritative goal. PyBullet-free (FakePB / estimate-only), so it runs everywhere."""
import numpy as np
import pytest

from src.planning.navigator import NavState, authoritative_goal
from src.simulation.gui_visuals import GoalMarker
from tests.test_gui_visuals import FakePB
from tests.test_navigation_logic import Rig, est, DT


def test_authoritative_goal_follows_the_navigator_and_falls_back_only_without_one():
    r = Rig()
    assert authoritative_goal(r.nav, (1, 2, -3)) == (30.0, 5.0, -5.0)
    r.nav.set_goal([20.0, 10.0, -4.0], 1.0)
    assert authoritative_goal(r.nav, (30.0, 5.0, -5.0)) == (20.0, 10.0, -4.0)
    r.nav.goal = None
    assert authoritative_goal(r.nav, (1, 2, -3)) == (1.0, 2.0, -3.0)


def test_goal_change_after_goal_reached_restarts_navigation_to_the_new_goal():
    r = Rig()
    r.step()
    r.step(pos=(30.0, 5.0, -5.0))
    assert r.nav.state == NavState.GOAL_REACHED
    r.nav.set_goal([40.0, 5.0, -5.0], r.t)
    assert r.nav.state == NavState.PLAN and tuple(r.nav.goal) == (40.0, 5.0, -5.0)
    r.step()
    assert r.nav.global_path.xy[-1] == pytest.approx([40.0, 5.0])


def test_goal_reached_station_keeps_toward_the_goal_using_only_the_estimate():
    r = Rig()
    r.step()
    r.step(pos=(29.7, 5.0, -5.0))
    assert r.nav.state == NavState.GOAL_REACHED
    out = r.step(pos=(29.7, 5.0, -5.0))
    assert out.info["mode"] == "station_keep"
    assert 0.0 < out.control["forward"] < 0.5                    # gently pushes toward the goal (+x), never hard
    out = r.step(pos=(30.05, 5.0, -5.0))                         # inside the deadband: no horizontal push
    assert abs(out.control["forward"]) < 0.05 and abs(out.control["lateral"]) < 0.05


def test_default_goal_speed_tolerance_blocks_a_fast_fly_through():
    r = Rig()
    r.step()
    r.nav.update(r.t, est(30.0, 5.0, -5.0, t=r.t, v=(0.6, 0, 0)), r.mapping, None)
    assert r.nav.state == NavState.FOLLOW_PATH
    r.nav.update(r.t + DT, est(30.0, 5.0, -5.0, t=r.t + DT, v=(0.05, 0, 0)), r.mapping, None)
    assert r.nav.state == NavState.GOAL_REACHED


def test_goal_marker_is_created_once_and_follows_goal_changes():
    pb = FakePB()
    pb.addUserDebugText = lambda text, pos, **k: (pb.texts.append((text, pos)), pb._id())[1]   # real PyBullet returns an item id
    m = GoalMarker(pb, 0, [35.0, 28.0, -5.0])
    assert len(pb.bodies) == 1 and pb.bodies[0]["basePosition"] == [35.0, 28.0, -5.0]
    assert m.update([35.0, 28.0, -5.0]) is False and not pb.marker_moves        # unchanged goal: nothing redrawn
    assert m.update([20.0, 10.0, -4.0]) is True
    assert pb.marker_moves[-1][1] == [20.0, 10.0, -4.0] and len(pb.bodies) == 1  # moved, not duplicated
    assert [t for t, _ in pb.texts].count("GOAL") == 2 and pb.removed             # label replaced
    assert pb.texts[-1][1] == [20.0, 10.0, -3.0]


def test_goal_marker_labels_the_goal_with_the_coordinates_that_were_entered():
    """The 3-D window shows the NUMBERS the operator set next to the marker, not just the word GOAL."""
    pb = FakePB()
    pb.addUserDebugText = lambda text, pos, **k: (pb.texts.append((text, pos)), pb._id())[1]
    m = GoalMarker(pb, 0, [35.0, 28.0, -5.0])
    assert ("goal 35.0, 28.0, -5.0", [35.0, 28.0, -4.45]) in pb.texts          # coordinates beside the marker
    assert pb.texts[-1] == ("GOAL", [35.0, 28.0, -4.0])                        # the marker label itself is unchanged
    m.update([30.0, 12.5, -4.0])
    labels = [t for t, _ in pb.texts]
    assert "goal 30.0, 12.5, -4.0" in labels and labels.count("GOAL") == 2     # follows the goal, one label per marker
    assert pb.texts[-1] == ("GOAL", [30.0, 12.5, -3.0])
    assert len(pb.removed) >= 2                                                # old label + old coordinates removed

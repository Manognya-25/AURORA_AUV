"""Backend 'scene feed': the world geometry published for the dashboard is the REAL generated world,
deterministic, the right size, and served only with conditions the simulation actually has."""
import json
from pathlib import Path

import pytest

from dashboard.app import create_app
from dashboard.simulation_manager import SimulationManager
from src.simulation.world_generator import generate_world
from tests.test_dashboard_api import CHILD, Client, wait_for

ROOT = Path(__file__).resolve().parents[1]


def _spec(scenario="mixed_terrain_obstacles", seed=3, count=None):
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text(encoding="utf-8"))
    env, sim = dict(cfg["environment"]), cfg["simulation"]
    env.update({"scenario": scenario, "seed": seed})
    if count is not None:
        env["obstacles"] = {"count": count}
    return generate_world(env, sim, (int(env["grid_size"]["rows"]), int(env["grid_size"]["cols"]), float(env["resolution"])))


def test_scene_payload_is_the_generated_world_and_obstacle_count_follows_the_request():
    for n in (10, 30, 53):
        sc = _spec("dense_obstacles", 5, n).scene_payload()
        assert len(sc["obstacles"]) == n                                   # every real body appears in the view, none invented
        assert len(sc["heights"]) <= 64 and len(sc["heights"][0]) <= 64
        assert sc["label"].startswith("GROUND TRUTH")


def test_scene_payload_is_deterministic_for_the_same_seed_and_scenario():
    a, b = _spec(seed=9).scene_payload(), _spec(seed=9).scene_payload()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert json.dumps(a, sort_keys=True) != json.dumps(_spec(seed=10).scene_payload(), sort_keys=True)


@pytest.fixture()
def env(tmp_path):
    mgr = SimulationManager(runtime_dir=tmp_path / "rt", child_module=CHILD, telemetry_timeout_s=1.5)
    yield Client(create_app(mgr).test_client()), mgr
    mgr.shutdown()


def test_api_scene_serves_the_running_worlds_geometry_and_only_real_conditions(env):
    cl, _ = env
    assert cl.get("/api/scene")[1]["available"] is False                    # nothing before a run
    cl.post("/api/environment/configure", {"scenario": "dense_obstacles", "seed": 11, "obstacle_count": 30})
    assert cl.post("/api/simulation/start")[0] == 202
    sc = wait_for(lambda: (lambda r: r[1] if r[1].get("available") else None)(cl.get("/api/scene")), timeout=60)
    assert len(sc["obstacles"]) == 30 and sc["scenario"] == "dense_obstacles" and sc["seed"] == 11
    cond = sc["conditions"]
    assert cond["wave_height_m"] is None                                   # waves are not simulated: never invented
    assert cond["current_mps"] == 0.0 and cond["sonar_max_range_m"] > 0


def test_page_no_longer_serves_the_underwater_view():
    html = (ROOT / "dashboard" / "templates" / "index.html").read_text(encoding="utf-8")
    assert "sceneCanvas" not in html and "scene.js" not in html and "Underwater view" not in html

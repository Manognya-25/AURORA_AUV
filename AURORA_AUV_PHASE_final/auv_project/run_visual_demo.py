"""Human demonstration of the REAL simulation in the real 3-D PyBullet window.

    python run_visual_demo.py                                   # mixed terrain + obstacles, seed 3
    python run_visual_demo.py --scenario dense_obstacles --seed 7 --obstacles 53
    python run_visual_demo.py --dynamics 6dof --no-underwater
    python run_visual_demo.py --list

THIS IS NOT A SECOND SCENE: the launcher calls exactly the same src.simulation.run_simulation.run() that the
dashboard's "3D PyBullet window" uses, with a config override. The same world generator builds the seabed and
the obstacles, the same sensors / EKF / mapper / navigator run, and the same collision + sonar geometry is used.

PRESENTATION ONLY: the orange AUV with its name tag, the thin dim sonar fan (hits in red), the small EKF dot, the
muted seabed/obstacle colours and the 1600x900 window change how the real scene is DRAWN; no body, sensor, EKF,
planner or controller value is affected.

You will see: an UNDERWATER scene (deep-blue water backdrop, one light with shadows, the water surface at z = 0),
the uneven seabed mesh, the generated pillars / walls / reef towers / wreck / boulders (real collision + visual
geometry), the AUV moving along its mission, the REAL sonar rays (red = measured hit, faint blue = no-return), a
yellow sphere at the EKF-estimated position, red points for the OCCUPIED cells of the sonar-built map, the planned
path (green) and the GOAL marker, which always sits on the navigator's authoritative goal.

Camera (src/simulation/gui_camera.py): third-person FOLLOW behind/above the AUV by default.
  Buttons in the side panel are numbered like the keys: 1 FOLLOW AUV, 2 TOP VIEW, 3 RESET VIEW, 4 ORBIT LEFT, 5 ORBIT RIGHT,
  6 ZOOM IN, 7 ZOOM OUT (PyBullet shows only ~2 characters of a button label, so the digit is the label)
  Keys: 1 follow, 2 top, 3 reset, 4 / 5 orbit, 6 / 7 zoom
  Mouse: drag = orbit, wheel = zoom (kept as an offset relative to the auto-camera; RESET VIEW clears it)
Close the window or press Ctrl+C to stop early.
"""
from __future__ import annotations

import argparse
import sys


LONG_ROUTE = [[5, 5, -5], [40, 12, -5], [62, 40, -5], [30, 62, -5], [75, 82, -5]]
STRAIGHT_ROUTE = [[5, 5, -5], [45, 45, -5], [85, 85, -5]]       # collinear: a straight corridor can follow it


def demo_route(scenario: str, mission: str = "long"):
    """Mission waypoints for the demo, or None to keep the short mission from configs/config.yaml.

    Obstacles are generated clear of the mission, so the route is chosen per scenario: the corridor is a
    straight pair of walls (needs a collinear route) and mapping_validation's fixed pillar/wall are defined
    relative to the default mission (so it keeps the default mission)."""
    if mission == "default" or scenario in ("mapping_validation", "legacy_week4"):
        return None
    return STRAIGHT_ROUTE if scenario == "narrow_corridor" else LONG_ROUTE


def build_override(scenario: str, seed: int, mission: str, vis: dict, dynamics: str | None = None,
                   obstacles: int | None = None, navigate: bool = False, goal: list | None = None) -> dict:
    """The config override the demo applies (also used by tools/verify_gui_camera.py so it runs the identical mission)."""
    override = {"environment": {"scenario": scenario, "seed": seed}, "visualization": vis}
    if obstacles is not None:
        # Exact number of PyBullet obstacle bodies generated for this run (deterministic for the seed above).
        override["environment"]["obstacles"] = {"count": int(obstacles)}
    route = demo_route(scenario, mission)
    if route is not None:
        # Obstacles are generated clear of whatever mission is configured, so the generator sees this route too.
        override["simulation"] = {"waypoints": route[1:], "goal_position": route[-1]}
        override["mapping"] = {"width_m": 110.0, "height_m": 110.0}      # keep the whole route inside the map
    if dynamics:
        override.setdefault("simulation", {})["dynamics"] = {"model": dynamics}
    if navigate:
        # Phase 5: the REAL EKF-only navigator (map -> cost map -> A* -> follower -> avoidance -> replanning) drives the AUV to a
        # goal the user can change while it runs (arrow keys + G in the window, or the dashboard). The obstacle generator keeps
        # the corridor to the goal clear, so the goal is also given to it as the mission goal.
        goal = [float(v) for v in (goal or (override.get("simulation", {}).get("goal_position")) or [35.0, 28.0, -5.0])]
        sim = override.setdefault("simulation", {})
        sim["goal_position"] = goal
        sim["waypoints"] = [goal]
        override["navigation"] = {"enabled": True, "goal_position": goal, "linger_after_terminal_s": 3600.0}
    return override


def main(argv=None) -> int:
    from src.simulation.world_generator import ALL_SCENARIOS, SCENARIOS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", default="mixed_terrain_obstacles", choices=list(ALL_SCENARIOS))
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--obstacles", type=int, default=None, metavar="N",
                    help="generate exactly N real PyBullet obstacle bodies (procedural scenarios only; "
                         "deterministic for the seed)")
    ap.add_argument("--goal", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="initial navigation goal in metres (z is negative = depth); change it live with the arrow keys + G")
    ap.add_argument("--no-panel-buttons", action="store_true",
                    help="hide PyBullet's side-panel camera buttons (their labels can be clipped on some displays); "
                         "keys 1-7, mouse drag/wheel and the dashboard camera buttons still work")
    ap.add_argument("--no-navigation", action="store_true",
                    help="old behaviour: follow the fixed waypoint route instead of the autonomous navigator (no live goal changes)")
    ap.add_argument("--dynamics", choices=["legacy", "6dof"], default=None)
    ap.add_argument("--speed", type=float, default=None, help="real-time factor (1.0 = real time)")
    ap.add_argument("--mission", choices=["long", "default"], default="long",
                    help="long: a ~200 m route across the whole world (default for the demo); default: the short config mission")
    ap.add_argument("--no-follow", action="store_true", help="do not move the camera with the AUV (free camera, whole-mission overview)")
    ap.add_argument("--camera", choices=["follow", "top"], default=None, help="starting camera view (switch live with the buttons / keys 1 2 3)")
    ap.add_argument("--cam-distance", type=float, default=None, help="follow-camera distance from the AUV in metres (default 3.8)")
    ap.add_argument("--no-underwater", action="store_true",
                    help="keep PyBullet's default background: no water colour, shadows or surface sheet")
    ap.add_argument("--no-rays", action="store_true")
    ap.add_argument("--hit-rays-only", action="store_true", help="draw only measured sonar hits, not the dim no-return fan")
    ap.add_argument("--ray-stride", type=int, default=None, metavar="N", help="draw every N-th no-return ray (default 6)")
    ap.add_argument("--window", default=None, metavar="WxH", help="3-D window size, e.g. 1920x1080 (default 1600x900)")
    ap.add_argument("--no-map-points", action="store_true")
    ap.add_argument("--list", action="store_true", help="list scenarios and exit")
    ap.add_argument("--config", default="configs/config.yaml")
    a = ap.parse_args(argv)
    if a.list:
        for n in ALL_SCENARIOS:
            print(f"{n:26s} {SCENARIOS[n]['description'] if n in SCENARIOS else 'Phase 1-3 world (Week 4 maps)'}")
        return 0
    if a.obstacles is not None:
        if a.scenario == "legacy_week4":
            print("--obstacles needs a procedural scenario: legacy_week4 builds its world from the Week-4 "
                  "occupancy map (choose e.g. dense_obstacles or mixed_terrain_obstacles).", file=sys.stderr)
            return 2
        if not 0 <= a.obstacles <= 200:
            print("--obstacles must be in [0, 200]", file=sys.stderr)
            return 2
    try:
        import pybullet  # noqa: F401
    except ImportError:
        print("PyBullet is not installed in this Python environment: pip install pybullet", file=sys.stderr)
        return 2
    from src.simulation.run_simulation import run
    vis = {"camera_follow": not a.no_follow, "show_sonar_rays": not a.no_rays,
           "show_mapped_obstacles": not a.no_map_points, "underwater_theme": not a.no_underwater}
    if a.speed is not None:
        vis["realtime_factor"] = a.speed
    if a.hit_rays_only:
        vis["show_free_rays"] = False
    if a.ray_stride is not None:
        vis["free_ray_stride"] = max(1, a.ray_stride)
    if a.window:
        try:
            w, h = (int(v) for v in a.window.lower().split("x"))
        except ValueError:
            print("--window must look like 1920x1080", file=sys.stderr)
            return 2
        vis["window"] = {"width": w, "height": h}
    cam = {k: v for k, v in (("mode", a.camera), ("distance_m", a.cam_distance)) if v is not None}
    if a.no_panel_buttons:
        cam["buttons"] = False
    cam["orbit_zoom_buttons"] = not a.no_panel_buttons                    # this demo offers the full side-panel camera controls
    vis["camera"] = cam
    override = build_override(a.scenario, a.seed, a.mission, vis, a.dynamics, a.obstacles,
                              navigate=not a.no_navigation, goal=a.goal)
    nav_goal = (override.get("navigation") or {}).get("goal_position")
    if nav_goal:
        print(f"Autonomous navigation to goal {nav_goal}. Change it live: arrow keys move the pending goal (Shift = 5 m), "
              "PgUp/PgDn depth, G or Enter applies, H cancels (or use the dashboard).")
    print(f"Opening the PyBullet window: scenario={a.scenario} seed={a.seed}"
          + (f" obstacles={a.obstacles}" if a.obstacles is not None else "")
          + (", underwater look ON" if not a.no_underwater else ", underwater look OFF"))
    print("Same run() / same world generator / same sensors as the dashboard's 3D window "
          "(this launcher does not build a second scene).")
    try:
        result = run(config_path=a.config, mode_override="GUI", cfg_override=override)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130
    m = (result.get("mapping") or {}).get("mapping", {})
    env = result.get("environment") or {}
    print(f"Done. final_status={result.get('final_status')} collisions={result.get('collisions')} "
          f"obstacles generated={env.get('obstacle_count')} "
          f"map known cells={m.get('known_cells')} occupied={m.get('occupied_cells')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

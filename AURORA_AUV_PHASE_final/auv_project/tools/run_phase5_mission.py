"""Run the REAL PyBullet autonomous mission and print / save a report.

    python tools/run_phase5_mission.py                 # DIRECT (headless), unpaced
    python tools/run_phase5_mission.py --gui           # PyBullet window: green path, orange detour, EKF marker
    python tools/run_phase5_mission.py --scenario narrow_corridor --goal 35 28 -5

AUV -> real sonar ray casts -> EKF -> online map -> A* -> follow -> avoid -> replan -> goal. Requires PyBullet.
Exit code 0 only if the navigator declared GOAL_REACHED from the ESTIMATE and the physics engine saw no collision.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

LAYOUT_A = [{"kind": "pillar", "x": 21, "y": 12.5, "radius_m": 0.8, "z_top_m": -2.0},
            {"kind": "pillar", "x": 24, "y": 11.2, "radius_m": 0.8, "z_top_m": -2.0}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--scenario", default="mapping_validation")
    ap.add_argument("--goal", type=float, nargs=3, default=[35.0, 28.0, -5.0])
    ap.add_argument("--no-extra-obstacles", action="store_true", help="do not add the two extra pillars")
    ap.add_argument("--dynamics", default=None, choices=["legacy", "6dof"])
    ap.add_argument("--max-steps", type=int, default=14000)
    ap.add_argument("--report", default="results/phase5_mission_report.json")
    a = ap.parse_args()
    try:
        import pybullet  # noqa: F401
    except ImportError:
        print("PyBullet is not installed: this mission cannot run (pip install pybullet).")
        return 2
    from src.simulation.run_simulation import run
    env = {"scenario": a.scenario}
    if not a.no_extra_obstacles and a.scenario == "mapping_validation":
        env["explicit_obstacles"] = LAYOUT_A
    over = {"environment": env, "navigation": {"enabled": True, "goal_position": a.goal, "linger_after_terminal_s": 1.0},
            "simulation": {"max_steps": a.max_steps}, "dashboard": {"realtime_factor": 1.0 if a.gui else 0.0}}
    if a.dynamics:
        over["simulation"]["dynamics"] = {"model": a.dynamics}
    res = run(mode_override="GUI" if a.gui else "DIRECT", cfg_override=over)
    nav = res["navigation"]
    summary = {
        "final_navigation_state": nav["state"], "failure_reason": nav["failure_reason"], "goal": nav["goal"],
        "goal_distance_estimate_m": nav["goal_distance_m"], "goal_distance_TRUE_evaluation_m": nav["final_true_goal_distance_m"],
        "physics_collisions": res["collisions"], "plans": nav["planner"]["plans"], "replans": nav["replanning"]["count"],
        "replan_reasons": nav["replanning"]["reasons"], "avoidance_maneuvers": nav["avoidance"]["count"],
        "last_path_length_m": nav["planner"]["path_length_m"], "mean_planning_time_ms": nav["planner"]["mean_planning_time_ms"],
        "state_history": nav["state_history"], "ekf": res["estimation"], "mapping_updates": res["mapping"]["mapping"]["update_count"],
        "sim_wall_time_s": res["metrics"].get("simulation_wall_time_s") if isinstance(res.get("metrics"), dict) else None,
    }
    os.makedirs(os.path.dirname(a.report) or ".", exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print("\n=== PHASE 5 MISSION REPORT ===")
    print(json.dumps({k: v for k, v in summary.items() if k != "state_history"}, indent=2, default=str))
    print("state history:")
    for h in nav["state_history"]:
        print(f"  t={h['t']:7.2f}s  {h['from']:>14s} -> {h['to']:<14s} ({h['reason']})")
    ok = nav["state"] == "GOAL_REACHED" and res["collisions"] == 0
    print("\nRESULT:", "SUCCESS" if ok else "NOT SUCCESSFUL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

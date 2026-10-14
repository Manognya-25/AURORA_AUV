"""TEMPORARY BENCHMARK (deleted after the check): how fast is the GUI 3-D window with/without overlays?

    python tools/bench_gui_overlays.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CASES = {
    "all overlays (baseline)": {},
    "no sonar rays": {"show_sonar_rays": False},
    "no rays + no trail/camera": {"show_sonar_rays": False, "show_mapped_obstacles": False, "camera_follow": False},
    "rays every 25 steps": {"update_every_n_steps": 25},
    "no mapped points": {"show_mapped_obstacles": False},
    "no underwater theme": {"underwater_theme": False},
}
STEPS = 700


def main() -> int:
    from src.simulation.run_simulation import run
    only = sys.argv[1:] or list(CASES)
    for name in only:
        vis = dict(CASES[name])
        ov = {"simulation": {"max_steps": STEPS},
              "navigation": {"enabled": True, "goal_position": [30.0, 28.0, -5.0]},
              "visualization": vis}
        t0 = time.perf_counter()
        res = run(config_path="configs/config.yaml", mode_override="GUI", cfg_override=ov)
        wall = time.perf_counter() - t0
        m = res.get("metrics") or {}
        print(f"RESULT {name!r}: {STEPS} steps in {wall:.1f} s wall -> {wall / STEPS * 1e3:.1f} ms/step "
              f"(real time needs <= 10 ms/step; sim time {m.get('simulated_time_s')})", flush=True)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYBULLET_GUI", "1")
    raise SystemExit(main())

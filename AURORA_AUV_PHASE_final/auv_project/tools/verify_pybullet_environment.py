"""Human-readable verification of the Phase 4 environment in REAL PyBullet.

    python tools/verify_pybullet_environment.py              # all scenarios: geometry, determinism, rays, sonar
    python tools/verify_pybullet_environment.py --mission    # also fly a real mission through every scenario
    python tools/verify_pybullet_environment.py --scenario mixed_terrain_obstacles --seed 3

Exit code 0 only if every check passes. Needs PyBullet (pip install pybullet).
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mission", action="store_true", help="also run a real bounded mission in each scenario")
    a = ap.parse_args(argv)
    try:
        import pybullet as pb
    except ImportError:
        print("PyBullet is not installed here: pip install pybullet", file=sys.stderr)
        return 2
    import numpy as np
    from src.sensors import Sonar, SonarConfig
    from src.simulation.auv import AUV
    from src.simulation.truth import snapshot_from_pybullet
    from tests.helpers import pybullet_checks as chk

    scenarios = [a.scenario] if a.scenario else chk.PROCEDURAL
    ok_all = True
    print(f"PyBullet {getattr(pb, '__version__', '?')}  seed={a.seed}\n")
    hdr = f"{'scenario':26s} {'bodies':>6s} {'coll+vis':>8s} {'mesh':>4s} {'determ.':>7s} {'ray agree':>9s} {'<0.5m':>6s} {'seabed err':>10s} {'sonar hits':>10s}"
    print(hdr)
    print("-" * len(hdr))
    for sc in scenarios:
        client, cfg, env = chk.build(pb, sc, a.seed)
        try:
            g = chk.geometry_report(pb, client, env)
            r = chk.ray_agreement(pb, client, env, cfg)
            try:
                s = chk.downward_ray_vs_seabed(pb, client, env)["max_abs_error_m"]
            except AssertionError:
                s = float("nan")
            auv = AUV(cfg["simulation"]["urdf_path"], tuple(cfg["simulation"]["start_position"]), cfg["simulation"], physics_client=client)
            sc_cfg = SonarConfig.from_config(cfg["sensors"]["sonar"])
            sc_cfg.dropout_probability, sc_cfg.range_noise_std_m, sc_cfg.detection_probability = 0.0, 0.0, 1.0
            m = Sonar(auv.body_id, client, sc_cfg).measure(snapshot_from_pybullet(auv.body_id, 0.0, client))
            hits = int(m.detected.sum())
        finally:
            pb.disconnect(client)
        c2, cfg2, env2 = chk.build(pb, sc, a.seed)
        try:
            again = chk.geometry_report(pb, c2, env2)["positions"]
        finally:
            pb.disconnect(c2)
        determ = again == g["positions"]
        geom_ok = (g["bodies"] == g["expected_bodies"] and not g["missing_collision"] and not g["missing_visual"])
        row_ok = geom_ok and g["seabed_is_mesh"] and determ and r["hit_agreement"] >= 0.95 and r["range_within_0.5m"] >= 0.90
        ok_all &= row_ok
        print(f"{sc:26s} {g['bodies']:>6d} {'ok' if geom_ok else 'FAIL':>8s} {'ok' if g['seabed_is_mesh'] else 'FAIL':>4s} "
              f"{'ok' if determ else 'FAIL':>7s} {r['hit_agreement']:>9.3f} {r['range_within_0.5m']:>6.2f} "
              f"{s:>10.2f} {hits:>10d}  {'PASS' if row_ok else 'FAIL'}")
    if a.mission:
        from src.simulation.run_simulation import run
        print("\nMission (real PyBullet, DIRECT, 4000 steps max):")
        for sc in scenarios:
            res = run(mode_override="DIRECT", cfg_override={"environment": {"scenario": sc, "seed": a.seed},
                                                            "simulation": {"max_steps": 4000}})
            m = res["mapping"]["mapping"] if res.get("mapping") else {}
            good = res["collisions"] == 0 and m.get("update_count", 0) > 20
            ok_all &= good
            print(f"  {sc:26s} status={res['final_status']:18s} collisions={res['collisions']} map_updates={m.get('update_count')} "
                  f"free={m.get('free_cells')} occupied={m.get('occupied_cells')}  {'PASS' if good else 'FAIL'}")
    print("\nRESULT:", "ALL CHECKS PASSED" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Measure how visible the AUV is from the follow camera in the REAL PyBullet visual demo world.

Nothing is simulated or faked. Three modes, all using the identical mission (run_visual_demo.build_override) and the
same Environment / AUV / GuiVisuals / CameraRig classes the demo uses:

  live    (default) run_visual_demo.main() itself in the PyBullet GUI, with the camera observed after every update.
          In p.GUI every PyBullet call is a GUI round trip (~12 ms under software GL), so a full mission can take hours
          on a machine without a GPU: use --max-wall to stop after N wall seconds (the report says how far it got).
  record  run the real mission headless (DIRECT) and save the AUV's true pose at every step (trajectory.npz).
  replay  open the real world in the PyBullet GUI, put the real AUV body at each recorded pose and let the real
          GuiVisuals / CameraRig follow it. Same world, same camera code, the whole mission in minutes. (Sonar rays, mapping
          and planner overlays are not re-run in replay; they do not affect the camera.)

At every sampled camera update it reads the camera PyBullet actually holds and measures, with PyBullet's own ray tests and a
segmentation render of the real scene from that exact camera:
  * line of sight : the first body hit by a ray from the camera to the AUV centre is the AUV itself
  * framing       : where the AUV centre lands in the image (NDC, 0,0 = centre)
  * AUV pixels    : pixels whose segmentation id is the AUV body in a 480x360 render
and writes metrics.json, render frames and (when a display exists) real window screenshots.
Exit code 0 only if the AUV passes every criterion at every sample.

    xvfb-run -a -s "-screen 0 1280x800x24 +extension GLX" python tools/verify_gui_camera.py --mode record
    xvfb-run -a -s "-screen 0 1280x800x24 +extension GLX" python tools/verify_gui_camera.py --mode replay
    xvfb-run -a -s "-screen 0 1280x800x24 +extension GLX" python tools/verify_gui_camera.py --mode live --max-wall 600
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

W, H = 480, 360


class Observer:
    """Measures the camera PyBullet really holds after each CameraRig.update."""

    def __init__(self, a, out: Path):
        self.a, self.out = a, out
        self.samples, self.shots = [], []
        self.next_t, self.last_shot, self.n = 0.0, -1e18, 0
        self.has_display = bool(os.environ.get("DISPLAY"))
        self.t0 = time.time()

    def measure(self, pb, rig, truth, sim_time):
        from PIL import Image
        cl = rig.client
        cam = pb.getDebugVisualizerCamera(physicsClientId=cl)
        V = np.array(cam[2], float).reshape(4, 4).T
        P = np.array(cam[3], float).reshape(4, 4).T
        eye = np.linalg.inv(V)[:3, 3]
        pos = np.asarray(truth.position, float)
        clip = P @ V @ np.append(pos, 1.0)
        ndc = (clip[:2] / clip[3]).tolist() if clip[3] > 1e-9 else [9.0, 9.0]
        hit = pb.rayTest(eye.tolist(), pos.tolist(), physicsClientId=cl)[0]
        los = hit[0] in (-1, rig.auv_id)
        img = pb.getCameraImage(W, H, cam[2], cam[3], renderer=pb.ER_TINY_RENDERER,
                                flags=pb.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX, physicsClientId=cl)
        mask = np.asarray(img[4]).reshape(H, W)
        px = int(np.sum((mask >= 0) & ((mask & 0xFFFFFF) == rig.auv_id)))
        s = {"t": round(float(sim_time), 2), "mode": rig.mode, "pos": [round(float(v), 2) for v in pos],
             "distance": round(float(cam[10]), 2), "pitch": round(float(cam[9]), 1), "yaw": round(float(cam[8]), 1),
             "eye_above_auv_m": round(float(eye[2] - pos[2]), 2), "ndc": [round(v, 3) for v in ndc],
             "los_clear": bool(los), "blocker_body": None if los else int(hit[0]), "auv_pixels": px,
             "rerouted": rig.stats["rerouted"], "pulled_in": rig.stats["pulled_in"]}
        self.samples.append(s)
        self.n += 1
        if self.n % 10 == 0:
            print(f"[verify] sim t={sim_time:6.1f}s wall={time.time() - self.t0:5.0f}s samples={self.n} los={s['los_clear']} "
                  f"auv_px={px} ndc={s['ndc']} dist={s['distance']}", flush=True)
        if len(self.shots) < self.a.shots and sim_time - self.last_shot >= self.a.shot_gap:
            idx = len(self.shots)
            rgb = np.asarray(img[2], np.uint8).reshape(H, W, 4)[:, :, :3]
            Image.fromarray(rgb).save(self.out / f"frame_{idx:02d}_t{sim_time:06.1f}.png")
            if self.has_display:
                subprocess.run(["import", "-window", "root", str(self.out / f"window_{idx:02d}_t{sim_time:06.1f}.png")],
                               check=False, capture_output=True)
            self.shots.append(s["t"])
            self.last_shot = sim_time
        return s

    def due(self, sim_time):
        if sim_time < self.next_t:
            return False
        self.next_t = sim_time + self.a.period
        return True

    def report(self, rc, note):
        a = self.a
        if not self.samples:
            print("NO CAMERA SAMPLES RECORDED (did the GUI start?)", file=sys.stderr)
            return 3
        px = np.array([s["auv_pixels"] for s in self.samples])
        ndc = np.array([max(abs(s["ndc"][0]), abs(s["ndc"][1])) for s in self.samples])
        los = np.array([s["los_clear"] for s in self.samples])
        last = self.samples[-1]
        summary = {"mode": a.mode, "note": note, "scenario": a.scenario, "seed": a.seed, "mission": a.mission,
                   "samples": len(self.samples), "sim_time_span_s": [self.samples[0]["t"], last["t"]],
                   "line_of_sight_clear_pct": round(100.0 * float(los.mean()), 2),
                   "auv_pixels_min": int(px.min()), "auv_pixels_median": int(np.median(px)), "auv_pixels_max": int(px.max()),
                   "ndc_abs_max": round(float(ndc.max()), 3), "ndc_abs_median": round(float(np.median(ndc)), 3),
                   "camera_distance_min": min(s["distance"] for s in self.samples),
                   "camera_distance_max": max(s["distance"] for s in self.samples),
                   "times_rerouted": last["rerouted"], "times_pulled_in": last["pulled_in"],
                   "criteria": {"min_pixels": a.min_pixels, "max_ndc": a.max_ndc, "line_of_sight": "100% clear"}}
        bad = [s for s in self.samples if (not s["los_clear"]) or s["auv_pixels"] < a.min_pixels
               or max(abs(s["ndc"][0]), abs(s["ndc"][1])) > a.max_ndc]
        summary["violations"] = len(bad)
        (self.out / "metrics.json").write_text(json.dumps({"summary": summary, "samples": self.samples, "violations": bad[:50]}, indent=1))
        print(json.dumps(summary, indent=1))
        if bad:
            print(f"{len(bad)} sample(s) violate the criteria, first: {bad[0]}", file=sys.stderr)
        return 0 if (not bad and rc in (0, 130)) else 1


def vis_overrides(a):
    vis = {"realtime_factor": 0.0}
    if a.camera:
        vis["camera"] = {"mode": a.camera}
    return vis


def do_record(a, out):
    import run_visual_demo
    from src.simulation.run_simulation import run
    override = run_visual_demo.build_override(a.scenario, a.seed, a.mission, {}, None)
    rows = []
    cb = lambda r: rows.append([r["time"], r["x"], r["y"], r["z"], r["roll"], r["pitch"], r["yaw"]])   # noqa: E731
    t0 = time.time()
    result = run(config_path=a.config, mode_override="DIRECT", telemetry_callback=cb, cfg_override=override)
    traj = np.array(rows, float)
    np.savez(out / "trajectory.npz", traj=traj, scenario=a.scenario, seed=a.seed, mission=a.mission)
    print(f"recorded {len(traj)} real steps ({traj[-1, 0]:.1f} sim s, wall {time.time() - t0:.0f}s) final_status={result.get('final_status')} "
          f"collisions={result.get('collisions')} -> {out / 'trajectory.npz'}")
    return 0


def do_replay(a, out):
    import pybullet as pb
    import pybullet_data
    import run_visual_demo
    from src.simulation import run_simulation as rs
    from src.simulation.auv import AUV
    from src.simulation.environment import Environment
    from src.simulation.gui_visuals import GuiVisuals

    traj = np.load(out / "trajectory.npz")["traj"] if (out / "trajectory.npz").exists() else np.load(a.trajectory)["traj"]
    override = run_visual_demo.build_override(a.scenario, a.seed, a.mission, vis_overrides(a), None)
    cfg = rs._deep_merge(rs.load_config(a.config), override)
    sim = cfg["simulation"]
    client = pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, float(sim["gravity"]))
    pb.setTimeStep(float(sim["time_step"]))
    env = Environment(cfg, physics_client=client)                  # the real generated world for this scenario / seed
    start = tuple(float(v) for v in sim["start_position"])
    goal = tuple(float(v) for v in sim["goal_position"])
    wps = [tuple(float(v) for v in w) for w in (sim.get("waypoints") or [list(goal)])]
    if wps[-1] != goal:
        wps.append(goal)
    auv = AUV(sim["urdf_path"], start, sim, physics_client=client)
    rs._setup_gui_visuals(start, wps, goal, env, client, cfg["visualization"])
    vis = GuiVisuals(client, cfg["visualization"], env, auv_body_id=auv.body_id, goal=goal)
    obs = Observer(a, out)
    rig = vis.camera
    print(f"replaying {len(traj)} real mission steps through the real world ({env.scenario}, seed {env.seed})", flush=True)
    i = 0
    for k in range(0, len(traj), a.stride):
        t, x, y, z, r, p_, yw = traj[k]
        q = pb.getQuaternionFromEuler([r, p_, yw])
        pb.resetBasePositionAndOrientation(auv.body_id, [x, y, z], q, physicsClientId=client)
        R = np.array(pb.getMatrixFromQuaternion(q)).reshape(3, 3)
        truth = SimpleNamespace(position=np.array([x, y, z]), rotation=R)
        vis.update(i, truth, None, None, None, float(t))
        i += 1
        if rig is not None and rig.stats["last"] is not None and obs.due(t):
            obs.measure(pb, rig, truth, t)
        if a.max_wall and time.time() - obs.t0 > a.max_wall:
            break
    rc = obs.report(0, f"replay of {len(traj)} recorded real steps; camera calls per update ~3")
    pb.disconnect()
    return rc


def do_live(a, out):
    import pybullet as pb
    from src.simulation import gui_camera
    import run_visual_demo
    obs = Observer(a, out)
    orig = gui_camera.CameraRig.update

    def observed(self, step, truth, sim_time):
        ok = orig(self, step, truth, sim_time)
        if ok and obs.due(sim_time):
            obs.measure(pb, self, truth, sim_time)
        if a.max_wall and time.time() - obs.t0 > a.max_wall:
            raise KeyboardInterrupt                                  # the demo handles this like Ctrl+C: clean stop
        return ok

    gui_camera.CameraRig.update = observed
    argv2 = ["--scenario", a.scenario, "--seed", str(a.seed), "--mission", a.mission, "--speed", "0"]
    if a.camera:
        argv2 += ["--camera", a.camera]
    rc = run_visual_demo.main(argv2)
    return obs.report(rc, "live run_visual_demo.main()" + (f", stopped after {a.max_wall:.0f} wall s" if a.max_wall else ""))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["live", "record", "replay"], default="live")
    ap.add_argument("--scenario", default="mixed_terrain_obstacles")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mission", choices=["long", "default"], default="long")
    ap.add_argument("--config", default="configs/config.yaml")
    ap.add_argument("--camera", choices=["follow", "top"], default=None)
    ap.add_argument("--period", type=float, default=1.0, help="sample every N simulated seconds")
    ap.add_argument("--shots", type=int, default=14, help="max number of frames / window screenshots kept")
    ap.add_argument("--shot-gap", type=float, default=15.0, help="simulated seconds between kept screenshots")
    ap.add_argument("--stride", type=int, default=4, help="replay: use every Nth recorded step (4 = the live camera cadence)")
    ap.add_argument("--max-wall", type=float, default=0.0, help="stop after N wall seconds (0 = run to the end)")
    ap.add_argument("--trajectory", default=None, help="replay: trajectory.npz to use (default: <out>/trajectory.npz)")
    ap.add_argument("--out", default="results/gui_camera_verification")
    ap.add_argument("--min-pixels", type=int, default=600, help="criterion: AUV pixels in the %dx%d render" % (W, H))
    ap.add_argument("--max-ndc", type=float, default=0.45, help="criterion: |NDC| of the AUV centre (0=centre, 1=edge)")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    return {"live": do_live, "record": do_record, "replay": do_replay}[a.mode](a, out)


if __name__ == "__main__":
    raise SystemExit(main())

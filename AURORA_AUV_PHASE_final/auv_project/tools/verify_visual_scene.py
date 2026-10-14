"""Objective check of what the 3-D window REALLY shows (real PyBullet GUI, real world, real AUV).

    python tools/verify_visual_scene.py                                   # the demo world (mixed terrain, seed 3)
    python tools/verify_visual_scene.py --scenario dense_obstacles --obstacles 30 --seed 7
    python tools/verify_visual_scene.py --no-underwater                   # compare with PyBullet's default look

It builds the world with the SAME code run_visual_demo.py / the dashboard's 3-D window use
(Environment + world_generator + underwater_look + gui_camera), renders frames with PyBullet's own OpenGL
renderer from the demo camera, saves them to results/gui_scene/ and checks:
  * the backdrop really is deep water (blue-dominant) when the underwater theme is on,
  * the seabed and the generated obstacles are visible (colour + body segmentation),
  * the AUV itself is visible in the frame,
  * the water surface sheet exists and created NO collision body (it must not exist for sonar/collision).
Exit code 0 only if every check passes. Needs a display (PyBullet GUI); 2 = no display / no PyBullet.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = os.path.join("results", "gui_scene")
W, H = 960, 600


def _as_np(a):
    import numpy as np
    return np.asarray(a).reshape(H, W, -1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="mixed_terrain_obstacles")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--obstacles", type=int, default=None, help="exact number of obstacle bodies (procedural scenarios)")
    ap.add_argument("--camera", choices=["follow", "top"], default="follow")
    ap.add_argument("--no-underwater", action="store_true", help="render PyBullet's default look for comparison")
    ap.add_argument("--frames", type=int, default=2, help="frames to save (first is the one analysed)")
    a = ap.parse_args(argv)
    try:
        import pybullet as p
    except ImportError:
        print("PyBullet is not installed: pip install pybullet", file=sys.stderr)
        return 2
    import numpy as np
    import run_visual_demo
    from src.mapping.map_types import MapConfig
    from src.simulation.commands import deep_merge
    from src.simulation.run_simulation import load_config
    from src.simulation import gui_camera

    vis = {"camera_follow": True, "underwater_theme": not a.no_underwater,
           "camera": {"mode": a.camera, "orbit_zoom_buttons": True}}
    override = run_visual_demo.build_override(a.scenario, a.seed, "long", vis, None, a.obstacles)
    cfg = deep_merge(load_config(), override)

    try:
        client = p.connect(p.GUI)
    except Exception as exc:                                     # noqa: BLE001
        print(f"Cannot open the PyBullet GUI here ({exc}); run this on a machine with a display.", file=sys.stderr)
        return 2
    if client < 0:
        print("Cannot open the PyBullet GUI here; run this on a machine with a display.", file=sys.stderr)
        return 2
    ok = True
    try:
        sim = cfg["simulation"]
        p.setGravity(0, 0, float(sim["gravity"]), physicsClientId=client)
        p.setTimeStep(float(sim["time_step"]), physicsClientId=client)
        from src.simulation.environment import Environment
        from src.simulation.underwater_look import apply as underwater_look
        env = Environment(cfg, physics_client=client)
        look = underwater_look(client, cfg.get("visualization"), world_bounds=env.bounds)
        from src.simulation.auv import AUV
        start = tuple(float(v) for v in sim["start_position"])
        route = run_visual_demo.demo_route(a.scenario, "long") or [list(sim["goal_position"])]
        auv = AUV(sim["urdf_path"], start, sim, physics_client=client)
        p.resetDebugVisualizerCamera(physicsClientId=client,
                                     **gui_camera.initial_follow_camera(start, route[0], (cfg.get("visualization") or {}).get("camera")))
        for _ in range(20):                                      # settle: render one real frame of the scene
            auv.step_underwater_physics()
            p.stepSimulation(physicsClientId=client)
        cam = gui_camera.initial_follow_camera(start, route[0], (cfg.get("visualization") or {}).get("camera"))
        os.makedirs(OUT_DIR, exist_ok=True)
        view = p.computeViewMatrixFromYawPitchRoll(cam["cameraTargetPosition"], cam["cameraDistance"],
                                                   cam["cameraYaw"], cam["cameraPitch"], 0, 2)
        proj = p.computeProjectionMatrixFOV(60.0, W / H, 0.1, 500.0)
        scen = "underwater" if not a.no_underwater else "default"
        paths = []
        for i in range(max(1, a.frames)):
            _w, _h, rgb, depth, seg = p.getCameraImage(W, H, viewMatrix=view, projectionMatrix=proj,
                                                       renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=client)
            img, segm = _as_np(rgb), _as_np(seg)
            path = os.path.join(OUT_DIR, f"scene_{a.scenario}_{a.seed}_{scen}_view{i}.png")
            _save_png(path, img)
            paths.append(path)
            if i == 0:
                report = _analyse(img, segm, env, auv, look, a)
        print("\n".join(f"  saved: {q}" for q in paths))
        for line in report:
            print(("  PASS  " if line[1] else "  FAIL  ") + line[0])
            ok &= bool(line[1])
        print(f"\nRESULT: {'ALL CHECKS PASSED' if ok else 'SOME CHECKS FAILED'}")
    finally:
        p.disconnect(client)
    return 0 if ok else 1


def _save_png(path, img):
    try:
        from PIL import Image
        Image.fromarray(img[..., :3].astype("uint8"), "RGB").save(path)
    except ImportError:                                          # fall back: ppm, readable everywhere
        raw = img[..., :3].astype("uint8").tobytes()
        with open(path.replace(".png", ".ppm"), "wb") as f:
            f.write(f"P6 {img.shape[1]} {img.shape[0]} 255\n".encode() + raw)


def _analyse(img, segm, env, auv, look, args):
    """Objective checks on the rendered frame (colour statistics + body segmentation)."""
    import numpy as np
    rgb = img[..., :3].astype(np.int32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    out = []
    theme_on = not args.no_underwater
    # 1. backdrop: deep dark water with the theme, PyBullet's bright void without it
    strip = np.s_[0:max(1, H // 6), :]
    br, bg_, bb = r[strip].mean(), g[strip].mean(), b[strip].mean()
    dark_water = (br + bg_ + bb) / 3.0 < 120.0 and bb > br
    out.append((f"backdrop {'is deep water' if theme_on else 'is PyBullet default'}: "
                f"mean rgb=({br:.0f},{bg_:.0f},{bb:.0f}) dark-and-blue={bool(dark_water)}",
                bool(dark_water) is theme_on))
    # 2. the seabed (sand) is really visible in the lower part of the frame
    low = np.s_[int(H * 0.55):, :]
    sand = ((r[low] > 100) & (r[low] > b[low] + 20) & (g[low] > b[low])).mean()
    out.append((f"seabed visible in the lower frame: {sand * 100:.1f}% sand-coloured pixels", sand > 0.02))
    # 3. the generated obstacles are really IN the picture (segmentation mask of their body ids)
    bodies = len(env.obstacle_ids)
    seen = [oid for oid in env.obstacle_ids if _body_pixels(segm, oid) > 0]
    out.append((f"{bodies} real obstacle bodies built, {len(seen)} of them visible in this frame", bodies > 0 and seen))
    # 4. the AUV is in the frame (segmentation mask of the AUV body id)
    auv_px = _body_pixels(segm, auv.body_id)
    out.append((f"AUV (body id {auv.body_id}) visible in the frame: {auv_px} pixels", auv_px > 20))
    # 5. the water surface: present with the theme, absent without it, and NEVER with collision geometry
    surf = look.get("surface_body")
    has_col = surf is not None and _has_collision_shape(surf)
    out.append((f"water surface body={surf} with the theme={theme_on}, collision shape={has_col} (must be False)",
                bool(surf is not None) is theme_on and not has_col))
    return out


def _body_pixels(segm, body_id: int) -> int:
    """Pixels whose segmentation value is this body (pybullet encodes objectUniqueId [+ link in the high bits])."""
    import numpy as np
    vals = segm[..., 0] if segm.ndim == 3 else segm
    m = (vals & 0xFFFFFF) == int(body_id)
    if m.sum() == 0:
        m = (vals & 0xFFFFFF) == int(body_id) + 1          # some builds use uniqueId + 1
    return int(m.sum())


def _has_collision_shape(body: int) -> bool:
    """True when the body really has collision geometry (what sonar / physics would hit)."""
    import pybullet as p
    try:
        return len(p.getCollisionShapeData(body, -1, physicsClientId=0)) > 0
    except Exception:                                        # noqa: BLE001 - no such body -> no collision
        return False


if __name__ == "__main__":
    raise SystemExit(main())

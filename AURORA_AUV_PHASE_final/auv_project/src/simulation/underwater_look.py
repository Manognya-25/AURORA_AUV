"""Underwater *look* for the 3-D PyBullet window (GUI mode only): a rendering of the REAL world, nothing else.

Everything drawn/configured here is how the water already behaves in the simulation, only made visible:
  * background colour : the water column around the arena (deep ocean blue instead of PyBullet's grey void)
  * light + shadows   : one light from above and shadow rendering, so relief, rocks and the hull read with
                        a depth/lighting feel instead of flat ambient light
  * water surface     : one flat sheet at z = 0 (the real air/water interface of the sim, where depth = 0)

HARD GUARANTEES (same rule as the other GUI-only modules: the 3-D overlays and the camera rig):
  * the water surface is a VISUAL-ONLY body (no collision shape), so sonar ray casts, DVL, collision
    detection, mapping, planning and physics are byte-for-byte unaffected;
  * nothing here is ever read back by the autonomy stack -- it renders, it never reports;
  * a failure prints one warning and disables only this feature, it can never crash or alter a run.

Config: `visualization.underwater_theme` (default on) in configs/config.yaml.
"""
from __future__ import annotations

# Deep-water backdrop (PyBullet clear colour). Water absorbs red first, so the distance reads blue-green.
DEFAULT_BACKGROUND = [0.024, 0.10, 0.17]
# One light, high and slightly toward the viewer: shadows fall below/behind rocks and terrain crests.
DEFAULT_LIGHT = [0.35, 0.45, -0.82]
SURFACE_RGBA = [0.45, 0.72, 0.86, 0.35]        # seen from below: a bright, semi-transparent ceiling
SURFACE_THICKNESS_M = 0.25


def apply(client: int, vis_cfg: dict | None, pb=None, world_bounds=None) -> dict:
    """Give the GUI window an underwater atmosphere. Returns a report of what was really applied
    (used by the dashboard label and by tests); never raises."""
    if pb is None:
        import pybullet as pb                                  # noqa: PLC0415 - GUI mode only
    cfg = vis_cfg or {}
    out = {"applied": False, "background": None, "shadows": False, "light": None,
           "surface_body": None, "reason": None}
    if not bool(cfg.get("underwater_theme", True)):
        out["reason"] = "visualization.underwater_theme is false"
        return out
    background = [float(v) for v in cfg.get("background_rgb", DEFAULT_BACKGROUND)]
    light = [float(v) for v in cfg.get("light_position", DEFAULT_LIGHT)]
    try:
        pb.configureDebugVisualizer(pb.COV_ENABLE_SHADOWS, 1, rgbBackground=background,
                                    lightPosition=light, physicsClientId=client)
        out["shadows"], out["background"], out["light"] = True, background, light
    except Exception as exc:                                   # noqa: BLE001 - reported once, never fatal
        out["reason"] = f"debug visualizer rejected the atmosphere: {type(exc).__name__}: {exc}"
        print(f"[underwater_look] {out['reason']}", flush=True)
        return out
    out["applied"] = True
    # Water surface: a thin, flat, VISUAL-ONLY slab at z = 0. Double-sided so it is visible from below
    # (and from above the surface in a top view) regardless of the renderer's face culling.
    if bool(cfg.get("show_surface", True)):
        try:
            cx, cy, half = _surface_frame(world_bounds)
            vis = pb.createVisualShape(pb.GEOM_BOX, halfExtents=[half, half, SURFACE_THICKNESS_M / 2.0],
                                       rgbaColor=list(SURFACE_RGBA), flags=getattr(pb, "VISUAL_SHAPE_DOUBLE_SIDED", 0),
                                       physicsClientId=client)
            out["surface_body"] = pb.createMultiBody(baseMass=0.0, baseCollisionShapeIndex=-1,
                                                     baseVisualShapeIndex=vis, basePosition=[cx, cy, 0.0],
                                                     physicsClientId=client)
        except Exception as exc:                               # noqa: BLE001 - reported once, never fatal
            out["surface_body"] = None
            out["reason"] = f"water surface not drawn: {type(exc).__name__}: {exc}"
            print(f"[underwater_look] {out['reason']}", flush=True)
    return out


def _surface_frame(world_bounds) -> tuple:
    """(centre_x, centre_y, half_size) covering the arena (plus the seabed padding) when bounds are known,
    otherwise a sheet big enough for any configured mission."""
    try:
        xmin, xmax = float(world_bounds.xmin), float(world_bounds.xmax)
        ymin, ymax = float(world_bounds.ymin), float(world_bounds.ymax)
    except Exception:                                          # noqa: BLE001 - no bounds: use a generous default
        return 75.0, 75.0, 400.0
    span = max(xmax - xmin, ymax - ymin, 1.0)
    cx, cy = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
    return cx, cy, float(max(2.0 * span, 200.0))

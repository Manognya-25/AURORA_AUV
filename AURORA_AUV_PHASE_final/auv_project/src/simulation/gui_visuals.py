"""3-D PyBullet window overlays for human demonstrations (GUI mode only).

Everything drawn here is REAL data, nothing is synthetic:
  * AUV, seabed and obstacles are the actual PyBullet bodies (not drawn here, just visible in the window)
  * sonar rays    : endpoints = sensor origin + MEASURED range * ray direction (hits in red, valid no-return
                    rays faint blue up to max range, invalid/dropped rays are not drawn at all)
  * estimate      : a translucent sphere at the EKF ESTIMATED position (the AUV body is the true state)
  * mapped points : OCCUPIED cells of the sensor-derived occupancy grid, at the estimated depth slice
  * trail         : the AUV's actual path
  * camera        : third-person follow camera behind/above the AUV (src/simulation/gui_camera.py)
This module is simulation/debug side: it may read the true pose to place overlays. It never feeds
anything back into mapping, the EKF or the controller. A drawing failure prints one warning and disables
only that overlay; it can never crash or alter the simulation.
"""
from __future__ import annotations

import numpy as np

SONAR_HIT_RGB = (1.0, 0.25, 0.1)
SONAR_FREE_RGB = (0.25, 0.55, 1.0)       # thin and sparse (stride 6) rather than recoloured: debug lines have no alpha
AUV_RGBA = (1.0, 0.45, 0.05, 1.0)         # high-contrast orange hull colour (rendering only; collision/inertia untouched)
TRAIL_RGB = (0.2, 0.9, 1.0)
ESTIMATE_RGBA = (1.0, 0.9, 0.1, 0.45)
MAPPED_RGB = (1.0, 0.1, 0.1)
GLOBAL_PATH_RGB = (0.2, 1.0, 0.3)       # Phase 5: planned global path (from the planner, EKF frame)
DETOUR_RGB = (1.0, 0.6, 0.1)            # Phase 5: local avoidance detour


GOAL_RGBA = (0.95, 0.1, 0.1, 1.0)


class GoalMarker:
    """Visual-only GOAL sphere + label that follows the AUTHORITATIVE goal (the navigator's current goal). It is moved only
    when the goal actually changes; no collision shape, so physics is untouched."""

    def __init__(self, pb, client: int, goal):
        self.p, self.client = pb, client
        self.goal = None
        self.body = self.text = self.coord = None
        self.moves = 0
        self.update(goal)

    def update(self, goal) -> bool:
        g = [float(v) for v in goal]
        if self.goal is not None and np.allclose(self.goal, g):
            return False
        p = self.p
        if self.body is None:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.55, rgbaColor=list(GOAL_RGBA), physicsClientId=self.client)
            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                          basePosition=g, physicsClientId=self.client)
        else:
            p.resetBasePositionAndOrientation(self.body, g, [0, 0, 0, 1], physicsClientId=self.client)
        if self.text is not None:
            p.removeUserDebugItem(self.text, physicsClientId=self.client)
        if self.coord is not None:
            p.removeUserDebugItem(self.coord, physicsClientId=self.client)
        # The NUMBERS the operator set, right next to the marker: the 3-D window shows the goal that was entered, not
        # just the word "GOAL". Added BEFORE the label so the label stays the last item drawn (rendering order only).
        self.coord = p.addUserDebugText(f"goal {g[0]:.1f}, {g[1]:.1f}, {g[2]:.1f}", [g[0], g[1], g[2] + 0.55],
                                        textColorRGB=[1.0, 0.78, 0.25], textSize=1.15, lifeTime=0,
                                        physicsClientId=self.client)
        self.text = p.addUserDebugText("GOAL", [g[0], g[1], g[2] + 1.0], textColorRGB=[1, 0.2, 0.2], textSize=1.3,
                                       lifeTime=0, physicsClientId=self.client)
        self.goal = g
        self.moves += 1
        return True


class GuiVisuals:
    def __init__(self, client: int, vis_cfg: dict | None, environment=None, pb=None, auv_body_id: int | None = None,
                 goal=None):
        if pb is None:
            import pybullet as pb                                  # noqa: PLC0415 - GUI mode only
        self.p, self.client = pb, client
        c = vis_cfg or {}
        self.show_rays = bool(c.get("show_sonar_rays", True))
        self.show_estimate = bool(c.get("show_estimate_marker", True))
        self.show_mapped = bool(c.get("show_mapped_obstacles", True))
        self.follow = bool(c.get("camera_follow", True))
        self.every_n = max(1, int(c.get("update_every_n_steps", 5)))
        self.mapped_period_s = float(c.get("mapped_update_period_s", 1.0))
        self.max_points = int(c.get("max_mapped_points", 20000))
        self.free_ray_stride = max(1, int(c.get("free_ray_stride", 6)))
        self.show_free_rays = bool(c.get("show_free_rays", True))
        self.hit_ray_width = float(c.get("hit_ray_width", 1.5))
        self.free_ray_width = float(c.get("free_ray_width", 1.0))
        self.estimate_radius = float(c.get("estimate_marker_radius_m", 0.15))
        self.auv_rgba = [float(v) for v in c.get("auv_rgba", AUV_RGBA)]
        self.auv_label = c.get("auv_label", "AURORA AUV")
        self.auv_body_id = auv_body_id
        self._disabled: set[str] = set()
        self._warned: set[str] = set()
        self._marker = None
        self._pts_id = None
        self._prev_pos = None
        self._last_scan_t = None
        self._last_mapped_t = -1e18
        self._path_version = None
        self._path_ids: list = []
        self._ray_life = 0.5
        self.draw_calls = {"lines": 0, "points": 0, "text": 0, "camera": 0}
        self.camera = None
        if self.follow:
            from src.simulation.gui_camera import CameraRig            # noqa: PLC0415
            self.camera = self._guard("camera", lambda: CameraRig(client, c.get("camera"), self.p, auv_body_id))
        if auv_body_id is not None and bool(c.get("highlight_auv", True)):
            self._guard("auv_look", self._style_auv)
        if self.show_estimate:
            self._guard("estimate", self._make_marker)
        self._guard("label", lambda: self._label(environment))
        self.goal_marker = None
        if goal is not None:
            self.goal_marker = self._guard("goal", lambda: GoalMarker(self.p, client, goal))

    def update_goal(self, goal) -> None:
        """Move the GOAL marker to the authoritative goal (no-op unless it changed)."""
        if self.goal_marker is not None:
            self._guard("goal", lambda: self.goal_marker.update(goal))

    # ------------------------------------------------------------------ helpers
    def _guard(self, feature: str, fn):
        if feature in self._disabled:
            return None
        try:
            return fn()
        except Exception as exc:                                    # noqa: BLE001 - reported once, never fatal
            self._disabled.add(feature)
            if feature not in self._warned:
                self._warned.add(feature)
                print(f"[gui_visuals] overlay '{feature}' disabled after error: {type(exc).__name__}: {exc}", flush=True)
            return None

    def _style_auv(self):
        """Rendering only: a high-contrast hull colour and a name tag that follows the body (PyBullet attaches the text to
        the body, so there is no per-step call). Shape, mass, inertia and collision geometry are not touched."""
        p, b = self.p, self.auv_body_id
        for link in [-1] + list(range(int(p.getNumJoints(b, physicsClientId=self.client)))):
            p.changeVisualShape(b, link, rgbaColor=list(self.auv_rgba), physicsClientId=self.client)
        if self.auv_label:
            p.addUserDebugText(str(self.auv_label), [0.0, 0.0, 0.75], textColorRGB=[1.0, 0.8, 0.3], textSize=1.2, lifeTime=0,
                               parentObjectUniqueId=b, physicsClientId=self.client)
            self.draw_calls["text"] += 1

    def _make_marker(self):
        p = self.p
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.estimate_radius, rgbaColor=list(ESTIMATE_RGBA), physicsClientId=self.client)
        self._marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                         basePosition=[0, 0, 0], physicsClientId=self.client)

    def _label(self, environment):
        p = self.p
        if environment is None:
            return
        d = environment.describe()
        b = environment.bounds
        text = (f"SCENARIO {d['scenario']} (seed {d['seed']}) | {d.get('obstacle_count', '?')} collision bodies | "
                f"seabed z {d.get('seabed_z_min', float('nan')):.1f}..{d.get('seabed_z_max', float('nan')):.1f} m")
        legend = "orange AUV | red = sonar hits & mapped obstacles | thin blue = no-return | small yellow dot = EKF estimate | green = path | cyan = trail"
        camline = "camera (keys = side-panel buttons 1-7): 1 follow | 2 top | 3 reset | 4/5 orbit | 6/7 zoom | goal: arrows move, PgUp/PgDn depth, G apply, H cancel"
        mx, my = 0.5 * (b.xmin + b.xmax), b.ymin
        p.addUserDebugText(text, [mx, my, -1.0], textColorRGB=[1, 1, 1], textSize=1.4, lifeTime=0, physicsClientId=self.client)
        p.addUserDebugText(legend, [mx, my, -2.2], textColorRGB=[0.8, 0.9, 1.0], textSize=1.1, lifeTime=0, physicsClientId=self.client)
        p.addUserDebugText(camline, [mx, my, -3.4], textColorRGB=[0.7, 0.85, 1.0], textSize=1.0, lifeTime=0, physicsClientId=self.client)
        self.draw_calls["text"] += 3

    # ------------------------------------------------------------------ per-step update
    def update(self, step: int, truth, sonar, estimate, mapping, sim_time: float) -> None:
        """truth: TruthSnapshot (placement of overlays only); sonar: SonarMeasurement or None;
        estimate: EstimatedState or None; mapping: MappingSystem or None."""
        if self.camera is not None and truth is not None:                   # own (faster) cadence: smooth follow
            if self._guard("camera", lambda: self.camera.update(step, truth, sim_time)):
                self.draw_calls["camera"] += 1
        if step % self.every_n != 0:
            return
        if truth is not None:
            self._guard("trail", lambda: self._trail(truth))
        if self.show_estimate and estimate is not None and self._marker is not None:
            self._guard("estimate", lambda: self.p.resetBasePositionAndOrientation(
                self._marker, [float(v) for v in estimate.position], [0, 0, 0, 1], physicsClientId=self.client))
        if self.show_rays and sonar is not None and truth is not None:
            self._guard("rays", lambda: self._rays(truth, sonar))
        if self.show_mapped and mapping is not None and estimate is not None \
                and sim_time - self._last_mapped_t >= self.mapped_period_s:
            self._last_mapped_t = sim_time
            self._guard("mapped", lambda: self._mapped(mapping, estimate))

    def update_path(self, payload: dict | None, z: float) -> None:
        """Phase 5: draw the planned global path (green) and the active detour (orange) at depth z. Redrawn only when the
        navigator's path version changes. The points are the planner's own output (the EKF/map frame); nothing is invented."""
        if payload is None or payload.get("version") == self._path_version:
            return
        self._path_version = payload.get("version")
        self._guard("path", lambda: self._draw_path(payload, float(z)))

    def _draw_path(self, payload: dict, z: float) -> None:
        for i in self._path_ids:
            self.p.removeUserDebugItem(i, physicsClientId=self.client)
        self._path_ids = []
        for key, rgb, width in (("global", GLOBAL_PATH_RGB, 4.0), ("detour", DETOUR_RGB, 5.0)):
            pts = payload.get(key)
            if not pts or len(pts) < 2:
                continue
            for a, b in zip(pts[:-1], pts[1:]):
                self._path_ids.append(self.p.addUserDebugLine([float(a[0]), float(a[1]), z], [float(b[0]), float(b[1]), z],
                                                              lineColorRGB=list(rgb), lineWidth=width, lifeTime=0,
                                                              physicsClientId=self.client))
                self.draw_calls["lines"] += 1

    def _trail(self, truth):
        pos = [float(v) for v in truth.position]
        if self._prev_pos is not None:
            self.p.addUserDebugLine(self._prev_pos, pos, lineColorRGB=list(TRAIL_RGB), lineWidth=3.0, lifeTime=0,
                                    physicsClientId=self.client)
            self.draw_calls["lines"] += 1
        self._prev_pos = pos

    @staticmethod
    def ray_geometry(truth, sonar):
        """World-frame (origin, unit directions) of every sonar ray, from the measurement's own extrinsics."""
        mount = np.zeros(3) if sonar.mount_position is None else np.asarray(sonar.mount_position, float)
        Rm = np.eye(3) if sonar.mount_rotation is None else np.asarray(sonar.mount_rotation, float)
        az, el = np.asarray(sonar.azimuths, float), np.asarray(sonar.elevations, float)
        d_s = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        R = np.asarray(truth.rotation, float)
        return np.asarray(truth.position, float) + R @ mount, (R @ Rm @ d_s.T).T

    def _rays(self, truth, sonar):
        t = float(sonar.timestamp)
        if self._last_scan_t is not None and t == self._last_scan_t:
            return                                                 # same scan as last time: nothing new to draw
        self._last_scan_t = t
        origin, dirs = self.ray_geometry(truth, sonar)
        status = sonar.ray_status()
        ranges = np.asarray(sonar.value, float)
        for i in range(len(status)):
            if status[i] < 0:
                continue                                           # dropped / blanked ray: no information, no line
            if status[i] == 1:
                end, rgb, width = origin + ranges[i] * dirs[i], SONAR_HIT_RGB, self.hit_ray_width
            elif self.show_free_rays and i % self.free_ray_stride == 0:
                end, rgb, width = origin + float(sonar.max_range) * dirs[i], SONAR_FREE_RGB, self.free_ray_width
            else:
                continue
            self.p.addUserDebugLine([float(v) for v in origin], [float(v) for v in end], lineColorRGB=list(rgb),
                                    lineWidth=width, lifeTime=self._ray_life, physicsClientId=self.client)
            self.draw_calls["lines"] += 1

    def _mapped(self, mapping, estimate):
        from src.mapping.map_types import OCCUPIED                   # noqa: PLC0415
        grid = mapping.grid
        rows, cols = np.nonzero(grid.state() == OCCUPIED)
        if len(rows) == 0:
            if self._pts_id is not None:
                self.p.removeUserDebugItem(self._pts_id, physicsClientId=self.client)
                self._pts_id = None
            return
        if len(rows) > self.max_points:
            sel = np.linspace(0, len(rows) - 1, self.max_points).astype(int)
            rows, cols = rows[sel], cols[sel]
        x, y = grid.geom.grid_to_world(rows, cols)
        z = np.full_like(x, float(estimate.position[2]))
        pts = np.stack([x, y, z], axis=1).astype(float).tolist()
        kw = {"replaceItemUniqueId": self._pts_id} if self._pts_id is not None else {}
        self._pts_id = self.p.addUserDebugPoints(pts, [list(MAPPED_RGB)] * len(pts), pointSize=7.0, lifeTime=0,
                                                 physicsClientId=self.client, **kw)
        self.draw_calls["points"] += 1

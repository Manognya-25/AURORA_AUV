"""Live 3-D PyBullet window for a RUNNING dashboard simulation that has none of its own.

WHY THIS EXISTS. `dashboard.headless: true` starts the run on PyBullet's DIRECT (windowless) connection.
PyBullet cannot attach a GUI window to a physics server that is already running, so the only way to SHOW a
run that is already going -- without stopping or restarting the vehicle -- is a separate viewer process
that renders that same run in its own GUI window. The dashboard's "3D PyBullet window" control starts and
stops exactly this process while the simulation keeps running untouched.

IT IS A VIEWER, NOT A SECOND SIMULATION.
  * It never steps the robot's physics: the run process owns the world, sensors, EKF, mapper, planner and
    controller. This process teleports a visual AUV along the pose the RUN publishes and renders overlays.
  * It reads only what the run already publishes (state.json, maps.json, scene.json, child.json/status.json)
    and writes only its own two files (viewer_status.json, viewer_commands.jsonl acks). It cannot change a
    single simulation value: gravity is switched off here and no force is ever applied.
  * Every drawn number is the run's own telemetry: true pose and heading (debug_ground_truth), the measured
    sonar rays (status + range per ray), the EKF estimate marker, the sensor-derived occupancy cells, the
    navigator's planned path and its authoritative goal.

The world geometry is rebuilt from the SAME config the run was started with (config.yaml + the run's
config_override.json), with the same src.simulation.world_builder: seabed mesh, obstacles, colours. The
underwater look, follow/top/orbit/zoom camera rig and goal marker are the same modules the in-process GUI
run uses (underwater_look.py, gui_camera.py, gui_visuals.GoalMarker).

Camera: the dashboard's 3-D camera buttons, the side-panel buttons and the keys 1-7 all act on THIS window
while it is open (dashboard -> viewer_commands.jsonl -> the rig below).

Exit: the window being closed, a stop command, `viewer_stop.request`, the dashboard process exiting, or (with
`visualization.keep_open_after_run: false`) the run ending. By DEFAULT a run that ends on its own -- the robot
reaching the goal, STOPPED, FAILED -- leaves the window open on the last frame, so the operator can still look at
where the vehicle ended up; the window is then closed explicitly (its own close button or the dashboard control).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.simulation.live_io import CommandChannel, atomic_write_json, read_json_retry
from src.utils.procutil import pid_alive
from src.utils.transforms import euler_to_quat

STATUS_FILE = "viewer_status.json"
COMMANDS_FILE = "viewer_commands.jsonl"
STOP_FILE = "viewer_stop.request"
TERMINAL_RUN_STATES = ("COMPLETED", "ENDED_BEFORE_GOAL", "FAILED", "STOPPED")
LOOP_HZ = 30.0
STATUS_HZ = 2.0
HIT_RGB = (1.0, 0.25, 0.1)
FREE_RGB = (0.25, 0.55, 1.0)
TRAIL_RGB = (0.2, 0.9, 1.0)
ESTIMATE_RGBA = (1.0, 0.9, 0.1, 0.45)
MAPPED_RGB = (1.0, 0.1, 0.1)
PATH_RGB = (0.2, 1.0, 0.3)
DETOUR_RGB = (1.0, 0.6, 0.1)
AUV_RGBA = (1.0, 0.45, 0.05, 1.0)
COMMAND_LOG_LEN = 20


def rotation_from_euler(rpy) -> np.ndarray:
    """3x3 body->world rotation from (roll, pitch, yaw) — same convention as euler_to_quat()."""
    r, p, y = (float(v) for v in rpy)
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


class Pose:
    """What the camera rig and the overlays need: a world position and a body->world rotation matrix."""

    __slots__ = ("position", "rotation", "rpy")

    def __init__(self, position, rpy=None, rotation=None):
        self.position = np.asarray(position, float)
        self.rpy = None if rpy is None else np.asarray(rpy, float)
        self.rotation = rotation if rotation is not None else rotation_from_euler(rpy if rpy is not None else (0.0, 0.0, 0.0))


class ViewerState:
    """Owns viewer_status.json: the dashboard reads what this window is really doing, never an assumption."""

    def __init__(self, run_dir: Path, run_id: str, parent_pid: int | None):
        self.dir, self.run_id, self.parent_pid = Path(run_dir), run_id, parent_pid
        self.state = "STARTING"
        self.error: str | None = None
        self.frames = 0
        self.command_log: list = []
        self.camera = None
        self.scene = None
        self._t = -1e18

    def write(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._t < 1.0 / STATUS_HZ:
            return
        self._t = now
        try:
            atomic_write_json(self.dir / STATUS_FILE, {
                "state": self.state, "open": self.state == "OPEN", "pid": os.getpid(),
                "run_id": self.run_id, "error": self.error, "frames": self.frames,
                "camera": self.camera, "scene": self.scene, "command_log": list(self.command_log)[-COMMAND_LOG_LEN:],
                "updated": time.time(),
            })
        except OSError as exc:                      # never let reporting the state kill the window
            print(f"[live_viewer] could not write {STATUS_FILE}: {exc}", file=sys.stderr, flush=True)


class _Stop(Exception):
    """Internal: the dashboard asked this window to close."""


class Mirror:
    """Draws the run's published state into the viewer's own PyBullet window (visual only, guarded)."""

    def __init__(self, p, client: int, cfg: dict, environment, auv_id: int, run_dir: Path):
        self.p, self.client = p, client
        self.run_dir = Path(run_dir)
        v = cfg.get("visualization") or {}
        self.show_rays = bool(v.get("show_sonar_rays", True))
        self.show_free_rays = bool(v.get("show_free_rays", True))
        self.free_ray_stride = max(1, int(v.get("free_ray_stride", 6)))
        self.hit_ray_width, self.free_ray_width = float(v.get("hit_ray_width", 1.5)), float(v.get("free_ray_width", 1.0))
        self.show_estimate = bool(v.get("show_estimate_marker", True))
        self.show_mapped = bool(v.get("show_mapped_obstacles", True))
        self.mapped_period_s = float(v.get("mapped_update_period_s", 1.0))
        self.max_mapped_points = int(v.get("max_mapped_points", 20000))
        self.estimate_radius = float(v.get("estimate_marker_radius_m", 0.15))
        self.every_n = max(1, int(v.get("update_every_n_steps", 5)))
        self.auv_id = auv_id
        self._disabled: set[str] = set()
        self._prev = None
        self._last_scan_t = None
        self._last_mapped_t = -1e18
        self._path_key = None
        self._path_ids: list = []
        self._marker = None
        self._pts_id = None
        self.draw_calls = {"lines": 0, "points": 0, "text": 0}
        self.goal_marker = None

    # ------------------------------------------------------------------ guarded drawing
    def guard(self, feature: str, fn):
        if feature in self._disabled:
            return None
        try:
            return fn()
        except Exception as exc:                                   # noqa: BLE001 - one overlay, never the window
            self._disabled.add(feature)
            print(f"[live_viewer] overlay '{feature}' disabled after error: {type(exc).__name__}: {exc}",
                  file=sys.stderr, flush=True)
            return None

    # ------------------------------------------------------------------ one-time setup
    def start(self, environment, mission: dict | None) -> None:
        p, c = self.p, self.client
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1, physicsClientId=c)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0, physicsClientId=c)
        for flag in ("COV_ENABLE_RGB_BUFFER_PREVIEW", "COV_ENABLE_DEPTH_BUFFER_PREVIEW"):
            if hasattr(p, flag):                                   # PyBullet's default preview panels: clutter
                p.configureDebugVisualizer(getattr(p, flag), 0, physicsClientId=c)
        if self.show_estimate:
            self.guard("estimate", self._make_marker)
        b = getattr(environment, "bounds", None)
        if b is not None:
            d = environment.describe()
            mx, my = 0.5 * (b.xmin + b.xmax), b.ymin
            self.guard("label", lambda: p.addUserDebugText(
                f"LIVE 3-D WINDOW of the running simulation | scenario {d.get('scenario')} (seed {d.get('seed')}) | "
                f"{d.get('obstacle_count', '?')} collision bodies | the simulation itself is untouched by this window",
                [mx, my, -1.0], textColorRGB=[1, 1, 1], textSize=1.4, lifeTime=0, physicsClientId=c))
            self.guard("label", lambda: p.addUserDebugText(
                "orange AUV = true pose (from the run's telemetry) | red = sonar hits & mapped obstacles | thin blue = "
                "no-return | yellow dot = EKF estimate | green = planned path | cyan = trail",
                [mx, my, -2.2], textColorRGB=[0.8, 0.9, 1.0], textSize=1.1, lifeTime=0, physicsClientId=c))
            self.guard("label", lambda: p.addUserDebugText(
                "camera: 1 follow | 2 top | 3 reset | 4/5 orbit | 6/7 zoom (side panel or keys; the dashboard's camera "
                "buttons act on THIS window)",
                [mx, my, -3.4], textColorRGB=[0.7, 0.85, 1.0], textSize=1.0, lifeTime=0, physicsClientId=c))
        if mission and mission.get("start"):
            s = [float(v) for v in mission["start"]]
            self.guard("start_marker", lambda: p.addUserDebugText("START", [s[0], s[1], s[2] + 0.9],
                                                                 textColorRGB=[0.1, 1, 0.2], textSize=1.3, lifeTime=0,
                                                                 physicsClientId=c))

    def _make_marker(self):
        p, c = self.p, self.client
        vis = p.createVisualShape(p.GEOM_SPHERE, radius=self.estimate_radius, rgbaColor=list(ESTIMATE_RGBA),
                                  physicsClientId=c)
        self._marker = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                        basePosition=[0, 0, 0], physicsClientId=c)

    def set_goal(self, goal) -> None:
        """Visual-only GOAL sphere that follows the navigator's authoritative goal (reused from the GUI run)."""
        if goal is None:
            return
        try:
            if self.goal_marker is None:
                from src.simulation.gui_visuals import GoalMarker        # noqa: PLC0415 - GUI only
                self.goal_marker = GoalMarker(self.p, self.client, goal)
            else:
                self.goal_marker.update([float(v) for v in goal])
        except Exception as exc:                                          # noqa: BLE001
            if "goal" not in self._disabled:
                self._disabled.add("goal")
                print(f"[live_viewer] goal marker disabled: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)

    # ------------------------------------------------------------------ per-frame drawing
    def place_auv(self, pose: Pose) -> None:
        if self.auv_id < 0:
            return
        rpy = pose.rpy if pose.rpy is not None else (0.0, 0.0, 0.0)
        self.p.resetBasePositionAndOrientation(self.auv_id, [float(v) for v in pose.position],
                                               [float(v) for v in euler_to_quat(*rpy)], physicsClientId=self.client)

    def frame(self, step: int, snap: dict, pose: Pose) -> None:
        """Draw this telemetry frame. `step` throttles the overlays exactly like the GUI run does."""
        p, c = self.p, self.client
        if step % self.every_n != 0:
            return
        self.guard("trail", lambda: self._trail(pose))
        est = snap.get("estimate") or {}
        if self.show_estimate and self._marker is not None and est.get("available") and est.get("position"):
            self.guard("estimate", lambda: p.resetBasePositionAndOrientation(
                self._marker, [float(v) for v in est["position"]], [0, 0, 0, 1], physicsClientId=c))
        so = snap.get("sonar") or {}
        if self.show_rays and pose.rpy is not None and so.get("available"):
            self.guard("rays", lambda: self._rays(pose, so))
        path = snap.get("navigation_path")
        if path:
            self.guard("path", lambda: self._path(path))
        nav = snap.get("navigation") or {}
        if nav.get("enabled") and nav.get("goal"):
            self.set_goal(nav["goal"])
        sim_t = float(snap.get("sim_time") or 0.0)
        if self.show_mapped and sim_t - self._last_mapped_t >= self.mapped_period_s and est.get("available"):
            self._last_mapped_t = sim_t
            self.guard("mapped", lambda: self._mapped(est))

    def _trail(self, pose: Pose) -> None:
        pos = [float(v) for v in pose.position]
        if self._prev is not None and float(np.linalg.norm(np.subtract(pos, self._prev))) < 5.0:
            self.p.addUserDebugLine(self._prev, pos, lineColorRGB=list(TRAIL_RGB), lineWidth=3.0, lifeTime=0,
                                    physicsClientId=self.client)
            self.draw_calls["lines"] += 1
        self._prev = pos

    def _rays(self, pose: Pose, so: dict) -> None:
        t = so.get("timestamp")
        if t is not None and t == self._last_scan_t:
            return                                               # same scan as last frame: nothing new to draw
        self._last_scan_t = t
        az = np.radians(np.asarray(so.get("azimuth_deg") or [], float))
        el = np.radians(np.asarray(so.get("elevation_deg") or [], float))
        status = np.asarray(so.get("status") or [], int)
        ranges = so.get("range_m") or []
        if not len(az) or len(az) != len(status):
            return
        dirs = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], axis=1)
        R = pose.rotation
        mount = np.asarray(so.get("mount_position_m") or [0.0, 0.0, 0.0], float)
        origin = pose.position + R @ mount
        max_range = float(so.get("max_range_m") or 0.0)
        for i in range(len(status)):
            if status[i] < 0:                                     # dropped / invalid ray: no information, no line
                continue
            d = R @ dirs[i]
            if status[i] == 1 and i < len(ranges) and ranges[i] is not None:
                end, rgb, width = origin + float(ranges[i]) * d, HIT_RGB, self.hit_ray_width
            elif self.show_free_rays and max_range > 0 and i % self.free_ray_stride == 0:
                end, rgb, width = origin + max_range * d, FREE_RGB, self.free_ray_width
            else:
                continue
            self.p.addUserDebugLine([float(v) for v in origin], [float(v) for v in end], lineColorRGB=list(rgb),
                                    lineWidth=width, lifeTime=0.5, physicsClientId=self.client)
            self.draw_calls["lines"] += 1

    def _path(self, payload: dict) -> None:
        key = (payload.get("version"), payload.get("global") is not None, bool(payload.get("detour")))
        if key == self._path_key:
            return
        self._path_key = key
        for i in self._path_ids:
            self.p.removeUserDebugItem(i, physicsClientId=self.client)
        self._path_ids = []
        for name, rgb, width in (("global", PATH_RGB, 4.0), ("detour", DETOUR_RGB, 5.0)):
            pts = payload.get(name) or []
            for a, b in zip(pts[:-1], pts[1:]):
                self._path_ids.append(self.p.addUserDebugLine(
                    [float(a[0]), float(a[1]), float(a[2])], [float(b[0]), float(b[1]), float(b[2])],
                    lineColorRGB=list(rgb), lineWidth=width, lifeTime=0, physicsClientId=self.client))
                self.draw_calls["lines"] += 1

    def _mapped(self, est: dict) -> None:
        """OCCUPIED cells of the SENSOR-derived occupancy grid the run published (maps.json), at the estimate depth."""
        try:
            m = read_json_retry(self.run_dir / "maps.json")
        except OSError:
            m = None
        if not m or "occupancy" not in m:
            return
        occ = m["occupancy"]
        from src.mapping.map_manager import rle_decode        # noqa: PLC0415
        code = rle_decode(occ["data"], tuple(occ["shape"]))
        rows, cols = np.nonzero(code == 2)
        if len(rows) == 0:
            if self._pts_id is not None:
                self.p.removeUserDebugItem(self._pts_id, physicsClientId=self.client)
                self._pts_id = None
            return
        if len(rows) > self.max_mapped_points:
            sel = np.linspace(0, len(rows) - 1, self.max_mapped_points).astype(int)
            rows, cols = rows[sel], cols[sel]
        res = float(occ["resolution"])
        ox, oy = (float(v) for v in occ["origin"])
        x = ox + (cols + 0.5) * res
        y = oy + (rows + 0.5) * res
        z = np.full_like(x, float(est["position"][2]))
        pts = np.stack([x, y, z], axis=1).astype(float).tolist()
        kw = {"replaceItemUniqueId": self._pts_id} if self._pts_id is not None else {}
        self._pts_id = self.p.addUserDebugPoints(pts, [list(MAPPED_RGB)] * len(pts), pointSize=7.0, lifeTime=0,
                                                physicsClientId=self.client, **kw)
        self.draw_calls["points"] += 1

def _load_cfg(config_path: Path, override_path: Path | None) -> dict:
    import yaml
    from src.simulation.commands import deep_merge
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if override_path is not None and Path(override_path).exists():
        cfg = deep_merge(cfg, json.loads(Path(override_path).read_text(encoding="utf-8")))
    return cfg


def _run_is_alive(run_dir: Path, parent_pid: int | None) -> tuple[bool, str]:
    """(alive, why not) — from the run's OWN files, never from a bare PID guess."""
    if parent_pid is not None and not pid_alive(parent_pid):
        return False, "dashboard process exited"
    try:
        child = read_json_retry(run_dir / "child.json") or {}
    except OSError:
        child = {}
    pid = child.get("pid")
    if pid is not None and not pid_alive(int(pid)):
        return False, f"the simulation process ({pid}) exited"
    try:
        st = read_json_retry(run_dir / "status.json") or {}
    except OSError:
        st = {}
    if str(st.get("status", "")).upper() in TERMINAL_RUN_STATES and child.get("pid") is None:
        return False, f"the run ended ({st.get('status')})"
    return True, ""


def run_ended_status(run_dir: Path, parent_pid: int | None = None) -> str | None:
    """The run's own TERMINAL status (COMPLETED / ENDED_BEFORE_GOAL / FAILED / STOPPED) if the run has finished,
    else None. `_run_is_alive` collapses every reason into False; this separates 'the mission finished' from 'the
    dashboard / simulation process disappeared', so a finished run keeps its window on the last frame while a
    vanished one still closes it. status.json is the run process's own final word (child.json keeps its pid, so the
    pid alone cannot tell a finished run from a dead one)."""
    if parent_pid is not None and not pid_alive(parent_pid):
        return None                                       # the dashboard is gone: nothing to keep the window for
    try:
        st = read_json_retry(run_dir / "status.json") or {}
    except OSError:
        return None
    status = str(st.get("status", "")).upper()
    return status if status in TERMINAL_RUN_STATES else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Live 3-D PyBullet window for a running dashboard simulation.")
    ap.add_argument("--config", required=True)
    ap.add_argument("--runtime", required=True, help="the RUN's own directory (state.json / maps.json live there)")
    ap.add_argument("--override", default=None, help="the config override that run was started with (same world)")
    ap.add_argument("--run-id", default="standalone")
    ap.add_argument("--parent-pid", type=int, default=None)
    a = ap.parse_args(argv)

    run_dir = Path(a.runtime).resolve()
    st = ViewerState(run_dir, a.run_id, a.parent_pid)
    st.write(force=True)
    try:
        import pybullet as p
        import pybullet_data
    except ImportError:
        st.state, st.error = "ERROR", "PyBullet is not installed in this Python environment: pip install pybullet"
        st.write(force=True)
        return 2
    try:
        cfg = _load_cfg(Path(a.config), Path(a.override) if a.override else None)
        vis = cfg.get("visualization") or {}
        keep_open_after_run = bool(vis.get("keep_open_after_run", True))
        win = vis.get("window") or {}
        options = f"--width={int(win.get('width', 1600))} --height={int(win.get('height', 900))}"
        try:
            client = p.connect(p.GUI, options=options)
        except Exception as exc:                        # noqa: BLE001 - e.g. no display / driver problem
            st.state = "ERROR"
            st.error = (f"could not open a PyBullet GUI window on this computer ({type(exc).__name__}: {exc}). "
                        f"It needs a display; the headless run itself is unaffected.")
            st.write(force=True)
            return 3
        st.state = "OPEN"
        st.write(force=True)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)                            # a viewer: nothing here is simulated
        p.setTimeStep(1.0 / LOOP_HZ)
        from src.simulation.environment import Environment
        from src.simulation.gui_camera import CameraRig
        from src.simulation.underwater_look import apply as apply_underwater
        environment = Environment(cfg, physics_client=client)
        apply_underwater(client, vis, world_bounds=getattr(environment, "bounds", None))
        sim_cfg = cfg.get("simulation") or {}
        auv_id = -1
        try:
            auv_id = p.loadURDF(str(sim_cfg.get("urdf_path", "assets/auv/auv.urdf")),
                                basePosition=[float(v) for v in (sim_cfg.get("start_position") or [0, 0, 0])],
                                useFixedBase=False, physicsClientId=client)
            if bool(vis.get("highlight_auv", True)):
                for link in [-1] + list(range(int(p.getNumJoints(auv_id, physicsClientId=client)))):
                    p.changeVisualShape(auv_id, link, rgbaColor=list(AUV_RGBA), physicsClientId=client)
                p.addUserDebugText(str(vis.get("auv_label", "AURORA AUV")), [0.0, 0.0, 0.75],
                                   textColorRGB=[1.0, 0.8, 0.3], textSize=1.2, lifeTime=0,
                                   parentObjectUniqueId=auv_id, physicsClientId=client)
        except Exception as exc:                         # noqa: BLE001 - the window is still usable without it
            print(f"[live_viewer] could not load the AUV mesh: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        # The same camera policy as the in-process GUI run: camera_follow on = the follow/top rig, off = the legacy
        # whole-mission overview with a free camera (which the dashboard controls cannot move - it says so).
        follow = bool(vis.get("camera_follow", True))
        rig = CameraRig(client, vis.get("camera"), p, auv_id if auv_id >= 0 else None) if follow else None
        st.camera = {"mode": rig.mode if rig is not None else "free (camera_follow: false)",
                     "distance_m": None if rig is None else rig.cfg["distance_m"]}
        mirror = Mirror(p, client, cfg, environment, auv_id, run_dir)
        commands = CommandChannel(run_dir / COMMANDS_FILE)
        cmd_id = 0
        started = False
        held_open_announced = False
        # A viewer must stay cheap: the run's own files are polled on a slower cadence than the window is drawn.
        poll_period = 0.1
        next_poll = 0.0
        period = 1.0 / LOOP_HZ
        print(f"[live_viewer] window open for run {a.run_id} in {run_dir}", file=sys.stderr, flush=True)

        while True:
            t0 = time.perf_counter()
            if not p.isConnected(physicsClientId=client):
                print("[live_viewer] window closed by the user.", file=sys.stderr, flush=True)
                break
            stop = run_dir / STOP_FILE
            if stop.exists():
                try:
                    content = stop.read_text(encoding="utf-8").strip()
                except OSError:
                    content = a.run_id
                if content in ("", a.run_id):
                    print("[live_viewer] close requested.", file=sys.stderr, flush=True)
                    break
            poll_now = time.perf_counter() >= next_poll
            if poll_now:
                next_poll = time.perf_counter() + poll_period
            for cmd in (commands.poll() if poll_now else []):
                cmd_id = int(cmd.get("id", cmd_id))
                name = str(cmd.get("cmd", ""))
                if name == "stop":
                    st.command_log.append({"id": cmd_id, "cmd": name, "ok": True, "detail": "window closing",
                                           "sim_time": time.time()})
                    st.write(force=True)
                    print("[live_viewer] stop command received.", file=sys.stderr, flush=True)
                    raise _Stop()
                if name == "set_camera":
                    if rig is None:
                        ok, detail = False, ("this window uses a free camera (visualization.camera_follow: false); the "
                                            "dashboard follow/top/orbit/zoom controls need the follow camera")
                    else:
                        ok, detail = rig.do(str(cmd.get("action", "")))
                    st.command_log.append({"id": cmd_id, "cmd": name, "ok": ok, "detail": detail, "sim_time": time.time()})
                    st.write(force=True)
                else:
                    st.command_log.append({"id": cmd_id, "cmd": name, "ok": False,
                                           "detail": f"unknown viewer command {name!r}", "sim_time": time.time()})
            try:
                snap = read_json_retry(run_dir / "state.json")
            except OSError:
                snap = None
            if snap is None:
                alive, why = (_run_is_alive(run_dir, a.parent_pid) if poll_now else (True, ""))
                if not alive and not (keep_open_after_run and run_ended_status(run_dir, a.parent_pid)):
                    st.state, st.error = "CLOSED", f"closed: {why} (nothing left to show)"
                    st.write(force=True)
                    return 0
                p.stepSimulation(physicsClientId=client)     # keep the window responsive while waiting
                time.sleep(period)
                continue
            if not started:
                mission = snap.get("mission") or {}
                mirror.start(environment, mission)
                if rig is None and mission.get("start"):          # free camera: frame the whole mission once
                    from src.simulation.gui_camera import overview_free_camera    # noqa: PLC0415
                    try:
                        p.resetDebugVisualizerCamera(physicsClientId=client, **overview_free_camera(
                            mission["start"], mission.get("waypoints") or [], mission.get("goal")))
                    except Exception as exc:                     # noqa: BLE001 - camera only
                        print(f"[live_viewer] overview camera failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                started = True
            gt = snap.get("debug_ground_truth") or {}
            est = snap.get("estimate") or {}
            pose = None
            if gt.get("position"):
                pose = Pose(gt["position"], rpy=gt.get("euler_rad") or (0.0, 0.0, 0.0))
            elif est.get("available") and est.get("position"):
                pose = Pose(est["position"], rpy=est.get("euler_rad"))
            if pose is not None:
                mirror.place_auv(pose)
                if rig is not None:
                    rig.update(int(snap.get("step") or 0), pose, float(snap.get("sim_time") or 0.0))
                mirror.frame(int(snap.get("step") or 0), snap, pose)
            p.stepSimulation(physicsClientId=client)
            st.frames += 1
            if st.frames % 15 == 0:
                if rig is not None:
                    st.camera = {"mode": rig.mode, "yaw": rig.user["yaw"], "zoom": rig.user["scale"],
                                 "stats": dict(rig.stats)}
                st.scene = {"scenario": (snap.get("environment") or {}).get("scenario"),
                            "sim_time": snap.get("sim_time"), "run_status": snap.get("run_status")}
                st.write()
            if poll_now:
                alive, why = _run_is_alive(run_dir, a.parent_pid)
                kept = keep_open_after_run and run_ended_status(run_dir, a.parent_pid)
                if kept:
                    # The mission finished on its own (the robot reached the goal, or the run stopped / failed):
                    # keep the window on the last frame instead of closing it. Nothing here changes the run.
                    if not held_open_announced:
                        held_open_announced = True
                        print(f"[live_viewer] the run ended ({kept}); keeping the last frame on screen "
                              f"(visualization.keep_open_after_run). Close this window, or use the dashboard's 3-D "
                              f"window control, to dismiss it.", file=sys.stderr, flush=True)
                elif not alive:
                    st.state, st.error = "CLOSED", f"closed: {why} (nothing left to show)"
                    st.write(force=True)
                    return 0
            left = period - (time.perf_counter() - t0)
            if left > 0:
                time.sleep(left)
        try:
            st.state = "CLOSED"
            st.error = None
            st.write(force=True)
        except Exception:                                  # noqa: BLE001
            pass
        return 0
    except _Stop:
        st.state, st.error = "CLOSED", None
        st.write(force=True)
        return 0
    except BaseException as exc:                           # noqa: BLE001 - reported, then non-zero exit
        st.state, st.error = "ERROR", f"{type(exc).__name__}: {exc}"
        st.write(force=True)
        traceback.print_exc()
        return 1
    finally:
        try:
            import pybullet as p
            p.disconnect()
        except Exception:                                  # noqa: BLE001
            pass
        sys.stdout.flush()


if __name__ == "__main__":
    raise SystemExit(main())

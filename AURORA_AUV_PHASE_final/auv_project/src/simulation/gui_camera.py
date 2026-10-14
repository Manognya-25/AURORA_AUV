"""Third-person follow camera for the 3-D PyBullet window (GUI mode only).

CAMERA ONLY. This module moves the debug-visualizer camera and nothing else: it never creates bodies, never
changes the environment, never touches physics / sensors / mapping / EKF / planning / navigation, and the
numbers it reads (the AUV's true pose, collision ray tests against the existing seabed and obstacles) are used
for placing the camera and are never fed back into anything. A failure disables only the camera rig (see
GuiVisuals._guard).

Views
  FOLLOW  default. Camera behind and above the AUV, looking along the AUV's heading, AUV (almost) centred.
  TOP     straight down above the AUV, +Y up on screen (same orientation as the dashboard maps).
  RESET   drop every mouse adjustment and return to the default FOLLOW view.

Controls: the side-panel buttons "FOLLOW AUV" / "TOP VIEW" / "RESET VIEW" (plus "ORBIT LEFT" / "ORBIT RIGHT" /
"ZOOM IN" / "ZOOM OUT" when camera.orbit_zoom_buttons is on, as in run_visual_demo.py), or the keys
1 / 2 / 3 (views) and 4 / 5 (orbit) / 6 / 7 (zoom) everywhere.
Mouse rotate / zoom still works: the change you make is remembered as an offset relative to the auto-camera
(so you can still look around while following); the ORBIT / ZOOM controls apply that same kind of offset, and
RESET VIEW clears every one of them.

Geometry safety: PyBullet's debug camera has no collision of its own, so the rig probes the REAL collision world
with ray tests. A candidate camera pose is accepted only if the sight line AUV -> camera (and four rays offset by
the camera clearance) is free of seabed / obstacle geometry and the camera is not about to touch the seabed.
If the preferred pose is blocked the rig tries higher / side poses, otherwise it moves the camera in towards the
AUV. Moves toward the AUV are fast, moves away are slow, so the picture never pops.
"""
from __future__ import annotations

import math

import numpy as np

FOLLOW, TOP = "follow", "top"

DEFAULTS = {
    "mode": FOLLOW,
    "distance_m": 5.0,          # camera-to-target distance (the AUV is 1.5 m long)
    "pitch_deg": -20.0,         # negative = looking down; 20 deg -> ~1.7 m above the AUV at 5 m
    "look_ahead_m": 0.8,        # aim slightly ahead of the AUV so the way forward is visible; AUV stays near centre
    "top_height_m": 18.0,       # TOP VIEW camera height above the AUV
    "heading_tau_s": 0.45,      # smoothing of the heading the camera follows (sim seconds)
    "target_tau_s": 0.06,       # smoothing of the aim point (small: the AUV must stay centred)
    "min_distance_m": 1.8,      # never closer than this while any acceptable pose exists
    "hard_min_distance_m": 1.0,  # absolute floor when the AUV is in a very tight spot
    "clearance_m": 0.45,        # camera radius used for the offset collision probes
    "ground_clearance_m": 0.5,  # keep the camera at least this far above the seabed
    "avoid_geometry": True,     # ray-test the real collision world so geometry never hides the AUV
    "update_every_n_steps": 4,   # 60 Hz at 240 Hz physics. In p.GUI every PyBullet call is a GUI round trip, so keep calls few
    "poll_every_n_updates": 6,   # buttons / keys are polled this often (each poll is 4 round trips)
    "buttons": True,            # FOLLOW AUV / TOP VIEW / RESET VIEW buttons in the PyBullet side panel
    "orbit_zoom_buttons": False,  # extra ORBIT LEFT/RIGHT + ZOOM IN/OUT buttons (cost: one round trip per poll each)
    "keys": True,               # keys 1 / 2 / 3 / 4 / 5 / 6 / 7 do the same
    "orbit_step_deg": 30.0,     # ORBIT LEFT/RIGHT yaw step (an offset relative to the auto-camera)
    "zoom_step": 0.75,          # ZOOM IN multiplies the distance by this, ZOOM OUT divides by it
}

# (yaw offset deg, pitch offset deg) tried in this order when the preferred pose is blocked
FOLLOW_ALTERNATIVES = [(0, 0), (0, -10), (30, 0), (-30, 0), (0, -25), (30, -25), (-30, -25), (60, -25), (-60, -25),
                       (0, -45), (45, -45), (-45, -45), (0, -70)]
KEY_FOLLOW, KEY_TOP, KEY_RESET = ord("1"), ord("2"), ord("3")
KEY_ORBIT_LEFT, KEY_ORBIT_RIGHT, KEY_ZOOM_IN, KEY_ZOOM_OUT = ord("4"), ord("5"), ord("6"), ord("7")

# PyBullet's side panel shows only ~2 characters of a button label (long names were clipped to "V", "W", "HT"), so the buttons
# carry the same digit as their key: 1 follow, 2 top, 3 reset, 4 orbit left, 5 orbit right, 6 zoom in, 7 zoom out. The legend
# drawn in the window says what each digit does.
BUTTON_FOLLOW, BUTTON_TOP, BUTTON_RESET = "1", "2", "3"
BUTTON_ORBIT_LEFT, BUTTON_ORBIT_RIGHT = "4", "5"
BUTTON_ZOOM_IN, BUTTON_ZOOM_OUT = "6", "7"
BASE_BUTTONS = (BUTTON_FOLLOW, BUTTON_TOP, BUTTON_RESET)
ORBIT_ZOOM_BUTTONS = (BUTTON_ORBIT_LEFT, BUTTON_ORBIT_RIGHT, BUTTON_ZOOM_IN, BUTTON_ZOOM_OUT)


# ---------------------------------------------------------------------------------------------- pure geometry
def wrap180(a: float) -> float:
    return (float(a) + 180.0) % 360.0 - 180.0


def heading_rad(rotation, previous: float | None = None, min_horizontal: float = 0.2) -> float:
    """Heading (yaw of the body +X axis projected on the horizontal plane). Holds `previous` when the hull points
    almost straight up / down, where the horizontal direction is meaningless."""
    R = np.asarray(rotation, float)
    fx, fy = float(R[0, 0]), float(R[1, 0])
    if math.hypot(fx, fy) < min_horizontal:
        return 0.0 if previous is None else float(previous)
    return math.atan2(fy, fx)


def yaw_deg_for_heading(heading: float) -> float:
    """PyBullet debug-camera yaw that makes the camera look along world heading `heading` (rad).
    PyBullet yaw 0 looks along +Y, so looking along heading h needs yaw = h - 90 deg."""
    return math.degrees(heading) - 90.0


def view_eye(target, yaw_deg: float, pitch_deg: float, distance: float) -> np.ndarray:
    """World position of the camera for PyBullet's (target, yaw, pitch, distance) description."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    forward = np.array([-math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
    return np.asarray(target, float) - float(distance) * forward


def _ease(dt: float, tau: float) -> float:
    return 1.0 if tau <= 0.0 else 1.0 - math.exp(-max(dt, 0.0) / tau)


def _angle_lerp(a: float, b: float, k: float) -> float:
    return a + math.atan2(math.sin(b - a), math.cos(b - a)) * k


def overview_free_camera(start, waypoints, goal):
    """The legacy whole-mission framing, used only when the camera follow is switched off (--no-follow)."""
    xs = [float(start[0]), float(goal[0])] + [float(w[0]) for w in waypoints]
    ys = [float(start[1]), float(goal[1])] + [float(w[1]) for w in waypoints]
    span = max(max(xs) - min(xs), max(ys) - min(ys), 10.0)
    return {"cameraDistance": max(22.0, span * 1.8), "cameraYaw": 45.0, "cameraPitch": -32.0,
            "cameraTargetPosition": [sum(xs) / len(xs), sum(ys) / len(ys), -5.0]}


def initial_follow_camera(start, first_waypoint, cfg: dict | None = None):
    """Camera placed behind the start position, looking toward the first waypoint (so the very first frame is
    already a good third-person view; the rig then takes over with the AUV's real heading)."""
    c = {**DEFAULTS, **(cfg or {})}
    d = np.asarray(first_waypoint[:2], float) - np.asarray(start[:2], float)
    h = math.atan2(d[1], d[0]) if np.hypot(*d) > 1e-6 else 0.0
    t = np.array([start[0], start[1], start[2]], float)
    if c["mode"] == TOP:
        return {"cameraDistance": float(c["top_height_m"]), "cameraYaw": 0.0, "cameraPitch": -89.0,
                "cameraTargetPosition": t.tolist()}
    t[:2] += float(c["look_ahead_m"]) * np.array([math.cos(h), math.sin(h)])
    return {"cameraDistance": float(c["distance_m"]), "cameraYaw": yaw_deg_for_heading(h),
            "cameraPitch": float(c["pitch_deg"]), "cameraTargetPosition": t.tolist()}


# ---------------------------------------------------------------------------------------------- the rig
class CameraRig:
    def __init__(self, client: int, cfg: dict | None, pb=None, auv_body_id: int | None = None):
        if pb is None:
            import pybullet as pb                                  # noqa: PLC0415 - GUI mode only
        self.p, self.client, self.auv_id = pb, client, auv_body_id
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.mode = TOP if str(self.cfg["mode"]).lower() == TOP else FOLLOW
        self.every_n = max(1, int(self.cfg["update_every_n_steps"]))
        self.user = {"yaw": 0.0, "pitch": 0.0, "scale": 1.0}      # mouse adjustments, relative to the auto-camera
        self._s = None                                             # smoothed state
        self._applied = None                                       # what was last sent to PyBullet
        self._last_t = None
        self._calls = 0
        self._avoid_disabled = not bool(self.cfg["avoid_geometry"])
        self._buttons, self._button_counts, self.notes = {}, {}, []
        self.stats = {"updates": 0, "rerouted": 0, "pulled_in": 0, "mode_switches": 0, "resets": 0,
                      "orbits": 0, "zooms": 0, "last": None, "min_clear_m": None}
        if self.cfg["buttons"]:
            self._make_buttons()

    # ----------------------------------------------------------------------- controls
    def _make_buttons(self):
        p = self.p
        names = BASE_BUTTONS + (ORBIT_ZOOM_BUTTONS if bool(self.cfg.get("orbit_zoom_buttons")) else ())
        try:
            for name in names:
                pid = p.addUserDebugParameter(name, 1, 0, 1, physicsClientId=self.client)       # start > end => a button
                self._buttons[name] = pid
                self._button_counts[name] = p.readUserDebugParameter(pid, physicsClientId=self.client)
        except Exception as exc:                                    # noqa: BLE001 - controls are optional
            self._buttons = {}
            self.notes.append(f"camera buttons unavailable: {type(exc).__name__}: {exc}")

    def _action(self, name: str) -> None:
        """One side-panel button press = one real view change (kept as an offset relative to the auto-camera)."""
        z = min(max(float(self.cfg["zoom_step"]), 0.05), 0.95)
        actions = {
            BUTTON_FOLLOW: lambda: self.set_mode(FOLLOW),
            BUTTON_TOP: lambda: self.set_mode(TOP),
            BUTTON_RESET: self.reset_view,
            BUTTON_ORBIT_LEFT: lambda: self.orbit(-float(self.cfg["orbit_step_deg"])),
            BUTTON_ORBIT_RIGHT: lambda: self.orbit(float(self.cfg["orbit_step_deg"])),
            BUTTON_ZOOM_IN: lambda: self.zoom(z),
            BUTTON_ZOOM_OUT: lambda: self.zoom(1.0 / z),
        }
        actions[name]()

    CAMERA_ACTIONS = {"follow": BUTTON_FOLLOW, "top": BUTTON_TOP, "reset": BUTTON_RESET, "orbit_left": BUTTON_ORBIT_LEFT,
                      "orbit_right": BUTTON_ORBIT_RIGHT, "zoom_in": BUTTON_ZOOM_IN, "zoom_out": BUTTON_ZOOM_OUT}

    def do(self, action: str) -> tuple[bool, str]:
        """Programmatic twin of the side-panel buttons / keys (used by the dashboard camera controls)."""
        name = self.CAMERA_ACTIONS.get(str(action).lower())
        if name is None:
            return False, f"unknown camera action {action!r} (one of {', '.join(self.CAMERA_ACTIONS)})"
        self._action(name)
        return True, f"camera {action}: mode={self.mode}, orbit offset {self.user['yaw']:.0f} deg, zoom x{self.user['scale']:.2f}"

    def set_mode(self, mode: str) -> None:
        mode = TOP if mode == TOP else FOLLOW
        if mode != self.mode:
            self.mode = mode
            self.stats["mode_switches"] += 1
        self.user = {"yaw": 0.0, "pitch": 0.0, "scale": 1.0}
        self._s = None if self._s is None else {**self._s, "snap": True}

    def reset_view(self) -> None:
        self.mode = FOLLOW if str(self.cfg["mode"]).lower() != TOP else TOP
        self.user = {"yaw": 0.0, "pitch": 0.0, "scale": 1.0}
        self.stats["resets"] += 1
        self._s = None if self._s is None else {**self._s, "snap": True}

    def orbit(self, delta_deg: float) -> None:
        """Rotate the view around the aim point (yaw offset relative to the auto-camera, so FOLLOW keeps following)."""
        self.user["yaw"] += float(delta_deg)
        self.stats["orbits"] += 1
        if self._s is not None:
            self._s["snap"] = True

    def zoom(self, factor: float) -> None:
        """Move the camera toward (factor < 1) or away from (factor > 1) the aim point, within the clamped range."""
        self.user["scale"] = float(np.clip(self.user["scale"] * float(factor), 0.4, 4.0))
        self.stats["zooms"] += 1
        if self._s is not None:
            self._s["snap"] = True

    def poll_controls(self) -> None:
        p = self.p
        for name, pid in self._buttons.items():
            v = p.readUserDebugParameter(pid, physicsClientId=self.client)
            if v != self._button_counts.get(name):
                self._button_counts[name] = v
                self._action(name)
        if self.cfg["keys"]:
            ev = p.getKeyboardEvents(physicsClientId=self.client)
            trig = getattr(p, "KEY_WAS_TRIGGERED", 2)
            for key, name in ((KEY_FOLLOW, BUTTON_FOLLOW), (KEY_TOP, BUTTON_TOP), (KEY_RESET, BUTTON_RESET),
                              (KEY_ORBIT_LEFT, BUTTON_ORBIT_LEFT), (KEY_ORBIT_RIGHT, BUTTON_ORBIT_RIGHT),
                              (KEY_ZOOM_IN, BUTTON_ZOOM_IN), (KEY_ZOOM_OUT, BUTTON_ZOOM_OUT)):
                if ev.get(key, 0) & trig:
                    self._action(name)

    def _absorb_mouse(self) -> None:
        """If the user dragged / zoomed the PyBullet camera since our last update, keep that as an offset."""
        a = self._applied
        if a is None:
            return
        cam = self.p.getDebugVisualizerCamera(physicsClientId=self.client)
        yaw, pitch, dist = float(cam[8]), float(cam[9]), float(cam[10])
        dyaw, dpitch = wrap180(yaw - a["yaw"]), pitch - a["pitch"]
        if abs(dyaw) > 0.5 or abs(dpitch) > 0.5 or abs(dist - a["dist"]) > 0.02 * max(a["dist"], 1.0):
            self.user["yaw"] += dyaw
            self.user["pitch"] = float(np.clip(self.user["pitch"] + dpitch, -60.0, 60.0))
            self.user["scale"] = float(np.clip(self.user["scale"] * dist / max(a["dist"], 1e-6), 0.4, 4.0))
            if self._s is not None and self._s.get("dist") is not None:
                self._s["dist"] = dist                              # the user's own zoom applies at once (no slow back-off)

    # ----------------------------------------------------------------------- geometry probes
    def _probe(self, target, poses) -> list[float]:
        """For each (yaw, pitch, dist): the largest camera distance along that view direction that keeps the sight line
        (and the camera's own clearance radius) free of collision geometry; `dist` itself when nothing is in the way.
        ALL rays for ALL poses go out in ONE rayTestBatch call (each PyBullet call is a GUI round trip)."""
        if self._avoid_disabled:
            return [float(d) for _y, _p, d in poses]
        c = self.cfg
        r, gc_ = float(c["clearance_m"]), float(c["ground_clearance_m"])
        offs = [(0.0, 0.0), (r, 0.0), (-r, 0.0), (0.0, r), (0.0, -r)]
        A, B, meta = [], [], []
        for yaw, pitch, dist in poses:
            eye = view_eye(target, yaw, pitch, dist)
            u = (eye - target) / max(np.linalg.norm(eye - target), 1e-9)
            right = np.cross(u, [0.0, 0.0, 1.0])
            right = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(right) < 1e-6 else right / np.linalg.norm(right)
            up = np.cross(right, u)
            start_off = min(0.9, 0.5 * dist)                       # start outside the hull (half length 0.75 m)
            for x, y in offs:
                A.append((target + u * start_off + right * x + up * y).tolist()); B.append((eye + right * x + up * y).tolist())
            A.append(eye.tolist()); B.append((eye - np.array([0.0, 0.0, gc_])).tolist())   # camera vs seabed
            meta.append((float(dist), start_off))
        try:
            res = self.p.rayTestBatch(A, B, physicsClientId=self.client)
        except Exception as exc:                                    # noqa: BLE001 - probing is best-effort
            self._avoid_disabled = True
            self.notes.append(f"camera collision probing disabled: {type(exc).__name__}: {exc}")
            return [float(d) for _y, _p, d in poses]
        out, k = [], 0
        for dist, start_off in meta:
            clear = dist
            for j in range(5):
                body, _link, frac = res[k + j][0], res[k + j][1], res[k + j][2]
                if body >= 0 and body != self.auv_id:
                    seg = float(np.linalg.norm(np.subtract(B[k + j], A[k + j])))
                    clear = min(clear, start_off + float(frac) * seg - 0.3)
            gbody = res[k + 5][0]
            if gbody >= 0 and gbody != self.auv_id:               # camera would (almost) touch the seabed / an obstacle
                clear = min(clear, 0.5 * dist)
            out.append(max(clear, 0.0))
            k += 6
        return out

    # ----------------------------------------------------------------------- per-update
    def update(self, step: int, truth, sim_time: float) -> bool:
        """Place the camera for this step; returns True when PyBullet's camera was set."""
        self._calls += 1
        if truth is None or (step % self.every_n) != 0:
            return False
        if self.stats["updates"] % max(1, int(self.cfg["poll_every_n_updates"])) == 0:
            self.poll_controls()
        self._absorb_mouse()
        c, p = self.cfg, self.p
        pos = np.asarray(truth.position, float)
        prev_h = None if self._s is None else self._s["heading"]
        h = heading_rad(truth.rotation, prev_h)
        dt = 0.0 if self._last_t is None else float(sim_time) - self._last_t
        snap = self._s is None or self._s.get("snap") or dt <= 0.0 or dt > 1.0
        self._last_t = float(sim_time)
        if snap:
            s = {"heading": h, "target": pos.copy(), "yaw_adj": 0.0, "pitch_adj": 0.0, "dist": None}
        else:
            s = self._s
            s["heading"] = _angle_lerp(s["heading"], h, _ease(dt, c["heading_tau_s"]))
        hs = s["heading"]
        top = self.mode == TOP

        ahead = 0.0 if top else float(c["look_ahead_m"])
        aim = pos + np.array([math.cos(hs) * ahead, math.sin(hs) * ahead, 0.0])
        s["target"] = aim if snap else s["target"] + (aim - s["target"]) * _ease(dt, c["target_tau_s"])
        target = s["target"]

        base_yaw = 0.0 if top else yaw_deg_for_heading(hs)
        base_pitch = -89.0 if top else float(c["pitch_deg"])
        pref = float(c["top_height_m"] if top else c["distance_m"]) * self.user["scale"]
        alts = [(0.0, 0.0)] if top else FOLLOW_ALTERNATIVES

        # 1. preferred view direction first (one batched ray call); only if it is blocked, try the alternatives (second call)
        accept = min(pref, max(0.85 * pref, float(c["min_distance_m"])))      # a pose this far out counts as "clear"

        def pose(dy, dp):
            return (base_yaw + self.user["yaw"] + dy, float(np.clip(base_pitch + self.user["pitch"] + dp, -89.0, -3.0)), pref)

        clear = self._probe(target, [pose(0, 0)])[0]
        best = (clear, 0, 0)
        if clear < accept and not top:
            alts = FOLLOW_ALTERNATIVES[1:]
            for cl, (dy, dp) in zip(self._probe(target, [pose(*a) for a in alts]), alts):
                if cl > best[0] + 1e-9:
                    best = (cl, dy, dp)
                if cl >= accept:
                    best = (cl, dy, dp)
                    break
        clear, dy, dp = best
        if (dy, dp) != (0, 0):
            self.stats["rerouted"] += 1
        want_dist = float(np.clip(clear, float(c["hard_min_distance_m"]), pref))
        if want_dist < pref - 1e-6:
            self.stats["pulled_in"] += 1

        # 2. smooth: yaw/pitch alternatives slowly, distance fast when closing in and slowly when backing off
        k = 1.0 if snap else _ease(dt, 0.35)
        s["yaw_adj"] += (dy - s["yaw_adj"]) * k
        s["pitch_adj"] += (dp - s["pitch_adj"]) * k
        if s["dist"] is None:
            s["dist"] = want_dist
        else:
            s["dist"] += (want_dist - s["dist"]) * _ease(dt, 0.10 if want_dist < s["dist"] else 1.2)
        yaw = base_yaw + self.user["yaw"] + s["yaw_adj"]
        pitch = float(np.clip(base_pitch + self.user["pitch"] + s["pitch_adj"], -89.0, -3.0))

        # 3. last line of defence: the SMOOTHED pose must also be clear (the smoothing can sweep through geometry)
        final_clear = s["dist"]
        if abs(s["yaw_adj"]) > 0.05 or abs(s["pitch_adj"]) > 0.05 or (dy, dp) != (0, 0):
            final_clear = self._probe(target, [(yaw, pitch, s["dist"])])[0]       # steady state: the evaluated pose IS the applied pose
            if final_clear < s["dist"]:
                s["dist"] = max(final_clear, float(c["hard_min_distance_m"]))
        else:
            final_clear = min(final_clear, clear)
        s["snap"] = False
        self._s = s

        p.resetDebugVisualizerCamera(cameraDistance=float(s["dist"]), cameraYaw=float(yaw), cameraPitch=pitch,
                                     cameraTargetPosition=[float(v) for v in target], physicsClientId=self.client)
        self._applied = {"yaw": float(yaw), "pitch": pitch, "dist": float(s["dist"])}
        self.stats["updates"] += 1
        mc = self.stats["min_clear_m"]
        self.stats["min_clear_m"] = float(final_clear) if mc is None else min(mc, float(final_clear))
        self.stats["last"] = {"mode": self.mode, "yaw": float(yaw), "pitch": pitch, "distance": float(s["dist"]),
                              "target": [float(v) for v in target], "heading_deg": math.degrees(hs)}
        return True

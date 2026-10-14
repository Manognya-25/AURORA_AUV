"""Change the navigation goal from inside the 3-D window (keyboard), through the SAME command path as the dashboard.

    arrow keys   move the PENDING goal 1 m in world X (left/right) and Y (down/up)      [Shift: 5 m]
    Page Up/Down pending goal shallower / deeper (1 m)
    G or Enter   apply: sends {"cmd": "nav_set_goal"} to the navigator (validated against the map exactly as the dashboard)
    H            cancel the pending goal

The pending goal is drawn as a small green marker + label. Nothing changes in the mission until it is applied; once applied
the navigator's goal is the ONE authoritative goal (planner, path, goal distance, GOAL marker and dashboard all follow it).
Visual/UI only: it never touches the vehicle, sensors, EKF or controller, and it only sends a command."""
from __future__ import annotations

import numpy as np

PENDING_RGBA = (0.2, 0.9, 0.35, 0.8)
STEP_M, BIG_STEP_M = 1.0, 5.0


class GoalKeyEditor:
    def __init__(self, client: int, pb=None, poll_every_n_steps: int = 5, message_steps: int = 300):
        if pb is None:
            import pybullet as pb                                  # noqa: PLC0415 - GUI mode only
        self.p, self.client = pb, client
        self.every = max(1, int(poll_every_n_steps))
        self.message_steps = int(message_steps)
        self.pending: np.ndarray | None = None
        self.body = self.text = None
        self.message, self._message_until = "", -1
        self.applied = 0
        g = lambda n, d: getattr(pb, n, d)                         # noqa: E731 - constants exist in real PyBullet
        self.K = dict(left=g("B3G_LEFT_ARROW", 65295), right=g("B3G_RIGHT_ARROW", 65296), up=g("B3G_UP_ARROW", 65297),
                      down=g("B3G_DOWN_ARROW", 65298), pgup=g("B3G_PAGE_UP", 65301), pgdn=g("B3G_PAGE_DOWN", 65302),
                      enter=g("B3G_RETURN", 65309), shift=g("B3G_SHIFT", 65306))
        self.G, self.H = ord("g"), ord("h")
        self._triggered = g("KEY_WAS_TRIGGERED", 2)

    # ---------------------------------------------------------------- drawing
    def _draw(self, label: str) -> None:
        p, g = self.p, self.pending
        if self.body is None:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.35, rgbaColor=list(PENDING_RGBA), physicsClientId=self.client)
            self.body = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=vis,
                                          basePosition=g.tolist(), physicsClientId=self.client)
        else:
            p.resetBasePositionAndOrientation(self.body, g.tolist(), [0, 0, 0, 1], physicsClientId=self.client)
        if self.text is not None:
            p.removeUserDebugItem(self.text, physicsClientId=self.client)
        self.text = p.addUserDebugText(label, [g[0], g[1], g[2] + 0.8], textColorRGB=[0.3, 1.0, 0.45], textSize=1.1,
                                       lifeTime=0, physicsClientId=self.client)

    def _clear(self) -> None:
        p = self.p
        if self.body is not None:
            p.removeBody(self.body, physicsClientId=self.client)
        if self.text is not None:
            p.removeUserDebugItem(self.text, physicsClientId=self.client)
        self.body = self.text = None
        self.pending = None

    @staticmethod
    def _label(g) -> str:
        return f"NEW GOAL ({g[0]:.0f}, {g[1]:.0f}, {g[2]:.0f}) - G/Enter apply, H cancel"

    # ---------------------------------------------------------------- per step
    def poll(self, step: int, current_goal, apply) -> str | None:
        """Read the keyboard every few steps. `apply(goal_list) -> (ok, detail)` sends the navigator command.
        Returns a status string when a goal was applied/rejected, else None."""
        if step % self.every:
            return None
        if self.message and step > self._message_until and self.pending is None:
            self.message = ""
        ev = self.p.getKeyboardEvents(physicsClientId=self.client)
        hit = lambda k: bool(ev.get(k, 0) & self._triggered)       # noqa: E731
        base = self.pending if self.pending is not None else (None if current_goal is None else np.asarray(current_goal, float))
        if base is None:
            return None
        big = bool(ev.get(self.K["shift"], 0) & 1)                 # KEY_IS_DOWN
        s = BIG_STEP_M if big else STEP_M
        d = np.zeros(3)
        d[0] += s * (hit(self.K["right"]) - hit(self.K["left"]))
        d[1] += s * (hit(self.K["up"]) - hit(self.K["down"]))
        d[2] += s * (hit(self.K["pgup"]) - hit(self.K["pgdn"]))   # z is negative-down: Page Up = shallower
        if np.any(d):
            self.pending = base + d
            self._draw(self._label(self.pending))
        if hit(self.H) and self.pending is not None:
            self._clear()
            return "pending goal cancelled"
        if (hit(self.G) or hit(self.K["enter"])) and self.pending is not None:
            goal = [float(v) for v in self.pending]
            ok, why = apply(goal)
            if ok:
                self.applied += 1
                self._clear()
                self.message = f"goal set to ({goal[0]:.0f}, {goal[1]:.0f}, {goal[2]:.0f})"
            else:
                self.message = f"goal rejected: {why}"
                self._draw(f"REJECTED: {why[:60]}")
            self._message_until = step + self.message_steps
            return self.message
        return None

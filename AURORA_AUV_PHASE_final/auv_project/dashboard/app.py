"""Flask application for the AURORA AUV dashboard (Phase 4).

Every endpoint returns strict JSON (NaN/Infinity -> null) built from the real simulation's
snapshots; errors are JSON too. Backend state is authoritative: the UI keeps no copy of it.
"""
from __future__ import annotations

import json
import time

from flask import Flask, Response, jsonify, make_response, render_template, request
from werkzeug.exceptions import HTTPException

from dashboard.simulation_manager import ApiError, SimulationManager
from src.sensors.health import FORCED_STATES, canonical_state
from src.simulation.commands import operator_z_to_world
from src.utils.jsonsafe import jsonsafe

SENSOR_NAMES = ("imu", "dvl", "depth", "sonar", "compass")


def _json(payload, status: int = 200) -> Response:
    body = json.dumps(jsonsafe(payload), allow_nan=False, separators=(",", ":"))
    return Response(body, status=status, mimetype="application/json")


def create_app(manager: SimulationManager | None = None) -> Flask:
    app = Flask(__name__)
    mgr = manager or SimulationManager()
    app.config["manager"] = mgr
    # A server that has been up for a while must not keep serving the page it read at start-up: with auto-reload
    # on, index.html is re-read when its file changes, so an edit shows up on the next reload instead of pairing
    # the OLD document (old chart canvases) with the CURRENT JS served from disk.
    app.config["TEMPLATES_AUTO_RELOAD"] = True

    # ---------------------------------------------------------------- errors -> JSON
    @app.errorhandler(ApiError)
    def _api_error(e: ApiError):
        return _json({"ok": False, "error": str(e)}, e.status)

    @app.errorhandler(HTTPException)
    def _http_error(e: HTTPException):
        return _json({"ok": False, "error": e.description or e.name}, e.code or 500)

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        app.logger.exception("unhandled error")
        return _json({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)

    def body_json() -> dict:
        data = request.get_json(silent=True)
        if data is None and request.data:
            raise ApiError("request body is not valid JSON")
        return data or {}

    def need_snapshot() -> dict:
        snap = mgr.state_snapshot()
        if snap is None:
            raise ApiError("no telemetry yet: the simulation has not published a snapshot "
                           f"(state: {mgr.status()['state']})", 404)
        return snap

    def with_status(section: dict | None, **extra) -> dict:
        st = mgr.status()
        return {"state": st["state"], "telemetry_age_s": st["telemetry_age_s"], **extra, "data": section}

    # ---------------------------------------------------------------- page
    @app.get("/")
    def index():
        # The document is never cached by the browser: replaying an older build is exactly how a page ends up with
        # chart canvases the current JS does not know about, which kills the console before it starts polling.
        resp = make_response(render_template("index.html"))
        resp.headers["Cache-Control"] = "no-store, must-revalidate"
        return resp

    # ---------------------------------------------------------------- read APIs
    @app.get("/api/status")
    def status():
        return _json(mgr.status())

    def wants_optional() -> bool:
        """?optional=1: a poller that races a reset/stop gets 200 + available:false instead of a 404."""
        return request.args.get("optional") in ("1", "true")

    @app.get("/api/telemetry")
    def telemetry():
        if wants_optional() and mgr.state_snapshot() is None:
            return _json({**mgr.status(), "available": False, "snapshot": None})
        snap = need_snapshot()
        return _json({**mgr.status(), "available": True, "snapshot": snap})

    @app.get("/api/sensors")
    def sensors():
        s = need_snapshot()
        return _json(with_status(s.get("sensors"), sensors_enabled=s.get("sensors_enabled")))

    @app.get("/api/ekf")
    def ekf():
        s = need_snapshot()
        return _json(with_status(s.get("estimate"), ground_truth_debug=s.get("debug_ground_truth")))

    @app.get("/api/sonar")
    def sonar():
        return _json(with_status(need_snapshot().get("sonar")))

    @app.get("/api/mapping")
    def mapping():
        s = need_snapshot()
        return _json(with_status({k: s.get(k) for k in ("mapping", "cost_map", "sonar_mapping", "timing")}))

    @app.get("/api/thrusters")
    def thrusters():
        s = need_snapshot()
        return _json(with_status({"thrusters": s.get("thrusters"), "dynamics": s.get("dynamics")}))

    @app.get("/api/environment")
    def environment():
        from src.simulation.world_generator import ALL_SCENARIOS, SCENARIOS
        snap = mgr.state_snapshot()
        return _json({"state": mgr.status()["state"], "active": None if snap is None else snap.get("environment"),
                      "pending_overrides": mgr.pending, "applies": "pending changes apply on next simulation start",
                      "scenarios": [{"name": n, "description": SCENARIOS[n]["description"] if n in SCENARIOS
                                     else "Phase 1-3 world built from the Week 4 maps"} for n in ALL_SCENARIOS]})

    def _map_response(key: str):
        m = mgr.maps()
        if (m is None or key not in m) and wants_optional():
            return _json({"available": False, "unchanged": False, "version": None})
        if m is None or key not in m:
            raise ApiError("no map published yet (mapping disabled, not started, or no scan integrated)", 404)
        payload = m[key]
        since = request.args.get("since", type=int)
        if since is not None and payload.get("version") == since:
            return _json({"unchanged": True, "version": since})
        return _json({"unchanged": False, "available": True, **payload})

    @app.get("/api/scene")
    def scene():
        sc = mgr.scene()
        if sc is None:
            return _json({"available": False})
        return _json({"available": True, **sc})

    @app.get("/api/occupancy")
    def occupancy():
        return _map_response("occupancy")

    @app.get("/api/costmap")
    def costmap():
        return _map_response("costmap")

    # ---------------------------------------------------------------- simulation control
    @app.post("/api/simulation/start")
    def sim_start():
        return _json({"ok": True, **mgr.start()}, 202)

    @app.post("/api/simulation/stop")
    def sim_stop():
        return _json({"ok": True, **mgr.stop()})

    @app.post("/api/simulation/restart")
    def sim_restart():
        return _json({"ok": True, **mgr.restart()}, 202)

    @app.post("/api/simulation/reset")
    def sim_reset():
        return _json({"ok": True, **mgr.reset()})

    @app.post("/api/simulation/speed")
    def sim_speed():
        b = body_json()
        if "value" not in b:
            raise ApiError("'value' is required")
        return _json({"ok": True, **mgr.command({"cmd": "set_realtime_factor", "value": b["value"]})})

    # ---------------------------------------------------------------- mapping control
    @app.post("/api/mapping/reset")
    def mapping_reset():
        return _json({"ok": True, **mgr.command({"cmd": "reset_mapping"})})

    @app.post("/api/mapping/enable")
    def mapping_enable():
        return _json({"ok": True, **mgr.command({"cmd": "set_mapping_enabled", "enabled": True})})

    @app.post("/api/mapping/disable")
    def mapping_disable():
        return _json({"ok": True, **mgr.command({"cmd": "set_mapping_enabled", "enabled": False})})

    # ---------------------------------------------------------------- sensor fault injection
    @app.post("/api/sensors/<name>/health")
    def sensor_health(name: str):
        """Inject (or clear) a fault on a REAL sensor of the running simulation. The response carries what the
        simulation reported after applying it; the next telemetry snapshot shows the resulting health + forced flag."""
        if name not in SENSOR_NAMES:
            raise ApiError(f"unknown sensor {name!r}; choose from {list(SENSOR_NAMES)}", 404)
        b = body_json()
        h = canonical_state(b.get("health", ""))
        if h not in FORCED_STATES:             # healthy / degraded / failed / discarded / link cut (RESET = healthy)
            raise ApiError("state must be one of " + ", ".join(FORCED_STATES)
                           + " (or RESET to clear an injected fault)")
        return _json({"ok": True, **mgr.command({"cmd": "set_sensor_health", "sensor": name, "health": h})})

    # ---------------------------------------------------------------- telemetry export + run history (CSV / analytics)
    @app.get("/api/telemetry/export")
    def telemetry_export():
        if not mgr.has_telemetry():
            raise ApiError("no telemetry recorded yet: start a simulation first", 404)
        run_id = mgr.run_id or "run"
        return Response(mgr.telemetry_csv(), mimetype="text/csv",
                        headers={"Content-Disposition": f'attachment; filename="aurora_telemetry_{run_id}.csv"'})

    @app.get("/api/runs")
    def runs_history():
        return _json({"runs": mgr.run_history(), "current_run_id": mgr.run_id,
                      "recorded_rows": mgr.recorded_row_count()})

    # ---------------------------------------------------------------- next-start configuration
    @app.post("/api/environment/configure")
    def environment_configure():
        return _json({"ok": True, **mgr.configure(body_json())})

    # ---------------------------------------------------------------- Phase 5: navigation / planner
    def nav_section(s: dict) -> dict:
        return s.get("navigation") or {"enabled": False}

    @app.get("/api/navigation")
    def navigation():
        s = need_snapshot()
        return _json(with_status(nav_section(s), path=s.get("navigation_path")))

    @app.get("/api/planner")
    def planner():
        s = need_snapshot()
        n = nav_section(s)
        return _json(with_status({"enabled": n.get("enabled", False), "planner": n.get("planner"),
                                  "replanning": n.get("replanning"), "state": n.get("state"),
                                  "goal": n.get("goal")}))

    @app.get("/api/path")
    def path():
        s = need_snapshot()
        return _json(with_status(s.get("navigation_path"), enabled=nav_section(s).get("enabled", False)))

    @app.post("/api/navigation/goal")
    def nav_goal():
        b = body_json()
        goal = b.get("goal")
        if goal is None and "x" in b and "y" in b:
            goal = [b["x"], b["y"], b.get("z")]
        if not isinstance(goal, (list, tuple)) or len(goal) != 3:
            raise ApiError("'goal' must be [x, y, depth] (or give x, y and optionally z); depth is metres below the surface, positive down")
        try:
            goal = [None if v is None else float(v) for v in goal]
        except (TypeError, ValueError):
            raise ApiError("goal coordinates must be numbers")
        if goal[2] is None:                      # depth defaults to the current goal's depth (set by the backend)
            snap = mgr.state_snapshot() or {}
            cur = (snap.get("navigation") or {}).get("goal")
            if not cur:
                raise ApiError("z is required when no goal is active")
            goal[2] = float(cur[2])
        # The z typed here is a DEPTH in metres below the surface (positive down), the same unit the Estimated
        # navigation panel shows; it becomes world z here, once (idempotent for an already-negative world z).
        goal[2] = operator_z_to_world(goal[2])
        return _json({"ok": True, **mgr.command({"cmd": "nav_set_goal", "goal": goal})})

    @app.post("/api/navigation/start")
    def nav_start():
        return _json({"ok": True, **mgr.command({"cmd": "nav_start"})})

    @app.post("/api/navigation/cancel")
    def nav_cancel():
        return _json({"ok": True, **mgr.command({"cmd": "nav_cancel"})})

    @app.post("/api/navigation/replan")
    def nav_replan():
        return _json({"ok": True, **mgr.command({"cmd": "nav_replan"})})

    @app.post("/api/camera/<action>")
    def camera(action):
        if action not in ("follow", "top", "reset", "orbit_left", "orbit_right", "zoom_in", "zoom_out"):
            raise ApiError(f"unknown camera action {action!r}", 404)
        return _json({"ok": True, **mgr.camera_command(action)})

    # ---------------------------------------------------------------- live 3-D window (while the run keeps going)
    @app.post("/api/visual/window")
    def visual_window():
        """Open or close the 3-D PyBullet window for the RUNNING simulation. The run itself is never stopped:
        opening starts the live viewer process that renders this run (PyBullet cannot attach a window to an
        already-running headless server), closing stops only that window."""
        b = body_json()
        if "open" not in b:
            raise ApiError("'open' is required: true opens the 3-D window for the running simulation, false closes it")
        if not isinstance(b["open"], bool):
            raise ApiError("'open' must be true or false")
        return _json({"ok": True, **(mgr.open_window() if b["open"] else mgr.close_window())})

    @app.get("/api/visual/window")
    def visual_window_state():
        return _json({"state": mgr.status()["state"], **mgr.window_state()})

    @app.post("/api/navigation/unknown_policy")
    def nav_policy():
        b = body_json()
        pol = str(b.get("policy", "")).lower()
        if pol not in ("inherit", "allow", "block"):
            raise ApiError("policy must be inherit, allow or block")
        return _json({"ok": True, **mgr.command({"cmd": "nav_set_unknown_policy", "policy": pol})})

    # ---------------------------------------------------------------- legacy aliases (Phase 1 API)
    @app.get("/api/mission")
    def mission():
        snap = mgr.state_snapshot()
        if snap and snap.get("mission"):
            return _json(snap["mission"])
        import yaml
        cfg = yaml.safe_load(mgr.config_path.read_text("utf-8"))["simulation"]
        goal = [float(v) for v in cfg["goal_position"]]
        wps = [[float(v) for v in w] for w in (cfg.get("waypoints") or [goal])]
        return _json({"start": [float(v) for v in cfg["start_position"]], "goal": goal, "waypoints": wps})

    @app.post("/api/start")
    def legacy_start():
        return sim_start()

    @app.post("/api/stop")
    def legacy_stop():
        return sim_stop()

    return app


app = create_app()

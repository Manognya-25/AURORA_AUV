# Dashboard (Phase 4)

## Architecture
```
browser  --HTTP/JSON-->  Flask (dashboard/app.py)  --SimulationManager-->  child process
                                                                          (dashboard/simulation_process.py -> run_simulation.run)
child  --state.json / maps.json (throttled, atomic)-->  manager  -->  API
API    --commands.jsonl / stop.request-->  child
```
* The simulation runs in its **own OS process** (see "Process and runtime-state robustness (Windows)" below), so a crash or a missing PyBullet cannot take Flask down,
  and *Stop* can be verified by checking the process really exited.
* Backend state is authoritative. `SimulationManager.status()` derives the state on every request from the
  OS process, `status.json`, and the age of the newest snapshot: `IDLE, STARTING, RUNNING, STALE (alive but
  no telemetry for > dashboard.telemetry_timeout_s), STOPPING, STOPPED, COMPLETED, ENDED_BEFORE_GOAL, FAILED`.
  A dead process is never shown as RUNNING; the page keeps no copy of simulation state.
* Snapshots (`src/simulation/telemetry_snapshot.build_snapshot`) are built from live objects (EKF, sensor
  suite, mapper, thrusters) at `dashboard.state_publish_hz`; maps at `map_publish_hz`, and only when the map
  version changed (`?since=<version>` returns `unchanged`). Maps are run-length-encoded class grids.
* All responses are strict JSON (`NaN/Infinity -> null`); errors are JSON with a meaningful status.
  `?optional=1` on `/api/telemetry|occupancy|costmap` returns `200 {available:false}` instead of 404 (used by
  the page's poller so a Stop/Reset race cannot produce failing requests).

## API
GET: `/api/status`, `/api/telemetry`, `/api/sensors`, `/api/ekf`, `/api/sonar`, `/api/mapping`,
`/api/thrusters`, `/api/environment`, `/api/occupancy`, `/api/costmap`, `/api/mission`.
POST: `/api/simulation/{start,stop,reset,speed}`, `/api/mapping/{enable,disable,reset}`,
`/api/sensors/<imu|dvl|depth|sonar|compass>/health`, `/api/environment/configure`, `/api/visual/window`
(open/close the 3-D window of the running simulation). `GET /api/visual/window` reads its state. (`/api/start`,
`/api/stop` are Phase 1 aliases.)

## Every control and what it does
| Control | API | Real effect | Shown afterwards |
|---|---|---|---|
| Start | `POST /api/simulation/start` | spawns the simulation process (409 if already live) | state follows STARTING -> RUNNING from real telemetry |
| Stop | `POST /api/simulation/stop` | stop request; waits for the process to exit, escalates terminate/kill | STOPPED only after the OS process is gone |
| Reset | `POST /api/simulation/reset` | stops any run, deletes snapshots/maps/log -> IDLE; next start rebuilds world, vehicle, sensors, EKF and map from config | page clears; maps 404/unavailable |
| Speed | `POST /api/simulation/speed` | changes the running loop's real-time pacing | header pace readout (measured sim s / wall s) |
| Enable / Disable mapping | `POST /api/mapping/{enable,disable}` | mapper stops/starts integrating scans | `enabled` and update count |
| Reset map | `POST /api/mapping/reset` | grid, cost map and counters return to UNKNOWN/zero; simulation continues | known cells 0, update count 0, canvas |
| Sensor state selectors (healthy / degraded / failed / discarded / link cut) | `POST /api/sensors/<name>/health` | changes the RUNNING sensor's own behaviour at once (no stop, no restart); see *Sensor states* below for what each state really does to the output | sensor state table (state, source **and the live reading**), the fault panel's *reading now* column, per-sensor charts, sonar rays, EKF counters |
| Goal x, y, **depth** ("Next start") | `POST /api/environment/configure` `{navigation_goal}` | the third field is a **depth in metres below the surface, typed positive** (7 = 7 m deep), converted once to world z by `src/simulation/commands.operator_xyz_to_world`; stores `navigation.goal_position`, which the simulation resolves the mission goal from in **both** control modes (`src/simulation/commands.resolve_mission_goal`): the planner flies to it with navigation on, and the Phase-4 waypoint mission ends at it with navigation off (blank = the config goal). `simulation.goal_position`/`waypoints` are left untouched so the generated world (whose obstacle-free corridor follows them) does not change. The legacy terminal approach then closes in on the goal (`controller.goal_tolerance`) | pending note (with the goal spelled out again as a depth), and the run's own `mission.start/goal` in the Environment and Mission statistics tables |
| Scenario, seed, obstacle count, dynamics, mapping-at-start + "Save for next start" | `POST /api/environment/configure` | validated (the scenario is really generated) and stored; **applied on next start**. `obstacle_count` is the exact number of PyBullet obstacle bodies the next run will build; on `legacy_week4` it is refused with a 422 (the field is disabled with that reason) | pending note, then the next run's scenario/seed and the Environment panel's *obstacle count requested / actual* |
| 3D window selector (running) | `POST /api/visual/window` `{open}` | **acts on the run that is going**: `on` starts the live viewer process (`src/simulation/live_viewer.py`), a real PyBullet window showing this run; `off` closes only that window. The simulation is never stopped or restarted. A run started with its own window can not be closed from here (409: it belongs to the run) | `window` block in `/api/status`, the note under the selector, the events log |
| 3D window selector (idle) + "Save for next start" | `POST /api/environment/configure` `{show_3d}` | stores `dashboard.headless = not show_3d`; the next start opens the real PyBullet GUI window. While a run IS going, the same call also applies it live (and still stores it) | pending note; `window` block of the response |
| "Auto-open the 3-D view for each run that is started" (checked by default) | `POST /api/visual/window` right after `POST /api/simulation/start`, and `dashboard.auto_open_3d` via `/api/environment/configure` | reveals the 3-D view of the run that was just started: the live viewer for a headless run, or the run's own window. An optional backend setting (`dashboard.auto_open_3d`) makes `start()` itself open it; a machine without a display reports a warning and the run continues | `window` block of `/api/status` (`open`, `owner`, `run_id`), the note under the selector, the events log |
| Sonar rays / ground-truth path checkboxes | none (client-side drawing options) | toggle overlays drawn from real data; truth path labelled DEBUG ONLY | canvas |

Controls that act on a running simulation are disabled when none is running; the backend also rejects
them with 409. A rejected request shows the backend error and never a success state.

### Depth, not world z, is what the operator types
Every z in the dashboard is a **depth in metres below the surface, positive down**: the goal and start fields say so,
the Estimated navigation panel reports `depth (m)`, and the Mission-statistics table prints `x / y / depth`. The
simulation itself works in world z (up-negative), so the two meet in exactly one place per layer:
`src/simulation/commands.operator_z_to_world` on the backend (used by `POST /api/environment/configure` for both the
goal and the start, and by `POST /api/navigation/goal`) and `toDepth`/`toWorldZ` in `dashboard/static/js/dashboard.js`
on the page.

The conversion reads the **magnitude**: `-abs(z)`. A typed `7` therefore means 7 m deep (world z -7), and the older
`-7` form selects the very same point - so no goal typed before this convention existed changes meaning, and a goal
above the surface is simply unrepresentable instead of being refused with a `400 navigation_goal z must be <= 0`. A
rejected field used to reject the **whole** `configure` body, which is why one out-of-range depth silently discarded
the scenario and seed changes submitted in the same click; nothing is rejected for depth any more.

`navigator.check_goal` (the in-simulation validation) keeps working in world z and is still the thing that refuses a
goal outside the map or inside an obstacle - the depth is converted before it is reached.

## Sensor states (applied live to the running sensor)
The selector is a request; the panel is re-read from telemetry afterwards, so it always shows the state the run
really applied (`forced`, plus `stopped`/`zeroed` for what that state does to the output). The API also accepts the
operator names `ENABLED` (=healthy), `DISABLED` (=discarded: no output) and `UNHEALTHY` (=degraded) as aliases of the
canonical states below (`src/sensors/health.canonical_state`).

When **every** configured sensor is FAILED or UNHEALTHY the vehicle has no usable navigation input, so the run stops
itself (`run_simulation.py`): the robot is commanded to stop, the run ends with status `STOPPED` and the final
telemetry carries a `safety` block (`{stopped: true, reason: "all configured sensors failed/unhealthy (...)"}`) plus a
`safety_stop` entry in the command log. DISABLED (discarded / link cut) does not trigger this stop.

| state | what the sensor model really does | what the readings show |
|---|---|---|
| healthy | normal noise/bias, samples carry information | real reading, changing every sample |
| degraded | samples keep coming, noise multiplied by `degraded_noise_scale` | real reading (worse), still changing; the EKF keeps using it |
| failed | a real sample every cycle, but its output is **stuck at 0.0** (`valid=False`, so the EKF rejects it) | `0.00` on every channel, and it never changes |
| discarded | the sensor produces **no sample at all** (`SensorSuite` skips it) | the last reading is frozen (same value, same sample time); `samples` stops growing |
| link cut | no data reaches the computer: same effect as discarded, reported as a link fault | frozen, as above |

* The health chart plots the real state per sensor (`HEALTHY/DEGRADED/FAILED/DISCARDED/LINK CUT` = 0..4).
* A chart fed by one sensor's raw output shows `0.0` flat while that sensor is failed, stops (no new points, no
  current value) while it is discarded / link cut, and never invents or carries over a number.
* The sonar is the one exception to "0.0": a failed scan reports **0 valid rays / 0 hits** and no range at all,
  because a `0.0 m` range would mean "obstacle touching the transducer".
* With **every** configured sensor discarded / link cut the snapshot carries `sensors_all_stopped: true` and the
  page shows no sensor value anywhere (the sensor system delivers nothing; the EKF has nothing to aid it).

## Where every displayed value comes from
Estimated pose/velocity/attitude/covariance/EKF counters: EKF. Sensor health/counts/ages: sensor suite
monitors. Sonar panel: the last `SonarMeasurement` (invalid rays are drawn as invalid, not as ranges).
Mapping/cost statistics and both maps: the mapper. Thrusters/dynamics: the vehicle's thruster model
(`available:false` with the reason in legacy mode). Estimate-vs-truth chart and the truth path: ground
truth, labelled **EVALUATION/DEBUG ONLY**. Environment block: generator metadata, labelled GROUND TRUTH.
Charts use only series accumulated from snapshots; nothing is random or synthetic.

### Sensor readings and failed sensors
Every snapshot carries, per sensor, the sample the sensor produced most recently:

| Field | Meaning |
|---|---|
| `value`, `value_time`, `value_age_s` | the reading of the **current** sample, or `null` when that sample was invalid (dropout/blanked/failed). `value_age_s` is its age at publish time. |
| `last_value`, `last_value_time` | the most recent **valid** reading (kept so history charts can hold on to the real samples they already have). |
| `health`, `forced` | the state the sensor really reports, and whether it was forced by the operator or detected automatically. |

The dashboard shows a number for a sensor only while it is not `FAILED` and its newest valid sample is fresher than
1 s (five times the slowest sensor period, so an ordinary single-sample dropout never blanks the cell). A failed or
silent sensor therefore shows **no value at all** - the column says why (`no data (failed ...)`, `no valid sample for
N s`, `N samples, no valid reading yet`) in the alert colour, and the value cell of `sensorTable` and of the fault
panel's *reading now* column both follow the same rule. The 105-ray sonar is summarised instead of stored raw
(`N hits · nearest X m of 30.0 m`); a failed scan shows `no data (failed)` and no range anywhere.

Charts fed by one sensor's raw output (`IMU acceleration`, `Sonar clearance`) declare that dependency to
`charts.js` through `sourceOk`/`sourceMsg`: while the sensor is failed the read-out shows `–`, the value tag reads
*no valid sample*, the stats line carries the real reason, and the pulse stops. The already-plotted line is real
recorded history and stays. The exported CSV uses the same `value` field, so a failed sensor leaves its sensor
columns empty instead of repeating an old reading.

## Controller gains (live) — adaptive P controller OR path follower
The `Controller gains` panel and both gain charts follow **whichever controller is really producing the commands**
(`gainSource()` in `dashboard.js`), read every telemetry snapshot — nothing is computed in the browser:

* **Legacy Phase-4 mission** → the `controller.adaptive` block written by
  ([`AdaptivePController`](../src/simulation/adaptive_controller.py)): `kp_position` nominal → effective, the active
gain multiplier, its two components (the gain **schedule** and the **self-tuning** term), `kp_yaw` nominal → effective
  and its multiplier, the position / heading errors that produced them, and the closing speed the self-tuning measured.
  The multipliers are capped by `simulation.controller.adaptive.gain_boost`.
* **Autonomous navigation** → the `controller.follower` block written by
  ([`PathFollower`](../src/planning/path_follower.py)): its loop gains in the units they are applied in (`kv`, `kp_yaw`
  / `kd_yaw`, `kp_depth` / `ki_depth`, `station_kp`, cruise speed), the live speed schedule (`speed_factor` and its
  `goal_factor` / `heading_factor` / `sonar_factor` components), the desired speed, lookahead, cross-track, heading and
  depth errors and the clamped depth-hold integral. The two gain charts switch to its series (speed / goal-distance /
  heading / sonar+avoidance over time) and their captions say which controller is being shown; the value read-outs, the
  event marks and the tracked history below follow the same source. Until this the panel and charts were dead in
  autonomous-navigation runs ("not driving this run"), which is the mode most runs use.

The same values are recorded in the exported CSV as the `ctrl_*` columns (adaptive controller) and the `follow_*`
columns (path follower), so the gains can be analysed off-line and not only watched.

## Gain tracking (live)
Below the live gains, `Gain tracking` shows the record the driving controller keeps of its **own** gains, cumulative for
the run and counted step by step in that controller — again nothing inferred in the browser:

* **adaptive P controller** (`controller.adaptive.tracking`):
  * **adaptation on the last step** — which way the self-tuning pushed (`ramping` / `relaxing` / `holding` / `off`) and
    the measured quantity that chose it (*stalled (closing 0.000 m/s)*, *tracking well (closing 0.250 m/s)*, *inside the
    settle deadband*, *new target — adaptation re-seeded*).
  * **time budget** — adaptation steps and simulated seconds tracked, and how much of it went to ramping, relaxing and
    holding; then how much was driven **boosted** (gain above nominal, schedule or self-tuning) and how much of that sat
    at the clamped **gain cap**. A step without a usable `dt` did not adapt and is not counted.
  * **tracked gain** — the multiplier in use now, the mean over the run so far and the peak it reached (plus the peak of
    the self-tuning term alone and the adaptation time that peak was reached at).
* **path follower** (`controller.follower.tracking`): commands and simulated time seen; time in each mode (following /
  holding / station-keeping) and in each speed state (full speed / slowed / braked); the now / mean / slowest speed
  factor and the slowest heading and sonar factors; and the slow-down / brake episodes (rising edges, so one long slow
  section is counted once). A new navigation mission resets it; a replan does not.
* **episodes / re-seeds** — how many times the gain started ramping again, and how many times a new target re-seeded
the history (adaptive controller only).

Real gain episodes are also marked on the multiplier chart and written to the Events log: the self-tuning starting to
ramp (`ramp ↑`), reaching the gain cap (`gain cap`) and relaxing back down (`relax ↓`), and — for the path follower —
the speed being scheduled down (`slow ↓`) and commanded to zero (`brake`).

## Live telemetry charts
The 20 time-series charts are drawn by `dashboard/static/js/charts.js`. That file only renders: the series are still
accumulated by `dashboard.js` from `/api/telemetry` snapshots exactly as before (monotonic sim time, 600-sample
buffer, cleared on reset/idle/sim-time rollback). Two of them belong to the controller that is driving, both fed from
the same block: **Gain multipliers** (adaptive: schedule / self-tuning / active / yaw; path follower: speed / goal
distance / heading / sonar+avoidance, with the
gain-episode event marks) and **Tracked gain history** (adaptive: cumulative time ramping, boosted and at the gain cap;
path follower: cumulative time at full speed, slowed and braked), so the gains are visible over time rather than only as
a current number. When a chart has no data at all its explanation is word-wrapped to the plot width and clipped to the
plot, so a long reason can no longer run over the axes.

* **Live indicator** driven by the real run state: LIVE (running), STALLED (telemetry timeout; amber, with an
  on-chart notice), CONNECTING/STOPPING, STOPPED/COMPLETED/ENDED/FAILED, NO DATA (idle), OFFLINE (backend unreachable),
  plus "updated N s ago" from `telemetry_age_s`.
* **Current value** (large readout, with unit) and window **min / mean / max / sample count / sim time**.
* **Rolling time window** selector: 30 s, 60 s, 120 s, all buffered. A window fills from t = 0, then scrolls.
* **Axes**: gridlines, nice tick labels, axis titles with units (`Position error (m)`, `Cells known (%)`,
  `Simulation time (s)`), auto-scaled Y with eased range changes.
* **Latest-point marker** (pulses only while LIVE), dashed guide line and a value tag showing the newest real sample.
* **Hover read-out**: nearest real sample's sim time and value.
* **Smooth updates**: a `requestAnimationFrame` loop glides the newest point from the previous sample to the new one over
  the measured arrival interval, so the axis scrolls continuously between 400 ms polls. No value is interpolated into
  the data, extrapolated, or invented; the animation is cosmetic. `prefers-reduced-motion` disables it.
* HiDPI-aware canvas sizing (ResizeObserver); `data-points` / `data-last` attributes on each canvas expose the real
  buffer size and latest value for tests. There is no battery
display because no battery model exists.

## Limitations
Fault injection exists for the configured sensors (IMU, DVL, depth, compass, sonar). No authentication (local engineering tool).
Maps are polled (1 s), not pushed. The dashboard process manager supports one simulation at a time.

## Process and runtime-state robustness (Windows)
Bugs found on Windows and how they are fixed (regression tests: `tests/test_windows_runtime.py`):

| Symptom | Root cause | Fix |
|---|---|---|
| `PermissionError: [WinError 5]` replacing `runtime\state.json`; child dies | `os.replace()` onto a file another process has open fails on Windows (no `FILE_SHARE_DELETE`); the dashboard polls the file constantly | `atomic_write_json` uses a unique temp file and a retried `os.replace` (WinError 5/32/33 backoff, ~5 s). A publish that still fails is counted and shown (`ipc` block in telemetry), never fatal. Readers retry too and keep the last good snapshot. |
| Test: `assert pid_alive(pid)` false right after Start | the test probe `os.kill(pid, 0)` **terminates** the process on Windows (and `/proc` does not exist) | `src/utils/procutil.pid_alive` only queries (`OpenProcess` + `GetExitCodeProcess`); a guard test forbids `os.kill` elsewhere |
| Start/Reset fail while a previous run's log/state is "in use" | files unlinked while a handle was still open | every run gets its own directory `runtime/runs/<run_id>/`; nothing from an earlier run is ever reused or deleted-while-open; old runs are pruned best-effort |
| Stale `stop.request` / state from an earlier or orphaned run | one shared directory | per-run directory + stop requests scoped to the run; children of an earlier dashboard are asked to stop on manager start; a child exits by itself when the dashboard process dies |
| Two Start clicks / races | unsynchronised start | start/stop/reset are serialized; a second concurrent Start gets HTTP 409 |
| A second `python dashboard.py` "starts" fine but the page never updates: charts blank, panels stuck on "no telemetry yet" | on Windows a second process **can** bind an address already in use, so both servers listen and the browser is answered by the older one — which still holds the page it read at start-up (an old `index.html`) while the static JS on disk is new, so the JS threw on the first chart canvas that old page did not have and the whole boot sequence (charts *and* polling) died | one dashboard per port, on purpose: `dashboard/serve.py` refuses to start when the port is already serving (`python dashboard.py --port 5001` runs it elsewhere, `DASHBOARD_HOST`/`DASHBOARD_PORT` also work); templates auto-reload and the document is served `Cache-Control: no-store`; every chart is built on its own, so one missing canvas can never take the console down (`tests/test_dashboard_serve.py`, `tests/test_dashboard_charts.py`) |

State is still derived from the real OS process and telemetry age; PID and status are never synthesized.
Only one dashboard may serve a given port: it owns the simulation child and the run-history file, and a second
server on the same port would answer some requests with a different, stale view of both.
A console that is already running can be verified in a real browser (no simulation of the checks):
`python tools/verify_live_console.py [--url http://127.0.0.1:5000/]` loads the page, starts a navigation run and
requires no page errors, the JS/page chart list to agree, and the charts to keep receiving points, with the gain
panel naming the controller that is driving. Exit code 0 means the console really is live.

## 3D PyBullet window (live, without stopping the run)

The selector is not only a next-start setting any more:

* **At start**: `dashboard.headless: false` ("3D window: on" saved for the next start) or `python run_visual_demo.py`
  opens the real PyBullet GUI inside the run itself. See `docs/visual_mode.md`.
* **While a run is going**: choosing "on" calls `POST /api/visual/window {"open": true}`. PyBullet cannot attach a
  GUI window to a physics server that is already running, so the backend starts `src/simulation/live_viewer.py`:
  a separate process that opens a real PyBullet window and renders **this run** -- the same world geometry rebuilt
  from the run's own config, the true pose from its telemetry, its measured sonar rays, the EKF estimate marker, the
  mapped occupied cells, the planned path and the authoritative goal. "off" closes only that window. In both cases
  the simulation process, its pid and its telemetry are untouched (verified in `tests/test_live_window.py`).
* **It stays open after the run ends**: `visualization.keep_open_after_run` (default `true`) keeps the window on the
  last frame when the run finishes on its own -- the robot reaching the goal, `STOPPED`, `ENDED_BEFORE_GOAL`,
  `FAILED` -- so the operator can still look at where the vehicle ended up. It is then closed explicitly (its own
  window close button or the dashboard's window control). A run that *vanished* instead of ending (its process
  died, or the dashboard exited) still closes it, so a dead simulation never leaves an orphan window behind.
  `false` restores the old behaviour (close as soon as the run ends). The viewer decides this from the run's own
  `status.json` (`live_viewer.run_ended_status`), never from a guess.
* **The window the RUN opened itself is kept too**: this applies to BOTH windows. The live viewer (above) keeps
  rendering once the run is gone; the run's OWN PyBullet window is kept by the run process itself, which after the
  mission ends stops simulating and only holds the window open (`run_simulation.hold_window_open`). While it is
  held the child writes `status.json` with `holding: true`, so `GET /api/status` reports the run's REAL terminal
  state (`COMPLETED` / `STOPPED` / ...) with a fresh snapshot -- *not* `RUNNING` and *not* `STALE` -- and
  `alive: true` (the window still exists). The dashboard disables **Start** and enables **Stop** while a window is
  held, and the window control (or Stop) releases it: the run disconnects and the window closes. **Apply & restart**
  also releases it and opens the new run's window with the NEW environment, so a scenario change is never shown in
  the old window. Just starting a fresh run needs the held window released first (it still owns the run slot).
* **The goal you set, in numbers**: the `GOAL` marker carries a second label with the goal coordinates
  (`goal x, y, z`, e.g. `goal 7.5, 5.0, -5.0`) in both the run's own window and the live viewer, so the window shows
  the goal that was entered and not just the word "GOAL". Visual only -- no collision shape, physics untouched.
* **The goal you entered, in the tables**: the Mission statistics table shows both the goal the run actually used
  (`goal x / y / depth (m), configured`, depth = -z) and `goal you entered, x / y / z (m; z is depth, negative)`
  (the numbers typed in the "Navigation goal x, y, z" fields). A goal typed while a run is already going applies to
  the NEXT start (or live via "Set goal" when navigation is on), so the two can legitimately differ -- showing both
  is how the operator sees that at a glance instead of mistaking one for the other.
* **Camera**: the 3-D camera buttons, the side-panel buttons and the keys 1-7 act on whichever window is open -- the
  live viewer (commands go through `viewer_commands.jsonl`) or the run's own window (the normal command channel).
* `GET /api/status` reports the truth in its `window` block (`open`, `owner` = `run`/`viewer`, `viewer_pid`,
  `can_open`/`can_close`, and any real failure such as "no display"). The viewer writes `viewer_status.json` and
  `viewer.log` in the run's own directory, exits when the run ends or the window is closed, and never edits a run file.
* The 2D occupancy/cost maps stay in the dashboard.


## Phase 5 additions
Navigation panel (state badge, goal / start / replan / cancel controls, unknown-cell policy, planner / following / avoidance /
replanning tables, state history) and a path overlay (green global path, orange detour, yellow current waypoint, magenta goal) on the
occupancy and cost-map canvases. Endpoints: `GET /api/navigation|planner|path`; `POST /api/navigation/goal|start|cancel|replan|unknown_policy`;
`POST /api/environment/configure` accepts `navigation_enabled`, `navigation_goal`, `unknown_policy`. See `docs/navigation.md`.

* The goal inputs (x / y / **depth**) **mirror the authoritative goal** (`navigator.goal`) on every poll, so what you edit
  is the goal the planner is actually using. *Set goal* posts it; on success the navigator re-plans (reason `goal_changed`),
  the path/trajectory/waypoint overlays and the 3-D GOAL marker move to it and the planner/replanning tables report it. An
  invalid goal is shown as an error and changes nothing.
* **When the robot reaches the goal, the goal that was entered is shown back with its coordinates**: the status
  banner (`Goal reached … the goal you set was 18.50, 13.50, depth 7.00 (x, y, depth in m)`), a row in the
  Mission-statistics table (`goal reached — the goal you set (x, y, depth m)`), **the goal rows of the Estimated
  navigation panel** (see below), a line in the Events & alerts log, and a coordinate label next to the magenta goal
  marker on the occupancy/cost-map canvases (it reads `GOAL REACHED …` once the goal is reached). The numbers come
  from the **authoritative goal** -- `navigation.goal` when navigation is on (so a goal changed mid-run with *Set goal*
  or the in-window goal keys is the one echoed), otherwise the run's own `mission.goal`. Nothing here is derived from
  the true state.
* **The goal in the Estimated navigation panel**: right under the estimate's own `x / y (m)` and `depth (m)` rows sit
  `goal this run x, y (m)`, `goal this run depth (m)` and `goal − estimate x, y, depth (m)`. The goal is printed in the
  *same units as the estimate above it*, so "did it arrive where I asked?" is answered by reading one panel instead of
  comparing the input form with a table. All three rows go through the single `toDepth` helper in `dashboard.js`, which
  is the only place in the UI where depth and world z meet.
* *Start navigation / Replan now / Cancel* are real state-machine commands: their result comes back from the simulation's
  own command log (acknowledged `detail`) and the state badge, state history and planner counters follow.
* The Environment panel reports the world the run actually built: `obstacle count requested / actual`, obstacle kinds and
  the collision-body / structure counts from `Environment.describe()`.

## Verifying the dashboard camera buttons against the real 3-D window

1. `python dashboard.py`, open the page and **Start** (headless is fine), then set **3D window: on**: the real
   PyBullet window opens next to the dashboard without restarting the run (the note under the selector shows
   "OPEN: live 3-D window showing this run (viewer PID ...)"). The other way is to set it before Start and press
   **Save for next start**.
2. In the dashboard click **Top view**: the PyBullet window must switch to the overhead view. **Follow AUV** returns it.
3. **Orbit** left/right must rotate the view around the AUV; **Zoom +/-** must change the distance; **Reset view** clears the offsets.
4. Every click is acknowledged in the dashboard event log ("camera <action>: mode=..., orbit offset ..., zoom x...") straight from the
   camera rig. Without a 3-D window the dashboard says so (HTTP 422) instead of pretending.

## Auto-opening the 3-D view on Start
Two independent mechanisms, both acting on the run that was just started (never a second simulation):

* **Web console (default)**: the "Auto-open the 3-D view for each run that is started" checkbox is on by default. After
  a successful Start the page calls `POST /api/visual/window {open:true}` and reveals the window note, so a headless
  run gets the live viewer and a run started with "3D window: on" reports its own window. The Start handler
  (`autoOpen3dForRun`) never blocks the run: an API error is shown as an event.
* **Backend (`dashboard.auto_open_3d: true`, next-start setting)**: `SimulationManager.start()` opens the live window
  itself after spawning the child, so a non-browser launcher gets the same behaviour. A missing display only appends a
  warning; the run is unaffected.

Both are measured against a real PyBullet session with `python tools/verify_dynamic_dashboard.py`.

## Goal distance: estimate vs ground truth
The Mission-statistics table shows the operational **EKF-estimate** AUV-to-goal distance and, on its own row, the
**ground-truth evaluation** distance (`debug_ground_truth.goal_distance_true`, labelled EVALUATION). They come from the
same snapshot (`navigation.goal_distance_m` vs the truth position to the authoritative goal) and are never mixed.

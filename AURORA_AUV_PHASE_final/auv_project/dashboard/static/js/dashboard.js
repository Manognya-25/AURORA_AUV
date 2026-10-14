/* AURORA AUV console. The page holds no simulation state of its own: every number comes from the
   backend (/api/status, /api/telemetry, /api/occupancy, /api/costmap) and every control is a real API call.
   Buttons never change the displayed state optimistically; the next poll shows what the backend reports. */
"use strict";
const $ = (id) => document.getElementById(id);
const DASH = "–";
const f = (v, d = 2) => (v === null || v === undefined || Number.isNaN(Number(v))) ? DASH : Number(v).toFixed(d);
const deg = (r) => (r === null || r === undefined) ? null : r * 180 / Math.PI;
const SENSORS = ["imu", "dvl", "depth", "compass", "sonar"];
const HEALTH_NUM = { HEALTHY: 0, DEGRADED: 1, FAILED: 2, DISCARDED: 3, LINK_CUT: 4 };   // sensor-state chart scale
const STOPPED_MODES = new Set(["DISCARDED", "LINK_CUT"]);                 // operator states that produce no sample at all
const AXIS_COLORS = ["#2b6cb0", "#2f855a", "#805ad5"];                   // X / Y / Z component colours
const HEALTH_COLORS = { imu: "#25708f", dvl: "#2f855a", depth: "#805ad5", sonar: "#c05621", compass: "#b7791f" };
const THR_COLORS = ["#25708f", "#b7791f", "#2f855a", "#c05621", "#805ad5", "#b83a2c", "#2b6cb0", "#86198f"];
const LIVE = new Set(["STARTING", "RUNNING", "STALE", "STOPPING"]);
const HAS_TELEMETRY = new Set(["RUNNING", "STALE", "STOPPING", "STOPPED", "COMPLETED", "ENDED_BEFORE_GOAL", "FAILED", "STARTING"]);

let status = null, snap = null, occ = null, cost = null, occVersion = -1, costVersion = -1;
let errSeries = [], covSeries = [], navEvents = [], lastNav = null, seenCmd = new Set(), lastState = null, lastSimTime = -1;
let adaptEvents = [], lastAdapt = null;                    // gain episodes tracked from the controller's telemetry
let lastFollow = null;                                     // ... and the follower's speed-schedule episodes
let lastHealth = {}, pendingFault = {}, thrustCount = 0, runs = [], goalEditUntil = 0;
let missionStats = { dist: 0, lastP: null };
/* Every chart series: [sim_time_s, value] pairs accumulated from REAL snapshots only. Bounded by push() so the
   buffers can never grow without limit; recreated on reset so a new run starts clean. */
function newSeries() {
  return {
    goal: [], speed: [], depth: [], yaw: [], clr: [], rpl: [],
    kpsched: [], ktune: [], kpmult: [], kyaw: [],        // adaptive P controller live gain multipliers
    kramp: [], kboost: [], kcap: [],                     // ... and its TRACKED cumulative adaptation time
    fspeed: [], fgoal: [], fhead: [], fsonar: [],        // Phase-5 path follower live speed schedule
    ffull: [], fslow: [], fbrake: [],                    // ... and its TRACKED cumulative time per speed state
    roll: [], pitch: [], estx: [], esty: [], estz: [],
    velx: [], vely: [], velz: [],
    imuax: [], imuay: [], imuaz: [],
    angx: [], angy: [], angz: [],
    hlImu: [], hlDvl: [], hlDepth: [], hlSonar: [], hlCompass: [],
    thr: [],
  };
}
let S = newSeries();

/* ---------------------------------------------------------------- API */
async function api(method, url, body) {
  try {
    const r = await fetch(url, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    let data = null;
    try { data = await r.json(); } catch (e) { data = { ok: false, error: `non-JSON response (HTTP ${r.status})` }; }
    return { ok: r.ok, status: r.status, data };
  } catch (e) {
    return { ok: false, status: 0, data: { ok: false, error: "backend unreachable: " + e.message } };
  }
}
function logEvent(text, isErr, simT) {
  const li = document.createElement("li");
  const stamp = simT === null || simT === undefined ? "" : `[t=${f(simT, 1)}s] `;
  li.textContent = `${new Date().toLocaleTimeString()}  ${stamp}${text}`;
  if (isErr) li.className = "err";
  const ol = $("events"); ol.prepend(li);
  while (ol.children.length > 200) ol.removeChild(ol.lastChild);          // bounded: the log never grows unbounded
}
async function act(label, method, url, body) {
  const r = await api(method, url, body);
  const msg = r.ok ? `${label}: ${r.data.detail || r.data.state || "ok"}` : `${label} failed: ${r.data.error}`;
  const el = $("lastAction"); el.textContent = msg; el.className = "note " + (r.ok ? "ok" : "err");
  logEvent(msg, !r.ok);
  await poll();
  return r;
}

/* ---------------------------------------------------------------- controls */
$("btnStart").onclick = async () => {
  const r = await act("Start", "POST", "/api/simulation/start");
  if (r.ok && $("chkAuto3d").checked) await autoOpen3dForRun();          // reveal the 3-D view of THIS run
  return r;
};

/* Auto-open the 3-D view for the run that was just started. It uses the SAME live-window endpoint as the manual
   switch, so no second simulation and no second environment are created: a run with its own window is a no-op, a
   headless run gets the live viewer that renders this run. A failure (no display, etc.) is reported as an event and
   never blocks the run. */
async function autoOpen3dForRun() {
  const st = status ? status.state : null;
  if (!["STARTING", "RUNNING", "STALE"].includes(st)) return;              // nothing to show yet
  const w = (status && status.window) || {};
  if (w.open) { revealWindowNote("3-D view already open for this run (" + (w.owner || "viewer") + ")"); return; }
  const r = await api("POST", "/api/visual/window", { open: true });
  if (r.ok && r.data.open) logEvent(`3-D view opened automatically for this run (${r.data.owner || "viewer"})`, false);
  else logEvent(`auto-open of the 3-D view failed: ${r.data.error}`, true);
  await poll();
  revealWindowNote(r.ok ? "3-D view opened automatically for this run" : "3-D view could not open");
}
/* "Reveal" the 3-D view control: bring the real state of this run's window into view and flash it, so the
   automatically opened view is unmissable (the PyBullet window itself has no browser tab to navigate to). */
function revealWindowNote(text) {
  const note = $("windowNote");
  if (!note) return;
  note.textContent = "3-D window: " + text;
  note.className = "note " + (text.includes("could not") ? "err" : "ok");
  if (note.scrollIntoView) note.scrollIntoView({ block: "center", behavior: "smooth" });
}
$("btnStop").onclick = () => act("Stop", "POST", "/api/simulation/stop");
$("btnReset").onclick = async () => { resetClientSeries(); await act("Reset", "POST", "/api/simulation/reset"); };
$("btnMapOn").onclick = () => act("Enable mapping", "POST", "/api/mapping/enable");
$("btnMapOff").onclick = () => act("Disable mapping", "POST", "/api/mapping/disable");
$("btnMapReset").onclick = async () => { occVersion = costVersion = -1; await act("Reset map", "POST", "/api/mapping/reset"); };
$("selSpeed").onchange = () => act("Set speed", "POST", "/api/simulation/speed", { value: Number($("selSpeed").value) });
$("btnApplyRestart").onclick = async () => {
  const saved = await $("btnApply").onclick();                    // validates + saves exactly like "Save for next start"
  if (!saved || !saved.ok) return;                                // invalid settings: nothing is restarted
  resetClientSeries();
  await act("Apply & restart", "POST", "/api/simulation/restart");
};
$("btnApply").onclick = () => {
  const body = { scenario: $("selScenario").value, seed: $("inSeed").value, dynamics_model: $("selDyn").value,
                 mapping_enabled: $("selMapStart").value === "true", show_3d: $("selShow3d").value === "true",
                 auto_open_3d: $("chkAuto3d").checked,
                 obstacle_count: $("inCount").disabled ? null : $("inCount").value,
                 navigation_enabled: $("selNavStart").value === "true", unknown_policy: $("selPolicyStart").value };
  const gx = $("inGoalX").value, gy = $("inGoalY").value, gz = $("inGoalZ").value;
  body.navigation_goal = (gx === "" && gy === "" && gz === "") ? null : [Number(gx), Number(gy), Number(gz)];
  const sx = $("inStartX").value, sy = $("inStartY").value, sz = $("inStartZ").value;
  body.start_position = (sx === "" || sy === "" || sz === "") ? null : [Number(sx), Number(sy), Number(sz)];
  return act("Save next-start settings", "POST", "/api/environment/configure", body).then(async (r) => { await loadEnvironment(false); return r; });
};
const numOrNull = (id) => { const v = $(id).value; return v === "" ? null : Number(v); };
$("btnGoal").onclick = () => {
  const x = numOrNull("goalX"), y = numOrNull("goalY"), z = numOrNull("goalZ");
  if (x === null || y === null) { act("Set goal", "POST", "/api/navigation/goal", {}); return; }   // backend rejects with the real reason
  return act("Set goal", "POST", "/api/navigation/goal", { goal: [x, y, z] });
};
$("btnNavStart").onclick = () => act("Start navigation", "POST", "/api/navigation/start");
$("btnNavReplan").onclick = () => act("Replan", "POST", "/api/navigation/replan");
document.querySelectorAll("#camRow button, [data-cam]").forEach((b) => {
  b.onclick = () => act("Camera " + b.dataset.cam, "POST", "/api/camera/" + b.dataset.cam);
});
$("btnNavCancel").onclick = () => act("Cancel navigation", "POST", "/api/navigation/cancel");
$("selPolicyLive").onchange = () => act("Set unknown policy", "POST", "/api/navigation/unknown_policy", { policy: $("selPolicyLive").value });

/* Clear every injected sensor fault (each request is acknowledged by the simulation before anything is claimed). */
/* ---------------------------------------------------------------- live 3-D PyBullet window
   While a run is going the switch acts on THAT run: "on" opens a real PyBullet window that shows the running
   simulation (PyBullet cannot attach a window to an already-running headless server, so the backend starts a
   viewer process that renders this run's own telemetry - the run itself is never stopped or restarted), "off"
   closes only that window. With no run going the value is a next-start setting stored by "Save for next start". */
function liveRun() { return !!(status && (status.state === "RUNNING" || status.state === "STALE")); }
function syncWindowControl() {
  const w = (status && status.window) || null;
  const sel = $("selShow3d"), note = $("windowNote");
  if (!w) { note.textContent = "3-D window: backend unreachable."; return; }
  const holding = !!(status && status.holding);
  if ((liveRun() || holding) && document.activeElement !== sel) sel.value = w.open ? "true" : "false";
  let txt;
  if (!liveRun() && !holding) txt = "No run: the switch is saved for the next start (that run opens the PyBullet window at start).";
  else if (holding && w.owner === "run") txt = "OPEN: the run finished and its own PyBullet window is kept on the last frame (visualization.keep_open_after_run). Close the window, or choose \"off\", to dismiss it.";
  else if (w.owner === "run") txt = "OPEN: this run opened its own PyBullet window at start; it closes when the run ends.";
  else if (w.open) txt = `OPEN: live 3-D window showing this run (viewer PID ${w.viewer_pid}); the camera buttons act on this window.`;
  else txt = "CLOSED: choose \"on\" to open a real PyBullet window showing this running simulation (the run keeps going).";
  if (w.error) txt += " — " + w.error;
  note.textContent = "3-D window " + txt;
  note.className = "note" + (w.error ? " err" : w.open ? " ok" : "");
}
$("selShow3d").onchange = async () => {
  const want = $("selShow3d").value === "true";
  const holding = !!(status && status.holding);
  if (!liveRun() && !holding) {                                         // idle: a next-start setting, saved explicitly
    $("windowNote").textContent = "3-D window: saved when you press \"Save for next start\" (no run is going).";
    $("windowNote").className = "note";
    return;
  }
  const r = await act(want ? "Open 3-D window" : "Close 3-D window", "POST", "/api/visual/window", { open: want });
  if (!r.ok) $("selShow3d").value = String(!want);                      // it was refused: show the real state again
  syncWindowControl();
};

$("btnClearFaults").onclick = async () => {
  const names = Object.keys((snap && snap.sensors) || {});
  if (!names.length) { const el = $("lastAction"); el.textContent = "Clear faults failed: no telemetry with configured sensors"; el.className = "note err"; logEvent(el.textContent, true); return; }
  const failed = [];
  for (const n of names) {
    const r = await api("POST", `/api/sensors/${n}/health`, { health: "HEALTHY" });
    if (!r.ok) failed.push(`${n}: ${r.data.error}`);
  }
  const msg = failed.length ? `Clear faults: ${failed.length} of ${names.length} rejected (${failed.join("; ")})` : `Clear faults: cleared on ${names.length} sensors`;
  const el = $("lastAction"); el.textContent = msg; el.className = "note " + (failed.length ? "err" : "ok");
  logEvent(msg, !!failed.length);
  await poll();
  syncFaultTable();
};

/* Export the telemetry recorded by the backend for this run as CSV (the backend is the source of truth). */
$("btnExport").onclick = async () => {
  try {
    const resp = await fetch("/api/telemetry/export");
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try { detail = (await resp.json()).error || detail; } catch (e) { /* not JSON */ }
      throw new Error(detail);
    }
    const text = await resp.text();
    const lines = text.trimEnd().split("\n").length - 1;
    const blob = new Blob([text], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url; a.download = "aurora_telemetry.csv";
    document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    const msg = `Export CSV: downloaded ${lines} rows of recorded telemetry`;
    const el = $("lastAction"); el.textContent = msg; el.className = "note ok"; logEvent(msg, false);
  } catch (e) {
    const msg = `Export CSV failed: ${e.message}`;
    const el = $("lastAction"); el.textContent = msg; el.className = "note err"; logEvent(msg, true);
  }
};

$("btnRunRefresh").onclick = () => loadRuns();
$("selRunA").onchange = () => renderCompare();
$("selRunB").onchange = () => renderCompare();
function resetSeries() {
  errSeries = []; covSeries = []; S = newSeries(); thrustCount = 0;
  navEvents = []; lastNav = null; adaptEvents = []; lastAdapt = null; lastFollow = null;
  lastHealth = {}; missionStats = { dist: 0, lastP: null };
}
function resetClientSeries() { resetSeries(); occ = cost = null; occVersion = costVersion = -1; }

/* ---------------------------------------------------------------- sensor fault controls
   The selector is ONLY a request form: the displayed value is always re-read from the backend telemetry
   (syncFaultTable), so what the operator sees is what the simulation actually applies. */

/* Live reading of one sensor, straight from the telemetry the simulation published (no client-side memory).
   The displayed value follows the operator state the run really applied:
     - DISCARDED / LINK_CUT: the sensor produces NO sample at all, so its readings STOP CHANGING - the last
       reading is shown frozen, with the fault that froze it and how long it has been frozen.
     - FAILED (injected, or detected automatically): the sensor keeps producing a sample at its own rate but the
       output is stuck at 0.0 on every channel, so the reading is 0.00 and it stays 0.00.
     - a single invalid sample (dropout / out of altitude / blanked ray) shows no number for that instant; the
       real history already charted is kept.
     - every sensor discarded / link cut: the sensor system delivers nothing, so no sensor value is shown at all.
   SENSOR_VALUE_STALE_S is 1.0 s: five times the slowest sensor period (compass/sonar run at 5 Hz), so ordinary
   single-sample dropouts never blank the cell while a really dead sensor does. */
const SENSOR_VALUE_STALE_S = 1.0;
const SENSOR_FMT = {
  imu: (v) => `a ${v.slice(0, 3).map((x) => f(x, 2)).join("/")} m/s²` + (v.length > 3 ? ` · ω ${v.slice(3, 6).map((x) => f(x, 3)).join("/")} rad/s` : ""),
  dvl: (v) => `v ${v.slice(0, 3).map((x) => f(x, 2)).join("/")} m/s`,
  depth: (v) => `${f(v[0], 2)} m`,
  compass: (v) => `${f(deg(v[0]), 1)}° (${f(v[0], 3)} rad)`,
};

function modeLabel(s) { return String((s && s.forced) || "").toLowerCase().replace("_", " "); }
function sensorStopped(s) { return !!(s && STOPPED_MODES.has(s.forced)); }
function sensorsAllStopped() { return !!(snap && snap.sensors_all_stopped); }
function fmtReading(name, v) {
  const fmt = SENSOR_FMT[name];
  return fmt ? fmt(v) : v.map((x) => f(x, 3)).join(" / ");
}

function sensorReading(name, s) {
  if (!s) return "not configured";
  if (sensorsAllStopped()) return "no sensor data (every sensor is discarded / link cut)";
  const so = (snap && snap.sonar) || {};
  if (sensorStopped(s)) {                                   // no sample at all: the reading is frozen where it was
    // Deliberately NO ticking age in this text: the value must visibly STOP changing. How long it has been frozen
    // is in the reading-age column and in the chart's message, not mixed into the frozen reading.
    const held = name === "sonar"
      ? (so.available ? `${so.valid_rays} valid rays · ${so.hit_rays} hits (last scan t=${f(so.timestamp, 1)} s)`
                      : "no scan was ever produced")
      : (Array.isArray(s.value) && s.value.length ? fmtReading(name, s.value) : "no reading was ever produced");
    return `${modeLabel(s)}: no new samples — output frozen at ${held}`;
  }
  if (s.zeroed) {                                           // FAILED: a real sample, stuck at 0.0 on every channel
    if (name === "sonar") return "0 valid rays · 0 hits — failed (scan stuck at 0)";
    return `${fmtReading(name, Array.isArray(s.value) && s.value.length ? s.value : [0])} — failed (output stuck at 0.0)`;
  }
  if (name === "sonar") {                                  // 105 rays: summarised by the sonar section, never stored raw
    if (!so.available) return "no scan yet";
    if (so.health === "FAILED" || !so.scan_valid) return "0 valid rays · 0 hits — failed (scan stuck at 0)";
    if (!so.valid_rays) return "no valid rays (every ray dropped / blanked)";
    if (!so.hit_rays) return `${so.valid_rays} valid rays · no return (clear) · max range ${f(so.max_range_m, 1)} m`;
    return `${so.hit_rays} hits · nearest ${f(so.min_valid_range_m, 2)} m of ${f(so.max_range_m, 1)} m`;
  }
  if (!Array.isArray(s.value) || !s.value.length) return s.samples ? `${s.samples} samples, no valid reading yet` : "no sample yet";
  if (s.value_age_s !== null && s.value_age_s !== undefined && s.value_age_s > SENSOR_VALUE_STALE_S)
    return `no valid sample for ${f(s.value_age_s, 1)} s (last valid t=${f(s.last_value_time, 1)} s)`;
  return fmtReading(name, s.value);
}

/* A chart fed by ONE sensor's raw output shows a live current value only while that sensor really produces one.
   FAILED: the samples ARE 0.0, so the chart gets a flat 0.0 line (fresh, not blanked). DISCARDED / LINK_CUT: no
   sample arrives at all, so the series simply stops (its real history stays on screen). */
function sensorFresh(name) {
  const s = snap && snap.sensors && snap.sensors[name];
  if (!s || sensorStopped(s) || sensorsAllStopped()) return false;
  if (!Array.isArray(s.value) || !s.value.length) return false;
  return s.value_age_s === null || s.value_age_s === undefined || s.value_age_s <= SENSOR_VALUE_STALE_S;
}
function sensorNoDataMsg(name) {
  if (!snap) return "No data – start the simulation";            // idle / no telemetry is not a sensor fault
  const s = snap.sensors && snap.sensors[name];
  if (sensorsAllStopped()) return "no sensor data: every sensor is discarded / link cut, so the sensor system delivers nothing";
  if (!s) return `no ${name} in this run's sensor suite`;
  if (sensorStopped(s)) {
    const age = (s.value_age_s === null || s.value_age_s === undefined) ? null : f(s.value_age_s, 1) + " s";
    return `${name} ${modeLabel(s)}: no new samples, so the chart stops here` + (age ? ` (frozen for ${age})` : "");
  }
  if (s.zeroed) return `${name} FAILED: every sample is a stuck-at-zero output, so the chart stays at 0.0`;
  if (s.value_age_s !== null && s.value_age_s !== undefined && s.value_age_s > SENSOR_VALUE_STALE_S)
    return `${name}: no valid sample for ${f(s.value_age_s, 1)} s (last valid t=${f(s.last_value_time, 1)} s)`;
  return `${name}: no valid sample yet (${s.samples} samples, ${s.invalid} invalid)`;
}
function sonarFresh() {
  const so = (snap && snap.sonar) || {};
  const s = snap && snap.sensors && snap.sensors.sonar;
  return !!(so.available && !sensorStopped(s) && so.health !== "FAILED" && so.scan_valid && so.valid_rays > 0);
}
function sonarNoDataMsg() {
  if (!snap) return "No data – start the simulation";
  const so = snap.sonar || {};
  const s = snap.sensors && snap.sensors.sonar;
  if (sensorStopped(s)) return `sonar ${modeLabel(s)}: no new scans arrive, so the scan and the clearance chart are frozen`;
  if (!so.available) return "no sonar scan published yet";
  if (so.health === "FAILED") return "sonar FAILED: every ray is a stuck-at-zero output (0 hits), so no range is shown – a 0.0 m range would mean an obstacle on the transducer";
  if (!so.valid_rays) return `sonar: every one of the ${so.n_rays} rays is invalid (${so.invalid_rays} dropped/blanked) – no range is shown`;
  return "sonar: valid scan with no return inside max range (clear water, no obstacle)";
}

/* The five injectable states, in the order the operator sees them. Each one is applied to the RUNNING sensor at
   once; what the sensor model really does with it is described on the option itself. */
const FAULT_STATES = ["HEALTHY", "DEGRADED", "FAILED", "DISCARDED", "LINK_CUT"];
const FAULT_LABELS = { HEALTHY: "healthy (clear)", DEGRADED: "degraded (noisier, still used)",
                       FAILED: "failed → output 0.0", DISCARDED: "discarded → no output",
                       LINK_CUT: "link cut → no data" };

function buildFaultTable() {
  const t = $("faultTable");
  t.innerHTML = '<tr><th>sensor</th><th>live state change</th><th>live state</th><th>reading now</th></tr>' +
    SENSORS.map((s) => `<tr><td>${s}</td><td><select data-sensor="${s}" aria-label="${s} state" title="Send this state to the running simulation now: it is applied at once, no stop and no restart">
    ${FAULT_STATES.map((o) => `<option value="${o}">${FAULT_LABELS[o]}</option>`).join("")}
    <option value="RESET">reset → healthy</option></select></td>
    <td class="faultsrc" data-src="${s}">–</td><td class="faultval" data-val="${s}">–</td></tr>`).join("");
  t.querySelectorAll("select").forEach((sel) => {
    sel.onchange = async () => {
      const name = sel.dataset.sensor;
      const sent = sel.value === "RESET" ? "HEALTHY" : sel.value;
      pendingFault[name] = true;                                          // block re-sync while the request is in flight
      const r = await act(`Set ${name} to ${sent}`, "POST", `/api/sensors/${name}/health`, { health: sent });
      pendingFault[name] = false;
      syncFaultTable();                                                   // success or failure: fall back to backend truth
    };
  });
}

/* The three goal inputs are edited as a group: while any of them (or Set goal) holds focus - or shortly after
   an edit - the telemetry mirror must not overwrite a sibling field mid-edit. */
const GOAL_FIELDS = ["goalX", "goalY", "goalZ"];
GOAL_FIELDS.forEach((id) => {
  ["focus", "input", "keydown", "pointerdown"].forEach((ev) => $(id).addEventListener(ev, () => { goalEditUntil = Date.now() + 2000; }));
});
$("btnGoal").addEventListener("click", () => { goalEditUntil = Date.now() + 500; });
function goalBeingEdited() {
  const a = document.activeElement;
  if (a && (GOAL_FIELDS.includes(a.id) || a.id === "btnGoal")) return true;
  return Date.now() < goalEditUntil;
}

function syncFaultTable() {
  const sens = (snap && snap.sensors) || {};
  // Editable whenever a run is alive: a fault injected here is applied to the RUNNING robot immediately (the
  // selector is only a request form - the value shown is always re-read from the backend telemetry afterwards).
  const enabled = !!(status && ["STARTING", "RUNNING", "STALE"].includes(status.state));
  $("faultTable").querySelectorAll("select").forEach((sel) => {
    const name = sel.dataset.sensor, s = sens[name];
    const cell = $("faultTable").querySelector(`td[data-src="${name}"]`);
    const vcell = $("faultTable").querySelector(`td[data-val="${name}"]`);
    if (!s) {
      sel.disabled = true;
      const why = !snap ? "no telemetry yet"
        : (snap.sensors_enabled === false ? "sensors disabled in this run" : "not configured");
      if (cell) cell.textContent = why;
      if (vcell) { vcell.textContent = "no reading (" + why + ")"; vcell.className = "faultval v-none"; }
      return;
    }
    sel.disabled = !enabled || !!pendingFault[name];
    if (!pendingFault[name]) {
      const want = s.forced || s.health;                                  // a forced state wins; otherwise the measured health
      if (want && FAULT_STATES.includes(want) && sel.value !== want) sel.value = want;
    }
    if (cell) cell.textContent = `${s.health || "no data"} · ${s.forced ? "forced " + modeLabel(s) : "automatic"}`;
    if (vcell) {
      vcell.textContent = sensorReading(name, s);
      vcell.className = "faultval" + (s.health === "FAILED" ? " v-none" : (sensorStopped(s) ? " v-frozen" : ""));
    }
  });
}

async function loadEnvironment(syncForm) {
  const r = await api("GET", "/api/environment");
  if (!r.ok) return;
  const sel = $("selScenario");
  if (!sel.options.length) {
    r.data.scenarios.forEach((s) => { const o = document.createElement("option"); o.value = s.name; o.textContent = s.name; o.title = s.description; sel.appendChild(o); });
  }
  const p = r.data.pending_overrides || {};
  const env = p.environment || {};
  // The form shows what the BACKEND has stored for the next start (pending value, else the active run's value).
  if (syncForm) {
    const act = r.data.active || {};
    if (env.scenario || act.scenario) sel.value = env.scenario || act.scenario;
    if (env.seed !== undefined || act.seed !== undefined) $("inSeed").value = env.seed !== undefined ? env.seed : act.seed;
    // obstacle count: pending override first, else the count the RUNNING world was actually built with
    const pendCount = (env.obstacles && env.obstacles.count !== undefined) ? env.obstacles.count : null;
    $("inCount").value = pendCount !== null ? pendCount
      : (act.procedural && act.obstacles_requested !== undefined ? act.obstacles_requested : "");
    const dyn = (p.simulation && p.simulation.dynamics && p.simulation.dynamics.model);
    if (dyn) $("selDyn").value = dyn;
    if (p.mapping && p.mapping.enabled !== undefined) $("selMapStart").value = String(p.mapping.enabled);
    if (p.dashboard && p.dashboard.headless !== undefined) $("selShow3d").value = String(!p.dashboard.headless);
    if (p.dashboard && p.dashboard.auto_open_3d !== undefined) $("chkAuto3d").checked = !!p.dashboard.auto_open_3d;
    const pn = p.navigation || {};
    if (pn.enabled !== undefined) $("selNavStart").value = String(pn.enabled);
    // The stored goal/start are world xyz; the form shows the depth again (rounded), so what the operator sees
    // after saving is the number they typed, not its sign-flipped world z.
    if (Array.isArray(pn.goal_position)) { $("inGoalX").value = pn.goal_position[0]; $("inGoalY").value = pn.goal_position[1]; $("inGoalZ").value = f(toDepth(pn.goal_position[2]), 2); }
    const st = (p.simulation && p.simulation.start_position) || null;
    if (Array.isArray(st)) { $("inStartX").value = st[0]; $("inStartY").value = st[1]; $("inStartZ").value = f(toDepth(st[2]), 2); }
    if (p.planning && p.planning.unknown_policy) $("selPolicyStart").value = p.planning.unknown_policy;
  }
  syncObstacleCountField();
  // The raw override JSON carries world z (negative down); the readable tail spells the same thing back out as
  // the depth that was typed, so the stored value is not mistaken for a mismatch.
  const pendGoal = (p.navigation || {}).goal_position, pendStart = (p.simulation || {}).start_position;
  const human = [];
  if (Array.isArray(pendGoal)) human.push(`goal x ${f(pendGoal[0], 2)}, y ${f(pendGoal[1], 2)}, depth ${f(toDepth(pendGoal[2]), 2)} m`);
  if (Array.isArray(pendStart)) human.push(`start x ${f(pendStart[0], 2)}, y ${f(pendStart[1], 2)}, depth ${f(toDepth(pendStart[2]), 2)} m`);
  $("pendingNote").textContent = (Object.keys(p).length ? "Pending for next start: " + JSON.stringify(p) : "Nothing pending (next start uses configs/config.yaml).")
    + (human.length ? "  ·  " + human.join("  ·  ") : "");
}

/* The obstacle-count control only means something on a procedural scenario (legacy_week4 builds its world from the
   Week-4 occupancy map). Say so on the control instead of letting it silently do nothing. */
function syncObstacleCountField() {
  const legacy = $("selScenario").value === "legacy_week4";
  $("inCount").disabled = legacy;
  $("countNote").textContent = legacy
    ? "legacy_week4 is built from the Week-4 occupancy map: the obstacle count does not apply to it (the backend rejects it)."
    : "Exact number of PyBullet obstacle bodies the next run will generate (scenario-fixed structures count towards it).";
}
$("selScenario").onchange = syncObstacleCountField;

/* ---------------------------------------------------------------- polling */
async function poll() {
  const s = await api("GET", "/api/status");
  if (!s.ok) { showBanner(s.data.error, false); status = null; render(); return; }
  status = s.data;
  // only ask for telemetry once the backend says a snapshot exists (telemetry_age_s is null before that)
  if (HAS_TELEMETRY.has(status.state) && status.telemetry_age_s !== null) {
    const t = await api("GET", "/api/telemetry?optional=1");
    snap = t.ok ? t.data.snapshot : snap;
  } else if (!HAS_TELEMETRY.has(status.state) || status.state === "STARTING") { snap = null; }
  if (status.state === "IDLE") { resetClientSeries(); }
  if (snap && snap.sim_time < lastSimTime) { resetSeries(); }
  // Remember the last sim time actually SEEN. While snap is null (before the first snapshot of a run) this is kept,
  // not cleared: a second run started without going through IDLE begins again at sim time ~0, and that rollback is
  // what tells the buffers to start clean. Clearing it here instead would leave the previous run's timestamps in the
  // series, and push() (which only appends increasing sim times) would silently drop the whole new run.
  if (snap) lastSimTime = snap.sim_time;
  if (lastState !== status.state) {
    logEvent(`state: ${lastState || "(start)"} -> ${status.state}${status.error ? " | " + status.error.split("\n").slice(-1)[0] : ""}`, status.state === "FAILED");
    // Legacy Phase-4 mission (navigation off) has no navigator state change to hook, so the goal it reached is
    // reported here. Navigation runs already log it at the GOAL_REACHED transition, so this does not duplicate it.
    if (status.state === "COMPLETED" && !(((snap || {}).navigation) || {}).enabled)
      logEvent(`goal reached — the goal you set: ${goalCoordsText(reachedGoal())} (x, y, depth in m)`, false);
    const ended = ["STOPPED", "COMPLETED", "ENDED_BEFORE_GOAL", "FAILED", "IDLE"].includes(status.state);
    const startedFresh = ["RUNNING", "STARTING"].includes(status.state);
    lastState = status.state;
    if (ended || startedFresh) loadRuns();             // run history changes when runs start and end
  }
  if (snap) {
    (snap.command_log || []).forEach((c) => { const k = `${c.id}`; if (!seenCmd.has(k)) { seenCmd.add(k); logEvent(`sim ack #${c.id} ${c.cmd}: ${c.detail}`, !c.ok); } });
    const e = snap.debug_ground_truth && snap.debug_ground_truth.estimate_error_norm;
    if (e !== null && e !== undefined) push(errSeries, snap.sim_time, e);
    const cov = snap.mapping && snap.mapping.coverage_percent;
    if (cov !== null && cov !== undefined) push(covSeries, snap.sim_time, cov);
    // every series is read from the same live snapshot (EKF estimate / navigator / sonar / sensors / thrusters);
    // nothing is synthesised and nothing is pushed while a value is unavailable.
    const t = snap.sim_time, est = snap.estimate || {}, nv = snap.navigation || {}, so = snap.sonar || {};
    const sens = snap.sensors || {};
    if (est.available) {
      push(S.speed, t, est.speed); push(S.depth, t, est.depth);
      const eu = est.euler_rad;
      if (Array.isArray(eu) && eu.length === 3) {
        push(S.yaw, t, deg(eu[2])); push(S.roll, t, deg(eu[0])); push(S.pitch, t, deg(eu[1]));
      }
      const p = est.position;
      if (Array.isArray(p) && p.length === 3) { push(S.estx, t, p[0]); push(S.esty, t, p[1]); push(S.estz, t, p[2]); }
      const vw = est.velocity_world;
      if (Array.isArray(vw) && vw.length === 3) { push(S.velx, t, vw[0]); push(S.vely, t, vw[1]); push(S.velz, t, vw[2]); }
      const w = est.angular_velocity_body;
      if (Array.isArray(w) && w.length === 3) { push(S.angx, t, w[0]); push(S.angy, t, w[1]); push(S.angz, t, w[2]); }
      updateMission(snap);
    }
    // Raw IMU accelerometer reading: the sensor's CURRENT sample, i.e. the sample it really output. A FAILED IMU
    // outputs 0.0, so the chart gets a flat 0.0 line; DISCARDED / LINK_CUT (and ordinary dropouts) deliver no
    // sample, so sensorFresh() is false and the series stops where the real data ended.
    const imu = sens.imu;
    if (imu && sensorFresh("imu") && Array.isArray(imu.value) && imu.value.length >= 3 &&
        imu.value_time !== null && imu.value_time !== undefined) {
      push(S.imuax, imu.value_time, imu.value[0]);
      push(S.imuay, imu.value_time, imu.value[1]);
      push(S.imuaz, imu.value_time, imu.value[2]);
    }
    // sensor state over time (0/1/2/3/4 = HEALTHY/DEGRADED/FAILED/DISCARDED/LINK CUT; unknown states not plotted)
    const HL = { imu: S.hlImu, dvl: S.hlDvl, depth: S.hlDepth, sonar: S.hlSonar, compass: S.hlCompass };
    SENSORS.forEach((n) => { const s = sens[n]; if (s && s.health && HEALTH_NUM[s.health] !== undefined) push(HL[n], t, HEALTH_NUM[s.health]); });
    trackSensorEvents(snap);
    // actual thruster forces (one line per thruster; unavailable on legacy/kinematic dynamics -> chart says so)
    const th = snap.thrusters || {};
    if (th.available && Array.isArray(th.force_n)) {
      thrustCount = th.force_n.length;
      th.force_n.forEach((fv, i) => { if (!S.thr[i]) S.thr[i] = []; push(S.thr[i], t, fv); });
    } else thrustCount = 0;
    // Adaptive P controller: the EFFECTIVE gains of the last command, straight from the controller telemetry block
    // (only present while the adaptive controller is the one driving, i.e. the Phase-4 waypoint mission).
    const ad = (snap.controller || {}).adaptive;
    if (ad && ad.adaptive) {
      if (ad.schedule_multiplier != null) push(S.kpsched, t, ad.schedule_multiplier);
      if (ad.self_tuning_gain != null) push(S.ktune, t, ad.self_tuning_gain);
      if (ad.kp_position_multiplier != null) push(S.kpmult, t, ad.kp_position_multiplier);
      if (ad.kp_yaw_multiplier != null) push(S.kyaw, t, ad.kp_yaw_multiplier);
      // The tracked gain history the controller keeps for itself: cumulative simulated time per adaptation mode.
      const tr = ad.tracking;
      if (tr) {
        if (tr.ramp_s != null) push(S.kramp, t, tr.ramp_s);
        if (tr.boosted_s != null) push(S.kboost, t, tr.boosted_s);
        if (tr.at_cap_s != null) push(S.kcap, t, tr.at_cap_s);
        trackAdaptEvents(ad, tr, t);
      }
    }
    // Phase-5 path follower: the gains that drive the vehicle when autonomous navigation is on (it publishes its own
    // loop gains and the live speed schedule, so the panel and the gain charts follow whichever controller is
    // really producing the commands - they are never empty for a run that is driving).
    const fol = (snap.controller || {}).follower;
    if (fol && fol.available) {
      if (fol.speed_factor != null) push(S.fspeed, t, fol.speed_factor);
      if (fol.goal_factor != null) push(S.fgoal, t, fol.goal_factor);
      if (fol.heading_factor != null) push(S.fhead, t, fol.heading_factor);
      if (fol.sonar_factor != null) push(S.fsonar, t, fol.sonar_factor);
      const ft = fol.tracking;
      if (ft) {
        if (ft.full_speed_s != null) push(S.ffull, t, ft.full_speed_s);
        if (ft.slowed_s != null) push(S.fslow, t, ft.slowed_s);
        if (ft.braked_s != null) push(S.fbrake, t, ft.braked_s);
        trackFollowEvents(fol, ft, t);
      }
    }
    if (nv.enabled && nv.goal_distance_m !== null && nv.goal_distance_m !== undefined) push(S.goal, t, nv.goal_distance_m);
    if (so.available && so.min_valid_range_m !== null && so.min_valid_range_m !== undefined) push(S.clr, t, so.min_valid_range_m);
    if (nv.enabled) {
      const rc = (nv.replanning || {}).count ?? 0, ac = (nv.avoidance || {}).count ?? 0;
      push(S.rpl, t, rc + ac);
      if (lastNav) {
        if (nv.state !== lastNav.state) {
          navEvents.push({ t, label: nv.state, color: nv.state === "GOAL_REACHED" ? "#2f855a" : nv.state === "FAILED" ? "#c53030" : "#6b46c1" });
          logEvent(`navigation: ${lastNav.state} → ${nv.state}${nv.failure_reason ? " (" + nv.failure_reason + ")" : ""}`, nv.state === "FAILED", t);
          if (nv.state === "GOAL_REACHED")                                    // the goal the operator gave, echoed back
            logEvent(`goal reached — the goal you set: ${goalCoordsText(reachedGoal())} (x, y, depth in m)`, false, t);
        }
        if (rc > lastNav.rc) {
          const why = (nv.replanning || {}).last_reason || "";
          navEvents.push({ t, label: "replan" + (why ? ": " + why : ""), color: "#b7791f" });
          logEvent(`navigation replan #${rc}${why ? ": " + why : ""}`, false, t);
        }
        if (ac > lastNav.ac) { navEvents.push({ t, label: "avoid", color: "#c05621" }); logEvent(`avoidance maneuver #${ac}`, false, t); }
        if (navEvents.length > 120) navEvents.splice(0, navEvents.length - 120);
      }
      lastNav = { state: nv.state, rc, ac };
    }
  }
  render();
}
function push(arr, t, v) { if (!arr.length || t > arr[arr.length - 1][0]) { arr.push([t, v]); if (arr.length > 600) arr.shift(); } }

/* Adaptive gain episodes the CONTROLLER reported (its own tracked counters, never inferred from the plotted line):
   a new self-tuning episode starting to ramp, the gain reaching its cap, and the gain relaxing back down. */
function trackAdaptEvents(ad, tr, t) {
  if (lastAdapt) {
    if (tr.episodes > lastAdapt.episodes) {
      adaptEvents.push({ t, label: "ramp ↑", color: "#2f855a" });
      logEvent(`adaptive gain: self-tuning episode ${tr.episodes} — ${tr.reason || "ramping up"} ` +
               `(position error ${f(ad.position_error_m, 2)} m)`, false, t);
    }
    if (tr.at_cap_s > 0 && !(lastAdapt.at_cap_s > 0)) {
      adaptEvents.push({ t, label: "gain cap", color: "#c53030" });
      logEvent(`adaptive gain: self-tuning reached the gain cap (${f(tr.peak_self_tuning)} ×)`, false, t);
    }
    if (tr.action === "relaxing" && lastAdapt.action === "ramping") {
      adaptEvents.push({ t, label: "relax ↓", color: "#2b6cb0" });
      logEvent(`adaptive gain: relaxing back from ${f(lastAdapt.gain)} × — ${tr.reason}`, false, t);
    }
    if (adaptEvents.length > 120) adaptEvents.splice(0, adaptEvents.length - 120);
  }
  lastAdapt = { episodes: tr.episodes, at_cap_s: tr.at_cap_s, action: tr.action, gain: ad.kp_position_multiplier };
}

/* Path-follower schedule episodes, from the FOLLOWER's own counters (never inferred from the plotted line): the
   speed being scheduled down below cruise, and the speed being commanded to zero (braking / holding). */
function trackFollowEvents(fol, ft, t) {
  if (lastFollow) {
    if (ft.slows > lastFollow.slows) {
      adaptEvents.push({ t, label: "slow ↓", color: "#b7791f" });
      logEvent(`path follower: speed scheduled down to ${f(100 * fol.speed_factor, 0)}% of cruise (${fol.mode})`, false, t);
    }
    if (ft.brakes > lastFollow.brakes) {
      adaptEvents.push({ t, label: "brake", color: "#c53030" });
      logEvent(`path follower: speed commanded to zero (${fol.mode})`, false, t);
    }
  }
  if (adaptEvents.length > 120) adaptEvents.splice(0, adaptEvents.length - 120);
  lastFollow = { slows: ft.slows, brakes: ft.brakes };
}

/* Which controller is driving, and the gain block it publishes: the legacy Phase-4 mission uses the adaptive P
   controller, autonomous navigation uses the Phase-5 path follower. The gain panel and both gain charts follow this. */
function gainSource() {
  const ctl = (snap && snap.controller) || {};
  if (ctl.adaptive) return { kind: "adaptive", block: ctl.adaptive };
  if (ctl.follower && ctl.follower.available) return { kind: "follower", block: ctl.follower };
  return null;
}

/* Sensor health transitions observed in the telemetry (both forced and automatic) become timestamped events. */
function trackSensorEvents(s) {
  const sens = s.sensors || {}, next = {};
  SENSORS.forEach((n) => { const e = sens[n]; if (e && e.health) next[n] = { h: e.health, f: e.forced }; });
  Object.keys(next).forEach((n) => {
    const prev = lastHealth[n];
    if (prev && prev.h !== next[n].h) {
      const src = next[n].f ? `forced ${String(next[n].f).toLowerCase()}` : "automatic";
      logEvent(`sensor ${n}: ${String(prev.h).toLowerCase()} → ${String(next[n].h).toLowerCase()} (${src})`,
               next[n].h === "FAILED", s.sim_time);
    }
  });
  lastHealth = next;
}

/* Estimated path length of the current run (EKF positions sampled at the poll rate; label says estimate). */
function updateMission(s) {
  const e = s && s.estimate;
  if (e && e.available && Array.isArray(e.position) && e.position.length === 3) {
    if (missionStats.lastP) {
      const d = Math.hypot(e.position[0] - missionStats.lastP[0], e.position[1] - missionStats.lastP[1],
                           e.position[2] - missionStats.lastP[2]);
      if (d < 50) missionStats.dist += d;                       // reject jumps (reset / estimate restart)
    }
    missionStats.lastP = e.position.slice();
  }
}

async function pollMaps() {
  if (!status || !HAS_TELEMETRY.has(status.state) || !snap || !snap.mapping_available) { return; }
  const o = await api("GET", `/api/occupancy?optional=1&since=${occVersion}`);
  if (o.ok && o.data.available === false) { occ = null; occVersion = -1; }
  else if (o.ok && !o.data.unchanged) { occ = o.data; occVersion = o.data.version; }
  const c = await api("GET", `/api/costmap?optional=1&since=${costVersion}`);
  if (c.ok && c.data.available === false) { cost = null; costVersion = -1; }
  else if (c.ok && !c.data.unchanged) { cost = c.data; costVersion = c.data.version; }
  drawMaps();
}

/* ---------------------------------------------------------------- render */
/* showBanner(text, kind): false = alert (red), true = warning (amber), "info" = neutral status (blue). */
function showBanner(text, kind) {
  const b = $("banner"); if (!text) { b.hidden = true; return; }
  b.hidden = false; b.textContent = text;
  b.className = "banner" + (kind === true ? " warn" : kind === "info" ? " info" : "");
}
function rows(el, pairs) { $(el).innerHTML = pairs.map(([k, v]) => `<tr><td>${k}</td><td>${v}</td></tr>`).join(""); }

/* ---------------------------------------------------------------- the goal the operator set
The goal that was ENTERED is shown back to the operator once the robot reaches it. The navigator's CURRENT goal is
authoritative (it follows every "Set goal" and every in-window goal-key change); without navigation the run targets
the mission goal it was started with. Nothing here is derived from the true state. */
function reachedGoal() {
  const nv = (snap && snap.navigation) || {}, ms = (snap && snap.mission) || {};
  const g = (nv.enabled && Array.isArray(nv.goal)) ? nv.goal : (Array.isArray(ms.goal) ? ms.goal : null);
  return (Array.isArray(g) && g.length === 3) ? g : null;
}
/* ------------------------------------------------------------------ depth <-> world z
   z is DEPTH. The operator types and reads depth in metres BELOW the surface (positive down) - the same unit the
   Estimated navigation panel reports - while the simulation works in world z, which is up. These two one-liners
   are the only place in the UI where the two conventions meet, so the goal typed in the form and the goal read
   back from telemetry can never be printed in different units. toWorldZ takes the magnitude, so a value that is
   already a world z (the older "-7" form) selects the very same point. */
const toDepth = (z) => ((z === null || z === undefined || z === "") ? null : -Number(z));
const toWorldZ = (depth) => ((depth === null || depth === undefined || depth === "") ? null : -Math.abs(Number(depth)));

/* "x, y, depth d" - the ONE way a goal is written in the UI, so every echo of it (banner, event log, tables,
   map label) shows the depth the operator entered instead of a sign-flipped world z. */
function goalCoordsText(g, dec) {
  const d = dec === undefined ? 2 : dec;
  if (!Array.isArray(g) || g.length !== 3) return DASH;
  return `${f(g[0], d)}, ${f(g[1], d)}, depth ${f(toDepth(g[2]), d)}`;
}
/* The goal the operator TYPED for the next start (the "Navigation goal x, y, depth" fields). Shown next to the goal
that the run actually used so the two can never be confused: a run that was already going when the goal was typed
keeps the goal it started with until it is restarted. Returned in WORLD xyz (the typed depth converted once) so it
is directly comparable with the goal the run uses. Null when the fields are blank (= the run's configured goal). */
function enteredGoal() {
  const v = ["inGoalX", "inGoalY", "inGoalZ"].map((id) => { const el = $(id); return el ? el.value.trim() : ""; });
  if (v.every((x) => x === "")) return null;
  const g = v.map((x) => (x === "" ? null : Number(x)));
  if (!g.every((x) => x !== null && Number.isFinite(x))) return null;
  return [g[0], g[1], toWorldZ(g[2])];
}
/* The run has reached the goal: the navigator says GOAL_REACHED (while the run lingers), or the run status is
COMPLETED (which also covers the Phase-4 waypoint mission, where navigation is off). */
function goalReachedNow() {
  if (status && status.state === "COMPLETED") return true;
  const nv = (snap && snap.navigation) || {};
  return !!(nv.enabled && nv.state === "GOAL_REACHED");
}
function pill(h) { const c = h ? "h-" + String(h).toLowerCase() : "h-none"; return `<span class="pill ${c}">${h ? String(h).toLowerCase() : "no data"}</span>`; }

function render() {
  const st = status ? status.state : "UNREACHABLE";
  const el = $("state"); el.textContent = st; el.className = "state s-" + st.toLowerCase();
  const live = status && LIVE.has(st);
  // A run that finished on its own may still be holding its own PyBullet window open on the last frame
  // (visualization.keep_open_after_run): the run slot is still taken until the window is released.
  const holding = !!(status && status.holding);
  $("btnStart").disabled = !status || live || holding;
  $("btnStop").disabled = !(live || holding) || st === "STOPPING";
  $("btnReset").disabled = !status;
  ["btnMapOn", "btnMapOff", "btnMapReset", "selSpeed"].forEach((id) => { $(id).disabled = !(status && (st === "RUNNING" || st === "STALE")); });
  const navLive = !!(snap && snap.navigation && snap.navigation.enabled) && (st === "RUNNING" || st === "STALE");
  ["btnGoal", "btnNavStart", "btnNavReplan", "btnNavCancel", "selPolicyLive", "goalX", "goalY", "goalZ"].forEach((id) => { $(id).disabled = !navLive; });
  $("btnExport").disabled = !snap;                                       // CSV export needs recorded telemetry
  syncFaultTable();
  syncWindowControl();                                                  // the real state of the 3-D PyBullet window
  if (!status) { showBanner("Backend unreachable: nothing below is live.", false); return; }
  if (st === "FAILED") showBanner("Simulation failed: " + (status.error || "unknown error"), false);
  else if (st === "STALE") showBanner(`Telemetry stalled: no snapshot for ${f(status.telemetry_age_s, 1)} s (timeout ${f(status.telemetry_timeout_s, 1)} s). Values below are the last received, not live.`, true);
  else if (st === "ENDED_BEFORE_GOAL") showBanner("Run ended before reaching the goal (step limit).", true);
  else if (st === "IDLE") showBanner("Idle: no simulation is running. Press Start to load live telemetry.", "info");
  else if (st === "STARTING") showBanner("Loading: starting the simulation process; first telemetry arrives in a moment.", "info");
  else if (st === "STOPPED") showBanner("Run stopped by the operator; the panels show the final snapshot of this run.", "info");
  else if (goalReachedNow()) showBanner(`Goal reached${st === "COMPLETED" ? " (run completed)" : ""}: the goal you set was ${goalCoordsText(reachedGoal())} (x, y, depth in m).`, "info");
  else showBanner(null);
  $("tAge").textContent = status.telemetry_age_s === null ? DASH : f(status.telemetry_age_s, 1) + " s";
  if (!snap) { clearPanels(); return; }

  $("simTime").textContent = f(snap.sim_time, 1) + " s";
  $("dynMode").textContent = snap.dynamics_mode || DASH;
  const perf = snap.performance || {};
  $("pace").textContent = perf.sim_seconds_per_wall_second != null ? f(perf.sim_seconds_per_wall_second, 1) + "× real time" : DASH;
  $("scenario").textContent = snap.environment ? `${snap.environment.scenario} (seed ${snap.environment.seed})` : DASH;
  if (perf.realtime_factor_setting !== undefined) { const v = String(perf.realtime_factor_setting); if ([...$("selSpeed").options].some(o => o.value === v) && document.activeElement !== $("selSpeed")) $("selSpeed").value = v; }

  const e = snap.estimate || {};
  if (e.available) {
    const eu = (e.euler_rad || []).map(deg);
    // The goal the RUN is steering to, in the very same units as the estimate above/below it (x, y and depth), so
    // "did the robot get to the goal I gave?" is answered by reading one panel instead of comparing a form with a
    // table. The goal is the navigator's authoritative goal when navigation is on (it follows every Set goal),
    // otherwise the mission goal the run was started with; the remaining row is what is left of it, from the
    // estimate - the difference is printed only when both sides are real numbers.
    const goal = reachedGoal();
    const goalRows = goal ? [["goal this run x, y (m)", `${f(goal[0])} / ${f(goal[1])}`], ["goal this run depth (m)", f(toDepth(goal[2]))],
      ["goal − estimate x, y, depth (m)", `${f(goal[0] - e.position[0])} / ${f(goal[1] - e.position[1])} / ${f(toDepth(goal[2]) - e.depth)}`]] : [];
    rows("navTable", [["x / y (m)", `${f(e.position[0])} / ${f(e.position[1])}`], ["depth (m)", f(e.depth)], ...goalRows,
      ["roll / pitch / yaw (°)", eu.map((v) => f(v, 1)).join(" / ")], ["velocity, world (m/s)", e.velocity_world.map((v) => f(v)).join(" / ")],
      ["speed (m/s)", f(e.speed)], ["position 1σ x/y/z (m)", e.position_std.map((v) => f(v, 3)).join(" / ")]]);
    const s = e.stats || {}; const ages = e.last_aiding_age_s || {};
    rows("ekfTable", [["filter status", e.filter_status], ["predictions", s.predictions], ["DVL updates / rejected(gate)", `${s.dvl_updates} / ${s.rejected_gate}`],
      ["depth updates", s.depth_updates], ["IMU missing (coasted)", s.imu_missing], ["rejected invalid / stale", `${s.rejected_invalid} / ${s.rejected_stale}`],
      ["last DVL / depth aiding (s ago)", `${f(ages.dvl, 1)} / ${f(ages.depth, 1)}`], ["covariance trace", f(e.covariance_trace, 4)]]);
  } else { rows("navTable", [["estimate", e.reason || "unavailable"]]); rows("ekfTable", []); }
  // Adaptive P controller: the proportional gains the vehicle is ACTUALLY using (nominal vs effective), plus the
  // two multipliers that produced them. Rendered only when the run really reports the block, so a navigation run
  // says so instead of showing stale numbers.
  const ctl = snap.controller || {};
  const src = gainSource();
  const ad = src && src.kind === "adaptive" ? src.block : null;
  const fol = src && src.kind === "follower" ? src.block : null;
  if (fol) {
    // Autonomous navigation: the PATH FOLLOWER drives, so its own gains are shown - the loop gains in the units they
    // are applied in and the speed schedule that shapes the command - instead of saying nothing is driving.
    const ft = fol.tracking || {};
    rows("adaptTable", [
      ["controller", ctl.type || DASH],
      ["gains driving this run", `Phase-5 path follower (${fol.mode} mode, ${fol.command_frame} frame)`],
      ["kv velocity loop (N per m/s)", f(fol.kv_n_per_mps, 1)],
      ["kp_yaw / kd_yaw (N·m per rad, N·m per rad/s)", `${f(fol.kp_yaw_nm_per_rad)} / ${f(fol.kd_yaw_nm_per_rad_s)}`],
      ["kp_depth / ki_depth (m/s per m, N per m·s)", `${f(fol.kp_depth_mps_per_m)} / ${f(fol.ki_depth_n_per_m_s, 3)}`],
      ["station-kp (m/s per m)", f(fol.station_kp_per_s)],
      ["speed schedule now (× cruise)", `${f(fol.speed_factor)} ×`],
      ["desired speed (m/s)", f(fol.desired_speed_mps, 3)],
      ["lookahead (m)", f(fol.lookahead_m)],
      ["cross-track (m)", f(fol.cross_track_m, 3)],
      ["heading error (°)", f(deg(fol.heading_error_rad), 1)],
      ["depth error (m)", f(fol.depth_error_m, 3)],
      ["depth hold integral (N)", f(fol.depth_integral_n, 3)]]);
    rows("adaptTrackTable", [
      ["schedule factors (×): goal / heading / sonar+avoidance", `${f(fol.goal_factor)} / ${f(fol.heading_factor)} / ${f(fol.sonar_factor)}`],
      ["commands / simulated time", `${f(ft.steps, 0)} / ${f(ft.elapsed_s, 1)} s`],
      ["time following / holding / station-keeping (s)", `${f(ft.follow_s, 1)} / ${f(ft.hold_s, 1)} / ${f(ft.station_s, 1)}`],
      ["time at full speed / slowed / braked (s)", `${f(ft.full_speed_s, 1)} / ${f(ft.slowed_s, 1)} / ${f(ft.braked_s, 1)}`],
      ["speed schedule now / mean / slowest (× cruise)", `${f(ft.speed_factor_now)} / ${f(ft.speed_factor_mean)} / ${f(ft.speed_factor_min)}`],
      ["slowest heading / sonar factor (×)", `${f(ft.heading_factor_min)} / ${f(ft.sonar_factor_min)}`],
      ["slow-down / brake episodes", `${f(ft.slows, 0)} / ${f(ft.brakes, 0)}`]]);
  } else if (ad) {
    rows("adaptTable", [
      ["controller", ctl.type || DASH],
      ["adaptive", String(ad.adaptive)],
      ["kp_position nominal → effective (N/m)", `${f(ad.kp_position_nominal)} → ${f(ad.kp_position_effective)}`],
      ["gain multiplier (active, ×)", `${f(ad.kp_position_multiplier)} ×`],
      ["gain schedule (×)", `${f(ad.schedule_multiplier)} ×`],
      ["self-tuning gain (×)", `${f(ad.self_tuning_gain)} ×`],
      ["kp_yaw nominal → effective", `${f(ad.kp_yaw_nominal)} → ${f(ad.kp_yaw_effective)}`],
      ["kp_yaw multiplier (×)", `${f(ad.kp_yaw_multiplier, 3)} ×`],
      ["position error (m)", f(ad.position_error_m, 3)],
      ["heading error (°)", f(deg(ad.heading_error_rad), 1)],
      ["closing speed (m/s)", f(ad.closing_speed_mps, 3)]]);
    // The TRACKED gain history: the controller's own cumulative record of the adaptation so far (it counts these
    // step by step, so the panel never has to infer them from what it happens to have polled).
    const tr = ad.tracking || {};
    rows("adaptTrackTable", [
      ["adaptation on the last step", tr.action === undefined ? DASH : `${tr.action} — ${tr.reason || ""}`],
      ["adaptation steps / simulated time", `${f(tr.steps, 0)} / ${f(tr.elapsed_s, 1)} s`],
      ["time ramping / relaxing / holding (s)", `${f(tr.ramp_s, 1)} / ${f(tr.relax_s, 1)} / ${f(tr.hold_s, 1)}`],
      ["time boosted / at gain cap (s)", `${f(tr.boosted_s, 1)} / ${f(tr.at_cap_s, 1)}`],
      ["gain now / mean / peak (×)", `${f(ad.kp_position_multiplier)} / ${f(tr.mean_active)} / ${f(tr.peak_active)}`],
      ["self-tuning now / peak (×)", `${f(ad.self_tuning_gain)} / ${f(tr.peak_self_tuning)}`],
      ["peak gain reached at (s of adaptation)", tr.peak_time_s === null || tr.peak_time_s === undefined ? DASH : f(tr.peak_time_s, 1)],
      ["adaptation episodes / re-seeds", `${f(tr.episodes, 0)} / ${f(tr.reseeds, 0)}`]]);
  } else {
    rows("adaptTable", [["gains driving this run", ctl.type ? `none yet — ${ctl.type}` : "none yet (no controller reported gains)"]]);
    rows("adaptTrackTable", [["gain tracking", ctl.type ? `no gains reported by ${ctl.type} yet` : "no gains reported yet"]]);
  }
  // The two gain charts describe whichever controller is driving, so their captions say which one that is.
  $("adaptChartNote").textContent = fol ? "path follower: commanded speed and the factors that shape it"
    : "adaptive P controller: gain schedule, self-tuning, active and yaw gain";
  $("adaptTrackNote").textContent = fol ? "path follower: simulated time at full speed, slowed and braked"
    : "adaptive P controller: ramping, boosted and at the gain cap";
  const gt = snap.debug_ground_truth;
  if (gt && e.available) rows("truthTable", [["true x / y / z (m)", gt.position.map((v) => f(v)).join(" / ")], ["estimate error x/y/z (m)", (gt.estimate_error || []).map((v) => f(v)).join(" / ")], ["error norm (m)", f(gt.estimate_error_norm)]]);
  else rows("truthTable", []);

  const sens = snap.sensors || {};
  // "live reading" is the sensor's CURRENT sample from this snapshot, shown according to the operator state:
  // 0.0 while failed (stuck at zero), frozen while discarded / link cut, and blank while every sensor is stopped.
  const allStopped = sensorsAllStopped();
  $("sensorTable").innerHTML = "<tr><th>sensor</th><th>state</th><th>source</th><th>live reading</th><th>reading age (s)</th><th>valid</th><th>invalid</th><th>last sample (s)</th></tr>" +
    SENSORS.map((n) => { const s = sens[n];
      if (!s) return `<tr><td>${n}</td><td>${pill(null)}</td><td colspan="6">not configured</td></tr>`;
      const src = s.forced ? `forced (${modeLabel(s)})` : "automatic";
      const cls = s.health === "FAILED" ? " class=\"v-none\"" : (sensorStopped(s) ? " class=\"v-frozen\"" : "");
      return `<tr><td>${n}</td><td>${pill(s.health)}</td><td>${src}</td>` +
        `<td${cls}>${sensorReading(n, s)}</td><td>${allStopped ? DASH : f(s.value_age_s, 2)}</td>` +
        `<td>${s.valid}</td><td>${s.invalid}</td><td>${f(s.age_s, 2)}</td></tr>`; }).join("");

  const so = snap.sonar || {};
  if (so.available) {
    $("sonarMeta").textContent = `t=${f(so.timestamp, 1)} s, ${so.n_rays} rays`;
    rows("sonarTable", [["scan health / valid", `${so.health} / ${so.scan_valid}`], ["valid / invalid rays", `${so.valid_rays} / ${so.invalid_rays}`], ["hits / no-return", `${so.hit_rays} / ${so.no_return_rays}`],
      ["min / max hit range (m)", `${f(so.min_valid_range_m)} / ${f(so.max_valid_range_m)}`]]);
  } else { $("sonarMeta").textContent = DASH; rows("sonarTable", [["sonar", "no scan yet / unavailable"]]); }
  drawSonar(so);

  const m = snap.mapping || {}, cm = snap.cost_map || {}, sm = snap.sonar_mapping || {};
  if (snap.mapping_available) {
    rows("mapTable", [["enabled", String(m.enabled)], ["map updates", m.update_count], ["known / unknown cells", `${m.known_cells} / ${m.unknown_cells}`], ["free / occupied cells", `${m.free_cells} / ${m.occupied_cells}`],
      ["coverage (%)", f(m.coverage_percent, 2)], ["map position x / y (m)", `${f(m.estimated_map_x)} / ${f(m.estimated_map_y)}`],
      ["valid hits / no-returns (total)", `${sm.valid_returns} / ${sm.no_returns}`], ["invalid rays (total)", sm.dropouts], ["scans rejected", m.scans_rejected + (m.last_reject_reason ? ` (${m.last_reject_reason})` : "")],
      ["resolution (m) / size", `${m.map_resolution} / ${(m.map_size || []).join("×")}`]]);
    rows("costTable", [["traversable cells", cm.traversable_count], ["occupied cells", cm.occupied_count], ["inflated cells", cm.inflated_count], ["high-cost cells", cm.high_cost_count],
      ["unknown cells", cm.unknown_count], ["inflation radius (m)", f(cm.inflation_radius_m, 2)], ["unknown policy", cm.unknown_policy], ["cost min / max", `${f(cm.min_cost, 1)} / ${f(cm.max_cost, 1)}`]]);
    $("occMeta").textContent = `${m.update_count} updates`; $("costMeta").textContent = `${cm.inflated_count} inflated cells`;
  } else { rows("mapTable", [["mapping", (m && m.unavailable_reason) || "unavailable"]]); rows("costTable", []); }
  const en = snap.environment || {};
  const kinds = en.obstacle_kinds ? Object.entries(en.obstacle_kinds).map(([k, v]) => `${k}×${v}`).join(", ") : DASH;
  const ms = snap.mission || {};
  const fmt3 = (a) => Array.isArray(a) ? a.map((v) => f(v, 1)).join(", ") : DASH;
  rows("envTable", [["scenario / seed", `${en.scenario} / ${en.seed}`], ["terrain (seabed type)", en.terrain_type], ["seabed z range (m)", `${f(en.seabed_z_min, 1)} to ${f(en.seabed_z_max, 1)} (variation ${f(en.seabed_z_max - en.seabed_z_min, 1)} m)`],
    ["start / goal (m), loaded config", `${fmt3(ms.start)} → ${fmt3(ms.goal)}`],
    ["unknown-cell policy (planning)", (snap.cost_map && snap.cost_map.unknown_policy) || ((snap.navigation && snap.navigation.planner && snap.navigation.planner.unknown_policy) || DASH)],
    ["PyBullet bodies built (obstacles / structures)", `${en.obstacle_count} / ${en.structure_count ?? DASH}`],
    ["obstacle count requested / actual", `${en.procedural ? (en.obstacles_requested ?? DASH) : "n/a (Week-4 map)"} / ${en.obstacle_count ?? DASH}`],
    ["fixed / randomly placed bodies", `${en.procedural ? (en.obstacles_fixed ?? DASH) : "n/a (Week-4 map)"} / ${en.procedural ? (en.obstacles_placed ?? DASH) : "n/a (Week-4 map)"}`],
    ["obstacle kinds", kinds], ["obstacles mapped (clusters)", m.mapped_obstacle_clusters ?? DASH]]);
  renderThrusters(snap.thrusters);
  renderNavigation(snap.navigation, st);
  renderMission(st);
  drawMaps();
}
const NAV_CLASS = { IDLE: "s-idle", PLAN: "s-starting", FOLLOW_PATH: "s-running", AVOID_OBSTACLE: "s-stale", REPLAN: "s-starting", GOAL_REACHED: "s-completed", FAILED: "s-failed" };
function renderNavigation(n, st) {
  const badge = $("navState");
  if (!n || !n.enabled) {
    badge.textContent = "DISABLED"; badge.className = "state s-idle";
    $("navNote").textContent = "Autonomous navigation is off in this run (Phase 4 fixed waypoint mission). Set 'Autonomous navigation: on' under Next start.";
    ["plannerTable", "followTable", "avoidTable", "replanTable"].forEach((id) => $(id).innerHTML = ""); $("navHistory").innerHTML = ""; return;
  }
  badge.textContent = n.state; badge.className = "state " + (NAV_CLASS[n.state] || "s-idle");
  $("navNote").textContent = n.failure_reason ? "FAILED: " + n.failure_reason : `pose source: ${n.pose_source}; goal ${n.goal ? goalCoordsText(n.goal, 1) : DASH}; goal distance ${f(n.goal_distance_m)} m (tolerance ${f(n.goal_tolerance_m, 2)} m, from the EKF estimate)`;
  // The goal inputs mirror the AUTHORITATIVE goal (navigator.goal) so the operator always edits the live value.
  // Never while the goal group is being edited: the mirror previously raced the poll and reverted a sibling
  // field's typed value before it was sent.
  if (Array.isArray(n.goal) && !goalBeingEdited()) {
    ["goalX", "goalY", "goalZ"].forEach((id, i) => {
      const el = $(id);
      // the first two are world x/y; the third is a DEPTH, which is what these fields hold (see toDepth)
      const v = String(f(i === 2 ? toDepth(n.goal[i]) : n.goal[i], 1));
      if (el.value !== v) el.value = v;                          // mirror the authoritative goal when not being edited
    });
  }
  const ms = snap.mission || {};
  if (Array.isArray(ms.start)) {                             // show the start the run really uses while it is empty
    ["inStartX", "inStartY", "inStartZ"].forEach((id, i) => {
      const el = $(id);
      const v = String(f(i === 2 ? toDepth(ms.start[i]) : ms.start[i], 1));
      if (el.value === "" && document.activeElement !== el) el.value = v;
    });
  }
  const pl = n.planner || {}, fo = n.following || {}, av = n.avoidance || {}, rp = n.replanning || {};
  rows("plannerTable", [["status", pl.status], ["detail", pl.detail || DASH], ["path length (m) / raw", `${f(pl.path_length_m)} / ${f(pl.raw_path_length_m)}`],
    ["vertices (smoothed: " + (pl.smoothed === null ? DASH : pl.smoothed) + ")", pl.path_vertices ?? DASH], ["collision-free (independent check)", pl.collision_free === null ? DASH : String(pl.collision_free)],
    ["planning time last / mean (ms)", `${f(pl.last_planning_time_ms, 1)} / ${f(pl.mean_planning_time_ms, 1)}`], ["A* expansions", pl.expansions ?? DASH],
    ["plans ok / failed", `${pl.plan_successes} / ${pl.plan_failures}`], ["unknown policy / connectivity", `${pl.unknown_policy} / ${pl.connectivity}`], ["planned on map version", pl.planned_on_map_version ?? DASH]]);
  rows("followTable", [["following", fo.path_kind || DASH], ["current waypoint (x, y, z)", fo.current_waypoint ? fo.current_waypoint.map((v) => f(v, 1)).join(", ") : DASH],
    ["waypoint index / count", `${fo.waypoint_index ?? DASH} / ${fo.waypoint_count ?? DASH}`], ["progress / remaining (m)", `${f(fo.progress_m)} / ${f(fo.remaining_m)}`],
    ["cross-track (m)", f(fo.cross_track_m)], ["desired speed (m/s)", f(fo.desired_speed_mps)], ["heading error (°)", f(deg(fo.heading_error_rad), 1)], ["depth error (m)", f(fo.depth_error_m, 2)]]);
  rows("avoidTable", [["avoidance state", av.state], ["maneuvers / retries", `${av.count} / ${av.retries}`], ["last detour ok", av.last_detour_ok === null ? DASH : `${av.last_detour_ok}${av.last_detour_reason ? " (" + av.last_detour_reason + ")" : ""}`],
    ["detour length (m)", f(av.detour_length_m)], ["rejoin point", av.rejoin_point ? av.rejoin_point.map((v) => f(v, 1)).join(", ") : DASH],
    ["sonar forward clearance (m)", f(av.sonar_front_clearance_m)], ["sonar speed scale", f(av.speed_scale, 2)]]);
  rows("replanTable", [["replans (budget " + rp.budget + ")", rp.count], ["last reason", rp.last_reason || DASH], ["reasons", Object.entries(rp.reasons || {}).map(([k, v]) => `${k}×${v}`).join(", ") || DASH],
    ["throttled (rate limit)", rp.throttled], ["goal changes", rp.goal_changes], ["path validations / blocked / map-change", `${rp.validations} / ${rp.blocked_events} / ${rp.map_change_events}`]]);
  $("navHistory").innerHTML = (n.history || []).slice().reverse().map((h) => `<li>t=${f(h.t, 1)} ${h.from} → ${h.to} <em>${h.reason}</em></li>`).join("");
  const sel = $("selPolicyLive"); if (document.activeElement !== sel && pl.unknown_policy) sel.value = pl.unknown_policy;
}
function drawNavOverlay(ctx, p, canvas) {
  if (!snap || !snap.navigation || !snap.navigation.enabled || !$("chkPath").checked) return;
  const np = snap.navigation_path || {};
  drawPath(ctx, p, canvas, np.global, "#1f9d55", np.detour ? [4, 3] : []);
  drawPath(ctx, p, canvas, np.detour, "#e07b00");
  const mark = (pt, color, r) => { if (!pt) return; const [x, y] = toPx(p, pt[0], pt[1], canvas); ctx.save(); ctx.fillStyle = color; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, r, 0, 7); ctx.fill(); ctx.stroke(); ctx.restore(); };
  mark(np.goal, "#d6246e", 6); mark(np.waypoint, "#f2c200", 4);
  if (np.goal) {                                                        // the goal numbers, next to the goal marker
    const [gx, gy] = toPx(p, np.goal[0], np.goal[1], canvas);
    const label = `${goalReachedNow() ? "GOAL REACHED" : "goal"} ${goalCoordsText(np.goal, 1)} (x, y, depth m)`;
    ctx.save(); ctx.font = "bold 11px ui-monospace, monospace"; ctx.textBaseline = "bottom";
    ctx.strokeStyle = "rgba(255,255,255,.9)"; ctx.lineWidth = 3; ctx.strokeText(label, gx + 8, gy - 5);
    ctx.fillStyle = "#d6246e"; ctx.fillText(label, gx + 8, gy - 5); ctx.restore();
  }
  const rp = snap.navigation.avoidance && snap.navigation.avoidance.rejoin_point; if (snap.navigation.avoidance && snap.navigation.avoidance.active) mark(rp, "#e07b00", 4);
}
function clearPanels() {
  ["navTable", "ekfTable", "adaptTable", "adaptTrackTable", "truthTable", "sonarTable", "mapTable", "costTable", "envTable", "plannerTable", "followTable", "avoidTable", "replanTable", "navHistory"].forEach((id) => $(id).innerHTML = "");
  $("navState").textContent = DASH; $("navState").className = "state s-idle";
  $("sensorTable").innerHTML = ""; $("thrusters").textContent = status && status.state === "IDLE" ? "Start the simulation to see live data." : "No telemetry.";
  rows("missionTable", [["mission", status && status.state === "IDLE" ? "no run yet - start a simulation" : "no telemetry"]]);
  ["simTime", "dynMode", "pace", "scenario"].forEach((id) => $(id).textContent = DASH);
  ["occCanvas", "costCanvas", "sonarCanvas"].forEach((id) => { const c = $(id); c.getContext("2d").clearRect(0, 0, c.width, c.height); });
}
function renderThrusters(t) {
  const el = $("thrusters");
  if (!t || !t.available) { el.textContent = (t && t.reason) || "No thruster data."; return; }
  let h = '<div class="thr"><b>#</b><b>force / max</b><b>cmd</b><b>fault</b>';
  for (let i = 0; i < t.count; i++) {
    const frac = Math.min(1, Math.abs(t.force_n[i]) / (t.max_thrust_n[i] || 1));
    h += `<span>${i}</span><div class="bar-h"><b style="left:0;width:${(frac * 100).toFixed(0)}%"></b></div><span>${f(t.command[i], 2)}</span><span>${t.fault[i]}</span>`;
  }
  h += "</div>";
  h += `<p class="note">saturated: ${t.saturated === null ? DASH : t.saturated}; allocation residual: ${f(t.allocation_residual_norm, 3)}</p>`;
  el.innerHTML = h;
}

/* ---------------------------------------------------------------- mission statistics + runs */
const firstDefined = (...v) => { for (const x of v) if (x !== null && x !== undefined) return x; return null; };
function renderMission(st) {
  const res = (status && status.result) || {}, met = res.metrics || {};
  const nv = snap.navigation || {}, gt = snap.debug_ground_truth, ctl = snap.controller || {};
  // Configured mission geometry (from the running simulation's own telemetry) against the live estimate, so the
  // operator sees the start, the goal and the current pose side by side - all three in x / y / depth.
  const ms = snap.mission || {};
  const est0 = snap.estimate || {};
  const cur = (est0.available && Array.isArray(est0.position) && est0.position.length === 3) ? est0.position : null;
  const xyd = (v) => (Array.isArray(v) && v.length === 3) ? `${f(v[0], 2)} / ${f(v[1], 2)} / ${f(-v[2], 2)}` : DASH;
  const goalDist = (nv.enabled && nv.goal_distance_m != null) ? nv.goal_distance_m
    : (cur && Array.isArray(ms.goal) && ms.goal.length === 3
        ? Math.hypot(cur[0] - ms.goal[0], cur[1] - ms.goal[1], cur[2] - ms.goal[2]) : null);
  let completion;
  if (st === "RUNNING" || st === "STALE" || st === "STARTING" || st === "STOPPING") completion = "in progress";
  else if (st === "COMPLETED") completion = "goal reached (run status)";
  else if (st === "ENDED_BEFORE_GOAL") completion = "ended before the goal (step limit)";
  else if (st === "FAILED") completion = "failed";
  else if (st === "STOPPED") completion = "stopped by the operator";
  else completion = "no run";
  if (res.goal_reached !== undefined && res.goal_reached !== null)
    completion += res.goal_reached ? " · truth evaluation: goal reached" : " · truth evaluation: goal not reached";
  const coll = firstDefined(met.collision_count, ctl.collisions, gt && gt.collisions);
  const cov = (snap.mapping || {}).coverage_percent;
  const rc = (nv.replanning || {}).count, ac = (nv.avoidance || {}).count;
  const dbg = (v) => `<b class="debug">EVALUATION</b> ${v}`;
  const goalDone = st === "COMPLETED" || (nv.enabled && nv.state === "GOAL_REACHED");
  rows("missionTable", [
    ["run state", st],
    ["start x / y / depth (m), configured", xyd(ms.start)],
    ["current x / y / depth (m), EKF estimate", cur ? `${f(cur[0], 2)} / ${f(cur[1], 2)} / ${f(est0.depth, 2)}` : DASH],
    ["goal x / y / depth (m), configured", xyd(ms.goal)],
    ["goal you entered x, y, depth (m)", goalCoordsText(enteredGoal() || reachedGoal())],
    ["goal reached — the goal you set (x, y, depth m)", goalDone ? goalCoordsText(reachedGoal()) : DASH],
    ["mission duration, simulated (s)", snap.sim_time == null ? DASH : f(snap.sim_time, 1)],
    ["distance travelled (m, EKF estimate)", f(missionStats.dist, 1)],
    ["distance travelled (m, truth path, whole run)", met.trajectory_length_m == null ? DASH : dbg(f(met.trajectory_length_m, 1))],
    ["completion", completion],
    ["collisions", coll == null ? DASH : dbg(String(coll))],
    ["goal distance (m, EKF estimate, to the configured goal)", goalDist == null ? DASH : f(goalDist, 2)],
    ["goal distance (m, ground-truth, evaluation only)", (gt && gt.goal_distance_true != null) ? dbg(f(gt.goal_distance_true, 2)) : DASH],
    ["mapping coverage (%)", cov == null ? DASH : f(cov, 2)],
    ["replans / avoidance maneuvers", nv.enabled ? `${f(rc, 0)} / ${f(ac, 0)}` : DASH],
    ["final position error (m)", (gt && gt.estimate_error_norm != null) ? dbg(f(gt.estimate_error_norm, 3)) : DASH],
  ]);
}

async function loadRuns() {
  const r = await api("GET", "/api/runs");
  if (!r.ok) return;
  runs = Array.isArray(r.data.runs) ? r.data.runs : [];
  const opts = runs.map((run) => {
    const nFaults = (run.faults || []).length;
    const o = document.createElement("option");
    o.value = run.run_id;
    o.textContent = `${run.run_id} · ${run.state || "?"} · ${nFaults ? nFaults + " fault" + (nFaults > 1 ? "s" : "") : "baseline"}`;
    return o;
  });
  [["selRunA", "A"], ["selRunB", "B"]].forEach(([id, which]) => {
    const sel = $(id), prev = sel.value;
    sel.innerHTML = "";
    opts.forEach((o) => sel.appendChild(o.cloneNode(true)));
    if (prev && runs.some((run) => run.run_id === prev)) sel.value = prev;
    else if (runs.length) {                        // default: A = newest baseline (no faults), B = newest fault run (or newest)
      const base = runs.find((run) => !(run.faults || []).length) || runs[0];
      const faulty = runs.find((run) => (run.faults || []).length) || runs[0];
      sel.value = which === "A" ? base.run_id : faulty.run_id;
    }
  });
  renderRunsTable();
  renderCompare();
}

function runById(id) { return runs.find((run) => run.run_id === id) || null; }

function renderRunsTable() {
  const t = $("runsTable");
  if (!runs.length) { t.innerHTML = "<tr><td colspan=\"8\">No finished runs yet.</td></tr>"; return; }
  let h = "<tr><th>run</th><th>state</th><th>scenario</th><th>duration (s)</th><th>distance est. (m)</th><th>collisions</th><th>coverage (%)</th><th>sensor faults</th></tr>";
  runs.forEach((run) => {
    const fl = (run.faults || []).map((x) => `${x.sensor}:${String(x.health).toLowerCase()}@${f(x.sim_time, 1)}s (${x.source})`).join(", ");
    const isBase = !(run.faults || []).length;
    h += `<tr class="${isBase ? "baseline" : "faulted"}"><td>${run.run_id}${run.live ? " (live)" : ""}</td><td>${run.state || DASH}</td>` +
      `<td>${run.scenario || DASH}</td><td>${f(run.duration_sim_s, 1)}</td><td>${f(run.distance_estimated_m, 1)}</td>` +
      `<td>${run.collisions == null ? DASH : run.collisions}</td><td>${f(run.coverage_percent, 1)}</td><td>${fl || (isBase ? "none (baseline)" : DASH)}</td></tr>`;
  });
  t.innerHTML = h;
}

function renderCompare() {
  const a = runById($("selRunA").value), b = runById($("selRunB").value), t = $("compareTable");
  if (!a || !b) { t.innerHTML = "<tr><td>Select two runs to compare them.</td></tr>"; return; }
  const rowsDef = [
    ["state", (r) => r.state || DASH],
    ["scenario / seed", (r) => `${r.scenario || DASH} / ${r.seed == null ? DASH : r.seed}`],
    ["dynamics", (r) => r.dynamics_mode || DASH],
    ["duration, simulated (s)", (r) => r.duration_sim_s],
    ["distance travelled, estimated (m)", (r) => r.distance_estimated_m],
    ["distance travelled, truth path (m)", (r) => r.distance_truth_m],
    ["collisions", (r) => r.collisions],
    ["goal reached", (r) => r.goal_reached == null ? DASH : (r.goal_reached ? "yes" : "no")],
    ["final goal distance (m)", (r) => r.final_goal_distance_m],
    ["mapping coverage (%)", (r) => r.coverage_percent],
    ["replans / avoidance", (r) => r.replans == null ? DASH : `${r.replans} / ${r.avoidance}`],
    ["sensor faults", (r) => { const fl = r.faults || []; return fl.length ? fl.map((x) => `${x.sensor} ${String(x.health).toLowerCase()} @ ${f(x.sim_time, 1)} s (${x.source})`).join(", ") : "none (baseline)"; }],
  ];
  const fmt = (v) => (v == null ? DASH : (typeof v === "number" ? f(v, typeof v === "number" && Number.isInteger(v) ? 0 : 2) : v));
  let h = "<tr><td>metric</td><td>A: " + a.run_id + "</td><td>B: " + b.run_id + "</td><td>B − A</td></tr>";
  rowsDef.forEach(([label, get]) => {
    const va = get(a), vb = get(b);
    let delta = DASH;
    if (typeof va === "number" && typeof vb === "number" && va != null && vb != null) {
      const d = vb - va;
      delta = `<span class="${d > 0 ? "delta-pos" : d < 0 ? "delta-neg" : ""}">${f(d, 2)}</span>`;
    }
    h += `<tr><td>${label}</td><td>${fmt(va)}</td><td>${fmt(vb)}</td><td>${delta}</td></tr>`;
  });
  t.innerHTML = h;
}

/* ---------------------------------------------------------------- canvases */
function thrustDefs() {
  const out = [];
  for (let i = 0; i < thrustCount; i++) out.push({ label: "#" + i, color: THR_COLORS[i % THR_COLORS.length], getSeries: () => S.thr[i] || [] });
  return out;
}
function rle(data, n) { const out = new Uint8Array(n); let p = 0; for (let i = 0; i < data.length; i += 2) { out.fill(data[i], p, p + data[i + 1]); p += data[i + 1]; } return out; }
const OCC_RGB = [[68, 82, 92], [220, 236, 241], [201, 64, 45]];                 // unknown, free, occupied
const COST_CLS = { 0: null, 1: [68, 82, 92], 2: [201, 64, 45], 3: [233, 154, 61] };
function paintGrid(canvas, p, colorFn) {
  const [rows_, cols] = p.shape; const img = new ImageData(cols, rows_);
  for (let r = 0; r < rows_; r++) for (let c = 0; c < cols; c++) {
    const rgb = colorFn(r * cols + c); const o = ((rows_ - 1 - r) * cols + c) * 4;    // flip: row grows with +Y
    img.data[o] = rgb[0]; img.data[o + 1] = rgb[1]; img.data[o + 2] = rgb[2]; img.data[o + 3] = 255;
  }
  const off = document.createElement("canvas"); off.width = cols; off.height = rows_; off.getContext("2d").putImageData(img, 0, 0);
  canvas.width = cols * 2; canvas.height = rows_ * 2;
  const ctx = canvas.getContext("2d"); ctx.imageSmoothingEnabled = false; ctx.drawImage(off, 0, 0, canvas.width, canvas.height);
  return ctx;
}
function toPx(p, x, y, canvas) { const k = canvas.width / p.shape[1]; return [((x - p.origin[0]) / p.resolution) * k, canvas.height - ((y - p.origin[1]) / p.resolution) * k]; }
function drawPath(ctx, p, canvas, pts, style, dash) {
  if (!pts || pts.length < 2) return; ctx.save(); ctx.strokeStyle = style; ctx.lineWidth = 2; ctx.setLineDash(dash || []); ctx.beginPath();
  pts.forEach((q, i) => { const [x, y] = toPx(p, q[0], q[1], canvas); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke(); ctx.restore();
}
function drawVehicle(ctx, p, canvas) {
  const e = snap && snap.estimate; if (!e || !e.available) return;
  const [x, y] = toPx(p, e.position[0], e.position[1], canvas); const yaw = e.euler_rad[2];
  ctx.save(); ctx.translate(x, y); ctx.rotate(-yaw); ctx.fillStyle = "#0f3d52"; ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(10, 0); ctx.lineTo(-7, 6); ctx.lineTo(-7, -6); ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.restore();
}
function drawRays(ctx, p, canvas) {
  const so = snap && snap.sonar, e = snap && snap.estimate; if (!so || !so.available || !e || !e.available || !$("chkRays").checked) return;
  const yaw = e.euler_rad[2]; const mp = so.mount_position_m || [0, 0, 0];
  const ox = e.position[0] + Math.cos(yaw) * mp[0] - Math.sin(yaw) * mp[1], oy = e.position[1] + Math.sin(yaw) * mp[0] + Math.cos(yaw) * mp[1];
  ctx.save(); ctx.lineWidth = 1;
  so.azimuth_deg.forEach((azd, i) => {
    const st = so.status[i]; if (st < 0) return;                       // invalid ray: no information, draw nothing
    const r = st === 1 ? so.range_m[i] : so.max_range_m; if (r === null) return;
    const a = yaw + azd * Math.PI / 180; const [x0, y0] = toPx(p, ox, oy, canvas), [x1, y1] = toPx(p, ox + r * Math.cos(a), oy + r * Math.sin(a), canvas);
    ctx.strokeStyle = st === 1 ? "rgba(37,112,143,.35)" : "rgba(37,112,143,.12)"; ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke();
    if (st === 1) { ctx.fillStyle = "#b83a2c"; ctx.fillRect(x1 - 1.5, y1 - 1.5, 3, 3); }
  });
  ctx.restore();
}
function drawMaps() {
  if (occ && !occ.unchanged && occ.data) {
    const cells = rle(occ.data, occ.shape[0] * occ.shape[1]);
    const ctx = paintGrid($("occCanvas"), occ, (i) => OCC_RGB[cells[i]]);
    const c = $("occCanvas");
    if (snap) { drawPath(ctx, occ, c, snap.trajectory_estimated, "#25708f"); 
      if ($("chkTruth").checked && snap.debug_ground_truth) drawPath(ctx, occ, c, snap.debug_ground_truth.trajectory, "#b7791f", [6, 4]);
      drawNavOverlay(ctx, occ, c); drawRays(ctx, occ, c); drawVehicle(ctx, occ, c); }
    $("occLegend").innerHTML = `<div><i style="background:rgb(${OCC_RGB[0]})"></i>unknown</div><div><i style="background:rgb(${OCC_RGB[1]})"></i>free</div><div><i style="background:rgb(${OCC_RGB[2]})"></i>occupied</div>
      <div><i style="background:#25708f"></i>EKF path</div><div><i style="background:#1f9d55"></i>planned path</div><div><i style="background:#e07b00"></i>avoidance detour</div><div><i style="background:#f2c200"></i>current waypoint</div><div><i style="background:#d6246e"></i>goal</div>${$("chkTruth").checked ? '<div><i style="background:#b7791f"></i>truth path (DEBUG ONLY)</div>' : ""}`;
  }
  if (cost && !cost.unchanged && cost.classes) {
    const n = cost.shape[0] * cost.shape[1]; const cls = rle(cost.classes, n), lvl = rle(cost.level, n);
    const ctx = paintGrid($("costCanvas"), cost, (i) => { const c = COST_CLS[cls[i]]; if (c) return c; const t = lvl[i] / 15; return [Math.round(235 - 60 * t), Math.round(243 - 90 * t), Math.round(245 - 150 * t)]; });
    if (snap) { drawPath(ctx, cost, $("costCanvas"), snap.trajectory_estimated, "#25708f"); drawNavOverlay(ctx, cost, $("costCanvas")); drawVehicle(ctx, cost, $("costCanvas")); }
    $("costLegend").innerHTML = `<div><i style="background:rgb(235,243,245)"></i>free, low cost</div><div><i style="background:rgb(175,153,95)"></i>free, higher cost (clearance)</div>
      <div><i style="background:rgb(68,82,92)"></i>unknown (policy: ${cost.stats ? cost.stats.unknown_policy : DASH})</div><div><i style="background:rgb(233,154,61)"></i>inflated obstacle</div><div><i style="background:rgb(201,64,45)"></i>occupied</div>`;
  }
}
function drawSonar(so) {
  const c = $("sonarCanvas"), ctx = c.getContext("2d"); ctx.clearRect(0, 0, c.width, c.height);
  if (!so || !so.available) return;
  const ox = c.width / 2, oy = c.height - 8, R = c.height - 22, k = R / so.max_range_m;
  ctx.strokeStyle = "#c9d3d7"; ctx.fillStyle = "#5c6f79"; ctx.font = "10px sans-serif";
  [0.5, 1].forEach((fr) => { ctx.beginPath(); ctx.arc(ox, oy, R * fr, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke(); ctx.fillText(f(so.max_range_m * fr, 0) + " m", ox + 4, oy - R * fr + 10); });
  so.azimuth_deg.forEach((azd, i) => {
    const st = so.status[i], a = azd * Math.PI / 180;
    if (st < 0) { ctx.strokeStyle = "#b83a2c"; ctx.beginPath(); ctx.moveTo(ox + 12 * Math.sin(a), oy - 12 * Math.cos(a)); ctx.lineTo(ox + 20 * Math.sin(a), oy - 20 * Math.cos(a)); ctx.stroke(); return; }
    const r = st === 1 ? so.range_m[i] : so.max_range_m;
    ctx.strokeStyle = st === 1 ? "rgba(37,112,143,.55)" : "rgba(37,112,143,.12)"; ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + r * k * Math.sin(a), oy - r * k * Math.cos(a)); ctx.stroke();
    if (st === 1) { ctx.fillStyle = "#b83a2c"; ctx.fillRect(ox + r * k * Math.sin(a) - 2, oy - r * k * Math.cos(a) - 2, 4, 4); }
  });
  ctx.fillStyle = "#5c6f79"; ctx.fillText("red tick near origin = invalid ray (no information)", 6, 12);
}
/* ---------------------------------------------------------------- boot */
/* The live data path starts FIRST. Polling status/telemetry, the sensor table, the environment and the run list are
   what make the console live, and they must come up even if the chart layer cannot start (see the try/catch below). */
buildFaultTable(); loadEnvironment(true); loadRuns(); poll();
setInterval(poll, 400);
setInterval(pollMaps, 1000);
/* Live telemetry charts (rendering only: charts.js draws the series accumulated above from real snapshots).
   Exactly 20 charts, each registered once, each fed from a real telemetry field (see docs/dashboard.md). */
const LIVE_CHARTS = {
  getStatus: () => status, windowSelectId: "selWindow", badgeId: "liveBadge", badgeTextId: "liveText", ageId: "liveAge",
  charts: [
    { canvasId: "errChart", nowId: "errNow", statsId: "errStats", getSeries: () => errSeries, unit: "m", axisTitle: "Position error (m)", color: "#b7791f", decimals: 3, yMin: 0, minSpan: 0.1 },
    { canvasId: "covChart", nowId: "covNow", statsId: "covStats", getSeries: () => covSeries, unit: "%", axisTitle: "Cells known (%)", color: "#25708f", decimals: 1, yMin: 0, yMax: 100, minSpan: 5 },
    { canvasId: "goalChart", nowId: "goalNow", statsId: "goalStats", getSeries: () => S.goal, unit: "m", axisTitle: "Goal distance (m)", color: "#2b6cb0", decimals: 2, yMin: 0, minSpan: 1, getEvents: () => navEvents },
    { canvasId: "speedChart", nowId: "speedNow", statsId: "speedStats", getSeries: () => S.speed, unit: "m/s", axisTitle: "Speed (m/s)", color: "#2f855a", decimals: 2, yMin: 0, minSpan: 0.2, getEvents: () => navEvents },
    { canvasId: "depthChart", nowId: "depthNow", statsId: "depthStats", getSeries: () => S.depth, unit: "m", axisTitle: "Depth (m, + down)", color: "#0987a0", decimals: 2, minSpan: 0.2 },
    { canvasId: "yawChart", nowId: "yawNow", statsId: "yawStats", getSeries: () => S.yaw, unit: "deg", axisTitle: "Heading (deg)", color: "#805ad5", decimals: 1, minSpan: 10, getEvents: () => navEvents },
    // sonar-fed: while the sonar is failed (or its scan carries no valid ray) no current range is shown anywhere
    { canvasId: "clrChart", nowId: "clrNow", statsId: "clrStats", getSeries: () => S.clr, unit: "m", axisTitle: "Nearest sonar hit (m)", color: "#c05621", decimals: 2, yMin: 0, minSpan: 1, getEvents: () => navEvents,
      sourceOk: () => sonarFresh(), sourceMsg: () => sonarNoDataMsg() },
    { canvasId: "rplChart", nowId: "rplNow", statsId: "rplStats", getSeries: () => S.rpl, unit: "", axisTitle: "Replans + avoidance (cumulative)", color: "#b7791f", decimals: 0, yMin: 0, minSpan: 2, getEvents: () => navEvents },
    { canvasId: "rollChart", nowId: "rollNow", statsId: "rollStats", getSeries: () => S.roll, unit: "deg", axisTitle: "Roll, EKF estimate (deg)", color: "#c05621", decimals: 2, minSpan: 5, getEvents: () => navEvents },
    { canvasId: "pitchChart", nowId: "pitchNow", statsId: "pitchStats", getSeries: () => S.pitch, unit: "deg", axisTitle: "Pitch, EKF estimate (deg)", color: "#dd6b20", decimals: 2, minSpan: 5, getEvents: () => navEvents },
    { canvasId: "estXChart", nowId: "estXNow", statsId: "estXStats", getSeries: () => S.estx, unit: "m", axisTitle: "X position, EKF estimate (m)", color: "#25708f", decimals: 2, minSpan: 2 },
    { canvasId: "estYChart", nowId: "estYNow", statsId: "estYStats", getSeries: () => S.esty, unit: "m", axisTitle: "Y position, EKF estimate (m)", color: "#2f855a", decimals: 2, minSpan: 2 },
    { canvasId: "estZChart", nowId: "estZNow", statsId: "estZStats", getSeries: () => S.estz, unit: "m", axisTitle: "Z position, EKF estimate (m, +up)", color: "#805ad5", decimals: 2, minSpan: 2 },
    { canvasId: "velChart", nowId: "velNow", statsId: "velStats", unit: "m/s", axisTitle: "Velocity, world (m/s)", decimals: 2, minSpan: 0.1,
      series: [
        { label: "X", color: AXIS_COLORS[0], getSeries: () => S.velx },
        { label: "Y", color: AXIS_COLORS[1], getSeries: () => S.vely },
        { label: "Z", color: AXIS_COLORS[2], getSeries: () => S.velz }],
      getEvents: () => navEvents },
    // IMU-fed (raw sensor output): a failed IMU shows no acceleration value at all, only why there is none
    { canvasId: "imuAccChart", nowId: "imuAccNow", statsId: "imuAccStats", unit: "m/s²", axisTitle: "IMU specific force, body (m/s²)", decimals: 2, minSpan: 1,
      emptyMsg: (info) => (snap && snap.sensors_enabled && snap.sensors && snap.sensors.imu && snap.sensors.imu.samples > 0)
        ? "No valid IMU sample yet (IMU failed)" : info.emptyMsg,
      sourceOk: () => sensorFresh("imu"), sourceMsg: () => sensorNoDataMsg("imu"),
      series: [
        { label: "X", color: "#b83a2c", getSeries: () => S.imuax },
        { label: "Y", color: "#2f855a", getSeries: () => S.imuay },
        { label: "Z", color: "#2b6cb0", getSeries: () => S.imuaz }] },
    { canvasId: "angVelChart", nowId: "angVelNow", statsId: "angVelStats", unit: "rad/s", axisTitle: "Angular velocity, body (rad/s)", decimals: 3, minSpan: 0.05,
      series: [
        { label: "X", color: "#805ad5", getSeries: () => S.angx },
        { label: "Y", color: "#c05621", getSeries: () => S.angy },
        { label: "Z", color: "#25708f", getSeries: () => S.angz }] },
    { canvasId: "healthChart", nowId: "healthNow", statsId: "healthStats", unit: "", axisTitle: "Sensor state (0-4)", decimals: 0, minSpan: 2,
      yMin: 0, yMax: 4, yTicks: [0, 1, 2, 3, 4],
      yTickLabels: { 0: "HEALTHY", 1: "DEGRADED", 2: "FAILED", 3: "DISCARDED", 4: "LINK CUT" }, step: true, marginLeft: 80,
      series: [
        { label: "imu", color: HEALTH_COLORS.imu, getSeries: () => S.hlImu },
        { label: "dvl", color: HEALTH_COLORS.dvl, getSeries: () => S.hlDvl },
        { label: "depth", color: HEALTH_COLORS.depth, getSeries: () => S.hlDepth },
        { label: "sonar", color: HEALTH_COLORS.sonar, getSeries: () => S.hlSonar },
        { label: "compass", color: HEALTH_COLORS.compass, getSeries: () => S.hlCompass }] },
    { canvasId: "thrustChart", nowId: "thrustNow", statsId: "thrustStats", unit: "N", axisTitle: "Thruster force, actual (N)", decimals: 1, minSpan: 1,
      emptyMsg: (info) => thrustCount ? info.emptyMsg : ((snap && snap.thrusters && snap.thrusters.reason) || "No thruster layer in this run (legacy dynamics)"),
      series: () => thrustDefs() },
    // Gain multipliers (unitless): the ADAPTIVE P controller's gain schedule / self-tuning / active / yaw gain, or -
    // when autonomous navigation is driving - the PHASE-5 PATH FOLLOWER's speed schedule and the factors that shape
    // it. One card, fed by whichever controller is really producing the commands (see gainSource()).
    { canvasId: "adaptChart", nowId: "adaptNow", statsId: "adaptStats", unit: "×", axisTitle: "Gain multiplier (×)", color: "#b7791f", decimals: 2, yMin: 0, minSpan: 0.5,
      emptyMsg: () => gainSource() ? "Waiting for the first gain sample…"
        : ((snap && snap.controller && snap.controller.type) ? `No gains reported by ${snap.controller.type}` : "No gain data yet (nothing is driving)"),
      getEvents: () => adaptEvents,
      series: () => ((gainSource() || {}).kind === "follower"
        ? [
          { label: "speed", color: "#2b6cb0", getSeries: () => S.fspeed },
          { label: "goal dist", color: "#2f855a", getSeries: () => S.fgoal },
          { label: "heading", color: "#b7791f", getSeries: () => S.fhead },
          { label: "sonar+avoid", color: "#c05621", getSeries: () => S.fsonar }]
        : [
          { label: "schedule", color: "#2b6cb0", getSeries: () => S.kpsched },
          { label: "self-tuning", color: "#2f855a", getSeries: () => S.ktune },
          { label: "active", color: "#b7791f", getSeries: () => S.kpmult },
          { label: "yaw", color: "#805ad5", getSeries: () => S.kyaw }]) },
    // Tracked gain history, in simulated seconds: cumulative time the ADAPTIVE controller spent ramping / boosted / at
    // the gain cap, or - for the follower - at full speed / slowed / braked.
    { canvasId: "adaptTrackChart", nowId: "adaptTrackNow", statsId: "adaptTrackStats", unit: "s", axisTitle: "Simulated time in mode (s)", color: "#2f855a", decimals: 1, yMin: 0, minSpan: 1,
      emptyMsg: () => gainSource() ? "Waiting for the first tracked sample…"
        : ((snap && snap.controller && snap.controller.type) ? `No tracked gains reported by ${snap.controller.type}` : "No tracked gain history yet (nothing is driving)"),
      series: () => ((gainSource() || {}).kind === "follower"
        ? [
          { label: "full speed", color: "#2f855a", getSeries: () => S.ffull },
          { label: "slowed", color: "#b7791f", getSeries: () => S.fslow },
          { label: "braked", color: "#c53030", getSeries: () => S.fbrake }]
        : [
          { label: "ramping", color: "#2f855a", getSeries: () => S.kramp },
          { label: "boosted", color: "#b7791f", getSeries: () => S.kboost },
          { label: "at cap", color: "#c53030", getSeries: () => S.kcap }]) },
  ],
};
/* A chart layer that cannot start must not stop the console: the numbers below keep updating and the banner says so. */
try { createLiveCharts(LIVE_CHARTS); }
catch (err) {
  console.error("the charts could not start:", err);
  showBanner(`the charts could not start (${err.message}) — reload the page (Ctrl+F5). The live panel data keeps updating.`, false);
}
